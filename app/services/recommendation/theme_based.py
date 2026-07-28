import asyncio
from typing import Any

from loguru import logger

from app.models.taste_profile import TasteProfile
from app.services.profile.constants import (
    RUNTIME_BUCKET_MEDIUM_MAX_MOVIE,
    RUNTIME_BUCKET_MEDIUM_MAX_SERIES,
    RUNTIME_BUCKET_SHORT_MAX_MOVIE,
    RUNTIME_BUCKET_SHORT_MAX_SERIES,
)
from app.services.profile.scorer import ProfileScorer
from app.services.recommendation.filtering import RecommendationFiltering
from app.services.recommendation.metadata import RecommendationMetadata
from app.services.recommendation.scoring import RecommendationScoring
from app.services.recommendation.utils import (
    apply_discover_filters,
    content_type_to_mtype,
    filter_by_genres,
    filter_watched_by_imdb,
)
from app.services.tmdb.service import TMDBService


class ThemeBasedService:
    """
    Handles theme-based recommendations using role-based axis recipes.

    Strategy:
    1. Parse role-based theme ID (a: anchor, f: flavor, b: fallback)
    2. Primary discovery using Anchors
    3. Weighted scoring: anchor (1.0) + flavor (0.7) + fallback (0.3)
    4. Profile-aware ranking
    5. Expansion logic if results are sparse
    """

    def __init__(self, tmdb_service: Any, user_settings: Any = None):
        self.tmdb_service: TMDBService = tmdb_service
        self.user_settings = user_settings
        self.scorer = ProfileScorer()

    async def get_recommendations_for_theme(
        self,
        theme_id: str,
        content_type: str,
        profile: TasteProfile | None = None,
        watched_tmdb: set[int] | None = None,
        watched_imdb: set[str] | None = None,
        limit: int = 20,
        whitelist: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Get recommendations for a role-based theme."""
        watched_tmdb = watched_tmdb or set()
        watched_imdb = watched_imdb or set()

        # 1. Parse roles and values
        anchors, flavors, fallbacks = self._parse_theme_id(theme_id)

        # 2. Prepare common excluded genres
        excluded_ids = RecommendationFiltering.get_excluded_genre_ids(self.user_settings, content_type)

        # 3. Extract mandatory filters (country/era from ANY role)
        # Union the roles. A plain dict merge let the LEAST important role (fallback)
        # overwrite the anchor for any shared axis, so the defining constraint of the
        # row was dropped before the query was built.
        all_constraints: dict = {}
        for _role in (anchors, flavors, fallbacks):
            for _k, _v in _role.items():
                _bucket = all_constraints.setdefault(_k, [])
                for _x in (_v if isinstance(_v, list) else [_v]):
                    if _x not in _bucket:
                        _bucket.append(_x)
        mandatory_filters = {}
        if "country" in all_constraints:
            mandatory_filters["country"] = all_constraints["country"]
        if "era" in all_constraints:
            mandatory_filters["era"] = all_constraints["era"]


        logger.info(f"Theme discovery for {theme_id}: anchors={anchors}, flavors={flavors}, fallbacks={fallbacks}")

        # ====================
        # PHASE 1: Combined Fetch (ALL constraints)
        # ====================
        fetch_tasks = []
        if all_constraints:
            combined_params = self._axes_to_params(all_constraints, content_type)
            if excluded_ids:
                with_ids = {int(g) for g in combined_params.get("with_genres", "").split("|") if g}
                without = [g for g in excluded_ids if g not in with_ids]
                if without:
                    combined_params["without_genres"] = "|".join(str(g) for g in without)
            fetch_tasks.append(self._fetch_discover_candidates(content_type, combined_params, pages=[1, 2, 3]))

        # Execute Phase 1
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        candidates = []
        for res in results:
            if isinstance(res, Exception):
                logger.debug(f"Error fetching combined: {res}")
                continue
            if isinstance(res, list):
                for item in res:
                    item["_discovery_tier"] = "combined"
                candidates.extend(res)

        logger.info(f"Phase 1 (combined): {len(candidates)} candidates")

        # ====================
        # PHASE 2: Individual Axes (if sparse)
        # ====================
        # Skip Phase 2 if anchor is a keyword — individual axis queries would dilute
        # the keyword constraint and return off-theme results (e.g. non-remakes in "Sci-Fi Remakes")
        # Any keyword in the recipe -- not just an anchor keyword -- makes per-axis
        # queries unsafe: they fetch items that match ONE axis and ignore the keyword
        # entirely, which is how unrelated titles reached keyword-named rows.
        has_keyword_anchor = "keyword" in all_constraints
        if not has_keyword_anchor and len(candidates) < limit * 2:
            fetch_tasks = []

            # Fire individual queries for all axes, but for flavor genre axes
            # force anchor genres as required filters so results stay on-theme
            anchor_genre_ids = [v for k, v in anchors.items() if k == "genre"]
            anchor_genres_str = "|".join(str(g) for g in anchor_genre_ids) if anchor_genre_ids else None

            for axis_name, axis_value in all_constraints.items():
                # Build params for this single axis
                params = self._axes_to_params({axis_name: axis_value}, content_type)

                # For flavor/fallback genre queries, pin anchor genres so the pool stays distinct
                if axis_name == "genre" and axis_value not in anchor_genre_ids and anchor_genres_str:
                    # Require at least one anchor genre alongside this flavor genre
                    existing = params.get("with_genres", "")
                    params["with_genres"] = f"{anchor_genres_str}|{existing}" if existing else anchor_genres_str

                # ALWAYS add mandatory filters (country/era) if they exist and are not the current axis
                for filter_name, filter_value in mandatory_filters.items():
                    if filter_name != axis_name:  # Don't duplicate
                        filter_params = self._axes_to_params({filter_name: filter_value}, content_type)
                        params.update(filter_params)

                # Apply excluded genres
                if excluded_ids:
                    with_ids = {int(g) for g in params.get("with_genres", "").split("|") if g}
                    without = [g for g in excluded_ids if g not in with_ids]
                    if without:
                        params["without_genres"] = "|".join(str(g) for g in without)

                fetch_tasks.append(self._fetch_discover_candidates(content_type, params, pages=[1, 2, 3, 4]))

            # Execute Phase 2
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.debug(f"Error fetching individual: {res}")
                    continue
                if isinstance(res, list):
                    for item in res:
                        item["_discovery_tier"] = "individual"
                    candidates.extend(res)

            logger.info(f"Phase 2 (individual): Total {len(candidates)} candidates")

        # 4. Expansion Logic if still sparse
        if len(candidates) < limit and anchors:
            # Expansion strategy: Relax constraints on the primary anchor
            primary_axis = dict([next(iter(anchors.items()))])
            base_p = self._axes_to_params(primary_axis, content_type)
            expanded = await self._expand_search(content_type, base_p, anchors, flavors)
            for item in expanded:
                item["_discovery_tier"] = "expanded"
            candidates.extend(expanded)

        # 5. Weighted Scoring
        scored = []
        rotation_seed = RecommendationScoring.generate_rotation_seed()
        mtype = content_type_to_mtype(content_type)

        for item in candidates:
            # Theme Match Score
            theme_match = self._calculate_theme_score(item, anchors, flavors, fallbacks)

            # Profile & Quality Score
            if profile:
                base_score = RecommendationScoring.calculate_final_score(
                    item=item,
                    profile=profile,
                    scorer=self.scorer,
                    mtype=mtype,
                    rotation_seed=rotation_seed,
                )
            else:
                base_score = RecommendationScoring.normalize(item.get("vote_average", 0))

            # Combine: theme match is the primary sorter for catalog rows
            final_score = (theme_match * 0.7) + (base_score * 0.3)

            # Apply tier multiplier
            tier = item.get("_discovery_tier", "individual")
            if tier == "combined":
                final_score *= 2.0  # Boost combined matches to appear first

            scored.append((final_score, item))

        # 6. Rank and Enrich
        scored.sort(key=lambda x: x[0], reverse=True)
        unique_results = []
        seen = set()
        for _, item in scored:
            if item["id"] not in seen and item["id"] not in watched_tmdb:
                unique_results.append(item)
                seen.add(item["id"])
            if len(unique_results) >= limit * 2:
                break

        enriched = await RecommendationMetadata.fetch_batch(
            self.tmdb_service, unique_results, content_type, user_settings=self.user_settings
        )
        final = filter_watched_by_imdb(enriched, watched_imdb)[:limit]

        # Register returned TMDB IDs into the cross-catalog exclusion set if provided
        if hasattr(self, '_cross_catalog_seen'):
            for item in final:
                tid = item.get("_tmdb_id")
                if tid:
                    self._cross_catalog_seen.add(tid)

        return final

    def _parse_theme_id(self, theme_id: str) -> tuple[dict, dict, dict]:
        """Parse role-based ID: watchly.theme.a:g123.f:k456.b:y1990"""
        parts = theme_id.replace("watchly.theme.", "").split(".")
        anchors, flavors, fallbacks = {}, {}, {}

        roles = {"a": anchors, "f": flavors, "b": fallbacks}

        for idx, part in enumerate(parts):
            if ":" not in part:
                # old format
                if idx == 0:
                    target = anchors
                elif idx == 1:
                    target = flavors
                elif idx == 2:
                    target = fallbacks
                else:
                    continue
                val = part
            else:
                role, val = part.split(":", 1)
                target = roles.get(role)

            if target is None:
                continue

            # Accumulate rather than assign. A row can legitimately carry two genres
            # or two keywords in the same role; the previous dict assignment silently
            # discarded all but the last, so "Drama + Squatting + True Crime" only
            # ever queried one keyword.
            def _add(tgt, key, value):
                bucket = tgt.setdefault(key, [])
                for piece in str(value).split("-"):
                    if piece and piece not in bucket:
                        bucket.append(piece)

            if val.startswith("g"):
                _add(target, "genre", val[1:])
            elif val.startswith("k"):
                _add(target, "keyword", val[1:])
            elif val.startswith("ct"):
                _add(target, "country", val[2:])
            elif val.startswith("y"):
                _add(target, "era", val[1:])
            elif val.startswith("r"):
                _add(target, "runtime", val[1:])
            elif val.startswith("cr"):
                _add(target, "creator", val[2:])

        return anchors, flavors, fallbacks

    def _axes_to_params(self, axes: dict, content_type: str) -> dict:
        """Convert axes to TMDB discover params."""
        def _vals(key):
            v = axes.get(key)
            if v is None:
                return []
            if isinstance(v, list):
                out = []
                for item in v:
                    for piece in str(item).split("-"):
                        if piece and piece not in out:
                            out.append(piece)
                return out
            return [p for p in str(v).split("-") if p]

        params = {}
        # TMDB treats "," as AND and "|" as OR. These were joined with "|", so a row
        # promising two genres or two keywords returned anything matching EITHER --
        # the direct cause of off-theme entries in themed rows.
        genres = _vals("genre")
        if genres:
            params["with_genres"] = ",".join(genres)
        keywords = _vals("keyword")
        if keywords:
            params["with_keywords"] = ",".join(keywords)
        countries = _vals("country")
        if countries:
            # Normalize common incorrect codes to ISO 3166-1 alpha-2
            country_code_fixes = {"UK": "GB", "EN": "GB"}
            country_val = countries[0]
            params["with_origin_country"] = country_code_fixes.get(country_val, country_val)
        if "era" in axes:
            try:
                # Value can be single year or range
                val = (axes["era"][0] if isinstance(axes["era"], list) else axes["era"])
                if "-" in val:
                    start_year = int(val.split("-")[0])
                    end_year = int(val.split("-")[1])
                else:
                    start_year = int(val)
                    end_year = start_year + 9

                    prefix = "first_air_date" if content_type in ("tv", "series") else "primary_release_date"
                    params[f"{prefix}.gte"] = f"{start_year}-01-01"
                    params[f"{prefix}.lte"] = f"{end_year}-12-31"
            except Exception:
                logger.error("Failed to parse era axis: {}", axes["era"])
                pass
        if "runtime" in axes:
            bucket = (axes["runtime"][0] if isinstance(axes["runtime"], list) else axes["runtime"])
            is_movie = content_type == "movie"
            s_max = RUNTIME_BUCKET_SHORT_MAX_MOVIE if is_movie else RUNTIME_BUCKET_SHORT_MAX_SERIES
            m_max = RUNTIME_BUCKET_MEDIUM_MAX_MOVIE if is_movie else RUNTIME_BUCKET_MEDIUM_MAX_SERIES
            if bucket == "short":
                params["with_runtime.lte"] = s_max
            elif bucket == "medium":
                params["with_runtime.gte"] = s_max
                params["with_runtime.lte"] = m_max
            elif bucket == "long":
                params["with_runtime.gte"] = m_max
        return params

    def _calculate_theme_score(self, item: dict, anchors: dict, flavors: dict, fallbacks: dict) -> float:
        """Calculate weighted score based on axis matches."""
        score = 0.0

        def _as_list(value):
            if isinstance(value, list):
                out = []
                for item_v in value:
                    for piece in str(item_v).split("-"):
                        if piece and piece not in out:
                            out.append(piece)
                return out
            return [p for p in str(value).split("-") if p]

        def check_match(axis_name, value):
            if axis_name == "genre":
                item_genres = [str(gid) for gid in item.get("genre_ids", [])]
                target_genres = _as_list(value)
                # ALL requested genres must be present. The query now uses AND
                # semantics, so scoring must agree or ranking contradicts the filter.
                return bool(target_genres) and all(tg in item_genres for tg in target_genres)
            if axis_name == "keyword":
                # Previously returned True unconditionally, which meant items fetched
                # WITHOUT the keyword constraint (see _expand_search) scored as full
                # matches and could collect the perfect-match bonus.
                return bool(item.get("_keyword_matched", False))
            if axis_name == "country":
                item_countries = item.get("origin_country", [])
                return any(c in item_countries for c in _as_list(value))
            if axis_name == "era":
                rel = item.get("release_date") or item.get("first_air_date")
                era_val = value[0] if isinstance(value, list) else value
                if rel:
                    try:
                        y = int(rel[:4])
                        if "-" in str(era_val):
                            start, end = map(int, str(era_val).split("-"))
                        else:
                            start = int(era_val)
                            end = start + 9
                        return start <= y <= end
                    except Exception:
                        logger.error("Failed to parse era axis: {}", value)
                        pass
            if axis_name == "runtime":
                # Runtimes are hard to match exactly from discover results without metadata enrichment
                return True
            return False

        total_anchors = len(anchors)
        matched_anchors = 0
        for name, val in anchors.items():
            if check_match(name, val):
                score += 1.0
                matched_anchors += 1

        # Perfect Match Bonus: Significant boost for items satisfying all anchors
        if total_anchors > 1 and matched_anchors == total_anchors:
            score += 2.0
        elif total_anchors > 0:
            # Proportion boost to differentiate partial matches
            score += (matched_anchors / total_anchors) * 0.5

        for name, val in flavors.items():
            if check_match(name, val):
                score += 0.7
        for name, val in fallbacks.items():
            if check_match(name, val):
                score += 0.3

        return score

    async def _expand_search(self, content_type: str, params: dict, anchors: dict, flavors: dict) -> list[dict]:
        """Expansion logic if results are sparse.

        A keyword is the most specific thing a themed row can promise, so it is the
        one constraint we never drop. Previously this deleted with_keywords entirely
        and back-filled with whatever matched the remaining genre/year filters, which
        is why niche rows ("Squatting") returned a handful of real matches followed by
        unrelated popular titles. Instead we keep the keyword and relax the SOFTER
        constraints (genre, runtime) to find more genuine matches.
        """
        if "with_keywords" in params:
            # Keep the keyword; drop the softer filters and search deeper.
            new_params = {
                k: v
                for k, v in params.items()
                if k in ("with_keywords", "language", "region", "include_adult")
                or k.startswith("primary_release_date")
                or k.startswith("first_air_date")
                or k.startswith("vote_")
                or k.startswith("sort_")
            }
            return await self._fetch_discover_candidates(
                content_type, new_params, pages=[1, 2, 3, 4]
            )

        # Non-keyword themes have no equivalently specific constraint to preserve,
        # so there is nothing safe to relax here.
        return []

    def _calculate_pages_to_fetch(self, num_excluded_genres: int) -> list[int]:
        """
        Calculate how many pages to fetch based on excluded genres.

        Args:
            num_excluded_genres: Number of excluded genres

        Returns:
            List of page numbers to fetch
        """
        if num_excluded_genres > 10:
            return list(range(1, 11))  # 10 pages
        elif num_excluded_genres > 5:
            return list(range(1, 6))  # 5 pages
        else:
            return [1, 2, 3]  # 3 pages

    async def _fetch_discover_candidates(
        self, content_type: str, params: dict[str, Any], pages: list[int]
    ) -> list[dict[str, Any]]:
        """
        Fetch candidates from TMDB discover API.

        Args:
            content_type: Content type
            params: Discover API parameters
            pages: List of page numbers to fetch

        Returns:
            List of candidate items
        """
        candidates = []

        # Apply global user filters (year range, popularity)
        params = apply_discover_filters(params, self.user_settings)

        # The vote_count floor is calibrated for broad browsing, but a narrow themed
        # query (e.g. true-crime documentaries) can lose 90%+ of its pool to it --
        # niche categories are watched widely and rated rarely. If the constrained
        # pool is too small to fill the row, retry once with a token floor so the
        # row is populated from genuinely on-theme titles rather than padded with
        # off-theme ones later.
        floor = params.get("vote_count.gte")
        if floor and int(floor) > 10:
            try:
                probe = await self.tmdb_service.get_discover(content_type, page=1, **params)
                if int(probe.get("total_results", 0) or 0) < 40:
                    relaxed = dict(params)
                    relaxed["vote_count.gte"] = 10
                    wider = await self.tmdb_service.get_discover(content_type, page=1, **relaxed)
                    if int(wider.get("total_results", 0) or 0) > int(
                        probe.get("total_results", 0) or 0
                    ):
                        logger.info(
                            f"[ThemeDiscover] relaxing vote floor {floor} -> 10 "
                            f"({probe.get('total_results')} -> {wider.get('total_results')} results)"
                        )
                        params = relaxed
            except Exception as e:
                logger.debug(f"[ThemeDiscover] vote-floor probe failed: {e}")

        tasks = [self.tmdb_service.get_discover(content_type, page=p, **params) for p in pages]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Record whether this particular query actually constrained on keywords.
        # TMDB discover results carry no keyword data, so this is the only reliable
        # signal that an item genuinely matches the theme's keyword axis. Items from
        # relaxed/expanded queries are tagged False and can be ranked down instead of
        # being treated as perfect matches.
        keyword_constrained = bool(params.get("with_keywords"))

        for res in results:
            if isinstance(res, Exception):
                logger.debug(f"Error fetching discover: {res}")
                continue
            for item in res.get("results", []):
                item["_keyword_matched"] = keyword_constrained
                candidates.append(item)

        return candidates

    def _filter_candidates(
        self,
        candidates: list[dict[str, Any]],
        watched_tmdb: set[int],
        whitelist: set[int],
        existing_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filter candidates by watched items and genre whitelist.

        Args:
            candidates: List of candidate items
            watched_tmdb: Set of watched TMDB IDs
            whitelist: Set of genre IDs in whitelist
            existing_ids: Set of IDs to exclude (for deduplication)

        Returns:
            Filtered list of items
        """
        existing = existing_ids or set()
        # First filter by genres (includes watched_tmdb check)
        filtered = filter_by_genres(candidates, watched_tmdb, whitelist, None)
        # Then deduplicate
        result = []
        for item in filtered:
            item_id = item.get("id")
            if item_id and item_id not in existing:
                result.append(item)
                existing.add(item_id)
        return result
