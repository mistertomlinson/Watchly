import asyncio
import re
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any

from loguru import logger

from app.core.constants import DEFAULT_CATALOG_LIMIT, MAX_CATALOG_ITEMS
from app.core.settings import UserSettings
from app.models.taste_profile import TasteProfile
from app.services.profile.constants import TOP_PICKS_CREATOR_CAP, TOP_PICKS_GENRE_CAP
from app.services.profile.sampling import SmartSampler
from app.services.profile.scorer import ProfileScorer
from app.services.recommendation.filtering import RecommendationFiltering
from app.services.recommendation.metadata import RecommendationMetadata
from app.services.recommendation.rotation import DailyRotation
from app.services.recommendation.scoring import RecommendationScoring
from app.services.recommendation.utils import (
    apply_discover_filters,
    content_type_to_mtype,
    filter_items_by_settings,
    filter_watched_by_imdb,
    resolve_tmdb_id,
)
from app.services.scoring import ScoringService
from app.services.openrouter import gemini_service
from app.services.simkl import simkl_service
from app.services.tmdb.service import TMDBService


class TopPicksService:
    """
    Generates top picks by combining multiple sources and applying diversity caps.
    """

    def __init__(self, tmdb_service: TMDBService, user_settings: UserSettings | None = None):
        self.tmdb_service: TMDBService = tmdb_service
        self.user_settings: UserSettings | None = user_settings
        self.scorer: ProfileScorer = ProfileScorer()
        self.scoring_service = ScoringService()
        self.smart_sampler = SmartSampler(self.scoring_service)

    async def get_top_picks(
        self,
        profile: TasteProfile,
        content_type: str,
        library_items: dict[str, list[dict[str, Any]]],
        watched_tmdb: set[int],
        watched_imdb: set[str],
        limit: int = 50,
        gemini_api_key: str | None = None,
        library_items_raw: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get top picks with diversity caps.

        Strategy:
        1. Fetch recommendations from top 8 library items - 1 page each
        2. Fetch discover with profile features (genres, keywords, cast, crew, era, country)
        3. Merge all candidates (deduped by TMDB ID)
        4. Score with ProfileScorer + Quality
        5. Apply diversity caps (relaxed: 50% genre, 50% era, 15% recent)
        6. Limit to 2x target before enrichment (performance optimization)
        7. Enrich metadata with full details
        8. Apply creator cap and final filters
        9. Return balanced results

        Args:
            profile: User taste profile
            content_type: Content type (movie/series)
            library_items: Library items dict
            watched_tmdb: Set of watched TMDB IDs
            watched_imdb: Set of watched IMDB IDs
            limit: Number of items to return

        Returns:
            List of recommended items
        """

        start_time = time.time()

        logger.info(f"Starting top picks generation for {content_type}, target limit={limit}")

        mtype = content_type_to_mtype(content_type)
        all_candidates = {}

        # 1. Fetch recommendations from top items
        # Use Simkl if API key available, otherwise fall back to TMDB
        simkl_api_key = self.user_settings.simkl_api_key if self.user_settings else None
        if simkl_api_key:
            rec_candidates = await self._fetch_simkl_recommendations(library_items, content_type, mtype)
            if not rec_candidates:
                # Fallback to TMDB if Simkl returns nothing
                logger.info("Simkl returned no results, falling back to TMDB")
                rec_candidates = await self._fetch_recommendations_from_top_items(library_items, content_type, mtype)
                # filter items
                rec_candidates = filter_items_by_settings(rec_candidates, self.user_settings, simkl=True)
        else:
            rec_candidates = await self._fetch_recommendations_from_top_items(library_items, content_type, mtype)
            # filter items
            rec_candidates = filter_items_by_settings(rec_candidates, self.user_settings)

        for item in rec_candidates:
            if item.get("id"):
                all_candidates[item["id"]] = item

        # 2. Fetch with Gemini if API key available, otherwise fall back to discover
        logger.info(f"Top picks Gemini check: api_key={bool(gemini_api_key)} library={bool(library_items_raw)} summary={bool(profile.interest_summary if profile else None)}")
        gemini_ids = set()
        use_gemini = gemini_api_key and library_items_raw and profile.interest_summary
        if use_gemini:
            gemini_candidates = await self._fetch_gemini_recommendations(
                profile, content_type, library_items_raw, gemini_api_key, limit
            )
            gemini_candidates = filter_items_by_settings(gemini_candidates, self.user_settings)
            for item in gemini_candidates:
                if item.get("id"):
                    item["_gemini_pick"] = True
                    all_candidates[item["id"]] = item
                    gemini_ids.add(item["id"])
            logger.info(f"Gemini returned {len(gemini_candidates)} candidates for top picks")

        # Only use discover as fallback if Gemini failed or unavailable
        if not use_gemini or len(gemini_ids) < limit // 2:
            logger.info(f"Falling back to discover (gemini_ids={len(gemini_ids)})")
            discover_candidates = await self._fetch_discover_with_profile(profile, content_type, mtype)
            discover_candidates = filter_items_by_settings(discover_candidates, self.user_settings)
            for item in discover_candidates:
                if item.get("id"):
                    all_candidates[item["id"]] = item

        # Filter out watched items
        filtered_candidates = [item for item in all_candidates.values() if item.get("id") not in watched_tmdb]

        logger.info(f"Found {len(filtered_candidates)} candidates after filtering out watched items and user settings")

        rotation_seed = RecommendationScoring.generate_rotation_seed()
        scored_candidates = []  # initialize for any downstream references

        # If Gemini succeeded, skip scoring/diversity caps — trust Gemini directly
        if gemini_ids and len(gemini_ids) >= limit // 2:
            result = [i for i in filtered_candidates if i.get("_gemini_pick")]
            logger.info(f"Using {len(result)} Gemini picks directly, skipping scoring/diversity caps")
        else:
            # Fall back to full scoring pipeline
            scored_candidates = []
            for item in filtered_candidates:
                try:
                    final_score = RecommendationScoring.calculate_final_score(
                        item=item,
                        profile=profile,
                        scorer=self.scorer,
                        mtype=mtype,
                        rotation_seed=rotation_seed,
                    )
                    scored_candidates.append((final_score, item))
                except Exception as e:
                    logger.debug(f"Failed to score item {item.get('id')}: {e}")
                    continue
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"Scored {len(scored_candidates)} candidates.")
            result_capped = self._apply_diversity_caps(scored_candidates, len(scored_candidates), mtype)
            logger.info(f"After diversity caps: {len(result_capped)} items")
            result = [i for i in result_capped]
            result = result[: limit * 3]
            logger.info(f"After pre-enrichment limit: {len(result)} items")

        # Enrich metadata
        enriched = await RecommendationMetadata.fetch_batch(
            self.tmdb_service, result, content_type, user_settings=self.user_settings
        )
        logger.info(f"Enriched {len(enriched)} items with full metadata")

        # Final filter
        filtered = filter_watched_by_imdb(enriched, watched_imdb)

        rotated = DailyRotation.rotate_items(filtered, rotation_seed)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Top picks complete: {len(rotated)} items returned in {elapsed_time:.2f}s "
            f"(target: {limit}, candidates: {len(all_candidates)}, scored: {len(scored_candidates)})"
        )

        return rotated[:MAX_CATALOG_ITEMS]

    async def _fetch_recommendations_from_top_items(
        self,
        library_items: dict[str, list[dict[str, Any]]],
        content_type: str,
        mtype: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch recommendations from top items (loved/watched/liked/added).

        Args:
            library_items: Library items dict
            content_type: Content type
            mtype: TMDB media type (movie/tv)

        Returns:
            List of candidate items
        """
        # Get top items (loved first, then liked, then added, then top watched)
        top_items = self.smart_sampler.sample_items(library_items, content_type, max_items=15)

        candidates = []
        tasks = []

        for item in top_items:
            item = item.item
            item_id = item.id
            if not item_id:
                continue

            # Resolve TMDB ID
            tmdb_id = await resolve_tmdb_id(item_id, self.tmdb_service)
            if not tmdb_id:
                continue

            # Fetch recommendations (1 page only)
            tasks.append(self.tmdb_service.get_recommendations(tmdb_id, mtype, page=1))
            # tasks.append(self.tmdb_service.get_similar(tmdb_id, mtype, page=1))

        # Execute all in parallel
        logger.info(f"Fetching recommendations from {len(tasks)} top library items")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_count = 0
        for res in results:
            if isinstance(res, Exception):
                failed_count += 1
                logger.debug(f"Recommendation fetch failed: {res}")
                continue
            candidates.extend(res.get("results", []))

        if failed_count > 0:
            logger.info(f"{failed_count}/{len(tasks)} recommendation fetches failed (expected for items with no recs)")
        logger.debug(f"Fetched {len(candidates)} candidates from top items")

        return candidates

    async def _fetch_simkl_recommendations(
        self,
        library_items: dict[str, list[dict[str, Any]]],
        content_type: str,
        mtype: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch recommendations from Simkl for top library items.

        Args:
            library_items: Library items dict
            content_type: Content type
            mtype: TMDB media type (movie/tv)

        Returns:
            List of candidate items in TMDB-compatible format
        """
        simkl_api_key = self.user_settings.simkl_api_key if self.user_settings else None
        if not simkl_api_key:
            logger.warning("Simkl API key not found, skipping Simkl recommendations")
            return []

        # Sample top items (same as TMDB flow - 15 items)
        top_items = self.smart_sampler.sample_items(library_items, content_type, max_items=15)

        # Extract IMDB IDs
        imdb_ids = []
        for scored_item in top_items:
            item_id = scored_item.item.id
            if item_id and item_id.startswith("tt"):
                imdb_ids.append(item_id)

        if not imdb_ids:
            logger.warning("No valid IMDB IDs found for Simkl recommendations")
            return []

        logger.info(f"Fetching Simkl recommendations for {len(imdb_ids)} items")

        # Get year range for filtering
        year_min = getattr(self.user_settings, "year_min", None)
        year_max = getattr(self.user_settings, "year_max", None)

        # Fetch from Simkl (batch optimized with early year filtering)
        try:
            candidates = await simkl_service.get_recommendations_batch(
                imdb_ids,
                mtype,
                simkl_api_key,
                max_per_item=8,
                year_min=year_min,
                year_max=year_max,
            )
        except Exception as e:
            logger.error(f"Error fetching Simkl recommendations: {e}")
            return []

        logger.info(f"Fetched {len(candidates)} candidates from Simkl")
        return candidates

    def _add_discover_task(self, tasks: list, mtype: str, without_genres: str | None, **kwargs: Any) -> None:
        """
        Add a discover task to the list of tasks with default parameters.
        """
        sort_by = RecommendationFiltering.get_sort_by_preference(self.user_settings)
        params = {
            "sort_by": sort_by,
            **kwargs,
        }
        if without_genres:
            params["without_genres"] = without_genres

        # Apply global user filters (year range, popularity)
        params = apply_discover_filters(params, self.user_settings)

        tasks.append(self.tmdb_service.get_discover(mtype, **params))


    async def _fetch_gemini_recommendations(
        self,
        profile: TasteProfile,
        content_type: str,
        library_items: dict,
        gemini_api_key: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Use Gemini to pick specific titles based on watch history and taste profile.
        Mirrors AI Search approach: pass history + preferences directly to LLM.
        """
        try:
            mtype = content_type_to_mtype(content_type)

            # Build watch history context
            loved = [i for i in library_items.get("loved", []) if i.get("type") == content_type]
            liked = [i for i in library_items.get("liked", []) if i.get("type") == content_type]
            watched = [i for i in library_items.get("watched", []) if i.get("type") == content_type]

            # Sort watched by last watched date
            watched.sort(key=lambda x: x.get("state", {}).get("lastWatched", ""), reverse=True)

            # Build loved/liked section separately
            loved_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in loved[:20]]
            liked_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in liked[:20]]
            watched_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in watched]

            # Top genres from profile
            top_genres = profile.get_top_genres(limit=6)
            genre_lines = [f"- Genre ID {gid} (score: {score:.2f})" for gid, score in top_genres]

            # Top directors and cast
            top_directors = profile.get_top_directors(limit=5)
            top_cast = profile.get_top_cast(limit=5)
            director_lines = [f"- Person ID {did} (score: {score:.2f})" for did, score in top_directors]
            cast_lines = [f"- Person ID {cid} (score: {score:.2f})" for cid, score in top_cast]

            interest_summary = profile.interest_summary or ""

            year_min = getattr(self.user_settings, "year_min", None) if self.user_settings else None
            year_max = getattr(self.user_settings, "year_max", None) if self.user_settings else None
            year_constraint = ""
            if year_min and year_max:
                year_constraint = f"\n- ONLY recommend titles released between {year_min} and {year_max}."
            elif year_min:
                year_constraint = f"\n- ONLY recommend titles released after {year_min}."
            elif year_max:
                year_constraint = f"\n- ONLY recommend titles released before {year_max}."

            popularity = getattr(self.user_settings, "popularity", "balanced") if self.user_settings else "balanced"
            popularity_instruction = {
                "mainstream": "Focus on popular, widely-known titles that are highly rated.",
                "balanced": "Include a mix of well-known titles and lesser-known quality titles.",
                "gems": "Focus on hidden gems — lesser-known, critically acclaimed titles that most people haven't seen.",
                "all": "Include any quality title regardless of popularity level.",
            }.get(popularity, "Include a mix of well-known titles and lesser-known quality titles.")

            gemini_request_limit = limit * 3
            all_watched_lines = list(dict.fromkeys(loved_lines + liked_lines + watched_lines))

            prompt = f"""You are an expert {content_type} recommendation engine.

User Interest Summary: {interest_summary}

Content they LOVED (10/10 — strongest signal, prioritize similarity to these above all else):
{chr(10).join(loved_lines) if loved_lines else "None recorded"}

Content they LIKED (7-9/10 — secondary signal):
{chr(10).join(liked_lines) if liked_lines else "None recorded"}

COMPLETE watch history — DO NOT recommend ANY of these titles:
{chr(10).join(all_watched_lines) if all_watched_lines else "None recorded"}

TASK: Recommend exactly {gemini_request_limit} {content_type}s this person has NOT watched yet.
- Strongly reflect their taste profile and interest summary
- {popularity_instruction}
- Prioritize quality and relevance
- DO NOT recommend ANYTHING from the watch history above — check every title before including it
- Lean toward their strongest preferences but include some variety{year_constraint}

RESPONSE FORMAT (one per line, no other text):
{content_type}|Title|Year

EXAMPLE:
{content_type}|Blade Runner 2049|2017
{content_type}|Annihilation|2018"""

            response = await gemini_service.generate_flash_content_async(
                prompt=prompt,
                system_instruction=f"You are a personalized {content_type} recommendation expert. Return ONLY the pipe-separated list, no explanations.",
                api_key=gemini_api_key,
            )

            if not response:
                return []

            # Parse response and resolve to TMDB IDs
            candidates = []
            lines = [l.strip() for l in response.strip().split("\n") if l.strip() and "|" in l]

            resolve_tasks = []
            parsed = []
            seen_titles = set()
            max_to_resolve = gemini_request_limit
            for line in lines:
                if len(parsed) >= max_to_resolve:
                    break
                parts = line.split("|")
                if len(parts) >= 3:
                    _, name, year = parts[0].strip(), parts[1].strip(), parts[2].strip()[:4]
                    # Strip year suffix from title if AI included it e.g. "Title (2004)"
                    name = re.sub(r'\s*\(\d{4}\)\s*$', '', name).strip()
                    title_key = name.lower().strip()
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    parsed.append((name, year))
                    resolve_tasks.append(self._resolve_title_to_tmdb(name, year, mtype))

            results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict) and result.get("id"):
                    candidates.append(result)

            logger.info(f"Gemini top picks: resolved {len(candidates)}/{len(parsed)} titles")
            logger.info(f"Gemini picks: {[c.get("title") or c.get("name") for c in candidates]}")
            return candidates

        except Exception as e:
            logger.warning(f"Gemini top picks failed, falling back to discover: {e}")
            return []

    async def _resolve_title_to_tmdb(self, name: str, year: str, mtype: str) -> dict | None:
        """Resolve a title+year to a TMDB item dict via TMDB search API."""
        try:
            search_type = "tv" if mtype == "tv" else "movie"
            params = {"query": name, "page": 1}
            if year:
                if search_type == "movie":
                    params["primary_release_year"] = year
                else:
                    params["first_air_date_year"] = year
            results = await self.tmdb_service.client.get(f"/search/{search_type}", params=params)
            items = results.get("results", [])
            if not items:
                # Try without year constraint
                params2 = {"query": name, "page": 1}
                results2 = await self.tmdb_service.client.get(f"/search/{search_type}", params=params2)
                items = results2.get("results", [])
            if not items:
                return None

            # TMDB search is fuzzy and always returns *something*, so taking
            # items[0] blindly meant a hallucinated or misremembered LLM title
            # resolved to whatever was closest -- e.g. a BJJ tournament recording
            # appearing among film recommendations. Require the result to actually
            # resemble what was asked for.
            def _norm(s: str) -> str:
                return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch == " ").strip()

            wanted = _norm(name)
            wanted_words = set(wanted.split())
            if not wanted_words:
                return None

            for item in items[:5]:
                candidate = _norm(item.get("title") or item.get("name") or "")
                if not candidate:
                    continue
                if candidate == wanted:
                    return item
                cand_words = set(candidate.split())
                overlap = len(wanted_words & cand_words) / len(wanted_words)

                # Full containment: every word of the query appears in the
                # candidate. This is the long-subtitle case -- "Birds of Prey" vs
                # "Birds of Prey (and the Fantabulous Emancipation of One Harley
                # Quinn)" -- and a length guard wrongly rejected it.
                if overlap == 1.0 and candidate.startswith(wanted):
                    return item

                # Otherwise accept a close match (punctuation, localised naming)
                # but reject a coincidental one-word collision.
                if overlap >= 0.7 and abs(len(cand_words) - len(wanted_words)) <= 3:
                    return item

            logger.debug(
                f"[TopPicks] no confident TMDB match for {name!r} "
                f"(best was {items[0].get('title') or items[0].get('name')!r})"
            )
            return None
        except Exception as e:
            logger.debug(f"Failed to resolve title {name}: {e}")
            return None

    async def _fetch_discover_with_profile(
        self, profile: TasteProfile, content_type: str, mtype: str
    ) -> list[dict[str, Any]]:
        """
        Fetch discover results using profile features.

        Args:
            profile: User taste profile
            content_type: Content type
            mtype: TMDB media type

        Returns:
            List of candidate items
        """

        excluded_genre_ids = RecommendationFiltering.get_excluded_genre_ids(self.user_settings, content_type)
        without_genres = "|".join(str(g) for g in excluded_genre_ids) if excluded_genre_ids else None

        logger.debug(f"Excluded genres for {content_type}: {excluded_genre_ids}")

        # Get top features from profile
        top_genres = profile.get_top_genres(limit=5)
        top_keywords = profile.get_top_keywords(limit=5)
        top_directors = profile.get_top_directors(limit=3)
        top_cast = profile.get_top_cast(limit=5)
        top_eras = profile.get_top_eras(limit=2)
        top_countries = profile.get_top_countries(limit=5)

        candidates = []
        tasks = []

        # Discover with genres — run individual queries per genre weighted by score
        # Higher scored genres get more pages so they dominate the candidate pool
        if top_genres:
            max_score = top_genres[0][1] if top_genres else 1.0
            for genre_id, score in top_genres[:5]:
                # Allocate 1-3 pages proportional to score relative to top genre
                ratio = score / max_score if max_score > 0 else 0
                pages = 3 if ratio >= 0.8 else 2 if ratio >= 0.5 else 1
                for page in range(1, pages + 1):
                    self._add_discover_task(
                        tasks,
                        mtype,
                        without_genres,
                        with_genres=str(genre_id),
                        page=page,
                    )

        # Discover with keywords
        if top_keywords:
            from app.services.row_generator import GENERIC_KEYWORD_BLACKLIST
            keyword_ids = [k[0] for k in top_keywords if k[0] not in GENERIC_KEYWORD_BLACKLIST]
            for page in range(1, 3):  # 2 pages
                self._add_discover_task(
                    tasks,
                    mtype,
                    without_genres,
                    with_keywords="|".join(str(k) for k in keyword_ids),
                    page=page,
                )

        # Discover with directors
        if top_directors:
            director_ids = [d[0] for d in top_directors]
            self._add_discover_task(
                tasks,
                mtype,
                without_genres,
                with_crew="|".join(str(d) for d in director_ids),
                page=1,
            )

        # Discover with cast
        if top_cast:
            cast_ids = [c[0] for c in top_cast]
            self._add_discover_task(
                tasks,
                mtype,
                without_genres,
                with_cast="|".join(str(c) for c in cast_ids),
                page=1,
            )

        # Discover with era (year range)
        if top_eras:
            era = top_eras[0][0]
            year_start = self._era_to_year_start(era)
            if year_start:
                prefix = "first_air_date" if mtype == "tv" else "primary_release_date"
                lte_prefix = (
                    date.today().isoformat() if year_start + 9 > date.today().year else f"{year_start + 9}-12-31"
                )
                params = {
                    f"{prefix}.gte": f"{year_start}-01-01",
                    f"{prefix}.lte": lte_prefix,
                    "page": 1,
                }

                self._add_discover_task(tasks, mtype, without_genres, **params)

        # Discover with countries
        if top_countries:
            country_codes = [c[0] for c in top_countries]
            params = {
                "with_origin_country": "|".join(country_codes),
                "page": 1,
            }
            self._add_discover_task(tasks, mtype, without_genres, **params)

        # Execute all in parallel
        logger.debug(f"Fetching {len(tasks)} discover queries with profile features")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_count = 0
        for res in results:
            if isinstance(res, Exception):
                failed_count += 1
                logger.warning(f"Discover query failed: {res}")
                continue
            candidates.extend(res.get("results", []))

        if failed_count > 0:
            logger.warning(f"{failed_count}/{len(tasks)} discover queries failed")
        logger.debug(f"Fetched {len(candidates)} candidates from discover")

        return candidates

    async def _fetch_trending_and_popular(self, content_type: str, mtype: str) -> list[dict[str, Any]]:
        """
        Fetch trending and popular items (for recent items injection).

        Args:
            content_type: Content type
            mtype: TMDB media type

        Returns:
            List of candidate items
        """
        candidates = []

        # Fetch trending (1 page)
        try:
            trending = await self.tmdb_service.get_trending(mtype, time_window="week", page=1)
            candidates.extend(trending.get("results", []))
        except Exception as e:
            logger.debug(f"Failed to fetch trending: {e}")

        return candidates

    def _apply_diversity_caps(
        self,
        scored_candidates: list[tuple[float, dict[str, Any]]],
        limit: int,
        mtype: str,
    ) -> list[dict[str, Any]]:
        """
        Apply diversity caps to ensure balanced results.

        Caps:
        - Genre: max 50% per genre
        - Era: max 50% per era
        - Quality: minimum vote_count and rating

        Args:
            scored_candidates: List of (score, item) tuples, sorted by score
            limit: Target number of items
            mtype: Media type for quality checks

        Returns:
            Filtered and capped list of items
        """
        result = []
        genre_counts = defaultdict(int)
        # era_counts = defaultdict(int)

        max_per_genre = int(limit * TOP_PICKS_GENRE_CAP)
        # max_per_era = int(limit * TOP_PICKS_ERA_CAP)

        for score, item in scored_candidates:
            if len(result) >= limit:
                break

            item_id = item.get("id")
            if not item_id:
                continue

            # Quality threshold
            vote_count = item.get("vote_count", 0)
            vote_avg = item.get("vote_average", 0)

            # Dynamic check
            min_rating, min_votes = RecommendationFiltering.get_quality_thresholds(self.user_settings)

            if vote_count < min_votes:
                continue

            # We keep weighted rating check but use dynamic base
            wr = RecommendationScoring.weighted_rating(vote_avg, vote_count, C=7.2 if mtype == "tv" else 6.8)
            if wr < min_rating:
                continue

            # Check genre cap (50% max per genre)
            genre_ids = item.get("genre_ids", [])
            top_genre = genre_ids[0] if genre_ids else None

            if top_genre:
                if genre_counts[top_genre] >= max_per_genre:
                    continue

            # Add item
            result.append(item)

            if top_genre:
                genre_counts[top_genre] += 1

        return result

    def _apply_creator_cap(self, items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """
        Apply creator cap (max 2 items per director/actor) after enrichment.

        Args:
            items: List of enriched items with full metadata
            limit: Target limit

        Returns:
            Filtered list respecting creator cap
        """
        result = []
        creator_counts = defaultdict(int)

        for item in items:
            if len(result) >= limit:
                break

            # Extract creators from credits
            credits = item.get("credits", {}) or {}
            crew = credits.get("crew", []) or []
            cast = credits.get("cast", []) or []

            # Check director cap
            directors = [c.get("id") for c in crew if c.get("job", "").lower() == "director" and c.get("id")]
            blocked_by_director = False
            for dir_id in directors:
                if creator_counts[dir_id] >= TOP_PICKS_CREATOR_CAP:
                    blocked_by_director = True
                    break

            # Check cast cap (top 3 only)
            top_cast = [c.get("id") for c in cast[:3] if c.get("id")]
            blocked_by_cast = False
            for cast_id in top_cast:
                if creator_counts[cast_id] >= TOP_PICKS_CREATOR_CAP:
                    blocked_by_cast = True
                    break

            if blocked_by_director or blocked_by_cast:
                continue

            # Add item
            result.append(item)

            # Update creator counts
            for dir_id in directors:
                creator_counts[dir_id] += 1
            for cast_id in top_cast:
                creator_counts[cast_id] += 1

        return result

    @staticmethod
    def _extract_year(item: dict[str, Any]) -> int | None:
        """Extract year from item."""
        release_date = item.get("release_date") or item.get("first_air_date")
        if release_date:
            try:
                return int(str(release_date)[:4])
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _is_recent_release(item: dict[str, Any], threshold: datetime, mtype: str) -> bool:
        """Check if item was released within the threshold (e.g., last 3 months)."""
        release_date_str = item.get("release_date") if mtype == "movie" else item.get("first_air_date")
        if not release_date_str:
            return False

        try:
            # Parse release date (format: YYYY-MM-DD)
            release_date = datetime.strptime(str(release_date_str)[:10], "%Y-%m-%d")
            return release_date >= threshold
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _year_to_era(year: int) -> str:
        """Convert year to era bucket."""
        if year < 1970:
            return "pre-1970s"
        elif year < 1980:
            return "1970s"
        elif year < 1990:
            return "1990s"
        elif year < 2000:
            return "2000s"
        elif year < 2010:
            return "2010s"
        elif year < 2020:
            return "2020s"
        else:
            return "2020s"

    @staticmethod
    def _era_to_year_start(era: str) -> int | None:
        """Convert era bucket to starting year."""
        era_map = {
            "pre-1970s": 1950,
            "1970s": 1970,
            "1980s": 1980,
            "1990s": 1990,
            "2000s": 2000,
            "2010s": 2010,
            "2020s": 2020,
        }
        return era_map.get(era)
