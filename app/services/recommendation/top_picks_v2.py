"""Unified candidate-pool pipeline for Watchly Top Picks.

The important invariant in this module is that recommendation sources only
*propose candidates*. Simkl, the LLM, and TMDB Discover all feed the same
identity-normalized, watched-filtered, enriched, scored pipeline. No source is
allowed to bypass ranking or discard the other sources merely because it
returned enough titles.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from datetime import date
from typing import Any

from loguru import logger

from app.core.constants import MAX_CATALOG_ITEMS
from app.core.settings import UserSettings
from app.models.taste_profile import TasteProfile
from app.services.openrouter import RECOMMENDATION_MAX_TOKENS, gemini_service
from app.services.profile.constants import TOP_PICKS_GENRE_CAP
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
from app.services.simkl import simkl_service
from app.services.tmdb.service import TMDBService


class TopPicksService:
    """Build a large multi-source pool, score it, then rotate a qualified set."""

    # Twenty displayed titles should be selected from a meaningfully larger pool.
    MIN_CANDIDATE_HEADROOM = 60
    MIN_ENRICHMENT_HEADROOM = 80
    QUALIFYING_POOL_FLOOR = 30

    def __init__(self, tmdb_service: TMDBService, user_settings: UserSettings | None = None):
        self.tmdb_service = tmdb_service
        self.user_settings = user_settings
        self.scorer = ProfileScorer()
        self.scoring_service = ScoringService()
        self.smart_sampler = SmartSampler(self.scoring_service)

    @staticmethod
    def _canonical_tmdb_id(item: dict[str, Any]) -> int | None:
        """Return one canonical integer TMDB identity for a raw candidate."""
        raw = item.get("_tmdb_id") or item.get("tmdb_id") or item.get("id")
        if raw in (None, ""):
            return None
        if isinstance(raw, str) and raw.startswith("tmdb:"):
            raw = raw.split(":", 1)[1]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _merge_candidate(
        cls,
        pool: dict[int, dict[str, Any]],
        item: dict[str, Any],
        source: str,
    ) -> bool:
        """Merge a source candidate by canonical TMDB identity.

        Duplicate Simkl/Gemini/Discover recommendations become one candidate
        with combined provenance instead of separate int/string dictionary keys.
        """
        tid = cls._canonical_tmdb_id(item)
        if tid is None:
            return False

        incoming = dict(item)
        incoming["id"] = tid

        existing = pool.get(tid)
        if existing is None:
            incoming["_watchly_sources"] = [source]
            pool[tid] = incoming
            return True

        sources = set(existing.get("_watchly_sources") or [])
        sources.add(source)
        existing["_watchly_sources"] = sorted(sources)

        # Keep the first source's useful metadata but fill any gaps from later
        # sources. This avoids a sparse search result overwriting richer data.
        for key, value in incoming.items():
            if key in {"id", "_watchly_sources"}:
                continue
            current = existing.get(key)
            if current in (None, "", [], {}, 0) and value not in (None, "", [], {}):
                existing[key] = value

        return False

    @classmethod
    def _merge_many(
        cls,
        pool: dict[int, dict[str, Any]],
        items: list[dict[str, Any]],
        source: str,
    ) -> int:
        added = 0
        for item in items:
            if cls._merge_candidate(pool, item, source):
                added += 1
        return added

    @classmethod
    def _pool_from_filtered(cls, items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        pool: dict[int, dict[str, Any]] = {}
        for item in items:
            tid = cls._canonical_tmdb_id(item)
            if tid is not None:
                item["id"] = tid
                pool[tid] = item
        return pool

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
        start_time = time.time()
        mtype = content_type_to_mtype(content_type)
        target = max(1, min(int(limit), MAX_CATALOG_ITEMS))
        candidate_target = max(self.MIN_CANDIDATE_HEADROOM, target * 3)
        enrichment_target = max(self.MIN_ENRICHMENT_HEADROOM, target * 4)
        qualifying_target = max(self.QUALIFYING_POOL_FLOOR, target + 10)

        logger.info(
            f"Starting unified top picks generation for {content_type}, "
            f"display={target}, candidate_headroom={candidate_target}, "
            f"qualifying_pool={qualifying_target}"
        )

        pool: dict[int, dict[str, Any]] = {}
        source_counts: dict[str, int] = defaultdict(int)

        # 1. Simkl (or TMDB recommendation fallback) proposes candidates.
        simkl_api_key = self.user_settings.simkl_api_key if self.user_settings else None
        if simkl_api_key:
            source_items = await self._fetch_simkl_recommendations(library_items, content_type, mtype)
            source_items = filter_items_by_settings(source_items, self.user_settings, simkl=True)
            source_name = "simkl"
            if not source_items:
                logger.info("Simkl returned no results, falling back to TMDB recommendations")
                source_items = await self._fetch_recommendations_from_top_items(
                    library_items, content_type, mtype
                )
                source_items = filter_items_by_settings(source_items, self.user_settings)
                source_name = "tmdb_recommendations"
        else:
            source_items = await self._fetch_recommendations_from_top_items(
                library_items, content_type, mtype
            )
            source_items = filter_items_by_settings(source_items, self.user_settings)
            source_name = "tmdb_recommendations"

        source_counts[source_name] = len(source_items)
        self._merge_many(pool, source_items, source_name)

        # 2. LLM recommendations are another source, never an override.
        use_gemini = bool(
            gemini_api_key
            and library_items_raw
            and profile
            and profile.interest_summary
        )
        if use_gemini:
            gemini_items = await self._fetch_gemini_recommendations(
                profile,
                content_type,
                library_items_raw or {},
                gemini_api_key or "",
                target,
            )
            gemini_items = filter_items_by_settings(gemini_items, self.user_settings)
            source_counts["gemini"] = len(gemini_items)
            self._merge_many(pool, gemini_items, "gemini")

        # Early watched exclusion now uses both preserved provider identities.
        before_watched = len(pool)
        early_filtered = RecommendationFiltering.filter_candidates(
            list(pool.values()), watched_imdb, watched_tmdb
        )
        pool = self._pool_from_filtered(early_filtered)
        logger.info(
            f"Top picks pool after Simkl/LLM merge: raw_unique={before_watched}, "
            f"unwatched={len(pool)}, watched_removed={before_watched - len(pool)}"
        )

        # 3. Discover supplies headroom only when the primary sources did not.
        discover_used = False
        if len(pool) < candidate_target:
            discover_used = True
            discover_items = await self._fetch_discover_with_profile(profile, content_type, mtype)
            discover_items = filter_items_by_settings(discover_items, self.user_settings)
            source_counts["discover"] = len(discover_items)
            self._merge_many(pool, discover_items, "discover")
            early_filtered = RecommendationFiltering.filter_candidates(
                list(pool.values()), watched_imdb, watched_tmdb
            )
            pool = self._pool_from_filtered(early_filtered)
            logger.info(f"Top picks pool after Discover headroom: unwatched={len(pool)}")

        if not pool:
            logger.warning("Top picks candidate pool is empty after watched/settings filtering")
            return []

        # 4. Cheap preliminary ranking limits expensive detail/image requests.
        # Final ranking happens *after* enrichment with keywords/credits available.
        rotation_seed = RecommendationScoring.generate_rotation_seed()
        preliminary: list[tuple[float, dict[str, Any]]] = []
        for item in pool.values():
            try:
                score = RecommendationScoring.calculate_final_score(
                    item=item,
                    profile=profile,
                    scorer=self.scorer,
                    mtype=mtype,
                    rotation_seed=rotation_seed,
                )
            except Exception as exc:
                logger.debug(f"Preliminary score failed for {item.get('id')}: {exc}")
                score = 0.0
            preliminary.append((score, item))
        preliminary.sort(key=lambda pair: pair[0], reverse=True)

        to_enrich = [item for _, item in preliminary[:enrichment_target]]
        logger.info(
            f"Top picks enriching {len(to_enrich)}/{len(pool)} candidates "
            "for full-feature ranking"
        )

        enriched = await RecommendationMetadata.fetch_batch(
            self.tmdb_service,
            to_enrich,
            content_type,
            user_settings=self.user_settings,
        )

        # Carry source provenance through formatting for diagnostics.
        source_by_tmdb = {
            tid: list(item.get("_watchly_sources") or [])
            for tid, item in pool.items()
        }
        for item in enriched:
            tid = self._canonical_tmdb_id(item)
            if tid is not None:
                item["_watchly_sources"] = source_by_tmdb.get(tid, [])

        # Final watched guard uses IMDb after full TMDB external IDs are known.
        enriched = filter_watched_by_imdb(enriched, watched_imdb)
        logger.info(f"Top picks full metadata survivors after IMDb guard: {len(enriched)}")

        # If late filtering unexpectedly starved the pool, Discover can still be
        # used as a reserve source. This is intentionally rare and only runs when
        # Discover was not already needed for ordinary headroom.
        if len(enriched) < qualifying_target and not discover_used:
            discover_used = True
            discover_items = await self._fetch_discover_with_profile(profile, content_type, mtype)
            discover_items = filter_items_by_settings(discover_items, self.user_settings)
            source_counts["discover"] = len(discover_items)

            existing_tmdb = {
                self._canonical_tmdb_id(item)
                for item in enriched
                if self._canonical_tmdb_id(item) is not None
            }
            reserve_pool: dict[int, dict[str, Any]] = {}
            self._merge_many(reserve_pool, discover_items, "discover")
            reserve_filtered = RecommendationFiltering.filter_candidates(
                list(reserve_pool.values()), watched_imdb, watched_tmdb
            )
            reserve_filtered = [
                item
                for item in reserve_filtered
                if self._canonical_tmdb_id(item) not in existing_tmdb
            ]
            reserve_enrich_limit = max(target * 2, qualifying_target - len(enriched) + 10)
            reserve_enriched = await RecommendationMetadata.fetch_batch(
                self.tmdb_service,
                reserve_filtered[:reserve_enrich_limit],
                content_type,
                user_settings=self.user_settings,
            )
            reserve_enriched = filter_watched_by_imdb(reserve_enriched, watched_imdb)
            enriched.extend(reserve_enriched)
            logger.info(
                f"Top picks reserve Discover added {len(reserve_enriched)} survivors; "
                f"combined={len(enriched)}"
            )

        # 5. Full-feature scoring is the only path to qualification.
        scored_candidates: list[tuple[float, dict[str, Any]]] = []
        seen_final: set[int | str] = set()
        for item in enriched:
            identity: int | str = self._canonical_tmdb_id(item) or item.get("id") or ""
            if not identity or identity in seen_final:
                continue
            seen_final.add(identity)
            try:
                final_score = RecommendationScoring.calculate_final_score(
                    item=item,
                    profile=profile,
                    scorer=self.scorer,
                    mtype=mtype,
                    rotation_seed=rotation_seed,
                )
                scored_candidates.append((final_score, item))
            except Exception as exc:
                logger.debug(f"Failed to score enriched item {item.get('id')}: {exc}")

        scored_candidates.sort(key=lambda pair: pair[0], reverse=True)
        logger.info(f"Scored {len(scored_candidates)} fully enriched top-picks candidates")

        # 6. Rank only to establish a quality bar. Presentation then rotates
        # within the qualifying pool instead of publishing rigid #1-#20 order.
        qualifying_limit = min(len(scored_candidates), qualifying_target)
        qualified = self._apply_diversity_caps(
            scored_candidates,
            qualifying_limit,
            mtype,
        )

        if len(qualified) < target:
            # Diversity/quality filtering can be stricter than expected. Continue
            # down the already-scored reserve while retaining the same quality
            # checks, but relax the genre ownership cap enough to fill the row.
            qualified_ids = {item.get("id") for item in qualified}
            min_rating, min_votes = RecommendationFiltering.get_quality_thresholds(self.user_settings)
            for _, item in scored_candidates:
                if len(qualified) >= target:
                    break
                if item.get("id") in qualified_ids:
                    continue
                vote_count = int(item.get("vote_count") or 0)
                vote_avg = float(item.get("vote_average") or 0.0)
                wr = RecommendationScoring.weighted_rating(
                    vote_avg,
                    vote_count,
                    C=7.2 if mtype == "tv" else 6.8,
                )
                if vote_count < min_votes or wr < min_rating:
                    continue
                qualified.append(item)
                qualified_ids.add(item.get("id"))

        rotated = DailyRotation.rotate_items(qualified, rotation_seed)
        result = rotated[:target]

        elapsed = time.time() - start_time
        logger.info(
            f"Top picks complete: returned={len(result)}, qualified={len(qualified)}, "
            f"scored={len(scored_candidates)}, enriched={len(enriched)}, "
            f"raw_sources={dict(source_counts)}, elapsed={elapsed:.2f}s"
        )
        return result

    async def _fetch_recommendations_from_top_items(
        self,
        library_items: dict[str, list[dict[str, Any]]],
        content_type: str,
        mtype: str,
    ) -> list[dict[str, Any]]:
        top_items = self.smart_sampler.sample_items(library_items, content_type, max_items=15)
        tasks = []
        for scored_item in top_items:
            item_id = scored_item.item.id
            if not item_id:
                continue
            tmdb_id = await resolve_tmdb_id(item_id, self.tmdb_service)
            if tmdb_id:
                tasks.append(self.tmdb_service.get_recommendations(tmdb_id, mtype, page=1))

        logger.info(f"Fetching TMDB recommendations from {len(tasks)} top library items")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Recommendation fetch failed: {result}")
                continue
            candidates.extend(result.get("results", []))
        return candidates

    async def _fetch_simkl_recommendations(
        self,
        library_items: dict[str, list[dict[str, Any]]],
        content_type: str,
        mtype: str,
    ) -> list[dict[str, Any]]:
        simkl_api_key = self.user_settings.simkl_api_key if self.user_settings else None
        if not simkl_api_key:
            return []

        top_items = self.smart_sampler.sample_items(library_items, content_type, max_items=15)
        imdb_ids = [
            scored.item.id
            for scored in top_items
            if scored.item.id and scored.item.id.startswith("tt")
        ]
        if not imdb_ids:
            logger.warning("No valid IMDb IDs found for Simkl recommendations")
            return []

        logger.info(f"Fetching Simkl recommendations for {len(imdb_ids)} items")
        year_min = getattr(self.user_settings, "year_min", None)
        year_max = getattr(self.user_settings, "year_max", None)
        try:
            candidates = await simkl_service.get_recommendations_batch(
                imdb_ids,
                mtype,
                simkl_api_key,
                max_per_item=8,
                year_min=year_min,
                year_max=year_max,
            )
        except Exception as exc:
            logger.error(f"Error fetching Simkl recommendations: {exc}")
            return []

        logger.info(f"Fetched {len(candidates)} candidates from Simkl")
        return candidates

    def _add_discover_task(
        self,
        tasks: list,
        mtype: str,
        without_genres: str | None,
        **kwargs: Any,
    ) -> None:
        params = {
            "sort_by": RecommendationFiltering.get_sort_by_preference(self.user_settings),
            **kwargs,
        }
        if without_genres:
            params["without_genres"] = without_genres
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
        """Ask the configured LLM for candidate titles, then resolve them to TMDB."""
        try:
            mtype = content_type_to_mtype(content_type)
            loved = [i for i in library_items.get("loved", []) if i.get("type") == content_type]
            liked = [i for i in library_items.get("liked", []) if i.get("type") == content_type]
            watched = [i for i in library_items.get("watched", []) if i.get("type") == content_type]
            plan_to_watch = [
                i
                for i in library_items.get("added", [])
                if i.get("type") == content_type
                and str(i.get("_provider_status") or "").lower() in {"plantowatch", "planning"}
            ]
            watched.sort(key=lambda x: x.get("state", {}).get("lastWatched", ""), reverse=True)

            loved_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in loved[:20]]
            liked_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in liked[:20]]
            plan_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in plan_to_watch]
            watched_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in watched]

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
                "gems": "Focus on hidden gems — lesser-known, critically acclaimed titles.",
                "all": "Include any quality title regardless of popularity level.",
            }.get(popularity, "Include a mix of well-known titles and lesser-known quality titles.")

            request_limit = max(limit * 3, 30)
            prompt = f"""You are an expert {content_type} recommendation engine.

User Interest Summary: {profile.interest_summary or ''}

Content they LOVED (3.0 — strongest positive signal):
{chr(10).join(loved_lines) if loved_lines else 'None recorded'}

Content they LIKED (2.0 — second strongest positive signal):
{chr(10).join(liked_lines) if liked_lines else 'None recorded'}

Content they PLAN TO WATCH (1.0):
{chr(10).join(plan_lines) if plan_lines else 'None recorded'}

Content they WATCHED (0.5 — weakest positive signal):
{chr(10).join(watched_lines) if watched_lines else 'None recorded'}

DO NOT recommend any title appearing in ANY section above.

TASK: Recommend exactly {request_limit} {content_type}s this person has NOT watched yet.
- Strongly reflect their taste profile and interest summary
- {popularity_instruction}
- Prioritize quality and relevance
- Do not repeat watched/loved/liked/plan-to-watch titles
- Lean toward strongest preferences but include some variety{year_constraint}

RESPONSE FORMAT (one per line, no other text):
{content_type}|Title|Year
"""

            response = await gemini_service.generate_flash_content_async(
                prompt=prompt,
                system_instruction=(
                    f"You are a personalized {content_type} recommendation expert. "
                    "Return ONLY the pipe-separated list, no explanations."
                ),
                api_key=gemini_api_key,
                max_tokens=RECOMMENDATION_MAX_TOKENS,
                minimum_pipe_lines=5,
            )
            if not response:
                return []

            resolve_tasks = []
            parsed = []
            seen_titles = set()
            for line in [line.strip() for line in response.splitlines() if "|" in line]:
                if len(parsed) >= request_limit:
                    break
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                name = re.sub(r"\s*\(\d{4}\)\s*$", "", parts[1].strip()).strip()
                year = parts[2].strip()[:4]
                title_key = name.lower()
                if not name or title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                parsed.append((name, year))
                resolve_tasks.append(self._resolve_title_to_tmdb(name, year, mtype))

            results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
            candidates = [
                result
                for result in results
                if isinstance(result, dict) and result.get("id")
            ]
            logger.info(f"Gemini top picks: resolved {len(candidates)}/{len(parsed)} titles")
            logger.info(
                f"Gemini picks: {[c.get('title') or c.get('name') for c in candidates]}"
            )
            return candidates
        except Exception as exc:
            logger.warning(f"Gemini top picks failed; continuing with other sources: {exc}")
            return []

    async def _resolve_title_to_tmdb(self, name: str, year: str, mtype: str) -> dict | None:
        try:
            search_type = "tv" if mtype == "tv" else "movie"
            params: dict[str, Any] = {"query": name, "page": 1}
            if year:
                params["first_air_date_year" if search_type == "tv" else "primary_release_year"] = year
            response = await self.tmdb_service.client.get(f"/search/{search_type}", params=params)
            items = response.get("results", [])
            if not items:
                response = await self.tmdb_service.client.get(
                    f"/search/{search_type}", params={"query": name, "page": 1}
                )
                items = response.get("results", [])
            if not items:
                return None

            def norm(value: str) -> str:
                return "".join(
                    ch for ch in (value or "").lower() if ch.isalnum() or ch == " "
                ).strip()

            wanted = norm(name)
            wanted_words = set(wanted.split())
            if not wanted_words:
                return None

            for item in items[:5]:
                candidate = norm(item.get("title") or item.get("name") or "")
                if not candidate:
                    continue
                if candidate == wanted:
                    return item
                candidate_words = set(candidate.split())
                overlap = len(wanted_words & candidate_words) / len(wanted_words)
                if overlap == 1.0 and candidate.startswith(wanted):
                    return item
                if overlap >= 0.7 and abs(len(candidate_words) - len(wanted_words)) <= 3:
                    return item

            logger.debug(
                f"[TopPicks] no confident TMDB match for {name!r} "
                f"(best was {items[0].get('title') or items[0].get('name')!r})"
            )
            return None
        except Exception as exc:
            logger.debug(f"Failed to resolve title {name}: {exc}")
            return None

    async def _fetch_discover_with_profile(
        self,
        profile: TasteProfile,
        content_type: str,
        mtype: str,
    ) -> list[dict[str, Any]]:
        excluded = RecommendationFiltering.get_excluded_genre_ids(self.user_settings, content_type)
        without_genres = "|".join(str(gid) for gid in excluded) if excluded else None
        top_genres = profile.get_top_genres(limit=5)
        top_keywords = profile.get_top_keywords(limit=5)
        top_directors = profile.get_top_directors(limit=3)
        top_cast = profile.get_top_cast(limit=5)
        top_eras = profile.get_top_eras(limit=2)

        tasks = []
        if top_genres:
            max_score = top_genres[0][1] or 1.0
            for genre_id, score in top_genres:
                ratio = score / max_score if max_score else 0.0
                pages = 3 if ratio >= 0.8 else 2 if ratio >= 0.5 else 1
                for page in range(1, pages + 1):
                    self._add_discover_task(
                        tasks,
                        mtype,
                        without_genres,
                        with_genres=str(genre_id),
                        page=page,
                    )

        if top_keywords:
            from app.services.row_generator import GENERIC_KEYWORD_BLACKLIST

            keyword_ids = [
                keyword_id
                for keyword_id, _ in top_keywords
                if keyword_id not in GENERIC_KEYWORD_BLACKLIST
            ]
            if keyword_ids:
                for page in range(1, 3):
                    self._add_discover_task(
                        tasks,
                        mtype,
                        without_genres,
                        with_keywords="|".join(str(k) for k in keyword_ids),
                        page=page,
                    )

        if top_directors:
            self._add_discover_task(
                tasks,
                mtype,
                without_genres,
                with_crew="|".join(str(director_id) for director_id, _ in top_directors),
                page=1,
            )

        if top_cast:
            self._add_discover_task(
                tasks,
                mtype,
                without_genres,
                with_cast="|".join(str(cast_id) for cast_id, _ in top_cast),
                page=1,
            )

        if top_eras:
            year_start = self._era_to_year_start(top_eras[0][0])
            if year_start:
                prefix = "first_air_date" if mtype == "tv" else "primary_release_date"
                end = (
                    date.today().isoformat()
                    if year_start + 9 > date.today().year
                    else f"{year_start + 9}-12-31"
                )
                self._add_discover_task(
                    tasks,
                    mtype,
                    without_genres,
                    **{
                        f"{prefix}.gte": f"{year_start}-01-01",
                        f"{prefix}.lte": end,
                        "page": 1,
                    },
                )

        logger.debug(f"Fetching {len(tasks)} discover queries with profile features")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Discover query failed: {result}")
                continue
            candidates.extend(result.get("results", []))
        logger.debug(f"Fetched {len(candidates)} candidates from discover")
        return candidates

    def _apply_diversity_caps(
        self,
        scored_candidates: list[tuple[float, dict[str, Any]]],
        limit: int,
        mtype: str,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        result: list[dict[str, Any]] = []
        genre_counts: dict[int, int] = defaultdict(int)
        max_per_genre = max(1, int(limit * TOP_PICKS_GENRE_CAP))
        min_rating, min_votes = RecommendationFiltering.get_quality_thresholds(self.user_settings)

        for _, item in scored_candidates:
            if len(result) >= limit:
                break

            vote_count = int(item.get("vote_count") or 0)
            vote_avg = float(item.get("vote_average") or 0.0)
            if vote_count < min_votes:
                continue

            wr = RecommendationScoring.weighted_rating(
                vote_avg,
                vote_count,
                C=7.2 if mtype == "tv" else 6.8,
            )
            if wr < min_rating:
                continue

            genre_ids = item.get("genre_ids") or []
            top_genre = genre_ids[0] if genre_ids else None
            if top_genre is not None and genre_counts[top_genre] >= max_per_genre:
                continue

            result.append(item)
            if top_genre is not None:
                genre_counts[top_genre] += 1

        return result

    @staticmethod
    def _era_to_year_start(era: str) -> int | None:
        return {
            "pre-1970s": 1950,
            "1970s": 1970,
            "1980s": 1980,
            "1990s": 1990,
            "2000s": 2000,
            "2010s": 2010,
            "2020s": 2020,
        }.get(era)
