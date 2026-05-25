from app.services.gemini import gemini_service
import asyncio
from typing import Any

from loguru import logger

from app.services.recommendation.filtering import RecommendationFiltering
from app.services.recommendation.metadata import RecommendationMetadata
from app.services.recommendation.utils import (
    content_type_to_mtype,
    filter_by_genres,
    filter_items_by_settings,
    filter_watched_by_imdb,
    resolve_tmdb_id,
)
from app.services.simkl import simkl_service
from app.services.tmdb.service import TMDBService


class ItemBasedService:
    """
    Handles item-based recommendations (Because you watched/loved).
    """

    def __init__(self, tmdb_service: Any, user_settings: Any = None):
        self.tmdb_service: TMDBService = tmdb_service
        self.user_settings = user_settings

    async def get_recommendations_for_item(
        self,
        item_id: str,
        content_type: str,
        watched_tmdb: set[int] | None = None,
        watched_imdb: set[str] | None = None,
        limit: int = 20,
        whitelist: set[int] | None = None,
        gemini_api_key: str | None = None,
        library_items: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get recommendations for a specific item.

        Strategy:
        1. Fetch similar + recommendations from TMDB (2 pages each)
        2. Filter watched items
        3. Filter excluded genres
        4. Apply genre whitelist
        5. Return top N

        Args:
            item_id: Item ID (tt... or tmdb:...)
            content_type: Content type (movie/series)
            watched_tmdb: Set of watched TMDB IDs
            watched_imdb: Set of watched IMDB IDs
            limit: Number of items to return

        Returns:
            List of recommended items
        """
        # Try Gemini-based recommendations first if API key available
        if gemini_api_key and library_items:
            gemini_results = await self._fetch_gemini_item_recommendations(
                item_id, content_type, library_items, gemini_api_key, limit
            )
            if len(gemini_results) >= limit // 2:
                logger.info(f"Using Gemini recommendations for {item_id}: {len(gemini_results)} results")
                return gemini_results
            logger.info(f"Gemini returned only {len(gemini_results)} for {item_id}, falling back to TMDB")

        # Resolve TMDB ID
        tmdb_id = await resolve_tmdb_id(item_id, self.tmdb_service)
        if not tmdb_id:
            return []

        # Exclude source item
        watched_tmdb = watched_tmdb.copy() if watched_tmdb else set()
        watched_tmdb.add(tmdb_id)

        mtype = content_type_to_mtype(content_type)

        # Fetch candidates (similar + recommendations, 2 pages each)
        tasks = [self._fetch_candidates_from_simkl(item_id, mtype), self._fetch_candidates(tmdb_id, mtype)]
        simkl_candidates, candidates = await asyncio.gather(*tasks)


        # extend candidates always include simkl candidates
        candidates = simkl_candidates + candidates

        # Filter by genres and watched items
        excluded_ids = RecommendationFiltering.get_excluded_genre_ids(self.user_settings, content_type)
        filtered = filter_by_genres(candidates, watched_tmdb, whitelist, excluded_ids, watched_imdb=watched_imdb or set())

        # Enrich metadata
        enriched = await RecommendationMetadata.fetch_batch(
            self.tmdb_service, filtered, content_type, user_settings=self.user_settings
        )

        # Final filter (remove watched by IMDB ID)
        final = filter_watched_by_imdb(enriched, watched_imdb or set())

        # Apply year and popularity filters from user settings
        final = filter_items_by_settings(final, self.user_settings)

        return final

    async def _fetch_gemini_item_recommendations(
        self,
        item_id: str,
        content_type: str,
        library_items: dict,
        gemini_api_key: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Use Gemini to recommend titles similar to a seed item."""
        try:
            # Resolve seed item title from TMDB
            tmdb_id = await resolve_tmdb_id(item_id, self.tmdb_service)
            if not tmdb_id:
                return []
            mtype = content_type_to_mtype(content_type)
            details = await self.tmdb_service.client.get(
                f"/{mtype}/{tmdb_id}", params={"language": "en-US"}
            )
            seed_title = details.get("title") or details.get("name", item_id)
            seed_year = (details.get("release_date") or details.get("first_air_date") or "")[:4]
            seed_overview = (details.get("overview") or "")[:400]
            # Detail endpoint returns 'genres' as objects, not 'genre_ids'
            genres_list = details.get("genres") or []
            genre_ids = [g["id"] for g in genres_list if isinstance(g, dict) and g.get("id")]
            if not genre_ids:
                genre_ids = details.get("genre_ids") or []
            from app.services.tmdb.genre import movie_genres, series_genres
            genre_map = movie_genres if mtype == "movie" else series_genres
            seed_genres = ", ".join([genre_map.get(gid, "") for gid in genre_ids if genre_map.get(gid)])
            watched = [i for i in library_items.get("watched", []) if i.get("type") == content_type]
            watched_lines = [f"- {i.get('name')} ({i.get('year', 'N/A')})" for i in watched]

            year_min = getattr(self.user_settings, "year_min", None)
            year_max = getattr(self.user_settings, "year_max", None)
            year_constraint = ""
            if year_min and year_max:
                year_constraint = f"\n- ONLY recommend titles released between {year_min} and {year_max}. Do not suggest anything outside this range."
            content_label = seed_genres if seed_genres else content_type
            prompt = f"""You are a {content_label} recommendation expert.

The user just watched: {seed_title} ({seed_year})

About this title: {seed_overview}
Genres: {seed_genres}


Their watch history (DO NOT recommend these):
{chr(10).join(watched_lines) if watched_lines else "None recorded"}
TASK: Recommend exactly {limit} {content_type}s that are similar to "{seed_title}" in theme, tone, style, AND genre. If the seed title is a documentary, only recommend documentaries. If it is a horror film, recommend horror films. Match the genre closely.
- Focus on similarity to the seed title
- Avoid anything in their watch history above
- Include both well-known and obscure titles{year_constraint}

RESPONSE FORMAT (one per line, no other text):
{content_type}|Title|Year"""

            # Request extra results to compensate for post-filtering losses
            gemini_limit = min(limit * 4, 120)
            prompt = prompt.replace(f"Recommend exactly {limit} {content_type}s", f"Recommend exactly {gemini_limit} {content_type}s")
            response = await gemini_service.generate_flash_content_async(
                prompt=prompt,
                system_instruction=f"You are a {content_type} recommendation expert specializing in {seed_genres if seed_genres else content_type} content. The seed title is a {seed_genres} title. ONLY recommend {seed_genres} titles. Return ONLY the pipe-separated list.",
                api_key=gemini_api_key,
            )

            if not response:
                return []

            candidates = []
            lines = [l.strip() for l in response.strip().splitlines() if l.strip() and "|" in l]
            resolve_tasks = []
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 3:
                    name, year = parts[1].strip(), parts[2].strip()[:4]
                    resolve_tasks.append(self._resolve_title(name, year, mtype))
                elif len(parts) == 2:
                    name, year = parts[0].strip(), parts[1].strip()[:4]
                    resolve_tasks.append(self._resolve_title(name, year, mtype))

            results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
            logger.info(f"Gemini raw lines for {seed_title}: {lines[:5]}")
            for result in results:
                if isinstance(result, dict) and result.get("id"):
                    candidates.append(result)

            # Enrich with IMDB IDs and full metadata
            enriched = await RecommendationMetadata.fetch_batch(
                self.tmdb_service, candidates, content_type, user_settings=self.user_settings
            )
            logger.info(f"Gemini item recs for {seed_title}: {len(enriched)} enriched")
            # Hard post-filter to enforce year constraints — Gemini doesn't always
            # respect the year constraint in the prompt alone.
            enriched = filter_items_by_settings(enriched, self.user_settings)
            logger.info(f"Gemini item recs for {seed_title}: {len(enriched)} after year filter")
            return enriched

        except Exception as e:
            logger.warning(f"Gemini item recommendations failed for {item_id}: {e}")
            return []

    async def _resolve_title(self, name: str, year: str, mtype: str) -> dict | None:
        """Resolve title+year to TMDB item."""
        try:
            search_type = "tv" if mtype == "tv" else "movie"
            params = {"query": name, "page": 1}
            if year:
                key = "first_air_date_year" if search_type == "tv" else "primary_release_year"
                params[key] = year
            results = await self.tmdb_service.client.get(f"/search/{search_type}", params=params)
            items = results.get("results", [])
            if not items:
                results2 = await self.tmdb_service.client.get(f"/search/{search_type}", params={"query": name, "page": 1})
                items = results2.get("results", [])
            if items:
                items[0]["media_type"] = search_type
                return items[0]
            return None
        except Exception:
            return None

    async def _fetch_candidates_from_simkl(self, imdb_id: str, mtype: str):
        # check if user_settings has simkl api key or not
        logger.info("Fetching recommendations from Simkl")
        simkl_api_key = self.user_settings.simkl_api_key
        if not simkl_api_key:
            logger.warning("Simkl API key not found. Using TMDB for recommendations")
            return []
        return await simkl_service.get_recommendations(imdb_id, mtype, simkl_api_key)

    async def _fetch_candidates(self, tmdb_id: int, mtype: str) -> list[dict[str, Any]]:
        """
        Fetch candidates from TMDB (similar + recommendations).

        Args:
            tmdb_id: TMDB ID
            mtype: Media type (movie/tv)

        Returns:
            List of candidate items
        """
        combined = {}

        async def fetch_and_combine(fetch_method, source_name, pages: list[int] = [1, 2, 3]):
            results = await asyncio.gather(
                *[fetch_method(tmdb_id, mtype, page=p) for p in pages],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"Error fetching {source_name} for {tmdb_id}: {res}")
                    continue
                for item in res.get("results", []):
                    item_id = item.get("id")
                    if item_id:
                        combined[item_id] = item

        await fetch_and_combine(self.tmdb_service.get_recommendations, "recommendations")

        if not combined or len(combined) < 30:
            await fetch_and_combine(self.tmdb_service.get_similar, "similar")

        # apply filter and check
        filtered = filter_items_by_settings(combined.values(), self.user_settings)

        if not filtered or len(filtered) < 30:
            # fetch more similar items if there are less than 30 items after user_settings filter
            await fetch_and_combine(self.tmdb_service.get_similar, "similar", pages=[4, 5, 6])

        return list(combined.values())
