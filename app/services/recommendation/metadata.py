import asyncio
from typing import Any

from loguru import logger

from app.core.constants import DEFAULT_CONCURRENCY_LIMIT
from app.services.content_preferences import lgbtq_content_preference_reason_values
from app.services.poster_ratings.factory import PosterProvider, poster_ratings_factory


class RecommendationMetadata:
    """
    Handles fetching and formatting metadata for Stremio.
    """

    @staticmethod
    def extract_year(item: dict[str, Any]) -> int | None:
        """Extract year from TMDB item."""
        date_str = item.get("release_date") or item.get("first_air_date")
        if not date_str:
            ri = item.get("releaseInfo")
            if isinstance(ri, str) and len(ri) >= 4 and ri[:4].isdigit():
                return int(ri[:4])
            return None
        try:
            return int(date_str[:4])
        except Exception:
            return None

    @classmethod
    async def format_for_stremio(
        cls,
        details: dict[str, Any],
        media_type: str,
        user_settings: Any = None,
        logo_url: str | None = None,
    ) -> dict[str, Any] | None:
        """Format TMDB details into Stremio metadata object."""
        external_ids = details.get("external_ids", {})
        imdb_id = external_ids.get("imdb_id")
        tmdb_id_raw = details.get("id")

        if imdb_id:
            stremio_id = imdb_id
        elif tmdb_id_raw:
            stremio_id = f"tmdb:{tmdb_id_raw}"
        else:
            return None

        title = details.get("title") or details.get("name")
        if not title:
            return None

        # Base Fields
        genres_full = details.get("genres", []) or []
        release_date = details.get("release_date") or details.get("first_air_date") or ""

        meta_data = {
            "id": stremio_id,
            "imdb_id": imdb_id,
            "type": "series" if media_type in ["tv", "series"] else "movie",
            "name": title,
            "poster": cls._get_poster_url(details, stremio_id, user_settings),
            "background": cls._get_backdrop_url(details),
            "description": details.get("overview"),
            "releaseInfo": release_date[:4] if release_date else None,
            "released": release_date if release_date else None,
            "imdbRating": str(details.get("vote_average", "")),
            "genres": [g.get("name") for g in genres_full if isinstance(g, dict)],
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "original_language": details.get("original_language"),
            "_external_ids": external_ids,
            "_tmdb_id": tmdb_id_raw,
            "genre_ids": [g.get("id") for g in genres_full if isinstance(g, dict) and g.get("id") is not None],
        }
        if logo_url:
            meta_data["logo"] = logo_url

        # Extensions
        runtime_str = cls._extract_runtime_string(details)
        if runtime_str:
            meta_data["runtime"] = runtime_str

        if media_type == "movie":
            coll = details.get("belongs_to_collection")
            if isinstance(coll, dict):
                meta_data["_collection_id"] = coll.get("id")

        # Cast & Crew
        cast = details.get("credits", {}).get("cast", []) or []
        meta_data["_top_cast_ids"] = [c.get("id") for c in cast[:3] if isinstance(c, dict) and c.get("id")]

        # Keywords & Credits for similarity re-ranking
        if details.get("keywords"):
            meta_data["keywords"] = details.get("keywords")
        if details.get("credits"):
            meta_data["credits"] = details.get("credits")

        return meta_data

    @staticmethod
    def _get_poster_url(details: dict, item_id: str, user_settings: Any) -> str | None:
        """Resolve poster URL using poster rating provider if configured, otherwise TMDB."""
        path = details.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{path}"

        if user_settings:
            poster_rating = user_settings.poster_rating
            if poster_rating and poster_rating.api_key:
                try:
                    provider_enum = PosterProvider(poster_rating.provider)
                    poster_url = poster_ratings_factory.get_poster_url(
                        provider_enum, poster_rating.api_key, "imdb", item_id, fallback=poster_url
                    )
                except ValueError as e:
                    logger.warning(f"Error getting poster URL for item ID {item_id}: {e}")
                    pass

        return poster_url

    @staticmethod
    def _get_backdrop_url(details: dict) -> str | None:
        """Construct full TMDB backdrop URL."""
        path = details.get("backdrop_path")
        return f"https://image.tmdb.org/t/p/original{path}" if path else None

    @staticmethod
    def _extract_runtime_string(details: dict) -> str | None:
        """Extract and format runtime from either movie or TV format."""
        runtime = details.get("runtime")
        if not runtime and details.get("episode_run_time"):
            runtime = details.get("episode_run_time")[0]
        return f"{runtime} min" if runtime else None

    @classmethod
    async def fetch_batch(
        cls,
        tmdb_service: Any,
        items: list[dict[str, Any]],
        media_type: str,
        user_settings: Any = None,
    ) -> list[dict[str, Any]]:
        """Fetch details for a batch of items in parallel with target-based short-circuiting."""
        final_results = []
        valid_items = [it for it in items if it.get("id")]
        query_type = "movie" if media_type == "movie" else "tv"
        sem = asyncio.Semaphore(DEFAULT_CONCURRENCY_LIMIT)

        async def _fetch_one(tid: int):
            async with sem:
                try:
                    if query_type == "movie":
                        return await tmdb_service.get_movie_details(tid)
                    return await tmdb_service.get_tv_details(tid)
                except Exception:
                    return None

        tasks = [_fetch_one(it.get("id")) for it in valid_items]
        details_list = await asyncio.gather(*tasks)

        language = getattr(user_settings, "language", None) or "en-US"
        mt = "movie" if media_type == "movie" else "tv"

        async def _images_one(d: dict[str, Any]) -> dict[str, str]:
            async with sem:
                try:
                    return await tmdb_service.get_images_for_title(mt, d["id"], language=language)
                except Exception:
                    return {}

        successful_details = [d for d in details_list if d]

        # Global content-preference gate. Full TMDB details include keyword names,
        # allowing the same corroboration threshold used by Seasonal Spotlight.
        # Run before image requests so excluded titles incur no extra artwork calls.
        preference_filtered_details = []
        for details in successful_details:
            reason = lgbtq_content_preference_reason_values(
                title=details.get("title") or details.get("name") or "",
                overview=details.get("overview") or "",
                keywords=details.get("keywords"),
            )
            if reason:
                logger.info(
                    "[ContentPreference] Excluding recommendation "
                    f"tmdb={details.get('id')} "
                    f"title={details.get('title') or details.get('name') or ''!r} "
                    f"reason={reason}"
                )
                continue
            preference_filtered_details.append(details)

        successful_details = preference_filtered_details
        image_tasks = [_images_one(d) for d in successful_details]
        images_list = await asyncio.gather(*image_tasks, return_exceptions=True)

        format_task = []
        for details, imgs in zip(successful_details, images_list):
            logo_url = None
            if isinstance(imgs, dict):
                logo_url = imgs.get("logo") or None
            format_task.append(cls.format_for_stremio(details, media_type, user_settings, logo_url=logo_url))

        formatted_list = await asyncio.gather(*format_task, return_exceptions=True)

        preferred_language = ((getattr(user_settings, "language", None) or "").split("-")[0].lower() if user_settings else "")

        for formatted in formatted_list:
            if isinstance(formatted, Exception):
                logger.warning(f"Error formatting metadata: {formatted}")
                continue
            if formatted:
                original_language = (formatted.get("original_language") or "").lower()
                if preferred_language and original_language and original_language != preferred_language:
                    continue
                final_results.append(formatted)

        return final_results
