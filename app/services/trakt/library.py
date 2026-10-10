import asyncio
from typing import Any

from loguru import logger

from app.services.library_provider import add_rated_item, empty_library, make_library_item
from app.services.trakt.client import TraktClient


class TraktLibraryService:
    """Normalize a Trakt account into Watchly's shared library contract."""

    def __init__(self, client: TraktClient):
        self.client = client

    async def get_library_items(self) -> dict[str, list[dict[str, Any]]]:
        library = empty_library()
        try:
            movies, shows, movie_ratings, show_ratings = await asyncio.gather(
                self._get_history("movies"),
                self._get_history("shows"),
                self._get_ratings("movies"),
                self._get_ratings("shows"),
            )

            watched_ids: set[str] = set()
            for raw in movies:
                self._append_history_item(library, raw, "movie", "movie", watched_ids)
            for raw in shows:
                self._append_history_item(library, raw, "series", "show", watched_ids)

            rated_ids: set[str] = set()
            for raw in movie_ratings + show_ratings:
                media_key = "movie" if "movie" in raw else "show"
                media = raw.get(media_key) or {}
                rating = self._coerce_rating(raw.get("rating"))
                item = make_library_item(
                    ids=media.get("ids") or {},
                    content_type="movie" if media_key == "movie" else "series",
                    title=media.get("title") or "",
                    year=media.get("year"),
                    provider="trakt",
                    watched_at=raw.get("rated_at"),
                    rating=rating,
                )
                if not item or item["_id"] in rated_ids:
                    continue
                rated_ids.add(item["_id"])
                add_rated_item(library, item)

            logger.info(
                f"[Trakt] library: {len(library['watched'])} watched, "
                f"{len(library['loved'])} loved, {len(library['liked'])} liked, "
                f"{len(library['disliked'])} disliked"
            )
            return library
        except Exception as exc:
            logger.exception(f"[Trakt] Failed to get library items: {exc}")
            return empty_library()

    async def _get_ratings(self, media_type: str) -> list[dict[str, Any]]:
        try:
            data = await self.client.get(f"/users/me/ratings/{media_type}")
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"[Trakt] Failed to fetch {media_type} ratings: {exc}")
            return []

    async def _get_history(self, media_type: str) -> list[dict[str, Any]]:
        try:
            data = await self.client.get(f"/users/me/watched/{media_type}")
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"[Trakt] Failed to fetch {media_type} history: {exc}")
            return []

    def _append_history_item(
        self,
        library: dict[str, list[dict[str, Any]]],
        raw: dict[str, Any],
        content_type: str,
        media_key: str,
        seen_ids: set[str],
    ) -> None:
        media = raw.get(media_key) or {}
        item = make_library_item(
            ids=media.get("ids") or {},
            content_type=content_type,
            title=media.get("title") or "",
            year=media.get("year"),
            provider="trakt",
            watched_at=raw.get("last_watched_at"),
            plays=int(raw.get("plays") or 1),
        )
        if item and item["_id"] not in seen_ids:
            seen_ids.add(item["_id"])
            library["watched"].append(item)

    @staticmethod
    def _coerce_rating(value: Any) -> int | None:
        try:
            rating = int(value)
        except (TypeError, ValueError):
            return None
        return rating if 1 <= rating <= 10 else None
