from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from app.services.library_provider import add_rated_item, empty_library, make_library_item


class SimklApiClient:
    """Authenticated Simkl client for user-library endpoints."""

    BASE_URL = "https://api.simkl.com"

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {access_token}",
                "simkl-api-key": client_id,
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def get(self, path: str, **params: Any) -> Any:
        response = await self.client.get(path, params={k: v for k, v in params.items() if v is not None})
        response.raise_for_status()
        return response.json()

    async def get_user(self) -> dict[str, Any]:
        data = await self.get("/users/settings")
        return data if isinstance(data, dict) else {}

    async def get_all_items(self, media_type: str) -> Any:
        # Simkl documents these path values as movie, tv, and anime.
        return await self.get(
            f"/sync/all-items/{media_type}",
            extended="full",
            episode_watched_at="yes",
        )


class SimklLibraryProvider:
    """Normalize a Simkl account into Watchly's shared library contract."""

    def __init__(self, client: SimklApiClient):
        self.client = client

    async def get_library_items(self) -> dict[str, list[dict[str, Any]]]:
        library = empty_library()
        try:
            movies_payload, shows_payload, anime_payload = await asyncio.gather(
                self.client.get_all_items("movie"),
                self.client.get_all_items("tv"),
                self.client.get_all_items("anime"),
            )

            seen_watched: set[str] = set()
            seen_rated: set[str] = set()
            seen_added: set[str] = set()

            self._consume_entries(
                library,
                self._extract_entries(movies_payload, "movies"),
                content_type="movie",
                media_keys=("movie",),
                seen_watched=seen_watched,
                seen_rated=seen_rated,
                seen_added=seen_added,
            )
            self._consume_entries(
                library,
                self._extract_entries(shows_payload, "shows", "tv"),
                content_type="series",
                media_keys=("show", "tv"),
                seen_watched=seen_watched,
                seen_rated=seen_rated,
                seen_added=seen_added,
            )
            self._consume_entries(
                library,
                self._extract_entries(anime_payload, "anime"),
                content_type="series",
                media_keys=("show", "anime"),
                seen_watched=seen_watched,
                seen_rated=seen_rated,
                seen_added=seen_added,
            )

            logger.info(
                "[Simkl] library: {} watched, {} loved, {} liked, {} disliked, {} added",
                len(library["watched"]),
                len(library["loved"]),
                len(library["liked"]),
                len(library["disliked"]),
                len(library["added"]),
            )
            return library
        except Exception as exc:
            logger.exception(f"[Simkl] Failed to get library items: {exc}")
            return empty_library()

    @staticmethod
    def _extract_entries(payload: Any, *keys: str) -> list[dict[str, Any]]:
        """Accept Simkl's list response and older/object-wrapped response shapes."""
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return [entry for entry in value if isinstance(entry, dict)]
            # Some API responses group entries by status.
            grouped: list[dict[str, Any]] = []
            for value in payload.values():
                if isinstance(value, list):
                    grouped.extend(entry for entry in value if isinstance(entry, dict))
            return grouped
        return []

    def _consume_entries(
        self,
        library: dict[str, list[dict[str, Any]]],
        entries: list[dict[str, Any]],
        *,
        content_type: str,
        media_keys: tuple[str, ...],
        seen_watched: set[str],
        seen_rated: set[str],
        seen_added: set[str],
    ) -> None:
        for raw in entries:
            media = self._find_media(raw, media_keys)
            ids = media.get("ids") or raw.get("ids") or {}
            status = str(raw.get("status") or raw.get("list") or "").lower().replace("_", "").replace(" ", "")
            rating = self._coerce_rating(
                raw.get("user_rating")
                or raw.get("rating")
                or (raw.get("ratings") or {}).get("user")
            )
            watched_at = (
                raw.get("last_watched_at")
                or raw.get("watched_at")
                or raw.get("watched_date")
                or raw.get("last_watched")
            )
            watched_count = (
                raw.get("watched_episodes_count")
                or raw.get("episodes_watched")
                or raw.get("plays")
                or (1 if watched_at or status == "completed" else 0)
            )

            item = make_library_item(
                ids=ids,
                content_type=content_type,
                title=media.get("title") or media.get("name") or raw.get("title") or "",
                year=media.get("year") or raw.get("year"),
                provider="simkl",
                watched_at=watched_at,
                plays=max(int(watched_count or 0), 1),
                rating=rating,
                status=status,
            )
            if not item:
                continue

            item_id = item["_id"]
            is_watched = bool(
                watched_at
                or watched_count
                or status in {"watching", "completed", "hold", "onhold", "dropped"}
            )
            is_added = status in {
                "plantowatch", "planning", "watching", "completed", "hold", "onhold", "dropped"
            }

            if is_watched and item_id not in seen_watched:
                seen_watched.add(item_id)
                library["watched"].append(item)

            if rating is not None and item_id not in seen_rated:
                seen_rated.add(item_id)
                add_rated_item(library, item)

            if is_added and item_id not in seen_added:
                seen_added.add(item_id)
                library["added"].append(item)

    @staticmethod
    def _find_media(raw: dict[str, Any], media_keys: tuple[str, ...]) -> dict[str, Any]:
        for key in media_keys:
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        return raw

    @staticmethod
    def _coerce_rating(value: Any) -> int | None:
        try:
            rating = int(value)
        except (TypeError, ValueError):
            return None
        return rating if 1 <= rating <= 10 else None
