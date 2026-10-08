from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


EMPTY_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "watched": [],
    "loved": [],
    "liked": [],
    "disliked": [],
    "added": [],
    "removed": [],
}


class RatingBucket(str, Enum):
    LOVED = "loved"
    LIKED = "liked"
    NEUTRAL = "neutral"
    DISLIKED = "disliked"


@dataclass(frozen=True)
class RatingPolicy:
    """Canonical Watchly rating semantics shared by every provider."""

    loved_rating: int = 10
    liked_min: int = 7
    liked_max: int = 9
    disliked_max: int = 2

    def classify(self, rating: int | None) -> RatingBucket:
        if rating == self.loved_rating:
            return RatingBucket.LOVED
        if rating is not None and self.liked_min <= rating <= self.liked_max:
            return RatingBucket.LIKED
        if rating is not None and 1 <= rating <= self.disliked_max:
            return RatingBucket.DISLIKED
        return RatingBucket.NEUTRAL

    def taste_weight(self, rating: int | None) -> float:
        bucket = self.classify(rating)
        if bucket == RatingBucket.LOVED:
            return 1.0
        if bucket == RatingBucket.LIKED:
            return 0.65
        return 0.0


RATING_POLICY = RatingPolicy()


class LibraryProvider(Protocol):
    async def get_library_items(self) -> dict[str, list[dict[str, Any]]]: ...


def empty_library() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in EMPTY_LIBRARY}


def best_external_id(ids: dict[str, Any]) -> str | None:
    imdb = ids.get("imdb") or ids.get("imdb_id")
    if imdb:
        return str(imdb)
    tmdb = ids.get("tmdb") or ids.get("tmdb_id")
    if tmdb:
        return f"tmdb:{tmdb}"
    simkl = ids.get("simkl") or ids.get("simkl_id")
    if simkl:
        return f"simkl:{simkl}"
    return None


def normalize_external_ids(ids: dict[str, Any]) -> dict[str, Any]:
    """Preserve all provider identities instead of collapsing to only ``_id``.

    Watchly still keeps a single canonical ``_id`` for compatibility with the
    existing library/profile code, but recommendation exclusion needs both IMDb
    and TMDB identities when a provider supplies them.
    """
    external: dict[str, Any] = {}

    imdb = ids.get("imdb") or ids.get("imdb_id")
    if imdb:
        external["imdb_id"] = str(imdb)

    tmdb = ids.get("tmdb") or ids.get("tmdb_id")
    if tmdb not in (None, ""):
        try:
            external["tmdb_id"] = int(str(tmdb).removeprefix("tmdb:"))
        except (TypeError, ValueError):
            pass

    simkl = ids.get("simkl") or ids.get("simkl_id")
    if simkl not in (None, ""):
        external["simkl_id"] = simkl

    return external


def make_library_item(
    *,
    ids: dict[str, Any],
    content_type: str,
    title: str,
    year: int | None,
    provider: str,
    watched_at: str | None = None,
    plays: int = 1,
    rating: int | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    canonical_id = best_external_id(ids)
    if not canonical_id:
        return None

    bucket = RATING_POLICY.classify(rating)
    return {
        "_id": canonical_id,
        "_external_ids": normalize_external_ids(ids),
        "type": content_type,
        "name": title,
        "year": year,
        "state": {
            "timesWatched": max(plays, 0),
            "flaggedWatched": 1 if plays > 0 else 0,
            "lastWatched": watched_at or "",
        },
        "temp": False,
        "removed": False,
        "_source": provider,
        "_provider_status": status,
        "_personal_rating": rating,
        "_rating_bucket": bucket.value,
        "_taste_weight": RATING_POLICY.taste_weight(rating),
        "_is_loved": bucket == RatingBucket.LOVED,
        "_is_liked": bucket == RatingBucket.LIKED,
        "_is_disliked": bucket == RatingBucket.DISLIKED,
        "_mtime": watched_at or "",
    }


def add_rated_item(library: dict[str, list[dict[str, Any]]], item: dict[str, Any]) -> None:
    bucket = item.get("_rating_bucket")
    if bucket == RatingBucket.LOVED.value:
        library["loved"].append(item)
    elif bucket == RatingBucket.LIKED.value:
        library["liked"].append(item)
    elif bucket == RatingBucket.DISLIKED.value:
        library["disliked"].append(item)
    # Neutral ratings intentionally remain watched-only.
