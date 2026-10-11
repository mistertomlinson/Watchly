"""Production entry point for the unified Top Picks V2 pipeline.

The ranking/candidate implementation lives in ``top_picks_v2``. This module
keeps the stable import used by CatalogService and adds the final movie
home-availability guard so a theatrical-only title cannot be published as a
Top Pick.
"""

from typing import Any

from loguru import logger

from app.core.constants import MAX_CATALOG_ITEMS
from app.models.taste_profile import TasteProfile
from app.services.recommendation.availability import filter_home_released_movies
from app.services.recommendation.top_picks_v2 import TopPicksService as UnifiedTopPicksService


UNIFIED_TOP_PICKS_V2 = True


class TopPicksService(UnifiedTopPicksService):
    """Unified V2 Top Picks plus a final digital/physical availability gate."""

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
        requested = max(1, min(int(limit), MAX_CATALOG_ITEMS))

        # Keep V2's validated display target unchanged. Asking V2 for a larger
        # target just to create replacement headroom would also raise its minimum
        # qualifying count and could admit weaker titles. Quality stays ahead of
        # row fullness here: generate the normal V2 result first, then remove any
        # recent movie that cannot be verified as home-released.
        picks = await super().get_top_picks(
            profile=profile,
            content_type=content_type,
            library_items=library_items,
            watched_tmdb=watched_tmdb,
            watched_imdb=watched_imdb,
            limit=requested,
            gemini_api_key=gemini_api_key,
            library_items_raw=library_items_raw,
        )

        if content_type != "movie":
            return picks[:requested]

        available = await filter_home_released_movies(
            self.tmdb_service,
            picks,
            self.user_settings,
        )

        # CatalogService's generic <8 emergency padding is intentionally not a
        # quality/availability-aware Top Picks source. Returning an empty result
        # here if the availability gate ever starves catastrophically prevents
        # that fallback from reintroducing the very theatrical-only titles this
        # guard removed. With the V2 pool this should be exceptional.
        if 0 < len(available) < 8:
            logger.warning(
                "[HomeAvailability] Top Picks fell below 8 verified movies; "
                "returning empty rather than allowing generic padding"
            )
            return []

        return available[:requested]


__all__ = ["TopPicksService", "UNIFIED_TOP_PICKS_V2"]
