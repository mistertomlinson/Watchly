from typing import Any

from app.models.scoring import ScoredItem
from app.services.profile.constants import SMART_SAMPLING_MAX_ITEMS
from app.services.scoring import ScoringService


class SmartSampler:
    """
    Smart sampling for profile building.

    Strategy:
    1. Always include all loved/liked/added items (strong signals)
    2. Fill remaining slots with top watched items by score
    3. Limit total to prevent excessive API calls
    """

    def __init__(self, scoring_service: ScoringService):
        """
        Initialize smart sampler.

        Args:
            scoring_service: Service for scoring items
        """
        self.scoring_service = scoring_service

    @staticmethod
    def _is_taste_eligible(item: dict[str, Any]) -> bool:
        """
        Keep existing unrated watched-item behavior, but prevent explicit
        neutral/disliked ratings from shaping the general taste profile.

        Rated 3-6 items remain available elsewhere for watched exclusion and
        "Because You Watched". Rated 1-2 items are negative signals only.
        """
        rating = item.get("_personal_rating")
        if rating is None:
            return not bool(item.get("_is_disliked"))

        try:
            numeric_rating = int(rating)
        except (TypeError, ValueError):
            return not bool(item.get("_is_disliked"))

        return numeric_rating >= 7

    def sample_items(
        self,
        library_items: dict[str, list[dict[str, Any]]],
        content_type: str,
        max_items: int = SMART_SAMPLING_MAX_ITEMS,
    ) -> list[ScoredItem]:
        """
        Sample items for profile building.

        Args:
            library_items: Library items dict with 'loved', 'liked', 'watched', 'added' keys
            content_type: Content type to filter (movie/series)
            max_items: Maximum items to return

        Returns:
            List of ScoredItem objects
        """
        # Get all items of the requested type
        all_items = (
            library_items.get("loved", [])
            + library_items.get("liked", [])
            + library_items.get("watched", [])
            + library_items.get("added", [])
        )
        typed_items = [
            it
            for it in all_items
            if it.get("type") == content_type and self._is_taste_eligible(it)
        ]

        if not typed_items:
            return []

        if len(typed_items) <= max_items:
            # score all typed items and return
            return [self.scoring_service.process_item(it) for it in typed_items]

        # De-duplicate by ID
        unique_items = {}
        for it in typed_items:
            item_id = it.get("_id")
            if item_id:
                unique_items[item_id] = it

        # If still within limit after de-duplication
        if len(unique_items) <= max_items:
            return [self.scoring_service.process_item(it) for it in unique_items.values()]

        # Get set of added item IDs for classification
        added_item_ids = {it.get("_id") for it in library_items.get("added", [])}

        # Separate items into pools and score them
        loved_liked_pool = []
        added_pool = []
        watched_pool = []

        for it in unique_items.values():
            scored = self.scoring_service.process_item(it)
            if scored.source_type in ["loved", "liked"]:
                loved_liked_pool.append(scored)
            elif it.get("_id") in added_item_ids:
                added_pool.append(scored)
            else:
                watched_pool.append(scored)

        # Sort pools by score to ensure we take the most relevant items first
        # If we sort this, we will get high scoring items, but if we don't sort this,
        # we will get recent items. Maybe recent is good? I think yeah. Lets do that...
        # it will likely by almost similar but not confirmed.
        # loved_liked_pool.sort(key=lambda x: x.score, reverse=True)
        # added_pool.sort(key=lambda x: x.score, reverse=True)
        # watched_pool.sort(key=lambda x: x.score, reverse=True)

        # Step 1: Fill quotas
        final_scored_items: list[ScoredItem] = []
        used_ids: set[str] = set()

        loved_quota = int(max_items * 0.40)
        added_quota = int(max_items * 0.20)
        watched_quota = max_items - loved_quota - added_quota

        # Add initial quotas
        for pool, quota in [
            (loved_liked_pool, loved_quota),
            (added_pool, added_quota),
            (watched_pool, watched_quota),
        ]:
            for scored in pool[:quota]:
                final_scored_items.append(scored)
                used_ids.add(scored.item.id)

        # Step 2: Backfill if we have remaining slots
        remaining_slots = max_items - len(final_scored_items)
        if remaining_slots > 0:
            # Priority for backfill: Loved > Added > Watched
            for pool in [loved_liked_pool, added_pool, watched_pool]:
                for scored in pool:
                    if remaining_slots <= 0:
                        break
                    if scored.item.id not in used_ids:
                        final_scored_items.append(scored)
                        used_ids.add(scored.item.id)
                        remaining_slots -= 1

        return final_scored_items
