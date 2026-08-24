from typing import Literal

from app.models.scoring import ScoredItem
from app.services.profile.constants import (
    EVIDENCE_WEIGHT_PLAN_TO_WATCH,
    EVIDENCE_WEIGHT_LIKED,
    EVIDENCE_WEIGHT_LOVED,
    EVIDENCE_WEIGHT_WATCHED,
)


class EvidenceCalculator:
    """
    Calculates evidence weights for user interactions.

    Pure function: no side effects, easy to test.
    """

    @staticmethod
    def get_interaction_type(item: ScoredItem) -> Literal["loved", "liked", "plan_to_watch", "watched"]:
        """Determine fixed taste evidence category."""
        if item.item.is_loved:
            return "loved"
        if item.item.is_liked:
            return "liked"
        if (item.item.provider_status or "").lower() in {"plantowatch", "planning"}:
            return "plan_to_watch"
        return "watched"

    @staticmethod
    def get_base_weight(interaction_type: str) -> float:
        """
        Get base evidence weight for interaction type.

        Args:
            interaction_type: Type of interaction

        Returns:
            Base weight value
        """
        weights = {
            "loved": EVIDENCE_WEIGHT_LOVED,
            "liked": EVIDENCE_WEIGHT_LIKED,
            "watched": EVIDENCE_WEIGHT_WATCHED,
            "plan_to_watch": EVIDENCE_WEIGHT_PLAN_TO_WATCH,
        }
        return weights.get(interaction_type, EVIDENCE_WEIGHT_WATCHED)

    @staticmethod
    def calculate_evidence_weight(item: ScoredItem) -> float:
        """Return the fixed evidence weight for the interaction type."""
        interaction_type = EvidenceCalculator.get_interaction_type(item)
        return EvidenceCalculator.get_base_weight(interaction_type)
