"""Compatibility import for the unified Top Picks pipeline.

The implementation lives in ``top_picks_v2`` so the candidate-pool redesign is
isolated and easy to compare/revert while existing imports keep working.
"""

from app.services.recommendation.top_picks_v2 import TopPicksService

__all__ = ["TopPicksService"]
