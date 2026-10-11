"""Production entry point for the unified Top Picks V2 pipeline.

The implementation lives in ``top_picks_v2``. Movie home-availability filtering
is enforced globally at the shared recommendation metadata boundary so every
Watchly movie recommendation row follows the same rule.
"""

from app.services.recommendation.top_picks_v2 import TopPicksService


UNIFIED_TOP_PICKS_V2 = True


__all__ = ["TopPicksService", "UNIFIED_TOP_PICKS_V2"]
