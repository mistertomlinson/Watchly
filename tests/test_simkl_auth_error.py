import pytest

from app.services.recommendation.catalog_service import (
    _provider_auth_error_catalog,
)
from app.services.simkl_provider import (
    SimklAuthorizationError,
    SimklLibraryProvider,
)


class RejectedSimklClient:
    async def get_all_items(self, media_type: str):
        raise SimklAuthorizationError("rejected")


@pytest.mark.asyncio
async def test_simkl_authorization_error_is_not_converted_to_empty_library():
    provider = SimklLibraryProvider(RejectedSimklClient())

    with pytest.raises(SimklAuthorizationError):
        await provider.get_library_items()


def test_simkl_error_card_is_message_only():
    message = "Simkl authorization expired. Reconnect Simkl in Watchly."
    result = _provider_auth_error_catalog("movie", message)

    assert len(result["metas"]) == 1
    card = result["metas"][0]
    assert card["type"] == "movie"
    assert card["name"] == message
    assert card["description"] == message
    assert "poster" not in card
    assert "background" not in card
    assert "behaviorHints" not in card
