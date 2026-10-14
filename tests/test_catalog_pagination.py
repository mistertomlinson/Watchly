import pytest
from fastapi import Response
from unittest.mock import AsyncMock, patch

from app.api.endpoints.catalogs import get_catalog


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra, expected",
    [
        (None, ["0", "1", "2"]),
        ("skip=0", ["0", "1", "2"]),
        ("skip=2", ["2"]),
        ("skip=99", []),
        ("skip=-1", ["0", "1", "2"]),
        ("skip=nope", ["0", "1", "2"]),
        ("foo=bar&skip=1", ["1", "2"]),
    ],
)
async def test_catalog_skip_paginates_without_mutating_shared_catalog(extra, expected):
    shared = {"metas": [{"id": str(i)} for i in range(3)]}

    with patch(
        "app.api.endpoints.catalogs.catalog_service.get_catalog",
        new=AsyncMock(
            return_value=(shared, {"Cache-Control": "test"})
        ),
    ) as mocked:
        result = await get_catalog(
            Response(),
            "movie",
            "watchly.rec",
            "test-token",
            extra,
        )

    assert [item["id"] for item in result["metas"]] == expected
    assert [item["id"] for item in shared["metas"]] == ["0", "1", "2"]
    mocked.assert_awaited_once_with(
        "test-token",
        "movie",
        "watchly.rec",
    )


@pytest.mark.asyncio
async def test_cached_published_catalog_keeps_stored_order_for_pagination():
    from app.services.recommendation import catalog_service as catalog_service_module
    from app.services.recommendation.catalog_service import CatalogService

    service = CatalogService()

    published = {
        "metas": [
            {"id": "tt0000001", "type": "movie"},
            {"id": "tt0000002", "type": "movie"},
            {"id": "tt0000003", "type": "movie"},
        ]
    }

    with (
        patch.object(
            catalog_service_module.token_store,
            "get_user_data",
            new=AsyncMock(return_value={"auth_provider": "stremio"}),
        ),
        patch.object(
            catalog_service_module.settings,
            "AUTO_UPDATE_CATALOGS",
            False,
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_catalog",
            new=AsyncMock(return_value=(published, 2_000_000_000, 7)),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_catalog_dirty_revision",
            new=AsyncMock(return_value=7),
        ),
        patch.object(
            catalog_service_module,
            "shuffle_data_if_needed",
        ) as reshuffle,
    ):
        data, _headers = await service._get_catalog_impl(
            "test-token",
            "movie",
            "watchly.rec",
        )

    assert [item["id"] for item in data["metas"]] == [
        "tt0000001",
        "tt0000002",
        "tt0000003",
    ]
    assert data is not published
    assert data["metas"] is not published["metas"]
    reshuffle.assert_not_called()
