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
