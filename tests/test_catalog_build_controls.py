import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.openrouter import RECOMMENDATION_MAX_TOKENS
from app.services.recommendation import item_based as item_based_module
from app.services.recommendation.catalog_service import CatalogService
from app.services.recommendation.item_based import ItemBasedService


@pytest.mark.asyncio
async def test_duplicate_catalog_requests_share_one_build_and_survive_cancellation():
    service = CatalogService()

    build_started = asyncio.Event()
    release_build = asyncio.Event()
    build_calls = 0

    async def fake_get_catalog_impl(token, content_type, catalog_id):
        nonlocal build_calls
        build_calls += 1
        build_started.set()
        await release_build.wait()

        return (
            {"metas": [{"id": "tt1234567", "type": content_type}]},
            {"Cache-Control": "test"},
        )

    service._get_catalog_impl = fake_get_catalog_impl

    first_waiter = asyncio.create_task(
        service.get_catalog(
            "test-token",
            "movie",
            "watchly.loved.tt1234567",
        )
    )

    await build_started.wait()

    second_waiter = asyncio.create_task(
        service.get_catalog(
            "test-token",
            "movie",
            "watchly.loved.tt1234567",
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert build_calls == 1
    assert len(service._inflight_catalogs) == 1

    first_waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    assert not second_waiter.done()
    assert build_calls == 1

    release_build.set()
    result = await second_waiter

    assert result[0]["metas"][0]["id"] == "tt1234567"
    assert build_calls == 1

    await asyncio.sleep(0)
    assert service._inflight_catalogs == {}


@pytest.mark.asyncio
async def test_gemini_output_is_deduplicated_and_hard_capped_before_tmdb_enrichment():
    class FakeTMDBClient:
        async def get(self, path, params=None):
            return {
                "title": "Seed Movie",
                "release_date": "2024-01-01",
                "overview": "Seed overview",
                "genres": [],
            }

    fake_tmdb_service = SimpleNamespace(
        client=FakeTMDBClient(),
    )

    user_settings = SimpleNamespace(
        year_min=None,
        year_max=None,
        popularity="balanced",
        language="en-US",
    )

    service = ItemBasedService(
        fake_tmdb_service,
        user_settings,
    )

    resolved_titles = []

    async def fake_resolve_title(name, year, media_type):
        resolved_titles.append((name, year, media_type))
        return {
            "id": len(resolved_titles),
            "title": name,
        }

    service._resolve_title = fake_resolve_title

    raw_lines = [
        "movie|Alpha|2001",
        "movie|alpha|2001",
    ]

    raw_lines.extend(
        f"movie|Title {index}|{2000 + index % 20}"
        for index in range(1, 101)
    )

    gemini_response = "\n".join(raw_lines)
    llm_mock = AsyncMock(return_value=gemini_response)

    async def fake_fetch_batch(
        tmdb_service,
        candidates,
        content_type,
        user_settings=None,
    ):
        return candidates

    with (
        patch.object(
            item_based_module,
            "resolve_tmdb_id",
            new=AsyncMock(return_value=123),
        ),
        patch.object(
            item_based_module.gemini_service,
            "generate_flash_content_async",
            new=llm_mock,
        ),
        patch.object(
            item_based_module.RecommendationMetadata,
            "fetch_batch",
            new=AsyncMock(side_effect=fake_fetch_batch),
        ),
        patch.object(
            item_based_module,
            "filter_items_by_settings",
            side_effect=lambda items, settings: items,
        ),
    ):
        result = await service._fetch_gemini_item_recommendations(
            item_id="tt1234567",
            content_type="movie",
            library_items={"watched": []},
            gemini_api_key="test-key",
            limit=20,
        )

    assert (
        llm_mock.await_args.kwargs["max_tokens"]
        == RECOMMENDATION_MAX_TOKENS
    )

    # limit=20 means the Gemini parsing cap is limit * 3 = 60.
    assert len(resolved_titles) == 60
    assert len(result) == 60

    normalized = {
        (name.casefold(), year)
        for name, year, _ in resolved_titles
    }

    assert len(normalized) == 60
    assert resolved_titles[0] == ("Alpha", "2001", "movie")
