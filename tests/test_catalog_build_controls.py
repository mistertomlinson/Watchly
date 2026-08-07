import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.openrouter import RECOMMENDATION_MAX_TOKENS
from app.services.recommendation import catalog_service as catalog_service_module
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


@pytest.mark.asyncio
async def test_published_dirty_catalog_returns_immediately_and_schedules_refresh():
    service = CatalogService()

    published = {
        "metas": [
            {"id": "tt1111111", "type": "movie", "name": "Published"},
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
            new=AsyncMock(return_value=(published, 2_000_000_000, 100)),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_catalog_dirty_revision",
            new=AsyncMock(return_value=101),
        ),
        patch.object(
            service,
            "_extract_settings",
            return_value=SimpleNamespace(),
        ),
        patch.object(
            catalog_service_module,
            "shuffle_data_if_needed",
            side_effect=lambda _settings, _catalog_id, metas: metas,
        ),
        patch.object(
            service,
            "_schedule_background_catalog_refresh",
        ) as schedule_refresh,
    ):
        data, _headers = await service._get_catalog_impl(
            "profile-one-token",
            "movie",
            "watchly.rec",
        )

    assert data == published
    schedule_refresh.assert_called_once_with(
        "profile-one-token",
        "movie",
        "watchly.rec",
    )


@pytest.mark.asyncio
async def test_background_catalog_refreshes_coalesce_per_profile_but_not_across_profiles():
    service = CatalogService()

    release_builds = asyncio.Event()
    started = []

    async def fake_get_catalog_impl(
        token,
        content_type,
        catalog_id,
        force_refresh=False,
    ):
        started.append(
            (token, content_type, catalog_id, force_refresh)
        )
        await release_builds.wait()
        return (
            {
                "metas": [
                    {
                        "id": f"{token}-result",
                        "type": content_type,
                    }
                ]
            },
            {},
        )

    service._get_catalog_impl = fake_get_catalog_impl

    service._schedule_background_catalog_refresh(
        "profile-one-token",
        "movie",
        "watchly.rec",
    )
    service._schedule_background_catalog_refresh(
        "profile-one-token",
        "movie",
        "watchly.rec",
    )
    service._schedule_background_catalog_refresh(
        "profile-two-token",
        "movie",
        "watchly.rec",
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert sorted(started) == sorted(
        [
            (
                "profile-one-token",
                "movie",
                "watchly.rec",
                True,
            ),
            (
                "profile-two-token",
                "movie",
                "watchly.rec",
                True,
            ),
        ]
    )

    assert set(service._background_catalog_refreshes) == {
        ("profile-one-token", "movie", "watchly.rec"),
        ("profile-two-token", "movie", "watchly.rec"),
    }

    release_builds.set()

    tasks = list(service._background_catalog_refreshes.values())
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)

    assert service._background_catalog_refreshes == {}


@pytest.mark.asyncio
async def test_superseded_background_build_does_not_replace_published_catalog():
    service = CatalogService()

    published = {
        "metas": [
            {"id": "tt1111111", "type": "movie", "name": "Published"},
        ]
    }

    generated = [
        {
            "id": f"tt{index:07d}",
            "type": "movie",
            "name": f"Generated {index}",
        }
        for index in range(1, 9)
    ]

    fake_profile = SimpleNamespace(
        interest_summary="Existing summary",
    )
    fake_user_settings = SimpleNamespace(
        language="en-US",
    )
    fake_integration = SimpleNamespace(
        get_genre_whitelist=AsyncMock(return_value=set()),
    )

    set_catalog = AsyncMock()

    # Calls occur in this order:
    #   1. initial cache-dirty check
    #   2. revision captured immediately before generation
    #   3. revision checked immediately before promotion
    #
    # The final revision differs, simulating the same profile changing
    # while a slow AI recommendation build is running.
    dirty_revisions = AsyncMock(
        side_effect=[100, 100, 101],
    )

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
            new=AsyncMock(
                return_value=(published, 2_000_000_000, 100)
            ),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_catalog_dirty_revision",
            new=dirty_revisions,
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_library_items",
            new=AsyncMock(return_value=[{"id": "library-item"}]),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_profile_and_watched_sets",
            new=AsyncMock(
                return_value=(fake_profile, set(), set())
            ),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "set_catalog",
            new=set_catalog,
        ),
        patch.object(
            service,
            "_resolve_auth",
            new=AsyncMock(return_value="auth-key"),
        ),
        patch.object(
            service,
            "_extract_settings",
            return_value=fake_user_settings,
        ),
        patch.object(
            service,
            "_initialize_services",
            return_value={
                "integration": fake_integration,
                "tmdb": SimpleNamespace(),
            },
        ),
        patch.object(
            service,
            "_get_recommendations",
            new=AsyncMock(return_value=generated),
        ),
        patch.object(
            catalog_service_module,
            "_clean_meta",
            side_effect=lambda meta: meta,
        ),
        patch.object(
            catalog_service_module,
            "shuffle_data_if_needed",
            side_effect=lambda _settings, _catalog_id, metas: metas,
        ),
    ):
        data, _headers = await service._get_catalog_impl(
            "profile-one-token",
            "movie",
            "watchly.rec",
            force_refresh=True,
        )

    assert data["metas"] == generated
    set_catalog.assert_not_awaited()


@pytest.mark.asyncio
async def test_current_background_build_publishes_persistent_snapshot_with_source_revision():
    service = CatalogService()

    published = {
        "metas": [
            {"id": "tt1111111", "type": "movie", "name": "Published"},
        ]
    }

    generated = [
        {
            "id": f"tt{index:07d}",
            "type": "movie",
            "name": f"Generated {index}",
        }
        for index in range(1, 9)
    ]

    fake_profile = SimpleNamespace(
        interest_summary="Existing summary",
    )
    fake_user_settings = SimpleNamespace(
        language="en-US",
    )
    fake_integration = SimpleNamespace(
        get_genre_whitelist=AsyncMock(return_value=set()),
    )

    set_catalog = AsyncMock()

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
            new=AsyncMock(
                return_value=(published, 2_000_000_000, 100)
            ),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_catalog_dirty_revision",
            new=AsyncMock(side_effect=[100, 100, 100]),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_library_items",
            new=AsyncMock(return_value=[{"id": "library-item"}]),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "get_profile_and_watched_sets",
            new=AsyncMock(
                return_value=(fake_profile, set(), set())
            ),
        ),
        patch.object(
            catalog_service_module.user_cache,
            "set_catalog",
            new=set_catalog,
        ),
        patch.object(
            service,
            "_resolve_auth",
            new=AsyncMock(return_value="auth-key"),
        ),
        patch.object(
            service,
            "_extract_settings",
            return_value=fake_user_settings,
        ),
        patch.object(
            service,
            "_initialize_services",
            return_value={
                "integration": fake_integration,
                "tmdb": SimpleNamespace(),
            },
        ),
        patch.object(
            service,
            "_get_recommendations",
            new=AsyncMock(return_value=generated),
        ),
        patch.object(
            catalog_service_module,
            "_clean_meta",
            side_effect=lambda meta: meta,
        ),
        patch.object(
            catalog_service_module,
            "shuffle_data_if_needed",
            side_effect=lambda _settings, _catalog_id, metas: metas,
        ),
    ):
        await service._get_catalog_impl(
            "profile-one-token",
            "movie",
            "watchly.rec",
            force_refresh=True,
        )

    set_catalog.assert_awaited_once_with(
        "profile-one-token",
        "movie",
        "watchly.rec",
        {"metas": generated},
        ttl=None,
        source_revision=100,
    )
