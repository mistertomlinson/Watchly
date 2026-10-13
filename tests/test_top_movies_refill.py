from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import manifest as manifest_module
from app.services.manifest import ManifestService
from app.services.recommendation.catalog_service import CatalogService


def _meta(media_id: int) -> dict:
    return {
        "id": f"tt{media_id:07d}",
        "type": "movie",
        "name": f"Movie {media_id}",
    }


async def _run_movie_dedupe(
    top_metas: list[dict],
    other_metas: list[dict],
    reserve: list[dict],
) -> dict[tuple[str, str], dict]:
    revision = 42
    visible = {
        ("movie", "watchly.rec"): {"metas": list(top_metas)},
        ("movie", "watchly.theme.test"): {"metas": list(other_metas)},
    }
    raw = {
        ("movie", "watchly.rec"): {
            "metas": list(top_metas),
            "_watchly_top_picks_reserve": list(reserve),
        },
        ("movie", "watchly.theme.test"): {"metas": list(other_metas)},
    }
    written: dict[tuple[str, str], dict] = {}

    async def get_staged_catalog(token, content_type, catalog_id):
        data = visible[(content_type, catalog_id)]
        return data, 1_000, revision

    async def get_staged_raw_catalog(token, content_type, catalog_id):
        data = raw[(content_type, catalog_id)]
        return data, 1_000, revision

    async def set_staged_catalog(
        token,
        content_type,
        catalog_id,
        catalog,
        source_revision=None,
    ):
        assert source_revision == revision
        written[(content_type, catalog_id)] = catalog

    candidate_manifest = {
        "catalogs": [
            {
                "type": "movie",
                "id": "watchly.rec",
                "name": "Top Movies for You",
            },
            {
                "type": "movie",
                "id": "watchly.theme.test",
                "name": "Theme Movies",
            },
        ]
    }

    with (
        patch.object(
            manifest_module.user_cache,
            "get_staged_catalog",
            new=AsyncMock(side_effect=get_staged_catalog),
        ),
        patch.object(
            manifest_module.user_cache,
            "get_staged_raw_catalog",
            new=AsyncMock(side_effect=get_staged_raw_catalog),
        ),
        patch.object(
            manifest_module.user_cache,
            "set_staged_catalog",
            new=AsyncMock(side_effect=set_staged_catalog),
        ),
    ):
        await ManifestService()._dedupe_prewarmed_catalogs(
            "test-token",
            candidate_manifest,
        )

    return written


@pytest.mark.asyncio
async def test_top_movies_refills_to_twenty_after_cross_row_dedupe():
    top = [_meta(i) for i in range(1, 21)]
    other = [_meta(19), _meta(20), _meta(101), _meta(102), _meta(103)]
    reserve = [_meta(21), _meta(22), _meta(23)]

    written = await _run_movie_dedupe(top, other, reserve)

    top_result = written[("movie", "watchly.rec")]
    other_result = written[("movie", "watchly.theme.test")]

    top_ids = [item["id"] for item in top_result["metas"]]
    other_ids = [item["id"] for item in other_result["metas"]]

    assert len(top_ids) == 20
    assert _meta(21)["id"] in top_ids
    assert _meta(22)["id"] in top_ids
    assert _meta(19)["id"] not in top_ids
    assert _meta(20)["id"] not in top_ids
    assert _meta(19)["id"] in other_ids
    assert _meta(20)["id"] in other_ids
    assert set(top_ids).isdisjoint(other_ids)
    assert "_watchly_top_picks_reserve" not in top_result


@pytest.mark.asyncio
async def test_top_movies_refill_skips_reserve_title_allocated_to_other_row():
    top = [_meta(i) for i in range(1, 21)]
    other = [_meta(20), _meta(21)]
    reserve = [_meta(21), _meta(22), _meta(23)]

    written = await _run_movie_dedupe(top, other, reserve)

    top_result = written[("movie", "watchly.rec")]
    other_result = written[("movie", "watchly.theme.test")]

    top_ids = [item["id"] for item in top_result["metas"]]
    other_ids = [item["id"] for item in other_result["metas"]]

    assert len(top_ids) == 20
    assert _meta(21)["id"] not in top_ids
    assert _meta(22)["id"] in top_ids
    assert _meta(21)["id"] in other_ids
    assert set(top_ids).isdisjoint(other_ids)
    assert "_watchly_top_picks_reserve" not in top_result


@pytest.mark.asyncio
async def test_top_picks_router_forwards_optional_reserve_sink():
    service = CatalogService()
    reserve_sink: list[dict] = []
    top_picks = SimpleNamespace(
        get_top_picks=AsyncMock(return_value=[_meta(1)]),
     )
    services = {"top_picks": top_picks}
    profile = SimpleNamespace()
    user_settings = SimpleNamespace(openrouter_api_key=None)

    result = await service._get_recommendations(
        catalog_id="watchly.rec",
        content_type="movie",
        services=services,
        profile=profile,
        watched_tmdb=set(),
        watched_imdb=set(),
        whitelist=set(),
        library_items={},
        limit=20,
        user_settings=user_settings,
        reserve_sink=reserve_sink,
    )

    assert result == [_meta(1)]
    top_picks.get_top_picks.assert_awaited_once()
    assert top_picks.get_top_picks.await_args.kwargs["reserve_sink"] is reserve_sink
