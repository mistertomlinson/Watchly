from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import manifest as manifest_module
from app.services.manifest import ManifestService
from app.services.recommendation.catalog_service import CatalogService


def _meta(
    media_id: int,
    content_type: str = "movie",
) -> dict:
    label = (
        "Movie"
        if content_type == "movie"
        else "Series"
    )

    return {
        "id": f"tt{media_id:07d}",
        "type": content_type,
        "name": f"{label} {media_id}",
    }


async def _run_top_picks_dedupe(
    content_type: str,
    top_metas: list[dict],
    other_metas: list[dict],
    reserve: list[dict],
) -> dict[tuple[str, str], dict]:
    revision = 42

    visible = {
        (
            content_type,
            "watchly.rec",
        ): {
            "metas": list(top_metas),
        },
        (
            content_type,
            "watchly.theme.test",
        ): {
            "metas": list(other_metas),
        },
    }

    raw = {
        (
            content_type,
            "watchly.rec",
        ): {
            "metas": list(top_metas),
            "_watchly_top_picks_reserve":
                list(reserve),
        },
        (
            content_type,
            "watchly.theme.test",
        ): {
            "metas": list(other_metas),
        },
    }

    written: dict[
        tuple[str, str],
        dict,
    ] = {}

    async def get_staged_catalog(
        token,
        requested_type,
        catalog_id,
    ):
        data = visible[
            (
                requested_type,
                catalog_id,
            )
        ]

        return (
            data,
            1_000,
            revision,
        )

    async def get_staged_raw_catalog(
        token,
        requested_type,
        catalog_id,
    ):
        data = raw[
            (
                requested_type,
                catalog_id,
            )
        ]

        return (
            data,
            1_000,
            revision,
        )

    async def set_staged_catalog(
        token,
        requested_type,
        catalog_id,
        catalog,
        source_revision=None,
    ):
        assert source_revision == revision

        written[
            (
                requested_type,
                catalog_id,
            )
        ] = catalog

    label = (
        "Movies"
        if content_type == "movie"
        else "Series"
    )

    candidate_manifest = {
        "catalogs": [
            {
                "type": content_type,
                "id": "watchly.rec",
                "name":
                    f"Top {label} for You",
            },
            {
                "type": content_type,
                "id":
                    "watchly.theme.test",
                "name":
                    f"Theme {label}",
            },
        ]
    }

    with (
        patch.object(
            manifest_module.user_cache,
            "get_staged_catalog",
            new=AsyncMock(
                side_effect=get_staged_catalog
            ),
        ),
        patch.object(
            manifest_module.user_cache,
            "get_staged_raw_catalog",
            new=AsyncMock(
                side_effect=
                    get_staged_raw_catalog
            ),
        ),
        patch.object(
            manifest_module.user_cache,
            "set_staged_catalog",
            new=AsyncMock(
                side_effect=set_staged_catalog
            ),
        ),
    ):
        await (
            ManifestService()
            ._dedupe_prewarmed_catalogs(
                "test-token",
                candidate_manifest,
            )
        )

    return written


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        "movie",
        "series",
    ],
)
async def test_top_picks_refills_to_twenty_after_cross_row_dedupe(
    content_type,
):
    top = [
        _meta(
            i,
            content_type,
        )
        for i in range(
            1,
            21,
        )
    ]

    other = [
        _meta(
            i,
            content_type,
        )
        for i in (
            19,
            20,
            101,
            102,
            103,
        )
    ]

    reserve = [
        _meta(
            i,
            content_type,
        )
        for i in (
            21,
            22,
            23,
        )
    ]

    written = (
        await _run_top_picks_dedupe(
            content_type,
            top,
            other,
            reserve,
        )
    )

    top_result = written[
        (
            content_type,
            "watchly.rec",
        )
    ]

    other_result = written[
        (
            content_type,
            "watchly.theme.test",
        )
    ]

    top_ids = [
        item["id"]
        for item
        in top_result["metas"]
    ]

    other_ids = [
        item["id"]
        for item
        in other_result["metas"]
    ]

    assert len(top_ids) == 20

    assert (
        _meta(
            21,
            content_type,
        )["id"]
        in top_ids
    )

    assert (
        _meta(
            22,
            content_type,
        )["id"]
        in top_ids
    )

    assert (
        _meta(
            19,
            content_type,
        )["id"]
        not in top_ids
    )

    assert (
        _meta(
            20,
            content_type,
        )["id"]
        not in top_ids
    )

    assert (
        _meta(
            19,
            content_type,
        )["id"]
        in other_ids
    )

    assert (
        _meta(
            20,
            content_type,
        )["id"]
        in other_ids
    )

    assert set(
        top_ids
    ).isdisjoint(
        other_ids
    )

    assert (
        "_watchly_top_picks_reserve"
        not in top_result
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        "movie",
        "series",
    ],
)
async def test_top_picks_refill_skips_reserve_title_allocated_to_other_row(
    content_type,
):
    top = [
        _meta(
            i,
            content_type,
        )
        for i in range(
            1,
            21,
        )
    ]

    other = [
        _meta(
            i,
            content_type,
        )
        for i in (
            20,
            21,
        )
    ]

    reserve = [
        _meta(
            i,
            content_type,
        )
        for i in (
            21,
            22,
            23,
        )
    ]

    written = (
        await _run_top_picks_dedupe(
            content_type,
            top,
            other,
            reserve,
        )
    )

    top_result = written[
        (
            content_type,
            "watchly.rec",
        )
    ]

    other_result = written[
        (
            content_type,
            "watchly.theme.test",
        )
    ]

    top_ids = [
        item["id"]
        for item
        in top_result["metas"]
    ]

    other_ids = [
        item["id"]
        for item
        in other_result["metas"]
    ]

    assert len(top_ids) == 20

    assert (
        _meta(
            21,
            content_type,
        )["id"]
        not in top_ids
    )

    assert (
        _meta(
            22,
            content_type,
        )["id"]
        in top_ids
    )

    assert (
        _meta(
            21,
            content_type,
        )["id"]
        in other_ids
    )

    assert set(
        top_ids
    ).isdisjoint(
        other_ids
    )

    assert (
        "_watchly_top_picks_reserve"
        not in top_result
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        "movie",
        "series",
    ],
)
async def test_top_picks_router_forwards_optional_reserve_sink(
    content_type,
):
    service = CatalogService()

    reserve_sink: list[dict] = []

    result_meta = _meta(
        1,
        content_type,
    )

    top_picks = SimpleNamespace(
        get_top_picks=AsyncMock(
            return_value=[
                result_meta,
            ]
        ),
    )

    services = {
        "top_picks":
            top_picks,
    }

    profile = SimpleNamespace()

    user_settings = SimpleNamespace(
        openrouter_api_key=None
    )

    result = await service._get_recommendations(
        catalog_id="watchly.rec",
        content_type=content_type,
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

    assert result == [
        result_meta,
    ]

    top_picks.get_top_picks.assert_awaited_once()

    assert (
        top_picks
        .get_top_picks
        .await_args
        .kwargs[
            "reserve_sink"
        ]
        is reserve_sink
    )
