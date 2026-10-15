import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.api.endpoints.simkl import SimklTokenRequest
from app.api.endpoints.tokens import TokenRequest
from app.api.endpoints.trakt import TraktTokenRequest
from app.core.settings import UserSettings, get_default_settings
from app.services.manifest import ManifestService
from app.services.row_generator import (
    RECENCY_BIAS_OFF_CACHE_SIGNATURE,
    RowGeneratorService,
)
from app.services.user_cache import user_cache


def test_existing_settings_without_recency_bias_default_off():
    raw = get_default_settings().model_dump()
    raw.pop("recency_bias", None)

    restored = UserSettings(**raw)

    assert restored.recency_bias is False


def test_default_settings_have_recency_bias_off():
    assert get_default_settings().recency_bias is False


def test_all_auth_request_models_default_recency_bias_off():
    assert TokenRequest().recency_bias is False

    assert (
        TraktTokenRequest(
            trakt_access_token="test"
        ).recency_bias
        is False
    )

    assert (
        SimklTokenRequest(
            simkl_access_token="test"
        ).recency_bias
        is False
    )


def test_all_auth_request_models_accept_recency_bias_on():
    assert TokenRequest(
        recency_bias=True
    ).recency_bias is True

    assert TraktTokenRequest(
        trakt_access_token="test",
        recency_bias=True,
    ).recency_bias is True

    assert SimklTokenRequest(
        simkl_access_token="test",
        recency_bias=True,
    ).recency_bias is True


@pytest.mark.asyncio
async def test_settings_change_cancels_old_manifest_prepare_and_marks_dirty():
    service = ManifestService()
    token = "toggle-test-token"

    started = asyncio.Event()

    async def pending_prepare():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        pending_prepare()
    )

    await started.wait()

    service._manifest_prepare_tasks[
        token
    ] = task

    with (
        patch.object(
            user_cache,
            "invalidate_manifest",
            new=AsyncMock(),
        ) as invalidate_manifest,
        patch.object(
            user_cache,
            "mark_catalogs_dirty",
            new=AsyncMock(),
        ) as mark_dirty,
    ):
        await service.invalidate_for_settings_change(
            token
        )

    assert task.cancelled()

    assert (
        token
        not in service._manifest_prepare_tasks
    )

    invalidate_manifest.assert_awaited_once_with(
        token
    )

    mark_dirty.assert_awaited_once_with(
        token
    )


@pytest.mark.asyncio
async def test_recency_off_theme_cache_rejects_recency_on_rows():
    service = RowGeneratorService(
        tmdb_service=object()
    )

    payload = (
        '{"generated_at":9999999999,'
        '"profile_signature":"recency-on-signal",'
        '"rows":[]}'
    )

    with patch(
        "app.services.redis_service.redis_service.get",
        new=AsyncMock(
            return_value=payload
        ),
    ):
        cached = await (
            service._get_cached_llm_rows(
                "token",
                "movie",
                max_age=None,
                expected_signature=(
                    RECENCY_BIAS_OFF_CACHE_SIGNATURE
                ),
            )
        )

    assert cached is None


@pytest.mark.asyncio
async def test_recency_off_accepts_unsigned_legacy_theme_cache():
    service = RowGeneratorService(
        tmdb_service=object()
    )

    payload = (
        '{"generated_at":9999999999,'
        '"profile_signature":null,'
        '"rows":[]}'
    )

    with patch(
        "app.services.redis_service.redis_service.get",
        new=AsyncMock(
            return_value=payload
        ),
    ):
        cached = await (
            service._get_cached_llm_rows(
                "token",
                "movie",
                max_age=None,
                expected_signature=(
                    RECENCY_BIAS_OFF_CACHE_SIGNATURE
                ),
            )
        )

    # Empty list is deliberately different from None:
    # the legacy cache was accepted rather than rejected
    # for a signature mismatch.
    assert cached == []


def test_web_config_exposes_and_persists_recency_bias():
    template = Path(
        "app/templates/components/"
        "section_config.html"
    ).read_text()

    assert 'id="recencyBiasToggle"' in template
    assert "Recency Bias" in template
    assert "recently watched titles" in template

    form_js = Path(
        "app/static/js/modules/form.js"
    ).read_text()

    auth_js = Path(
        "app/static/js/modules/auth.js"
    ).read_text()

    trakt_js = Path(
        "app/static/js/modules/trakt.js"
    ).read_text()

    simkl_js = Path(
        "app/static/js/modules/simkl.js"
    ).read_text()

    assert form_js.count(
        "recency_bias: recencyBias"
    ) == 2

    assert (
        "s.recency_bias === true"
        in auth_js
    )

    assert (
        "s.recency_bias === true"
        in trakt_js
    )

    assert (
        "recency_bias:"
        in simkl_js
    )

    assert (
        "s.recency_bias === true"
        in simkl_js
    )


def test_all_account_endpoints_rebuild_when_mode_changes():
    for path in (
        "app/api/endpoints/tokens.py",
        "app/api/endpoints/trakt.py",
        "app/api/endpoints/simkl.py",
    ):
        text = Path(path).read_text()

        assert (
            "previous_recency_bias"
            in text
        )

        assert (
            "invalidate_for_settings_change"
            in text
        )


def test_manifest_url_shape_remains_token_based():
    for path in (
        "app/api/endpoints/tokens.py",
        "app/api/endpoints/trakt.py",
        "app/api/endpoints/simkl.py",
    ):
        text = Path(path).read_text()

        assert (
            '/{token}/manifest.json'
            in text
        )
