from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import openrouter as openrouter_module
from app.services import row_generator as row_generator_module
from app.services.openrouter import (
    DEFAULT_MODEL,
    GROQ_MODEL,
    OpenRouterService,
    _is_approved_free_model_id,
)
from app.services.row_generator import (
    RowComponents,
    RowGeneratorService,
)


TEST_FREE_FALLBACK = "google/gemma-4-31b-it:free"
TEST_PAID_MODEL = "openai/gpt-oss-20b"


class FakeResponse:
    def __init__(
        self,
        status_code=200,
        model="test-model",
        content="Test response",
        finish_reason="stop",
        text="",
        headers=None,
    ):
        self.status_code = status_code
        self.model = model
        self.content = content
        self.finish_reason = finish_reason
        self.text = text
        self.headers = headers or {}

    def json(self):
        return {
            "model": self.model,
            "choices": [
                {
                    "finish_reason": self.finish_reason,
                    "message": {
                        "content": self.content,
                    },
                }
            ],
        }


class FakeCatalogResponse:
    text = ""

    def __init__(
        self,
        data,
        status_code=200,
    ):
        self.data = data
        self.status_code = status_code

    def json(self):
        return {
            "data": self.data,
        }


class FakeAsyncClient:
    def __init__(
        self,
        post_responses=None,
        get_responses=None,
        calls=None,
        timeout=None,
    ):
        # Keep the caller's response queues shared across successive
        # FakeAsyncClient instances. OpenRouterService creates a fresh HTTP
        # client for each model attempt, but the mocked provider response
        # sequence must continue advancing rather than restart at item zero.
        self.post_responses = (
            post_responses
            if post_responses is not None
            else []
        )
        self.get_responses = (
            get_responses
            if get_responses is not None
            else []
        )
        self.calls = (
            calls
            if calls is not None
            else []
        )

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    async def get(
        self,
        url,
        headers,
    ):
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
            }
        )

        response = self.get_responses.pop(0)

        if isinstance(
            response,
            Exception,
        ):
            raise response

        return response

    async def post(
        self,
        url,
        headers,
        json,
    ):
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return self.post_responses.pop(0)


def free_model(
    model_id,
    *,
    prompt_price="0",
    completion_price="0",
    context_length=131072,
    inputs=None,
    outputs=None,
):
    return {
        "id": model_id,
        "pricing": {
            "prompt": prompt_price,
            "completion": completion_price,
        },
        "context_length": context_length,
        "architecture": {
            "input_modalities": (
                inputs
                if inputs is not None
                else ["text"]
            ),
            "output_modalities": (
                outputs
                if outputs is not None
                else ["text"]
            ),
        },
    }


@pytest.mark.asyncio
async def test_primary_openrouter_model_is_used_first():
    calls = []
    responses = [
        FakeResponse(
            model=DEFAULT_MODEL,
            content="movie|Heat|1995\n" * 5,
        )
    ]
    service = OpenRouterService()
    resolver = AsyncMock(
        return_value=[
            DEFAULT_MODEL,
            TEST_FREE_FALLBACK,
        ]
    )

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=resolver,
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = (
            await service.generate_flash_content_async(
                prompt="Prompt",
                system_instruction="System",
                api_key="sk-or-test",
                max_tokens=321,
                minimum_pipe_lines=5,
            )
        )

    assert result
    resolver.assert_awaited_once_with(
        "sk-or-test"
    )
    assert len(calls) == 1
    assert calls[0]["method"] == "POST"
    assert calls[0]["json"]["model"] == DEFAULT_MODEL
    assert calls[0]["json"]["max_tokens"] == 321
    assert "models" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_invalid_primary_output_retries_live_free_fallback():
    calls = []
    responses = [
        FakeResponse(
            model=DEFAULT_MODEL,
            content="I cannot provide that list.",
        ),
        FakeResponse(
            model=TEST_FREE_FALLBACK,
            content="movie|Heat|1995\n" * 5,
        ),
    ]
    service = OpenRouterService()

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=AsyncMock(
                return_value=[
                    DEFAULT_MODEL,
                    TEST_FREE_FALLBACK,
                ]
            ),
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = (
            await service.generate_flash_content_async(
                prompt="Prompt",
                system_instruction="System",
                api_key="sk-or-test",
                minimum_pipe_lines=5,
            )
        )

    assert result
    assert [
        call["json"]["model"]
        for call in calls
    ] == [
        DEFAULT_MODEL,
        TEST_FREE_FALLBACK,
    ]


@pytest.mark.asyncio
async def test_invalid_primary_json_retries_live_free_fallback():
    calls = []
    responses = [
        FakeResponse(
            model=DEFAULT_MODEL,
            content='{"rows": [}',
        ),
        FakeResponse(
            model=TEST_FREE_FALLBACK,
            content='{"rows": []}',
        ),
    ]
    service = OpenRouterService()

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=AsyncMock(
                return_value=[
                    DEFAULT_MODEL,
                    TEST_FREE_FALLBACK,
                ]
            ),
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = await service.generate_structured_async(
            prompt="Prompt",
            response_schema=dict,
            system_instruction="System",
            api_key="sk-or-test",
        )

    assert result == {"rows": []}
    assert [
        call["json"]["model"]
        for call in calls
    ] == [
        DEFAULT_MODEL,
        TEST_FREE_FALLBACK,
    ]


@pytest.mark.asyncio
async def test_wrapped_rate_limit_retries_same_free_model_before_fallback():
    calls = []
    responses = [
        FakeResponse(
            status_code=502,
            model=DEFAULT_MODEL,
            text=(
                '{"error":{"code":502,'
                '"metadata":{"previous_errors":['
                '{"code":429,"message":'
                '"temporarily rate-limited"}]}}}'
            ),
        ),
        FakeResponse(
            model=DEFAULT_MODEL,
            content="movie|Heat|1995\n" * 5,
        ),
    ]
    service = OpenRouterService()
    sleep_mock = AsyncMock()

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=AsyncMock(
                return_value=[
                    DEFAULT_MODEL,
                    TEST_FREE_FALLBACK,
                ]
            ),
        ),
        patch.object(
            openrouter_module.asyncio,
            "sleep",
            new=sleep_mock,
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = (
            await service.generate_flash_content_async(
                prompt="Prompt",
                system_instruction="System",
                api_key="sk-or-test",
                minimum_pipe_lines=5,
            )
        )

    assert result
    assert [
        call["json"]["model"]
        for call in calls
    ] == [
        DEFAULT_MODEL,
        DEFAULT_MODEL,
    ]
    sleep_mock.assert_awaited_once_with(
        30.0
    )


@pytest.mark.asyncio
async def test_persistent_wrapped_rate_limit_moves_to_next_free_model():
    calls = []
    rate_limited = (
        '{"error":{"code":502,'
        '"metadata":{"raw":"rate limited",'
        '"previous_errors":[{"code":429,'
        '"message":"temporarily rate-limited"}]}}}'
    )
    responses = [
        FakeResponse(
            status_code=502,
            model=DEFAULT_MODEL,
            text=rate_limited,
        ),
        FakeResponse(
            status_code=502,
            model=DEFAULT_MODEL,
            text=rate_limited,
        ),
        FakeResponse(
            model=TEST_FREE_FALLBACK,
            content="movie|Heat|1995\n" * 5,
        ),
    ]
    service = OpenRouterService()
    sleep_mock = AsyncMock()

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=AsyncMock(
                return_value=[
                    DEFAULT_MODEL,
                    TEST_FREE_FALLBACK,
                ]
            ),
        ),
        patch.object(
            openrouter_module.asyncio,
            "sleep",
            new=sleep_mock,
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = (
            await service.generate_flash_content_async(
                prompt="Prompt",
                system_instruction="System",
                api_key="sk-or-test",
                minimum_pipe_lines=5,
            )
        )

    assert result
    assert [
        call["json"]["model"]
        for call in calls
    ] == [
        DEFAULT_MODEL,
        DEFAULT_MODEL,
        TEST_FREE_FALLBACK,
    ]
    sleep_mock.assert_awaited_once_with(
        30.0
    )


@pytest.mark.asyncio
async def test_live_discovery_admits_only_approved_zero_cost_text_models():
    calls = []
    catalog = [
        free_model(
            DEFAULT_MODEL
        ),
        free_model(
            TEST_FREE_FALLBACK
        ),
        free_model(
            TEST_PAID_MODEL,
            prompt_price="0.000001",
            completion_price="0.000001",
        ),
        free_model(
            "qwen/qwen-test:free"
        ),
        free_model(
            "google/gemma-tiny:free",
            context_length=8192,
        ),
        free_model(
            "google/gemma-image:free",
            outputs=["image"],
        ),
    ]
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            get_responses=[
                FakeCatalogResponse(
                    catalog
                )
            ],
            calls=calls,
            timeout=timeout,
        ),
    ):
        models = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )

    assert models == [
        DEFAULT_MODEL,
        TEST_FREE_FALLBACK,
    ]
    assert all(
        model.endswith(":free")
        for model in models
    )
    assert TEST_PAID_MODEL not in models
    assert "qwen/qwen-test:free" not in models
    assert calls[0]["method"] == "GET"


@pytest.mark.asyncio
async def test_dead_primary_is_skipped_when_live_catalog_has_free_fallback():
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            get_responses=[
                FakeCatalogResponse([
                    free_model(
                        TEST_FREE_FALLBACK
                    )
                ])
            ],
            timeout=timeout,
        ),
    ):
        models = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )

    assert models == [
        TEST_FREE_FALLBACK
    ]


@pytest.mark.asyncio
async def test_catalog_failure_falls_back_only_to_explicit_free_slugs():
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            get_responses=[
                RuntimeError(
                    "catalog unavailable"
                )
            ],
            timeout=timeout,
        ),
    ):
        models = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )

    assert models
    assert DEFAULT_MODEL in models
    assert all(
        _is_approved_free_model_id(
            model
        )
        for model in models
    )
    assert all(
        model.endswith(":free")
        for model in models
    )
    assert TEST_PAID_MODEL not in models


@pytest.mark.asyncio
async def test_no_approved_free_models_returns_empty_chain():
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            get_responses=[
                FakeCatalogResponse([
                    free_model(
                        TEST_PAID_MODEL,
                        prompt_price="0.000001",
                        completion_price="0.000001",
                    ),
                    free_model(
                        "qwen/qwen-test:free"
                    ),
                ])
            ],
            timeout=timeout,
        ),
    ):
        models = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )

    assert models == []


@pytest.mark.asyncio
async def test_free_model_catalog_is_cached():
    calls = []
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            get_responses=[
                FakeCatalogResponse([
                    free_model(
                        DEFAULT_MODEL
                    )
                ])
            ],
            calls=calls,
            timeout=timeout,
        ),
    ):
        first = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )
        second = (
            await service
            ._resolve_free_model_candidates(
                "sk-or-test"
            )
        )

    assert first == [
        DEFAULT_MODEL
    ]
    assert second == first
    assert [
        call["method"]
        for call in calls
    ] == ["GET"]


@pytest.mark.asyncio
async def test_groq_keys_keep_direct_groq_routing():
    calls = []
    responses = [
        FakeResponse(
            model=GROQ_MODEL,
            content="movie|Heat|1995\n" * 5,
        )
    ]
    service = OpenRouterService()
    resolver = AsyncMock(
        return_value=[
            DEFAULT_MODEL
        ]
    )

    with (
        patch.object(
            service,
            "_resolve_free_model_candidates",
            new=resolver,
        ),
        patch.object(
            openrouter_module.httpx,
            "AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_responses=responses,
                calls=calls,
                timeout=timeout,
            ),
        ),
    ):
        result = (
            await service.generate_flash_content_async(
                prompt="Prompt",
                system_instruction="System",
                api_key="gsk_test",
                max_tokens=222,
                minimum_pipe_lines=5,
            )
        )

    assert result
    resolver.assert_not_awaited()
    assert len(calls) == 1
    assert calls[0]["json"]["model"] == GROQ_MODEL
    assert calls[0]["json"]["max_tokens"] == 222


@pytest.mark.asyncio
async def test_tiered_row_titles_forward_users_key():
    title_mock = AsyncMock(
        return_value="British Crime Thrillers"
    )
    service = RowGeneratorService(
        tmdb_service=object(),
        user_settings=SimpleNamespace(
            openrouter_api_key="sk-or-user-key",
        ),
    )
    row = RowComponents(
        prompt_parts=[
            "British + Crime + Thriller"
        ],
        fallback_parts=[
            "British Crime Thriller"
        ],
    )

    with patch.object(
        row_generator_module.gemini_service,
        "generate_content_async",
        new=title_mock,
    ):
        result = await service._generate_titles([row])

    title_mock.assert_awaited_once_with(
        "British + Crime + Thriller",
        api_key="sk-or-user-key",
    )
    assert result[0].title == (
        "British Crime Thrillers"
    )
