from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import openrouter as openrouter_module
from app.services import row_generator as row_generator_module
from app.services.openrouter import (
    DEFAULT_MODEL,
    GROQ_MODEL,
    OPENROUTER_FALLBACK_MODEL,
    OpenRouterService,
)
from app.services.row_generator import (
    RowComponents,
    RowGeneratorService,
)


class FakeResponse:
    text = ""

    def __init__(
        self,
        status_code=200,
        model="test-model",
        content="Test response",
        finish_reason="stop",
    ):
        self.status_code = status_code
        self.model = model
        self.content = content
        self.finish_reason = finish_reason

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


class FakeAsyncClient:
    def __init__(self, responses, calls, timeout=None):
        self.responses = responses
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    async def post(self, url, headers, json):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return self.responses.pop(0)


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

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            responses,
            calls,
            timeout,
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
    assert len(calls) == 1
    assert calls[0]["json"]["model"] == DEFAULT_MODEL
    assert calls[0]["json"]["max_tokens"] == 321
    assert "models" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_invalid_primary_output_retries_fixed_fallback():
    calls = []
    responses = [
        FakeResponse(
            model=DEFAULT_MODEL,
            content="I cannot provide that list.",
        ),
        FakeResponse(
            model=OPENROUTER_FALLBACK_MODEL,
            content="movie|Heat|1995\n" * 5,
        ),
    ]
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            responses,
            calls,
            timeout,
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
        OPENROUTER_FALLBACK_MODEL,
    ]


@pytest.mark.asyncio
async def test_invalid_primary_json_retries_fallback():
    calls = []
    responses = [
        FakeResponse(
            model=DEFAULT_MODEL,
            content='{"rows": [}',
        ),
        FakeResponse(
            model=OPENROUTER_FALLBACK_MODEL,
            content='{"rows": []}',
        ),
    ]
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            responses,
            calls,
            timeout,
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
        OPENROUTER_FALLBACK_MODEL,
    ]


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

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            responses,
            calls,
            timeout,
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
