from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import openrouter as openrouter_module
from app.services.openrouter import (
    DEFAULT_MODEL,
    GROQ_MODEL,
    OPENROUTER_FREE_FALLBACK_MODEL,
    OpenRouterService,
)
from app.services import row_generator as row_generator_module
from app.services.row_generator import (
    RowComponents,
    RowGeneratorService,
)


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, resolved_model: str):
        self.resolved_model = resolved_model

    def json(self):
        return {
            "model": self.resolved_model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Test response",
                    },
                }
            ],
        }


class FakeAsyncClient:
    def __init__(self, calls, resolved_model):
        self.calls = calls
        self.resolved_model = resolved_model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, headers, json):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return FakeResponse(self.resolved_model)


@pytest.mark.asyncio
async def test_openrouter_uses_gemma_then_free_router_fallback():
    calls = []
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            calls,
            DEFAULT_MODEL,
        ),
    ):
        result = await service.generate_flash_content_async(
            prompt="Prompt",
            system_instruction="System",
            api_key="sk-or-test",
            max_tokens=321,
        )

    assert result == "Test response"
    assert len(calls) == 1

    payload = calls[0]["json"]
    assert payload["models"] == [
        DEFAULT_MODEL,
        OPENROUTER_FREE_FALLBACK_MODEL,
    ]
    assert "model" not in payload
    assert payload["max_tokens"] == 321


@pytest.mark.asyncio
async def test_groq_keys_keep_direct_groq_routing():
    calls = []
    service = OpenRouterService()

    with patch.object(
        openrouter_module.httpx,
        "AsyncClient",
        side_effect=lambda timeout: FakeAsyncClient(
            calls,
            GROQ_MODEL,
        ),
    ):
        result = await service.generate_flash_content_async(
            prompt="Prompt",
            system_instruction="System",
            api_key="gsk_test",
            max_tokens=222,
        )

    assert result == "Test response"
    payload = calls[0]["json"]
    assert payload["model"] == GROQ_MODEL
    assert "models" not in payload
    assert payload["max_tokens"] == 222


@pytest.mark.asyncio
async def test_tiered_row_titles_forward_users_openrouter_key():
    title_mock = AsyncMock(return_value="British Crime Thrillers")
    service = RowGeneratorService(
        tmdb_service=object(),
        user_settings=SimpleNamespace(
            openrouter_api_key="sk-or-user-key",
        ),
    )
    row = RowComponents(
        prompt_parts=["British + Crime + Thriller"],
        fallback_parts=["British Crime Thriller"],
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
    assert result[0].title == "British Crime Thrillers"
