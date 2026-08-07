from __future__ import annotations

import ast
import json
import os
import re
import time
from collections.abc import Callable

import httpx
from loguru import logger

from app.core.config import settings


DEFAULT_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemma-4-26b-a4b-it:free",
)
OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "openai/gpt-oss-20b:free",
)

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TIMEOUT = 60.0

DEFAULT_MAX_TOKENS = 4000

# Recommendation responses are short pipe-delimited title lists. Limiting this
# path prevents malformed or repetitive output from delaying catalog completion.
RECOMMENDATION_MAX_TOKENS = 1200

STRUCTURED_MAX_TOKENS_GROQ = 1200
STRUCTURED_MAX_TOKENS_OPENROUTER = 8000
STRUCTURED_MAX_TOKENS = STRUCTURED_MAX_TOKENS_OPENROUTER
TITLE_MAX_TOKENS = 60


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return str(content or "")


def _parse_json(text: str):
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    starts = [
        position
        for position in (
            cleaned.find("{"),
            cleaned.find("["),
        )
        if position >= 0
    ]

    if starts:
        cleaned = cleaned[min(starts):]

    object_end = cleaned.rfind("}")
    array_end = cleaned.rfind("]")
    final_end = max(object_end, array_end)

    if final_end >= 0:
        cleaned = cleaned[: final_end + 1]

    attempts = [
        cleaned,
        re.sub(r",\s*([}\]])", r"\1", cleaned),
    ]

    last_error: Exception | None = None

    for candidate in attempts:
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc

    try:
        parsed = ast.literal_eval(cleaned)

        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    raise ValueError(
        f"Invalid JSON response: {last_error}"
    )


def _ordered_models(primary_model: str) -> list[str]:
    return list(
        dict.fromkeys(
            [
                primary_model,
                OPENROUTER_FALLBACK_MODEL,
            ]
        )
    )


def _pipe_list_validator(minimum_lines: int) -> Callable[[str], None]:
    def validate(content: str) -> None:
        valid_lines = 0

        for line in content.splitlines():
            parts = [
                part.strip()
                for part in line.strip().split("|")
            ]

            if len(parts) >= 3 and all(parts[:3]):
                valid_lines += 1

        if valid_lines < minimum_lines:
            raise ValueError(
                "Recommendation response contained "
                f"{valid_lines} valid pipe-delimited lines; "
                f"minimum is {minimum_lines}"
            )

    return validate


def _json_validator(content: str) -> None:
    parsed = _parse_json(content)

    if not isinstance(parsed, (dict, list)):
        raise ValueError(
            "Structured response was not a JSON object or array"
        )


class OpenRouterService:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL

    def _get_api_key(
        self,
        api_key: str | None = None,
    ) -> str | None:
        return api_key or self.api_key

    @staticmethod
    def get_catalog_title_prompt():
        return """
        You are a content catalog naming expert.
        Given filters like genre, keywords, countries, or years, generate natural,
        engaging catalog row titles that streaming platforms would use.

        Examples:
        - Genre: Action, Country: South Korea → "Korean Action Thrillers"
        - Keyword: "space", Genre: Sci-Fi → "Space Exploration Adventures"
        - Genre: Drama, Country: France → "Acclaimed French Cinema"
        - Country: "USA" + Genre: "Sci-Fi and Fantasy" → "Hollywood Sci-Fi and Fantasy"
        - Keywords: "revenge" + "martial arts" → "Revenge & Martial Arts"

        Keep titles:
        - Short (2-5 words)
        - Natural and engaging
        - Focused on what makes the content appealing
        - Only return a single best title and nothing else.
        """

    async def _call(
        self,
        prompt: str,
        system_instruction: str,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        attempt_models: list[str] | None = None,
        validate_content: Callable[[str], None] | None = None,
    ) -> str:
        key = self._get_api_key(api_key)

        if not key:
            logger.warning("No OpenRouter API key available.")
            return ""

        effective_base_url = base_url or self.base_url

        models = (
            attempt_models
            if attempt_models is not None
            else [model]
        )
        models = list(dict.fromkeys(models))

        errors: list[str] = []

        for attempt, attempt_model in enumerate(models, 1):
            payload = {
                "model": attempt_model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            }

            started_at = time.monotonic()

            logger.info(
                "LLM attempt starting "
                f"attempt={attempt}/{len(models)} "
                f"model={attempt_model} "
                f"max_tokens={max_tokens}"
            )

            try:
                async with httpx.AsyncClient(
                    timeout=TIMEOUT,
                ) as client:
                    response = await client.post(
                        f"{effective_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": (
                                "https://github.com/"
                                "mistertomlinson/Watchly"
                            ),
                            "X-Title": "Watchly",
                        },
                        json=payload,
                    )

                elapsed = time.monotonic() - started_at

                if response.status_code != 200:
                    raise RuntimeError(
                        "HTTP "
                        f"{response.status_code}: "
                        f"{response.text[:500]}"
                    )

                data = response.json()
                choices = data.get("choices") or []

                if not choices:
                    raise ValueError(
                        "LLM response had no choices"
                    )

                choice = choices[0]
                message = choice.get("message") or {}
                content = _content_to_text(
                    message.get("content")
                ).strip()

                if not content:
                    raise ValueError(
                        "LLM returned empty content "
                        f"(finish_reason="
                        f"{choice.get('finish_reason')})"
                    )

                if validate_content is not None:
                    validate_content(content)

                logger.info(
                    "LLM call completed "
                    f"attempt={attempt}/{len(models)} "
                    f"requested_model={attempt_model} "
                    f"resolved_model="
                    f"{data.get('model') or attempt_model} "
                    f"finish_reason="
                    f"{choice.get('finish_reason')} "
                    f"duration={elapsed:.2f}s "
                    f"max_tokens={max_tokens} "
                    f"content_chars={len(content)}"
                )

                return content

            except Exception as exc:
                elapsed = time.monotonic() - started_at
                error = (
                    f"attempt {attempt} "
                    f"model={attempt_model}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(error)

                logger.warning(
                    "LLM attempt failed "
                    f"attempt={attempt}/{len(models)} "
                    f"model={attempt_model} "
                    f"duration={elapsed:.2f}s "
                    f"error={type(exc).__name__}: {exc}"
                )

        logger.error(
            "LLM failed after all attempts: "
            + " | ".join(errors)
        )
        return ""

    async def generate_content_async(
        self,
        prompt: str,
        api_key: str | None = None,
    ) -> str:
        """Generate a catalog title using BYOK when supplied."""
        key = self._get_api_key(api_key)

        if key and key.startswith("gsk_"):
            return await self._call(
                prompt=prompt,
                system_instruction=(
                    self.get_catalog_title_prompt()
                ),
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=TITLE_MAX_TOKENS,
            )

        return await self._call(
            prompt=prompt,
            system_instruction=(
                self.get_catalog_title_prompt()
            ),
            api_key=api_key,
            max_tokens=TITLE_MAX_TOKENS,
            attempt_models=_ordered_models(DEFAULT_MODEL),
        )

    async def generate_flash_content_async(
        self,
        prompt: str,
        system_instruction: str,
        api_key: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        minimum_pipe_lines: int = 0,
    ) -> str:
        """Generate recommendations or summaries using BYOK."""
        validator = (
            _pipe_list_validator(minimum_pipe_lines)
            if minimum_pipe_lines > 0
            else None
        )

        if api_key and api_key.startswith("gsk_"):
            return await self._call(
                prompt=prompt,
                system_instruction=system_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=max_tokens,
                validate_content=validator,
            )

        return await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            api_key=api_key,
            max_tokens=max_tokens,
            attempt_models=_ordered_models(DEFAULT_MODEL),
            validate_content=validator,
        )

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: type | dict,
        system_instruction: str,
        api_key: str,
    ) -> dict | list | None:
        """Generate and validate a structured JSON response."""
        structured_instruction = (
            system_instruction
            + "\n\nRespond ONLY with valid JSON matching "
            "the requested schema. No other text."
        )

        key = self._get_api_key(api_key)

        if key and key.startswith("gsk_"):
            result = await self._call(
                prompt=prompt,
                system_instruction=structured_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=STRUCTURED_MAX_TOKENS_GROQ,
                validate_content=_json_validator,
            )
        else:
            result = await self._call(
                prompt=prompt,
                system_instruction=structured_instruction,
                api_key=api_key,
                max_tokens=(
                    STRUCTURED_MAX_TOKENS_OPENROUTER
                ),
                attempt_models=_ordered_models(
                    DEFAULT_MODEL
                ),
                validate_content=_json_validator,
            )

        if not result:
            return None

        try:
            parsed = _parse_json(result)

            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception as exc:
            logger.error(
                "Failed to parse validated OpenRouter "
                f"response: {exc}"
            )

        return None


openrouter_service = OpenRouterService()
gemini_service = openrouter_service
