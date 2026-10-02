from __future__ import annotations

import ast
import asyncio
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
# OpenRouter free slugs are not stable enough to hardcode as a permanent
# fallback. Discover the currently available zero-cost models instead, but keep
# the allowlist intentionally narrow so a provider catalog change can never
# route Watchly onto an unrelated or paid model.
APPROVED_FREE_MODEL_PREFIXES = (
    "google/gemma-",
    "openai/gpt-oss-",
    "apodex/apodex-",
)
PREFERRED_FREE_MODELS = (
    "google/gemma-4-31b-it:free",
    "apodex/apodex-1.1-mini:free",
)
FREE_MODEL_MIN_CONTEXT = 64_000
FREE_MODEL_CATALOG_TTL_SECONDS = 15 * 60

# Avoid repeatedly paying the same retry delay when a provider has already
# confirmed that a model is temporarily rate limited. Cooldowns are deliberately
# process-local and short-lived: a restart clears them, and the preferred model
# automatically re-enters the chain after 90 seconds. Since the normal same-model
# retry already consumes roughly 30-60 seconds, this usually skips only the next
# one or two AI jobs instead of sidelining a preferred model for a whole prewarm.
MODEL_RATE_LIMIT_COOLDOWN_SECONDS = 90

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GOOGLE_MODELS_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models?pageSize=1000"
)
GOOGLE_OPENAI_CHAT = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/openai/chat/completions"
)
GOOGLE_MODEL_PREFERENCES = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
)
GOOGLE_MIN_CONTEXT = 64_000
GOOGLE_MIN_OUTPUT = 1_800

TIMEOUT = 60.0

DEFAULT_MAX_TOKENS = 4000

# Recommendation responses are short pipe-delimited title lists. Limiting this
# path prevents malformed or repetitive output from delaying catalog completion.
RECOMMENDATION_MAX_TOKENS = 1200

STRUCTURED_MAX_TOKENS_GROQ = 1200
STRUCTURED_MAX_TOKENS_OPENROUTER = 8000
STRUCTURED_MAX_TOKENS = STRUCTURED_MAX_TOKENS_OPENROUTER
# Some free reasoning models spend part of the completion budget internally.
# 60 tokens was enough for Gemma but caused Apodex title calls to terminate
# with finish_reason=length before emitting visible content.
TITLE_MAX_TOKENS = 256


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


def _is_approved_free_model_id(model_id: object) -> bool:
    return (
        isinstance(model_id, str)
        and model_id.endswith(":free")
        and any(
            model_id.startswith(prefix)
            for prefix in APPROVED_FREE_MODEL_PREFIXES
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
        self._free_model_cache: tuple[
            float,
            list[str],
        ] | None = None
        self._rate_limit_cooldowns: dict[
            tuple[str, str],
            float,
        ] = {}

    def _get_api_key(
        self,
        api_key: str | None = None,
    ) -> str | None:
        return api_key or self.api_key

    def _mark_rate_limited_model(
        self,
        provider: str,
        model: str,
    ) -> None:
        until = (
            time.monotonic()
            + MODEL_RATE_LIMIT_COOLDOWN_SECONDS
        )
        self._rate_limit_cooldowns[
            (provider, model)
        ] = until

        logger.info(
            "AI model cooldown set "
            f"provider={provider} "
            f"model={model} "
            f"cooldown="
            f"{MODEL_RATE_LIMIT_COOLDOWN_SECONDS:.0f}s"
        )

    def _clear_rate_limited_model(
        self,
        provider: str,
        model: str,
    ) -> None:
        self._rate_limit_cooldowns.pop(
            (provider, model),
            None,
        )

    def _filter_rate_limited_models(
        self,
        provider: str,
        models: list[str],
    ) -> list[str]:
        now = time.monotonic()
        available: list[str] = []
        skipped: list[str] = []

        for model in models:
            key = (provider, model)
            until = self._rate_limit_cooldowns.get(
                key
            )

            if (
                until is not None
                and until > now
            ):
                skipped.append(
                    f"{model} "
                    f"({until - now:.0f}s remaining)"
                )
                continue

            if until is not None:
                self._rate_limit_cooldowns.pop(
                    key,
                    None,
                )

            available.append(
                model
            )

        if skipped:
            logger.info(
                "Skipping temporarily rate-limited "
                f"{provider} model(s): "
                + ", ".join(skipped)
            )

        return available

    @staticmethod
    def _openrouter_headers(key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": (
                "https://github.com/"
                "mistertomlinson/Watchly"
            ),
            "X-Title": "Watchly",
        }

    async def _resolve_free_model_candidates(
        self,
        api_key: str | None = None,
    ) -> list[str]:
        """Return a live, approved, zero-cost OpenRouter fallback chain.

        OpenRouter free model slugs can disappear while similarly named paid
        variants remain available. Model discovery therefore verifies both the
        :free suffix and zero prompt/completion pricing before a model can be
        attempted. Paid models are never admitted to this chain.
        """
        now = time.monotonic()

        if (
            self._free_model_cache is not None
            and now < self._free_model_cache[0]
        ):
            return list(
                self._free_model_cache[1]
            )

        key = self._get_api_key(api_key)

        if not key:
            return []

        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
            ) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._openrouter_headers(
                        key
                    ),
                )

            if response.status_code != 200:
                raise RuntimeError(
                    "HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            catalog = (
                response.json().get("data")
                or []
            )

            eligible: set[str] = set()

            for item in catalog:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                model_id = item.get("id")

                if not _is_approved_free_model_id(
                    model_id
                ):
                    continue

                pricing = (
                    item.get("pricing")
                    or {}
                )

                try:
                    prompt_price = float(
                        pricing.get(
                            "prompt",
                            "1",
                        )
                    )
                    completion_price = float(
                        pricing.get(
                            "completion",
                            "1",
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    prompt_price != 0
                    or completion_price != 0
                ):
                    continue

                try:
                    context_length = int(
                        item.get(
                            "context_length"
                        )
                        or 0
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    context_length = 0

                if (
                    context_length
                    < FREE_MODEL_MIN_CONTEXT
                ):
                    continue

                architecture = (
                    item.get("architecture")
                    or {}
                )

                inputs = set(
                    architecture.get(
                        "input_modalities"
                    )
                    or []
                )

                outputs = set(
                    architecture.get(
                        "output_modalities"
                    )
                    or []
                )

                if (
                    inputs
                    and "text" not in inputs
                ):
                    continue

                if (
                    outputs
                    and "text" not in outputs
                ):
                    continue

                eligible.add(
                    model_id
                )

            models: list[str] = []

            for model_id in (
                DEFAULT_MODEL,
                *PREFERRED_FREE_MODELS,
            ):
                if (
                    model_id in eligible
                    and model_id not in models
                ):
                    models.append(
                        model_id
                    )

            for prefix in (
                APPROVED_FREE_MODEL_PREFIXES
            ):
                discovered = sorted(
                    model_id
                    for model_id in eligible
                    if (
                        model_id.startswith(
                            prefix
                        )
                        and model_id not in models
                    )
                )

                models.extend(
                    discovered
                )

            if DEFAULT_MODEL not in eligible:
                logger.warning(
                    "Configured OpenRouter model is not "
                    "currently available as an approved "
                    f"zero-cost model: {DEFAULT_MODEL}"
                )

            logger.info(
                "OpenRouter approved free model chain: "
                f"{models}"
            )

            self._free_model_cache = (
                now
                + FREE_MODEL_CATALOG_TTL_SECONDS,
                list(models),
            )

            return models

        except Exception as exc:
            # Discovery failure must never broaden into a paid fallback. These
            # are only explicitly free slugs from approved families; a stale
            # slug can fail harmlessly, but it cannot spend credit.
            fallback = []

            for model_id in (
                DEFAULT_MODEL,
                *PREFERRED_FREE_MODELS,
            ):
                if (
                    _is_approved_free_model_id(
                        model_id
                    )
                    and model_id not in fallback
                ):
                    fallback.append(
                        model_id
                    )

            logger.warning(
                "OpenRouter free-model discovery failed; "
                "using free-only safe fallback slugs "
                f"{fallback}: "
                f"{type(exc).__name__}: {exc}"
            )

            return fallback


    def _get_google_api_key(
        self,
        api_key: str | None = None,
    ) -> str | None:
        return api_key or settings.GEMINI_API_KEY

    async def _resolve_google_model_candidates(
        self,
        api_key: str | None = None,
    ) -> list[str]:
        """Resolve approved direct-Google fallback models for this job."""
        key = str(
            self._get_google_api_key(api_key)
            or ""
        ).strip()

        if not key:
            return []

        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
            ) as client:
                response = await client.get(
                    GOOGLE_MODELS_URL,
                    headers={
                        "x-goog-api-key": key,
                    },
                )

            if response.status_code != 200:
                raise RuntimeError(
                    "HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            catalog = (
                response.json().get("models")
                or []
            )

            eligible: set[str] = set()

            for item in catalog:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                model_id = (
                    item.get("baseModelId")
                    or str(
                        item.get("name")
                        or ""
                    ).removeprefix("models/")
                )

                if (
                    model_id
                    not in GOOGLE_MODEL_PREFERENCES
                ):
                    continue

                methods = set(
                    item.get(
                        "supportedGenerationMethods"
                    )
                    or []
                )

                if (
                    methods
                    and "generateContent"
                    not in methods
                ):
                    continue

                try:
                    input_limit = int(
                        item.get(
                            "inputTokenLimit"
                        )
                        or 0
                    )
                    output_limit = int(
                        item.get(
                            "outputTokenLimit"
                        )
                        or 0
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    input_limit
                    and input_limit
                    < GOOGLE_MIN_CONTEXT
                ):
                    continue

                if (
                    output_limit
                    and output_limit
                    < GOOGLE_MIN_OUTPUT
                ):
                    continue

                eligible.add(
                    model_id
                )

            models = [
                model_id
                for model_id in GOOGLE_MODEL_PREFERENCES
                if model_id in eligible
            ]

            logger.info(
                "Google AI approved fallback chain: "
                f"{models}"
            )

            return models

        except Exception as exc:
            # Discovery failure should not create an unbounded provider storm.
            # Keep the same small, known-safe set Seasonal Spotlight uses.
            fallback = list(
                GOOGLE_MODEL_PREFERENCES[:3]
            )

            logger.warning(
                "Google AI model discovery failed; "
                f"using bounded fallback {fallback}: "
                f"{type(exc).__name__}: {exc}"
            )

            return fallback

    @staticmethod
    def _google_headers(
        key: str,
    ) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    async def _call_google(
        self,
        prompt: str,
        system_instruction: str,
        google_api_key: str | None,
        max_tokens: int,
        validate_content: Callable[[str], None] | None = None,
        structured: bool = False,
    ) -> str:
        """Use direct Google AI only after the primary provider chain fails."""
        key = str(
            self._get_google_api_key(
                google_api_key
            )
            or ""
        ).strip()

        if not key:
            return ""

        models = (
            await self
            ._resolve_google_model_candidates(
                key
            )
        )

        if not models:
            logger.warning(
                "No approved direct Google AI "
                "fallback models are available."
            )
            return ""

        models = self._filter_rate_limited_models(
            "google-ai",
            models,
        )

        if not models:
            logger.info(
                "All direct Google AI fallback models "
                "are temporarily rate limited."
            )
            return ""

        errors: list[str] = []

        # Match Seasonal Spotlight's bounded provider fallback: at most two
        # direct-Google models, with one retry per model.
        for attempt, attempt_model in enumerate(
            models[:2],
            1,
        ):
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

            if structured:
                payload["response_format"] = {
                    "type": "json_object",
                }

            logger.info(
                "Google AI fallback attempt starting "
                f"attempt={attempt}/{min(len(models), 2)} "
                f"model={attempt_model} "
                f"max_tokens={max_tokens}"
            )

            started_at = time.monotonic()

            try:
                async with httpx.AsyncClient(
                    timeout=TIMEOUT,
                ) as client:
                    for http_attempt in range(
                        1,
                        3,
                    ):
                        response = await client.post(
                            GOOGLE_OPENAI_CHAT,
                            headers=self._google_headers(
                                key
                            ),
                            json=payload,
                        )

                        if (
                            response.status_code
                            in {
                                429,
                                503,
                                504,
                            }
                        ):
                            self._mark_rate_limited_model(
                                "google-ai",
                                attempt_model,
                            )

                            if http_attempt < 2:
                                retry_after_raw = (
                                    response.headers.get(
                                        "retry-after",
                                        "",
                                    )
                                )

                                try:
                                    retry_after = float(
                                        retry_after_raw
                                    )
                                except (
                                    TypeError,
                                    ValueError,
                                ):
                                    retry_after = 0.0

                                wait_seconds = min(
                                    max(
                                        retry_after,
                                        30.0,
                                    ),
                                    120.0,
                                )

                                logger.warning(
                                    "Google AI model rate limited "
                                    f"model={attempt_model} "
                                    f"http_attempt={http_attempt}/2 "
                                    f"waiting={wait_seconds:.1f}s"
                                )

                                await asyncio.sleep(
                                    wait_seconds
                                )
                                continue

                        if response.status_code != 200:
                            raise RuntimeError(
                                "HTTP "
                                f"{response.status_code}: "
                                f"{response.text[:500]}"
                            )

                        try:
                            data = response.json()
                            choices = (
                                data.get("choices")
                                or []
                            )

                            if not choices:
                                raise ValueError(
                                    "Google AI response had no choices"
                                )

                            choice = choices[0]
                            message = (
                                choice.get("message")
                                or {}
                            )
                            content = (
                                _content_to_text(
                                    message.get(
                                        "content"
                                    )
                                ).strip()
                            )

                            if not content:
                                raise ValueError(
                                    "Google AI returned empty content "
                                    f"(finish_reason="
                                    f"{choice.get('finish_reason')})"
                                )

                            if (
                                validate_content
                                is not None
                            ):
                                validate_content(
                                    content
                                )

                            self._clear_rate_limited_model(
                                "google-ai",
                                attempt_model,
                            )

                            logger.info(
                                "Google AI fallback completed "
                                f"attempt={attempt}/"
                                f"{min(len(models), 2)} "
                                f"requested_model={attempt_model} "
                                f"resolved_model="
                                f"{data.get('model') or attempt_model} "
                                f"finish_reason="
                                f"{choice.get('finish_reason')} "
                                f"duration="
                                f"{time.monotonic() - started_at:.2f}s "
                                f"content_chars={len(content)}"
                            )

                            return content

                        except Exception as exc:
                            if http_attempt >= 2:
                                raise

                            logger.warning(
                                "Google AI response unusable "
                                f"model={attempt_model} "
                                f"http_attempt={http_attempt}/2 "
                                f"error={type(exc).__name__}: {exc}; "
                                "retrying same model"
                            )

            except Exception as exc:
                error = (
                    f"attempt {attempt} "
                    f"model={attempt_model}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(
                    error
                )

                logger.warning(
                    "Google AI fallback attempt failed "
                    + error
                )

        logger.error(
            "Google AI fallback failed after all attempts: "
            + " | ".join(errors)
        )

        return ""

    @staticmethod
    def get_catalog_title_prompt():
        return """
        You are a content catalog naming expert.
        Given filters like genres, keywords, or years, generate natural,
        engaging catalog row titles that streaming platforms would use.
        Do not infer geography, nationality, language, or production country
        unless it is explicitly present in the supplied filters.

        Examples:
        - Keyword: "space", Genre: Sci-Fi → "Space Exploration Adventures"
        - Genre: Crime, Keyword: "based on novel or book" → "Literary Crime"
        - Genre: Sci-Fi, Keyword: "artificial intelligence" → "Rogue AI Thrillers"
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

        cooldown_provider = (
            "openrouter"
            if effective_base_url
            == self.base_url
            else None
        )

        if cooldown_provider:
            models = self._filter_rate_limited_models(
                cooldown_provider,
                models,
            )

        if not models:
            logger.info(
                "All candidate models are temporarily "
                "rate limited; skipping this provider "
                "for the current AI job."
            )
            return ""

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

            logger.info(
                "LLM attempt starting "
                f"attempt={attempt}/{len(models)} "
                f"model={attempt_model} "
                f"max_tokens={max_tokens}"
            )

            started_at = time.monotonic()

            try:
                async with httpx.AsyncClient(
                    timeout=TIMEOUT,
                ) as client:
                    for http_attempt in range(
                        1,
                        3,
                    ):
                        request_started_at = (
                            time.monotonic()
                        )

                        response = await client.post(
                            f"{effective_base_url}/chat/completions",
                            headers=self._openrouter_headers(
                                key
                            ),
                            json=payload,
                        )

                        elapsed = (
                            time.monotonic()
                            - request_started_at
                        )

                        response_text = (
                            response.text[:4000]
                        )

                        compact_response = (
                            response_text.replace(
                                " ",
                                "",
                            )
                        )

                        wrapped_rate_limit = (
                            response.status_code
                            in {
                                502,
                                503,
                                504,
                            }
                            and (
                                '"code":429'
                                in compact_response
                                or (
                                    "temporarily rate-limited"
                                    in response_text.lower()
                                )
                                or (
                                    "temporarily rate limited"
                                    in response_text.lower()
                                )
                            )
                        )

                        if (
                            response.status_code
                            == 429
                            or wrapped_rate_limit
                        ):
                            if cooldown_provider:
                                self._mark_rate_limited_model(
                                    cooldown_provider,
                                    attempt_model,
                                )

                            if http_attempt < 2:
                                retry_after_raw = (
                                    response.headers.get(
                                        "retry-after",
                                        "",
                                    )
                                )

                                try:
                                    retry_after = float(
                                        retry_after_raw
                                    )
                                except (
                                    TypeError,
                                    ValueError,
                                ):
                                    retry_after = 0.0

                                wait_seconds = min(
                                    max(
                                        retry_after,
                                        30.0,
                                    ),
                                    120.0,
                                )

                                logger.warning(
                                    "LLM model rate limited "
                                    f"model={attempt_model} "
                                    f"http_attempt="
                                    f"{http_attempt}/2 "
                                    f"waiting="
                                    f"{wait_seconds:.1f}s"
                                )

                                await asyncio.sleep(
                                    wait_seconds
                                )

                                continue

                        if response.status_code != 200:
                            raise RuntimeError(
                                "HTTP "
                                f"{response.status_code}: "
                                f"{response.text[:500]}"
                            )

                        data = response.json()
                        choices = (
                            data.get("choices")
                            or []
                        )

                        if not choices:
                            raise ValueError(
                                "LLM response had no choices"
                            )

                        choice = choices[0]
                        message = (
                            choice.get("message")
                            or {}
                        )
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

                        if cooldown_provider:
                            self._clear_rate_limited_model(
                                cooldown_provider,
                                attempt_model,
                            )

                        logger.info(
                            "LLM call completed "
                            f"attempt={attempt}/"
                            f"{len(models)} "
                            f"requested_model="
                            f"{attempt_model} "
                            f"resolved_model="
                            f"{data.get('model') or attempt_model} "
                            f"finish_reason="
                            f"{choice.get('finish_reason')} "
                            f"duration={elapsed:.2f}s "
                            f"max_tokens={max_tokens} "
                            f"content_chars="
                            f"{len(content)}"
                        )

                        return content

            except Exception as exc:
                total_elapsed = (
                    time.monotonic()
                    - started_at
                )
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
                    f"duration={total_elapsed:.2f}s "
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
        google_api_key: str | None = None,
        system_instruction: str | None = None,
    ) -> str:
        """Generate a catalog title with bounded provider fallback.

        A caller may supply a specialized naming instruction. Leaving it unset
        preserves the historical catalog-title prompt exactly.
        """
        title_system_instruction = (
            system_instruction
            or self.get_catalog_title_prompt()
        )
        key = self._get_api_key(
            api_key
        )
        result = ""

        if key and key.startswith("gsk_"):
            result = await self._call(
                prompt=prompt,
                system_instruction=title_system_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=TITLE_MAX_TOKENS,
            )
        elif key:
            attempt_models = (
                await self._resolve_free_model_candidates(
                    api_key
                )
            )

            if attempt_models:
                result = await self._call(
                    prompt=prompt,
                    system_instruction=title_system_instruction,
                    api_key=api_key,
                    max_tokens=TITLE_MAX_TOKENS,
                    attempt_models=attempt_models,
                )

        if result:
            return result

        google_key = self._get_google_api_key(
            google_api_key
        )

        if google_key:
            logger.info(
                "Primary AI provider chain exhausted; "
                "switching_provider=google-ai"
            )

            return await self._call_google(
                prompt=prompt,
                system_instruction=title_system_instruction,
                google_api_key=google_key,
                max_tokens=TITLE_MAX_TOKENS,
            )

        return ""

    async def generate_flash_content_async(
        self,
        prompt: str,
        system_instruction: str,
        api_key: str | None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        minimum_pipe_lines: int = 0,
        google_api_key: str | None = None,
    ) -> str:
        """Generate recommendations or summaries with provider fallback."""
        validator = (
            _pipe_list_validator(
                minimum_pipe_lines
            )
            if minimum_pipe_lines > 0
            else None
        )
        key = self._get_api_key(
            api_key
        )
        result = ""

        if key and key.startswith("gsk_"):
            result = await self._call(
                prompt=prompt,
                system_instruction=system_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=max_tokens,
                validate_content=validator,
            )
        elif key:
            attempt_models = (
                await self._resolve_free_model_candidates(
                    api_key
                )
            )

            if attempt_models:
                result = await self._call(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    attempt_models=attempt_models,
                    validate_content=validator,
                )

        if result:
            return result

        google_key = self._get_google_api_key(
            google_api_key
        )

        if google_key:
            logger.info(
                "Primary AI provider chain exhausted; "
                "switching_provider=google-ai"
            )

            return await self._call_google(
                prompt=prompt,
                system_instruction=system_instruction,
                google_api_key=google_key,
                max_tokens=max_tokens,
                validate_content=validator,
            )

        if not key:
            logger.warning(
                "No primary AI API key available."
            )
        else:
            logger.warning(
                "Primary AI provider chain exhausted "
                "and no Google AI fallback key is configured."
            )

        return ""

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: type | dict,
        system_instruction: str,
        api_key: str | None,
        google_api_key: str | None = None,
    ) -> dict | list | None:
        """Generate and validate JSON with bounded provider fallback."""
        structured_instruction = (
            system_instruction
            + "\n\nRespond ONLY with valid JSON matching "
            "the requested schema. No other text."
        )

        key = self._get_api_key(
            api_key
        )
        result = ""

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
        elif key:
            attempt_models = (
                await self._resolve_free_model_candidates(
                    api_key
                )
            )

            if attempt_models:
                result = await self._call(
                    prompt=prompt,
                    system_instruction=structured_instruction,
                    api_key=api_key,
                    max_tokens=(
                        STRUCTURED_MAX_TOKENS_OPENROUTER
                    ),
                    attempt_models=attempt_models,
                    validate_content=_json_validator,
                )

        google_key = self._get_google_api_key(
            google_api_key
        )

        if (
            not result
            and google_key
        ):
            logger.info(
                "Primary AI provider chain exhausted; "
                "switching_provider=google-ai"
            )

            result = await self._call_google(
                prompt=prompt,
                system_instruction=structured_instruction,
                google_api_key=google_key,
                max_tokens=(
                    STRUCTURED_MAX_TOKENS_OPENROUTER
                ),
                validate_content=_json_validator,
                structured=True,
            )

        if not result:
            return None

        try:
            parsed = _parse_json(
                result
            )

            if isinstance(
                parsed,
                (dict, list),
            ):
                return parsed
        except Exception as exc:
            logger.error(
                "Failed to parse validated AI "
                f"response: {exc}"
            )

        return None


openrouter_service = OpenRouterService()
gemini_service = openrouter_service
