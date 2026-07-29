import httpx
import os

from loguru import logger

from app.core.config import settings

# "openrouter/auto" routes to PAID models and bills per token. "openrouter/free"
# is OpenRouter's free-tier auto-router: it picks a currently-free model matching
# the request's needs (including structured output), so this keeps working as
# individual free models are retired. Override with OPENROUTER_MODEL to pin one.
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TIMEOUT = 60.0

# Providers count the REQUESTED max_tokens against per-minute token budgets, not the
# tokens actually returned. 4000 was reserved for every call regardless of size, which
# on Groq's free tier (12k TPM) meant a handful of concurrent calls exhausted the
# budget and returned 429 before the model ever ran. These are sized to the real
# responses, with headroom.
DEFAULT_MAX_TOKENS = 4000
# 1200 was sized for Groq, which counts the REQUESTED ceiling against a 12k
# tokens-per-minute budget. OpenRouter meters requests, not tokens, so a higher
# ceiling is free there -- and necessary: models that reason before answering
# were spending the whole allowance thinking and returning finish_reason=length
# with no content at all.
# Groq meters the REQUESTED ceiling against a 12k tokens-per-minute budget, so a
# large value there exhausts the quota before the model runs. OpenRouter meters
# requests, not tokens, so headroom is free -- and needed, because models that
# reason before answering otherwise spend the whole allowance thinking and return
# finish_reason=length with no content.
STRUCTURED_MAX_TOKENS_GROQ = 1200
STRUCTURED_MAX_TOKENS_OPENROUTER = 8000
STRUCTURED_MAX_TOKENS = STRUCTURED_MAX_TOKENS_OPENROUTER  # a JSON array of ~5 themed rows, plus reasoning headroom
TITLE_MAX_TOKENS = 60         # a single 2-5 word catalog title


class OpenRouterService:
    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1"

    def _get_api_key(self, api_key: str | None = None) -> str | None:
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
    ) -> str:
        key = self._get_api_key(api_key)
        if not key:
            logger.warning("No OpenRouter API key available.")
            return ""

        effective_base_url = base_url or self.base_url
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(
                    f"{effective_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/mistertomlinson/Watchly",
                        "X-Title": "Watchly",
                    },
                    json=payload,
                )
                if response.status_code != 200:
                    logger.error(f"OpenRouter API error {response.status_code}: {response.text[:500]}")
                    return ""
                data = response.json()
                # Some models (reasoning models in particular) return a null
                # content field and put their output elsewhere, or return an
                # empty message with a finish_reason. Crashing on .strip() here
                # surfaced as an opaque "unusable response" much further up.
                choices = data.get("choices") or []
                if not choices:
                    logger.error(f"LLM response had no choices: {str(data)[:300]}")
                    return ""
                message = choices[0].get("message") or {}
                content = message.get("content")
                # NB: do NOT fall back to message["reasoning"] here. On reasoning
                # models that field holds the chain-of-thought, not the answer, and
                # it ends up rendered verbatim as a catalog row title.
                if not content:
                    logger.error(
                        f"LLM returned empty content (model={data.get('model')}, "
                        f"finish_reason={choices[0].get('finish_reason')})"
                    )
                    return ""
                return content.strip()
        except Exception as e:
            logger.exception(f"OpenRouter API error: {e}")
            return ""

    async def generate_content_async(self, prompt: str) -> str:
        """Used for catalog title generation (uses server-side key)."""
        return await self._call(
            prompt=prompt,
            system_instruction=self.get_catalog_title_prompt(),
            max_tokens=TITLE_MAX_TOKENS,
        )

    async def generate_flash_content_async(
        self,
        prompt: str,
        system_instruction: str,
        api_key: str,
    ) -> str:
        """Used for recommendations and interest summaries (uses user key)."""
        # If key looks like a Groq key, use Groq API
        if api_key and api_key.startswith("gsk_"):
            return await self._call(
                prompt=prompt,
                system_instruction=system_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
            )
        return await self._call(
            prompt=prompt,
            system_instruction=system_instruction,
            api_key=api_key,
        )

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: type | dict,
        system_instruction: str,
        api_key: str,
    ) -> dict | list | None:
        """Used for structured JSON responses."""
        import json
        structured_instruction = (
            system_instruction
            + "\n\nRespond ONLY with valid JSON matching the requested schema. No other text."
        )
        # Route by key type, matching generate_flash_content_async. Without this a
        # Groq key was sent to the OpenRouter endpoint and rejected with
        # "401 Missing Authentication header", so structured row generation silently
        # fell back to tiered sampling while plain-text calls kept working.
        key = self._get_api_key(api_key)
        if key and key.startswith("gsk_"):
            result = await self._call(
                prompt=prompt,
                system_instruction=structured_instruction,
                api_key=api_key,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                max_tokens=STRUCTURED_MAX_TOKENS_GROQ,
            )
        else:
            result = await self._call(
                prompt=prompt,
                system_instruction=structured_instruction,
                api_key=api_key,
                max_tokens=STRUCTURED_MAX_TOKENS_OPENROUTER,
            )
        if not result:
            return None
        try:
            clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except Exception as e:
            logger.error(f"Failed to parse structured OpenRouter response: {e}")
            return None


openrouter_service = OpenRouterService()
# Alias for drop-in compatibility
gemini_service = openrouter_service
