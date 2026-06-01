import httpx
from loguru import logger

from app.core.config import settings

DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"
TIMEOUT = 60.0


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
    ) -> str:
        key = self._get_api_key(api_key)
        if not key:
            logger.warning("No OpenRouter API key available.")
            return ""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/mistertomlinson/Watchly",
                        "X-Title": "Watchly",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception(f"OpenRouter API error: {e}")
            return ""

    async def generate_content_async(self, prompt: str) -> str:
        """Used for catalog title generation (uses server-side key)."""
        return await self._call(
            prompt=prompt,
            system_instruction=self.get_catalog_title_prompt(),
        )

    async def generate_flash_content_async(
        self,
        prompt: str,
        system_instruction: str,
        api_key: str,
    ) -> str:
        """Used for recommendations and interest summaries (uses user key)."""
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
        result = await self._call(
            prompt=prompt,
            system_instruction=structured_instruction,
            api_key=api_key,
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
