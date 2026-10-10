import asyncio
from typing import Any

import httpx
from loguru import logger


class BaseClient:
    """
    Base asynchronous HTTP client with built-in retry logic and logging.
    """

    def __init__(
        self, base_url: str = "", timeout: float = 10.0, max_retries: int = 3, headers: dict[str, str] | None = None
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or {}
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx.AsyncClient instance."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, headers=self.headers, follow_redirects=True
            )
        return self._client

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _before_request_attempt(self) -> None:
        """Hook for subclasses that need per-attempt pacing."""
        return None

    async def _request(self, method: str, url: str, max_tries: int | None = None, **kwargs) -> httpx.Response:
        """Internal request handler with retry logic."""
        client = await self.get_client()
        tries = max_tries or self.max_retries

        for attempt in range(1, tries + 1):
            try:
                await self._before_request_attempt()
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.RequestError) as e:

                # Check if the error is retryable
                is_retryable = True
                if isinstance(e, httpx.HTTPStatusError):
                    # Only retry on 429 (Rate Limit) and 5xx (Server Errors)
                    # 404, 400, 401, etc. are not retryable
                    is_retryable = e.response.status_code in (429, 500, 502, 503, 504)

                if is_retryable and attempt < tries:
                    wait_time = 0.5 * (2 ** (attempt - 1))  # Exponential backoff

                    # A 429 may tell us exactly when the upstream wants the
                    # next request. Honor a numeric Retry-After when present.
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 429
                    ):
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = max(wait_time, float(retry_after))
                            except ValueError:
                                pass

                    logger.warning(
                        f"Request failed ({method} {url}): {str(e)}. "
                        f"Retrying in {wait_time}s... (Attempt {attempt}/{tries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # If not retryable or no more attempts left, log and raise
                    if not is_retryable:
                        logger.error(f"Non-retryable request failure ({method} {url}): {str(e)}")
                    else:
                        logger.error(f"Request failed after {tries} attempts ({method} {url}): {str(e)}")
                    raise e

        raise httpx.RequestError(f"Request failed for {method} {url} with 0 attempts configured")

    async def get(self, url: str, params: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """Perform a GET request and return the JSON response."""
        response = await self._request("GET", url, params=params, **kwargs)
        return response.json()

    async def post(self, url: str, json: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """Perform a POST request and return the JSON response."""
        response = await self._request("POST", url, json=json, **kwargs)
        return response.json()
