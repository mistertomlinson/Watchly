import asyncio
from typing import Any

import httpx
from loguru import logger

from app.core.security import redact_url_query


class BaseClient:
    """
    Base asynchronous HTTP client with built-in retry logic and logging.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 10.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
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
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
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

    @staticmethod
    def _sanitize_request_exception(
        error: httpx.HTTPStatusError | httpx.RequestError,
        method: str,
        fallback_url: str,
    ) -> str:
        """Make an HTTPX exception safe to log and safe to re-raise.

        HTTPX status exceptions embed the complete request URL in their string
        representation. For clients such as TMDB, that URL contains api_key in
        the query string. Sanitize both the request object and exception text so
        downstream logging cannot accidentally reveal those query values.
        """
        request = getattr(
            error,
            "request",
            None,
        )

        request_url = getattr(
            request,
            "url",
            None,
        )

        safe_url = redact_url_query(
            request_url
            if request_url is not None
            else fallback_url
        )

        # HTTPX Request.url is writable. Sanitizing it protects downstream code
        # that logs error.request.url or error.response.url after we re-raise.
        if request is not None:
            try:
                request.url = httpx.URL(
                    safe_url
                )
            except (TypeError, ValueError):
                pass

        if isinstance(
            error,
            httpx.HTTPStatusError,
        ):
            status_code = (
                error.response.status_code
            )
            reason = (
                error.response.reason_phrase
                or ""
            ).strip()

            status = f"HTTP {status_code}"

            if reason:
                status += f" {reason}"

            safe_message = (
                f"{status} "
                f"({method} {safe_url})"
            )
        else:
            safe_message = (
                f"{type(error).__name__} "
                f"({method} {safe_url})"
            )

        # Exception.__str__ uses args. Replacing the original HTTPX message
        # prevents a later logger.exception(... {error}) from resurrecting the
        # original URL containing secrets.
        error.args = (
            safe_message,
        )

        return safe_message

    async def _request(
        self,
        method: str,
        url: str,
        max_tries: int | None = None,
        **kwargs,
    ) -> httpx.Response:
        """Internal request handler with retry logic."""
        client = await self.get_client()
        tries = max_tries or self.max_retries

        for attempt in range(
            1,
            tries + 1,
        ):
            try:
                await self._before_request_attempt()

                response = await client.request(
                    method,
                    url,
                    **kwargs,
                )

                response.raise_for_status()
                return response

            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
            ) as error:
                safe_error = (
                    self
                    ._sanitize_request_exception(
                        error,
                        method,
                        url,
                    )
                )

                # Check if the error is retryable.
                is_retryable = True

                if isinstance(
                    error,
                    httpx.HTTPStatusError,
                ):
                    # Only retry on 429 and selected 5xx errors.
                    is_retryable = (
                        error.response.status_code
                        in (
                            429,
                            500,
                            502,
                            503,
                            504,
                        )
                    )

                if (
                    is_retryable
                    and attempt < tries
                ):
                    wait_time = (
                        0.5
                        * (
                            2
                            ** (
                                attempt - 1
                            )
                        )
                    )

                    # A 429 may tell us exactly when the upstream wants the
                    # next request. Honor numeric Retry-After when present.
                    if (
                        isinstance(
                            error,
                            httpx.HTTPStatusError,
                        )
                        and (
                            error
                            .response
                            .status_code
                            == 429
                        )
                    ):
                        retry_after = (
                            error
                            .response
                            .headers
                            .get(
                                "Retry-After"
                            )
                        )

                        if retry_after:
                            try:
                                wait_time = max(
                                    wait_time,
                                    float(
                                        retry_after
                                    ),
                                )
                            except ValueError:
                                pass

                    logger.warning(
                        f"Request failed: "
                        f"{safe_error}. "
                        f"Retrying in "
                        f"{wait_time}s... "
                        f"(Attempt "
                        f"{attempt}/{tries})"
                    )

                    await asyncio.sleep(
                        wait_time
                    )

                else:
                    if not is_retryable:
                        logger.error(
                            "Non-retryable "
                            "request failure: "
                            f"{safe_error}"
                        )
                    else:
                        logger.error(
                            "Request failed after "
                            f"{tries} attempts: "
                            f"{safe_error}"
                        )

                    # The caught exception's message/request URL have already
                    # been sanitized. Bare raise also preserves its traceback.
                    raise

        raise httpx.RequestError(
            "Request failed for "
            f"{method} "
            f"{redact_url_query(url)} "
            "with 0 attempts configured"
        )

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform a GET request and return the JSON response."""
        response = await self._request(
            "GET",
            url,
            params=params,
            **kwargs,
        )
        return response.json()

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform a POST request and return the JSON response."""
        response = await self._request(
            "POST",
            url,
            json=json,
            **kwargs,
        )
        return response.json()
