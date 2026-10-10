import asyncio
import time
from collections import deque
from typing import Any

from app.core.base_client import BaseClient
from app.core.version import __version__


# TMDB removed its old 40-requests/10-seconds rule, but still documents
# protective limits around 40 requests/second. P1 and P2 share one VPS/IP, so
# keep each process comfortably below that ceiling.
TMDB_REQUESTS_PER_SECOND = 15
TMDB_RATE_WINDOW_SECONDS = 1.0

# Concurrency and request rate are different controls: the semaphore bounds
# in-flight sockets, while the sliding window bounds how quickly new requests
# are allowed to start.
_tmdb_semaphore = asyncio.Semaphore(30)
_tmdb_rate_lock = asyncio.Lock()
_tmdb_request_times: deque[float] = deque()


async def _wait_for_tmdb_rate_slot() -> None:
    while True:
        async with _tmdb_rate_lock:
            now = time.monotonic()

            while (
                _tmdb_request_times
                and now - _tmdb_request_times[0] >= TMDB_RATE_WINDOW_SECONDS
            ):
                _tmdb_request_times.popleft()

            if len(_tmdb_request_times) < TMDB_REQUESTS_PER_SECOND:
                _tmdb_request_times.append(now)
                return

            wait_time = (
                TMDB_RATE_WINDOW_SECONDS
                - (now - _tmdb_request_times[0])
            )

        await asyncio.sleep(max(wait_time, 0.001))


class TMDBClient(BaseClient):
    """
    Client for interacting with the TMDB API.
    """

    def __init__(self, api_key: str, language: str = "en-US", timeout: float = 10.0, max_retries: int = 3):
        headers = {
            "User-Agent": f"Watchly/{__version__} (+https://github.com/TimilsinaBimal/Watchly)",
            "Accept": "application/json",
        }
        super().__init__(
            base_url="https://api.themoviedb.org/3", timeout=timeout, max_retries=max_retries, headers=headers
        )
        self.api_key = api_key
        self.language = language

    async def _before_request_attempt(self) -> None:
        """Pace every TMDB HTTP attempt, including retries."""
        await _wait_for_tmdb_rate_slot()

    async def _request(self, method: str, url: str, **kwargs) -> Any:
        """Override request to always include API key and language."""
        params = kwargs.get("params", {})
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        params["language"] = self.language
        kwargs["params"] = params
        async with _tmdb_semaphore:
            return await super()._request(method, url, **kwargs)
