import httpx
import pytest
from loguru import logger

from app.core.base_client import BaseClient
from app.core.security import redact_url_query
from app.services.tmdb.client import TMDBClient


SECRET = "tmdb-super-secret-value-DO-NOT-LOG"


def _capture_logs():
    messages = []

    sink = logger.add(
        lambda message: messages.append(
            str(message)
        ),
        format="{message}",
    )

    return messages, sink


def test_redact_url_query_redacts_every_value():
    url = (
        "https://example.test/path"
        "?api_key=super-secret"
        "&language=en-US"
        "&page=4"
    )

    safe = redact_url_query(
        url
    )

    assert "super-secret" not in safe
    assert "en-US" not in safe
    assert "page=4" not in safe

    assert (
        "https://example.test/path"
        in safe
    )

    assert "api_key=" in safe
    assert "language=" in safe
    assert "page=" in safe


@pytest.mark.asyncio
async def test_tmdb_404_never_logs_or_reraises_api_key():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            404,
            request=request,
            json={
                "status_message":
                    "Not found"
            },
        )

    client = TMDBClient(
        api_key=SECRET,
        max_retries=1,
    )

    client._client = httpx.AsyncClient(
        base_url=(
            "https://"
            "api.themoviedb.org/3"
        ),
        transport=httpx.MockTransport(
            handler
        ),
    )

    messages, sink = (
        _capture_logs()
    )

    try:
        with pytest.raises(
            httpx.HTTPStatusError
        ) as captured:
            await client.get(
                "/tv/569547"
            )
    finally:
        logger.remove(
            sink
        )
        await client.close()

    logs = "".join(
        messages
    )

    error = captured.value

    assert SECRET not in logs
    assert SECRET not in str(
        error
    )

    assert SECRET not in str(
        error.request.url
    )

    assert SECRET not in str(
        error.response.url
    )

    # We retain useful diagnostics:
    # endpoint + parameter names + status.
    assert "/3/tv/569547" in logs
    assert "api_key=" in logs
    assert "language=" in logs
    assert "HTTP 404" in logs


@pytest.mark.asyncio
async def test_request_error_never_logs_or_reraises_query_secret():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    client = BaseClient(
        base_url=(
            "https://example.test"
        ),
        max_retries=1,
    )

    client._client = httpx.AsyncClient(
        base_url=(
            "https://example.test"
        ),
        transport=httpx.MockTransport(
            handler
        ),
    )

    messages, sink = (
        _capture_logs()
    )

    try:
        with pytest.raises(
            httpx.ConnectError
        ) as captured:
            await client.get(
                "/resource",
                params={
                    "access_token":
                        SECRET,
                    "page": 3,
                },
            )
    finally:
        logger.remove(
            sink
        )
        await client.close()

    logs = "".join(
        messages
    )

    error = captured.value

    assert SECRET not in logs
    assert SECRET not in str(
        error
    )

    assert SECRET not in str(
        error.request.url
    )

    assert "/resource" in logs
    assert "access_token=" in logs
    assert "page=" in logs
    assert "ConnectError" in logs


@pytest.mark.asyncio
async def test_full_url_query_is_sanitized_too():
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
        )

    client = BaseClient(
        max_retries=1,
    )

    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            handler
        ),
    )

    messages, sink = (
        _capture_logs()
    )

    try:
        with pytest.raises(
            httpx.HTTPStatusError
        ) as captured:
            await client.get(
                "https://example.test/"
                "resource"
                "?api_key="
                + SECRET
                + "&language=en-US"
            )
    finally:
        logger.remove(
            sink
        )
        await client.close()

    logs = "".join(
        messages
    )

    assert SECRET not in logs
    assert SECRET not in str(
        captured.value
    )

    assert SECRET not in str(
        captured.value.request.url
    )
