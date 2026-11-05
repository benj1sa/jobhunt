"""Shared async HTTP client with bounded concurrency and retries.

Every outbound request in the pipeline goes through `HttpClient` so rate limiting and retry policy live in one place.
"""

import asyncio
from logging import getLogger
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..settings import settings

logger = getLogger(__name__)

# 529 is TypeSafe's "temporarily overloaded".
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}


class NotFoundError(Exception):
    """The resource returned 404/410. For job postings this usually means the job closed."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


class HttpClient:
    """Async context manager wrapping `httpx.AsyncClient`.

    Concurrency is bounded by a semaphore so a large board or a big Simplify batch can't flood any host.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            follow_redirects=True,
            headers={"User-Agent": settings.http_user_agent},
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(settings.http_concurrency)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send a request with retries on transient failures. Raises `NotFoundError` on 404/410."""
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:
                    response = await self._client.request(method, url, **kwargs)
                if response.status_code in (404, 410):
                    raise NotFoundError(f"{response.status_code} for {url}")
                response.raise_for_status()
                return response
        raise RuntimeError("unreachable: tenacity reraises on final failure")

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = await self.request("GET", url, params=params)
        return response.json()

    async def get_text(self, url: str) -> tuple[str, str]:
        """Return (final_url_after_redirects, body_text)."""
        response = await self.request("GET", url)
        return str(response.url), response.text
