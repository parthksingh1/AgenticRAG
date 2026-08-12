"""The AgenticRAG Python client.

    from agrag import AgRag

    client = AgRag(api_key="agr_...")
    answer = client.ask("What is the carry-over limit for annual leave?")
    print(answer.content)
    for citation in answer.citations:
        print(f"  [{citation.index}] {citation.document_title}")

Sync and async clients share their logic. The async one is the real
implementation; the sync one wraps it, because maintaining two copies of retry
and error handling guarantees they diverge and the one nobody uses is the one
that is wrong.

Every call retries on transient failures with exponential backoff and jitter.
Jitter is not decoration: without it, every client that failed at the same moment
retries at the same moment, and a service recovering from a blip is immediately
knocked over by its own clients.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.agrag.dev"
DEFAULT_TIMEOUT = 120.0

#: Status codes worth retrying. 429 and 5xx are transient; 4xx is the caller's
#: problem and retrying it just makes the same mistake more times.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


class AgRagError(Exception):
    """Base class for every error this client raises."""


class AuthenticationError(AgRagError):
    """The API key is missing, wrong, or lacks the required scope."""


class RateLimitError(AgRagError):
    """The tenant's rate limit or token budget is exhausted."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Carry the server's Retry-After, so a caller need not guess."""
        super().__init__(message)
        self.retry_after = retry_after


class APIError(AgRagError):
    """The API returned an error. Carries the status so callers can branch."""

    def __init__(self, message: str, status: int, code: str | None = None) -> None:
        """Record the status and the machine-readable code."""
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class Citation:
    """One source supporting part of an answer."""

    index: int
    chunk_id: str
    document_id: str
    document_title: str
    snippet: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    score: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Citation:
        """Build from the API's JSON, tolerating fields added later."""
        return cls(
            index=int(raw.get("index", 0)),
            chunk_id=str(raw.get("chunk_id", "")),
            document_id=str(raw.get("document_id", "")),
            document_title=str(raw.get("document_title", "")),
            snippet=str(raw.get("snippet", "")),
            page_number=raw.get("page_number"),
            section_path=tuple(raw.get("section_path") or ()),
            score=raw.get("score"),
        )


@dataclass(frozen=True, slots=True)
class Answer:
    """A complete answer with its sources and cost."""

    content: str
    citations: tuple[Citation, ...] = ()
    conversation_id: str | None = None
    message_id: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def refused(self) -> bool:
        """Whether the system declined to answer.

        Worth checking rather than assuming: a refusal is a successful response
        with a 200, and treating it as an answer is how "I don't have that
        information" ends up in a report.

        Example:
            >>> Answer(content="x", stop_reason="guardrail_blocked").refused
            True
        """
        return self.stop_reason in {"guardrail_blocked", "refused"}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Answer:
        """Build from the API's JSON."""
        return cls(
            content=str(raw.get("content", "")),
            citations=tuple(Citation.from_dict(c) for c in raw.get("citations") or ()),
            conversation_id=raw.get("conversation_id"),
            message_id=raw.get("message_id"),
            model=raw.get("model"),
            stop_reason=raw.get("stop_reason"),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            latency_ms=int(raw.get("latency_ms", 0)),
        )


@dataclass(frozen=True, slots=True)
class Document:
    """An indexed document."""

    id: str
    title: str
    status: str
    chunk_count: int = 0
    error_message: str | None = None
    tags: tuple[str, ...] = field(default=())

    @property
    def is_ready(self) -> bool:
        """Whether the document can be retrieved from.

        Example:
            >>> Document(id="d", title="t", status="indexed").is_ready
            True
        """
        return self.status == "indexed"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Document:
        """Build from the API's JSON."""
        return cls(
            id=str(raw["id"]),
            title=str(raw.get("title", "")),
            status=str(raw.get("status", "unknown")),
            chunk_count=int(raw.get("chunk_count", 0)),
            error_message=raw.get("error_message"),
            tags=tuple(raw.get("tags") or ()),
        )


class AsyncAgRag:
    """The asynchronous client.

    Example:
        >>> async with AsyncAgRag(api_key="agr_...") as client:  # doctest: +SKIP
        ...     answer = await client.ask("What is the notice period?")
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Create a client.

        Raises:
            ValueError: when the API key is empty. Failing here rather than on
                the first request means the mistake is reported where it was
                made.
        """
        if not api_key:
            msg = "an API key is required; pass api_key= or set AGRAG_API_KEY"
            raise ValueError(msg)

        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "agrag-python/0.1.0",
            },
        )

    async def __aenter__(self) -> AsyncAgRag:
        """Enter the context manager."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the underlying connection pool."""
        await self.close()

    async def close(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()

    async def ask(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
    ) -> Answer:
        """Ask a question and wait for the complete answer."""
        body: dict[str, Any] = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        if model:
            body["model"] = model
        return Answer.from_dict(await self._request("POST", "/api/chat", json=body))

    async def stream(
        self, message: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[str]:
        """Stream an answer token by token.

        Yields only content tokens. Callers wanting the citations should await
        :meth:`ask`, or read the ``citations`` event from the raw stream — a
        generator of strings cannot carry structured metadata without making the
        common case awkward.
        """
        body: dict[str, Any] = {"message": message, "include_thinking": False}
        if conversation_id:
            body["conversation_id"] = conversation_id

        async with self._client.stream("POST", "/api/chat/stream", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                self._raise_for(response)

            event = "message"
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event == "token":
                    import json

                    try:
                        text = json.loads(line[5:].strip()).get("text")
                    except ValueError:
                        continue
                    if text:
                        yield text

    async def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve without generating an answer."""
        body = await self._request("POST", "/api/search", json={"query": query, "top_k": top_k})
        return list(body.get("results", []))

    async def documents(self) -> list[Document]:
        """List the workspace's documents."""
        body = await self._request("GET", "/api/documents")
        return [Document.from_dict(d) for d in body]

    async def upload(self, path: str, *, title: str | None = None) -> Document:
        """Upload a file and queue it for ingestion.

        Three steps, because the bytes go straight to object storage rather than
        through the API — which would tie up a request handler for the whole
        transfer and cap the file size at the platform's request limit.
        """
        from pathlib import Path

        file_path = Path(path)
        data = file_path.read_bytes()

        slot = await self._request(
            "POST",
            "/api/documents/upload",
            json={
                "filename": file_path.name,
                "content_type": _mime_of(file_path.name),
                "size_bytes": len(data),
                "title": title or file_path.stem,
            },
        )

        # A plain client, without the Authorization header: presigned URLs carry
        # their own credentials, and sending ours to a storage host is a
        # credential leak to a third party.
        async with httpx.AsyncClient(timeout=300.0) as uploader:
            put = await uploader.put(
                slot["upload_url"],
                content=data,
                headers=slot.get("required_headers") or {},
            )
            if put.status_code >= 400:
                msg = f"the upload to storage failed with {put.status_code}"
                raise APIError(msg, put.status_code)

        return Document.from_dict(
            await self._request("POST", f"/api/documents/{slot['document_id']}/confirm")
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make one request, retrying transient failures."""
        last: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last = APIError(f"the request to {path} timed out", 408)
                if attempt < self._max_retries:
                    await _async_sleep(_backoff(attempt))
                    continue
                raise last from exc
            except httpx.HTTPError as exc:
                last = APIError(f"could not reach the API: {exc}", 0)
                if attempt < self._max_retries:
                    await _async_sleep(_backoff(attempt))
                    continue
                raise last from exc

            if response.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                # Honour Retry-After when the server sends one. Backing off less
                # than asked is how a rate limit becomes a rate-limit loop.
                delay = _retry_after(response) or _backoff(attempt)
                await _async_sleep(delay)
                continue

            if response.status_code >= 400:
                self._raise_for(response)

            return response.json() if response.content else {}

        raise last or APIError("the request failed", 0)  # pragma: no cover - unreachable

    def _raise_for(self, response: httpx.Response) -> None:
        """Convert an error response into the right exception type.

        Raises:
            AuthenticationError: on 401 and 403.
            RateLimitError: on 429.
            APIError: on anything else.
        """
        try:
            body = response.json()
        except ValueError:
            body = {}

        message = body.get("detail") or body.get("message") or response.text[:300]
        status = response.status_code

        if status in {401, 403}:
            raise AuthenticationError(message or "the API key was rejected")
        if status == 429:
            raise RateLimitError(
                message or "rate limit or token budget exhausted", _retry_after(response)
            )
        raise APIError(message or f"the API returned {status}", status, body.get("code"))


class AgRag:
    """The synchronous client.

    A thin wrapper over :class:`AsyncAgRag`. Two independent implementations of
    retry and error handling would diverge, and the one nobody exercises is the
    one that would be wrong.

    Example:
        >>> client = AgRag(api_key="agr_...")  # doctest: +SKIP
        >>> client.ask("What is the notice period?").content  # doctest: +SKIP
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Create a client with the same options as the async one."""
        self._async = AsyncAgRag(api_key, **kwargs)

    def __enter__(self) -> AgRag:
        """Enter the context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the underlying client."""
        self.close()

    def close(self) -> None:
        """Release the connection pool."""
        _run(self._async.close())

    def ask(self, message: str, **kwargs: Any) -> Answer:
        """Ask a question and wait for the complete answer."""
        return _run(self._async.ask(message, **kwargs))

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Retrieve without generating an answer."""
        return _run(self._async.search(query, **kwargs))

    def documents(self) -> list[Document]:
        """List the workspace's documents."""
        return _run(self._async.documents())

    def upload(self, path: str, **kwargs: Any) -> Document:
        """Upload a file and queue it for ingestion."""
        return _run(self._async.upload(path, **kwargs))

    def stream(self, message: str, **kwargs: Any) -> Iterator[str]:
        """Stream an answer token by token.

        Bridges the async generator by draining it into a list first. Correct,
        and it forfeits incremental delivery — which is why the async client is
        the right one for a UI. This exists so a script can use one API.
        """
        return iter(_run(_drain(self._async.stream(message, **kwargs))))


async def _drain(source: AsyncIterator[str]) -> list[str]:
    """Collect an async iterator into a list."""
    return [chunk async for chunk in source]


def _run(coro: Any) -> Any:
    """Run a coroutine from synchronous code.

    Raises:
        RuntimeError: when called from inside a running event loop. Silently
            spawning a second loop there deadlocks in ways that are very hard to
            diagnose, so this says what to do instead.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    msg = "AgRag is the synchronous client; inside an event loop, use AsyncAgRag"
    raise RuntimeError(msg)


async def _async_sleep(seconds: float) -> None:
    """Sleep without blocking the loop."""
    import asyncio

    await asyncio.sleep(seconds)


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Full jitter rather than a fixed multiplier: without it every client that
    failed at the same moment retries at the same moment, and a service
    recovering from a blip is immediately knocked over by its own clients.

    Example:
        >>> 0 <= _backoff(0) <= 0.5
        True
        >>> _backoff(99) <= MAX_BACKOFF_SECONDS
        True
    """
    ceiling = min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    return random.uniform(0, ceiling)  # noqa: S311 - jitter, not cryptography


def _retry_after(response: httpx.Response) -> float | None:
    """Read the Retry-After header, if the server sent a usable one.

    Example:
        >>> _retry_after(httpx.Response(429, headers={"Retry-After": "2"}))
        2.0
        >>> _retry_after(httpx.Response(429)) is None
        True
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # The header also allows an HTTP date. Falling back to normal backoff is
        # better than parsing dates for a value that is advisory anyway.
        return None


def _mime_of(filename: str) -> str:
    """Guess a content type from a filename.

    Example:
        >>> _mime_of("handbook.pdf")
        'application/pdf'
        >>> _mime_of("mystery.qqq")
        'application/octet-stream'
    """
    import mimetypes

    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


__all__: Sequence[str] = (
    "AgRag",
    "AsyncAgRag",
    "Answer",
    "APIError",
    "AuthenticationError",
    "Citation",
    "Document",
    "RateLimitError",
)
