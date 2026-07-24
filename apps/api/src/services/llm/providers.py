"""Provider adapters.

Each adapter maps the provider-agnostic :class:`CompletionRequest` onto one
vendor SDK and maps the response back. Vendor SDKs are imported lazily inside
the adapters so that importing this module — which the whole application does —
never requires every provider's package to be installed.

:class:`FakeProvider` is a first-class citizen, not a test afterthought: the unit
suite, the CI eval smoke run and the offline demo all drive the full agent
through it, which is what makes those runs deterministic and free.

Example:
    >>> provider = FakeProvider(responses=["hello"])
    >>> provider.name
    'fake'
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from src.core.errors import ProviderError
from src.core.logging import get_logger
from src.services.llm.pricing import price_for
from src.services.llm.types import (
    Completion,
    CompletionRequest,
    Message,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)

log = get_logger(__name__)


class Provider(ABC):
    """One LLM vendor, adapted to the internal contract."""

    #: Stable provider name, used in usage records and error messages.
    name: str

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> Completion:
        """Return a full completion."""

    @abstractmethod
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Yield streaming events for a completion."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release any underlying client resources."""

    @staticmethod
    def _split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
        """Separate leading system messages from the conversation.

        Anthropic takes the system prompt as a top-level parameter rather than a
        message, so every adapter needs this split.

        Example:
            >>> system, rest = Provider._split_system(
            ...     [Message.system("be brief"), Message.user("hi")]
            ... )
            >>> system, [m.content for m in rest]
            ('be brief', ['hi'])
        """
        systems = [m.content for m in messages if m.role is Role.SYSTEM]
        rest = [m for m in messages if m.role is not Role.SYSTEM]
        return ("\n\n".join(systems) or None), rest


class AnthropicProvider(Provider):
    """Adapter for the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, *, api_key: str, timeout: float = 60.0, max_retries: int = 0) -> None:
        """Create the adapter.

        ``max_retries`` defaults to zero because retries are owned by
        :mod:`src.services.llm.router`, which also handles cross-provider
        fallback. Letting the SDK retry as well would multiply the two.
        """
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def complete(self, request: CompletionRequest) -> Completion:
        """Call the Messages API and adapt the response."""
        system, messages = self._split_system(request.messages)
        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system or NOT_GIVEN_SENTINEL,
                messages=[self._to_anthropic(m) for m in messages],
                tools=[self._tool_to_anthropic(t) for t in request.tools] or NOT_GIVEN_SENTINEL,
                stop_sequences=list(request.stop) or NOT_GIVEN_SENTINEL,
            )
        except Exception as exc:
            raise ProviderError(provider=self.name, reason=str(exc)) from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = tuple(
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )
        return Completion(
            content=text,
            model=request.model,
            provider=self.name,
            usage=usage,
            cost_usd=price_for(request.model).cost_for(usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="tool_calls" if tool_calls else "stop",
            tool_calls=tool_calls,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then a final usage event."""
        system, messages = self._split_system(request.messages)
        try:
            async with self._client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system or NOT_GIVEN_SENTINEL,
                messages=[self._to_anthropic(m) for m in messages],
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamEvent(type=StreamEventType.TEXT, text=text)
                final = await stream.get_final_message()
                usage = Usage(
                    prompt_tokens=final.usage.input_tokens,
                    completion_tokens=final.usage.output_tokens,
                )
                yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
                yield StreamEvent(type=StreamEventType.DONE)
        except Exception as exc:  # noqa: BLE001 - surfaced as a stream error event
            yield StreamEvent(type=StreamEventType.ERROR, error=str(exc))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    @staticmethod
    def _to_anthropic(message: Message) -> dict[str, Any]:
        """Map one internal message onto the Anthropic wire shape."""
        if message.role is Role.TOOL:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        if message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in message.tool_calls
            )
            return {"role": "assistant", "content": blocks}
        return {"role": message.role.value, "content": message.content}

    @staticmethod
    def _tool_to_anthropic(tool: Any) -> dict[str, Any]:
        """Map a ToolSpec onto the Anthropic tool shape."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters or {"type": "object", "properties": {}},
        }


class OpenAICompatibleProvider(Provider):
    """Adapter for OpenAI and any API that speaks the same protocol.

    Groq and Together both expose OpenAI-compatible endpoints, so they are the
    same adapter with a different ``base_url`` — which is why the class is named
    for the protocol rather than the vendor.
    """

    def __init__(
        self,
        *,
        api_key: str,
        name: str = "openai",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Create the adapter for one OpenAI-compatible endpoint."""
        from openai import AsyncOpenAI

        self.name = name
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )

    async def complete(self, request: CompletionRequest) -> Completion:
        """Call chat.completions and adapt the response."""
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [self._to_openai(m) for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters or {"type": "object", "properties": {}},
                    },
                }
                for t in request.tools
            ]
        if request.response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.response_schema},
            }

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(provider=self.name, reason=str(exc)) from exc

        choice = response.choices[0]
        tool_calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_safe_json(call.function.arguments),
            )
            for call in (choice.message.tool_calls or [])
        )
        raw_usage = response.usage
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        )
        return Completion(
            content=choice.message.content or "",
            model=request.model,
            provider=self.name,
            usage=usage,
            cost_usd=price_for(request.model).cost_for(usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=_map_finish_reason(choice.finish_reason),
            tool_calls=tool_calls,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream token deltas, then usage."""
        try:
            stream = await self._client.chat.completions.create(
                model=request.model,
                messages=[self._to_openai(m) for m in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT, text=chunk.choices[0].delta.content
                    )
                if getattr(chunk, "usage", None):
                    yield StreamEvent(
                        type=StreamEventType.USAGE,
                        usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        ),
                    )
            yield StreamEvent(type=StreamEventType.DONE)
        except Exception as exc:  # noqa: BLE001 - surfaced as a stream error event
            yield StreamEvent(type=StreamEventType.ERROR, error=str(exc))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    @staticmethod
    def _to_openai(message: Message) -> dict[str, Any]:
        """Map one internal message onto the OpenAI wire shape."""
        if message.role is Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in message.tool_calls
                ],
            }
        return {"role": message.role.value, "content": message.content}


class GoogleProvider(Provider):
    """Adapter for the Google Gemini API."""

    name = "google"

    def __init__(self, *, api_key: str, timeout: float = 60.0) -> None:
        """Create the adapter."""
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._timeout = timeout

    async def complete(self, request: CompletionRequest) -> Completion:
        """Call generate_content and adapt the response."""
        system, messages = self._split_system(request.messages)
        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model,
                contents=[
                    {
                        "role": "model" if m.role is Role.ASSISTANT else "user",
                        "parts": [{"text": m.content}],
                    }
                    for m in messages
                ],
                config={
                    "system_instruction": system,
                    "max_output_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
            )
        except Exception as exc:
            raise ProviderError(provider=self.name, reason=str(exc)) from exc

        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        )
        return Completion(
            content=response.text or "",
            model=request.model,
            provider=self.name,
            usage=usage,
            cost_usd=price_for(request.model).cost_for(usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream text chunks from Gemini."""
        system, messages = self._split_system(request.messages)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=[{"role": "user", "parts": [{"text": m.content}]} for m in messages],
                config={"system_instruction": system, "max_output_tokens": request.max_tokens},
            )
            async for chunk in stream:
                if chunk.text:
                    yield StreamEvent(type=StreamEventType.TEXT, text=chunk.text)
            yield StreamEvent(type=StreamEventType.DONE)
        except Exception as exc:  # noqa: BLE001 - surfaced as a stream error event
            yield StreamEvent(type=StreamEventType.ERROR, error=str(exc))

    async def aclose(self) -> None:
        """No persistent client resources to release."""
        return


class FakeProvider(Provider):
    """Deterministic in-process provider.

    Drives the full agent without network access, which is what makes the unit
    suite fast, the CI eval smoke run free, and the offline demo reproducible.
    Responses are consumed in order and the last one repeats, so a test that
    makes one more call than expected gets a sensible answer instead of an
    IndexError from inside the graph.

    Example:
        >>> import asyncio
        >>> from src.services.llm.types import CompletionRequest, Message
        >>> provider = FakeProvider(responses=["first", "second"])
        >>> req = CompletionRequest(messages=(Message.user("q"),), model="fake-model")
        >>> asyncio.run(provider.complete(req)).content
        'first'
    """

    name = "fake"

    def __init__(
        self,
        *,
        responses: Sequence[str] | None = None,
        tool_calls: Sequence[Sequence[ToolCall]] | None = None,
        fail_times: int = 0,
    ) -> None:
        """Create a fake provider.

        Args:
            responses: Text replies, consumed in order; the last one repeats.
            tool_calls: Tool calls to attach to the matching response index.
            fail_times: Raise :class:`ProviderError` for the first N calls, to
                exercise the router's retry and fallback paths.
        """
        self._responses = list(responses or ["fake response"])
        self._tool_calls = [tuple(c) for c in (tool_calls or [])]
        self._fail_times = fail_times
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Completion:
        """Return the next scripted response."""
        self.calls.append(request)
        if len(self.calls) <= self._fail_times:
            raise ProviderError(provider=self.name, reason="scripted failure")

        index = min(len(self.calls) - 1, len(self._responses) - 1)
        content = self._responses[index]
        tools = self._tool_calls[index] if index < len(self._tool_calls) else ()
        usage = Usage(
            prompt_tokens=sum(len(m.content) for m in request.messages) // 4,
            completion_tokens=max(len(content) // 4, 1),
        )
        return Completion(
            content=content,
            model=request.model,
            provider=self.name,
            usage=usage,
            cost_usd=price_for(request.model).cost_for(usage),
            latency_ms=1,
            finish_reason="tool_calls" if tools else "stop",
            tool_calls=tools,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream the scripted response one word at a time."""
        completion = await self.complete(request)
        for word in completion.content.split(" "):
            yield StreamEvent(type=StreamEventType.TEXT, text=word + " ")
        yield StreamEvent(type=StreamEventType.USAGE, usage=completion.usage)
        yield StreamEvent(type=StreamEventType.DONE)

    async def aclose(self) -> None:
        """Nothing to release."""
        return


class _NotGiven:
    """Sentinel that vendor SDKs interpret as "parameter omitted"."""

    def __bool__(self) -> bool:
        """Always falsy, so ``value or NOT_GIVEN_SENTINEL`` reads naturally."""
        return False

    def __repr__(self) -> str:
        """Readable in tracebacks."""
        return "NOT_GIVEN"


NOT_GIVEN_SENTINEL = _NotGiven()


def _safe_json(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments, tolerating a model that emits invalid JSON.

    A malformed tool call is a normal event, not an exception: the executor
    turns the ``_parse_error`` key into a corrective message back to the model.

    Example:
        >>> _safe_json('{"a": 1}')
        {'a': 1}
        >>> _safe_json("not json")["_parse_error"]
        True
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _map_finish_reason(reason: str | None) -> str:
    """Normalise a provider finish reason onto the internal vocabulary.

    Example:
        >>> _map_finish_reason("tool_calls")
        'tool_calls'
        >>> _map_finish_reason("something_new")
        'stop'
    """
    mapping = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(reason or "stop", "stop")
