"""Provider-agnostic LLM contracts.

Every provider (OpenAI, Anthropic, Google, Groq, Together) is adapted to these
types, so the agent, guardrails and eval harness are written once. The important
design choice is that :class:`Completion` always carries usage and cost: a call
that cannot be attributed to a tenant and a price is a call that will eventually
show up as an unexplained bill.

Example:
    >>> Message.user("hello").role.value
    'user'
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    """Who authored a message in a completion request."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """One turn in a completion request."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    #: Populated on assistant turns that requested tools.
    tool_calls: tuple[ToolCall, ...] = ()
    #: Populated on tool-result turns; matches a ToolCall id.
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        """Build a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Build a user message.

        Example:
            >>> Message.user("hi").content
            'hi'
        """
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, *, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        """Build an assistant message, optionally requesting tools."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, *, tool_call_id: str, content: str, name: str | None = None) -> Message:
        """Build a tool-result message answering a specific tool call."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class ToolCall(BaseModel):
    """A model's request to invoke a tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    """A tool offered to the model, in JSON-Schema form."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token accounting for one provider call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens.

        Example:
            >>> Usage(prompt_tokens=10, completion_tokens=5).total_tokens
            15
        """
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum two usage records, for aggregating a multi-call turn."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class Completion(BaseModel):
    """A finished (non-streaming) model response."""

    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "error"] = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    #: True when the primary provider failed and a fallback answered.
    was_fallback: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def requested_tools(self) -> bool:
        """True when the model asked to call at least one tool."""
        return bool(self.tool_calls)


class StreamEventType(StrEnum):
    """Kinds of event emitted while streaming a completion."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    """One event in a streaming completion."""

    model_config = ConfigDict(frozen=True)

    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    error: str | None = None


class CompletionRequest(BaseModel):
    """Everything needed to make one model call."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...]
    model: str
    max_tokens: int = Field(default=2048, ge=1, le=200_000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    #: JSON-Schema the response must conform to, when the provider supports it.
    response_schema: dict[str, Any] | None = None
    #: Attribution, carried into usage records and Langfuse traces.
    tenant_id: str | None = None
    node: str | None = None
    prompt_version: str | None = None

    @model_validator(mode="after")
    def _check_messages(self) -> Self:
        """A request with no messages is always a caller bug."""
        if not self.messages:
            msg = "completion request must contain at least one message"
            raise ValueError(msg)
        return self


class ModelPricing(BaseModel):
    """USD price per million tokens for one model.

    Prices are data, not code: they live in :mod:`src.services.llm.pricing` and
    are the single place a cost figure can come from, so no dashboard number is
    ever a hand-typed guess.
    """

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0.0)
    output_per_mtok: float = Field(ge=0.0)
    cached_input_per_mtok: float | None = Field(default=None, ge=0.0)

    def cost_for(self, usage: Usage) -> float:
        """USD cost of a call with this usage.

        Cached input tokens are billed at the cached rate when the provider
        publishes one, and at the normal input rate otherwise.

        Example:
            >>> p = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
            >>> round(p.cost_for(Usage(prompt_tokens=1_000_000, completion_tokens=0)), 4)
            3.0
        """
        cached_rate = (
            self.cached_input_per_mtok
            if self.cached_input_per_mtok is not None
            else self.input_per_mtok
        )
        billable_input = max(usage.prompt_tokens - usage.cached_tokens, 0)
        return (
            billable_input * self.input_per_mtok
            + usage.cached_tokens * cached_rate
            + usage.completion_tokens * self.output_per_mtok
        ) / 1_000_000
