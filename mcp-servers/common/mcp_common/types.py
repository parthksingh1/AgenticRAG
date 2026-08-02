"""Shared contracts for the MCP servers.

Every server exposes the same surface — ``/healthz``, ``/metrics``, ``/tools``
and ``/call`` — so the client in ``apps/api/src/mcp_clients`` is written once and
a new server is a deployment, not a code change.

Tool results carry an ``ok`` flag rather than relying on HTTP status codes. A
tool that fails is normal and useful information for the model ("that query was
invalid, try a different one"), so a failed tool call is a 200 with ``ok:
false``. A non-200 means the *server* failed, which is a different problem and
gets retried rather than shown to the model.

Example:
    >>> ToolResult.failure("division by zero").ok
    False
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """A tool this server offers, in JSON-Schema form."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    #: Whether identical arguments always produce an identical result. Drives the
    #: client's caching decision, so it must be honest: marking a live-data tool
    #: deterministic serves stale answers.
    deterministic: bool = False
    #: Whether the tool can change state. Read-only tools are safe to retry.
    read_only: bool = True


class ToolCallRequest(BaseModel):
    """A request to invoke one tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Required on every call. The servers scope their data by it rather than
    #: trusting the caller to have filtered already.
    tenant_id: str
    #: Propagated so a tool call appears under the turn that caused it.
    trace_id: str | None = None


class ToolResult(BaseModel):
    """The outcome of one tool invocation."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    content: Any = None
    error: str | None = None
    #: Human-readable note shown in the thinking panel, e.g. "3 rows, 12ms".
    summary: str | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, content: Any, *, summary: str | None = None, **extra: Any) -> ToolResult:
        """Build a successful result.

        Example:
            >>> ToolResult.success(4, summary="2+2").content
            4
        """
        return cls(ok=True, content=content, summary=summary, **extra)

    @classmethod
    def failure(cls, error: str, **extra: Any) -> ToolResult:
        """Build a failed result.

        Failure is data the model can act on, not an exception, which is why this
        is a normal return value.

        Example:
            >>> ToolResult.failure("bad input").error
            'bad input'
        """
        return cls(ok=False, error=error, **extra)


class HealthResponse(BaseModel):
    """Healthcheck payload."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    server: str
    version: str
    tools: int
    #: Populated when a dependency the server needs is unreachable. The server
    #: still reports ``ok`` if it can serve *some* tools, because taking the whole
    #: server out of rotation for one degraded backend is an overreaction.
    degraded: list[str] = Field(default_factory=list)
