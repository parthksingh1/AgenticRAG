"""Shared MCP server scaffolding.

Builds the FastAPI application every server exposes, so each server file
contains only its tools. The handler contract is deliberately narrow — an async
callable taking ``(arguments, tenant_id)`` and returning a
:class:`~mcp_common.types.ToolResult` — because that is the smallest thing that
can be unit-tested without HTTP.

Three behaviours are shared by every server and therefore live here:

* **Unhandled exceptions become failed results, not 500s.** A tool that raises is
  a tool the model should be told about, and a 500 would make the client retry
  an operation that will fail identically.
* **Every call is timed and counted**, so ``/metrics`` is real without each
  server remembering to instrument itself.
* **A tenant id is mandatory.** The server scopes its own data rather than
  trusting the caller, which is what makes these safe to run as separate
  processes.

Example:
    >>> app = build_app(name="demo", version="1.0.0", tools={})
    >>> app.title
    'demo MCP server'
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from mcp_common.types import HealthResponse, ToolCallRequest, ToolResult, ToolSpec

#: An async tool handler: takes validated arguments and the calling tenant.
ToolHandler = Callable[[dict[str, Any], str], Awaitable[ToolResult]]

#: name -> (spec, handler)
ToolTable = dict[str, tuple[ToolSpec, ToolHandler]]


class Metrics:
    """Per-tool call counters, exposed in Prometheus text format."""

    def __init__(self) -> None:
        """Create empty counters."""
        self.calls: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.duration_ms: dict[str, float] = {}

    def record(self, tool: str, *, ok: bool, duration_ms: float) -> None:
        """Record one call."""
        self.calls[tool] = self.calls.get(tool, 0) + 1
        self.duration_ms[tool] = self.duration_ms.get(tool, 0.0) + duration_ms
        if not ok:
            self.failures[tool] = self.failures.get(tool, 0) + 1

    def render(self, server: str) -> str:
        """Render the counters as Prometheus exposition text.

        Example:
            >>> m = Metrics()
            >>> m.record("calc", ok=True, duration_ms=5)
            >>> "mcp_tool_calls_total" in m.render("calculator")
            True
        """
        lines = [
            "# HELP mcp_tool_calls_total Tool invocations.",
            "# TYPE mcp_tool_calls_total counter",
        ]
        lines.extend(
            f'mcp_tool_calls_total{{server="{server}",tool="{tool}"}} {count}'
            for tool, count in sorted(self.calls.items())
        )
        lines += [
            "# HELP mcp_tool_failures_total Tool invocations that returned ok=false.",
            "# TYPE mcp_tool_failures_total counter",
        ]
        lines.extend(
            f'mcp_tool_failures_total{{server="{server}",tool="{tool}"}} {count}'
            for tool, count in sorted(self.failures.items())
        )
        lines += [
            "# HELP mcp_tool_duration_ms_sum Cumulative tool duration in milliseconds.",
            "# TYPE mcp_tool_duration_ms_sum counter",
        ]
        lines.extend(
            f'mcp_tool_duration_ms_sum{{server="{server}",tool="{tool}"}} {total:.1f}'
            for tool, total in sorted(self.duration_ms.items())
        )
        return "\n".join(lines) + "\n"


def build_app(
    *,
    name: str,
    version: str,
    tools: ToolTable,
    description: str = "",
    health_check: Callable[[], Awaitable[list[str]]] | None = None,
) -> FastAPI:
    """Build the FastAPI application for one MCP server.

    Args:
        name: Server name, used in metrics labels and the healthcheck.
        version: Server version, reported by the healthcheck.
        tools: The tools this server offers.
        description: Shown in the generated OpenAPI document.
        health_check: Optional async probe returning the names of degraded
            dependencies. Returning names does not fail the healthcheck: a server
            that can still serve some tools should stay in rotation.
    """
    app = FastAPI(
        title=f"{name} MCP server",
        version=version,
        description=description,
        docs_url="/docs",
    )
    metrics = Metrics()
    app.state.metrics = metrics
    app.state.tools = tools

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness and readiness probe."""
        degraded = await health_check() if health_check else []
        return HealthResponse(
            status="ok", server=name, version=version, tools=len(tools), degraded=degraded
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> str:
        """Prometheus scrape endpoint."""
        return metrics.render(name)

    @app.get("/tools")
    async def list_tools() -> list[ToolSpec]:
        """Advertise this server's tools so the client can build its registry."""
        return [spec for spec, _ in tools.values()]

    @app.post("/call", response_model=ToolResult)
    async def call_tool(request: ToolCallRequest) -> ToolResult:
        """Invoke one tool.

        Always returns 200 with an ``ok`` flag. A tool failure is information for
        the model, not a transport error, and returning 5xx would make the client
        retry something guaranteed to fail again.
        """
        entry = tools.get(request.name)
        if entry is None:
            available = ", ".join(sorted(tools)) or "none"
            return ToolResult.failure(f"unknown tool {request.name!r}; available: {available}")

        _, handler = entry
        started = time.perf_counter()
        try:
            result = await handler(request.arguments, request.tenant_id)
        except Exception as exc:  # noqa: BLE001 - a raising tool is a failed result
            duration = (time.perf_counter() - started) * 1000
            metrics.record(request.name, ok=False, duration_ms=duration)
            return ToolResult.failure(f"{type(exc).__name__}: {exc}", duration_ms=int(duration))

        duration = (time.perf_counter() - started) * 1000
        metrics.record(request.name, ok=result.ok, duration_ms=duration)
        return result.model_copy(update={"duration_ms": int(duration)})

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        """Return a structured error rather than an HTML traceback page."""
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"{type(exc).__name__}: {exc}", "server": name},
        )

    return app


def tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    *,
    deterministic: bool = False,
    read_only: bool = True,
) -> Callable[[ToolHandler], tuple[ToolSpec, ToolHandler]]:
    """Decorator pairing a handler with its spec.

    Example:
        >>> schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        >>> @tool("double", "Doubles a number.", schema, deterministic=True)
        ... async def double(args, tenant_id):
        ...     return ToolResult.success(args["x"] * 2)
        >>> double[0].name
        'double'
    """

    def decorate(handler: ToolHandler) -> tuple[ToolSpec, ToolHandler]:
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            deterministic=deterministic,
            read_only=read_only,
        )
        return spec, handler

    return decorate


def build_table(*entries: tuple[ToolSpec, ToolHandler]) -> ToolTable:
    """Assemble decorated handlers into a tool table.

    Raises:
        ValueError: on a duplicate tool name, which would otherwise make the
            active implementation depend on declaration order.

    Example:
        >>> schema = {"type": "object"}
        >>> @tool("a", "A.", schema)
        ... async def handler(args, tenant_id):
        ...     return ToolResult.success(None)
        >>> sorted(build_table(handler))
        ['a']
    """
    table: ToolTable = {}
    for spec, handler in entries:
        if spec.name in table:
            msg = f"duplicate tool name: {spec.name}"
            raise ValueError(msg)
        table[spec.name] = (spec, handler)
    return table


def require(arguments: dict[str, Any], key: str, kind: type = str) -> Any:
    """Read a required argument, raising a clear error when it is missing.

    Models omit and mistype arguments routinely, so the message names the tool's
    expectation rather than surfacing a KeyError.

    Example:
        >>> require({"q": "select 1"}, "q")
        'select 1'
        >>> require({}, "q")
        Traceback (most recent call last):
        ...
        ValueError: missing required argument 'q'
    """
    if key not in arguments or arguments[key] is None:
        msg = f"missing required argument {key!r}"
        raise ValueError(msg)
    value = arguments[key]
    if not isinstance(value, kind):
        msg = f"argument {key!r} must be {kind.__name__}, got {type(value).__name__}"
        raise ValueError(msg)
    return value
