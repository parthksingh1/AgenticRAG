"""MCP tool registry and client.

Discovers tools from the configured servers at startup and presents them to the
agent as one flat namespace. The agent should not know which process serves a
tool, and a new server should be a deployment rather than a code change.

Four behaviours the agent depends on:

* **A tool failure is a result, not an exception.** The executor feeds it back to
  the model, which usually corrects itself. Raising would end a turn over
  something recoverable.
* **A server that is down is skipped, not fatal.** Its tools simply are not
  offered, so the agent answers without them rather than failing. Discovery is
  retried in the background.
* **Deterministic tools are cached** by ``(tool, arguments, tenant)``; the server
  declares which of its tools are deterministic, and a wrong declaration would
  serve stale data, so it is treated as part of the contract.
* **Every call is timed and traced**, and carries the tenant, because the servers
  scope their data by it rather than trusting a filter in the arguments.

Example:
    >>> registry = ToolRegistry(clients=[])
    >>> registry.describe()
    'No tools are available.'
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.errors import ToolExecutionError
from src.core.logging import get_logger
from src.services.llm.types import ToolSpec

if TYPE_CHECKING:
    from src.caching.base import ToolResultCache

log = get_logger(__name__)

DISCOVERY_TIMEOUT = 5.0
CALL_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """A tool offered by one server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server: str
    deterministic: bool = False
    read_only: bool = True

    def to_spec(self) -> ToolSpec:
        """Convert to the provider-facing tool specification."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.input_schema or {"type": "object", "properties": {}},
        )


@dataclass(slots=True)
class ToolCallOutcome:
    """What one tool call produced."""

    tool: str
    ok: bool
    content: Any = None
    error: str | None = None
    summary: str | None = None
    duration_ms: int = 0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def for_model(self) -> str:
        """Render the outcome as the text the model sees.

        A failure is phrased as something the model can act on rather than as a
        stack trace, because the next thing it does is decide whether to retry
        with different arguments.

        Example:
            >>> ToolCallOutcome(tool="calc", ok=False, error="bad input").for_model()
            'calc failed: bad input'
        """
        if not self.ok:
            return f"{self.tool} failed: {self.error}"
        import json

        return json.dumps(self.content, default=str)[:8000]


class MCPClient:
    """HTTP client for one MCP server."""

    def __init__(self, *, name: str, base_url: str, timeout: float = CALL_TIMEOUT) -> None:
        """Create a client for one server."""
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Any | None = None

    async def _http(self) -> Any:
        """Return the shared HTTP client, opening it on first use."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def discover(self) -> list[RemoteTool]:
        """Ask the server what it offers.

        Returns an empty list when the server is unreachable, so a down server
        removes its tools rather than taking the agent down with it.
        """
        try:
            client = await self._http()
            response = await client.get(f"{self._base_url}/tools", timeout=DISCOVERY_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - a down server is not fatal
            log.warning(
                "MCP server unavailable during discovery", server=self.name, reason=str(exc)
            )
            return []

        return [
            RemoteTool(
                name=entry["name"],
                description=entry.get("description", ""),
                input_schema=entry.get("input_schema", {}),
                server=self.name,
                deterministic=bool(entry.get("deterministic")),
                read_only=bool(entry.get("read_only", True)),
            )
            for entry in payload
        ]

    async def call(
        self, tool: str, arguments: dict[str, Any], *, tenant_id: str, trace_id: str | None = None
    ) -> ToolCallOutcome:
        """Invoke a tool on this server."""
        started = time.perf_counter()
        try:
            client = await self._http()
            response = await client.post(
                f"{self._base_url}/call",
                json={
                    "name": tool,
                    "arguments": arguments,
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an outcome
            return ToolCallOutcome(
                tool=tool,
                ok=False,
                error=f"could not reach {self.name}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        duration = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 500:
            # A 5xx is the server failing, not the tool: worth surfacing
            # differently because it is retryable and the tool failure is not.
            return ToolCallOutcome(
                tool=tool,
                ok=False,
                error=f"{self.name} returned {response.status_code}",
                duration_ms=duration,
            )

        body = response.json()
        return ToolCallOutcome(
            tool=tool,
            ok=bool(body.get("ok")),
            content=body.get("content"),
            error=body.get("error"),
            summary=body.get("summary"),
            duration_ms=duration,
            metadata=body.get("metadata", {}),
        )

    async def healthy(self) -> bool:
        """Whether the server answers its healthcheck."""
        try:
            client = await self._http()
            response = await client.get(f"{self._base_url}/healthz", timeout=DISCOVERY_TIMEOUT)
        except Exception:  # noqa: BLE001 - unreachable means unhealthy
            return False
        return bool(response.status_code == 200)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class ToolRegistry:
    """Presents every server's tools to the agent as one namespace."""

    def __init__(
        self,
        *,
        clients: Sequence[MCPClient],
        cache: ToolResultCache | None = None,
        max_concurrency: int = 4,
    ) -> None:
        """Create the registry.

        Args:
            clients: One client per configured server.
            cache: Result cache for deterministic tools.
            max_concurrency: Simultaneous in-flight tool calls per turn.
        """
        self._clients = {client.name: client for client in clients}
        self._cache = cache
        self._tools: dict[str, RemoteTool] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.calls = 0
        self.failures = 0
        self.cache_hits = 0

    async def discover(self) -> int:
        """Discover tools from every server, returning the number registered.

        A name offered by two servers is a configuration error: the agent would
        get whichever won the race. The later one is refused and logged rather
        than silently overriding.
        """
        results = await asyncio.gather(*(c.discover() for c in self._clients.values()))

        self._tools.clear()
        for tools in results:
            for remote in tools:
                existing = self._tools.get(remote.name)
                if existing is not None:
                    log.error(
                        "duplicate tool name across MCP servers; keeping the first",
                        tool=remote.name,
                        kept=existing.server,
                        ignored=remote.server,
                    )
                    continue
                self._tools[remote.name] = remote

        log.info(
            "discovered MCP tools",
            count=len(self._tools),
            servers=sorted(self._clients),
        )
        return len(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Tool specifications to offer the model."""
        return tuple(tool.to_spec() for tool in sorted(self._tools.values(), key=lambda t: t.name))

    def describe(self) -> str:
        """A human-readable tool list for the planner prompt.

        Example:
            >>> ToolRegistry(clients=[]).describe()
            'No tools are available.'
        """
        if not self._tools:
            return "No tools are available."
        return "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in sorted(self._tools.values(), key=lambda t: t.name)
        )

    def names(self) -> tuple[str, ...]:
        """Every registered tool name."""
        return tuple(sorted(self._tools))

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        tenant_id: str,
        trace_id: str | None = None,
    ) -> Any:
        """Invoke a tool, returning content the model can read.

        Raises:
            ToolExecutionError: only when the tool does not exist. Every other
                failure is returned as text, because the model can recover from
                a bad argument but not from an exception.
        """
        remote = self._tools.get(tool)
        if remote is None:
            available = ", ".join(self.names()) or "none"
            raise ToolExecutionError(tool=tool, reason=f"unknown tool; available: {available}")

        if self._cache is not None and remote.deterministic:
            cached = await self._cache.get(tool=tool, arguments=arguments, tenant_id=tenant_id)
            if cached is not None:
                self.cache_hits += 1
                return cached

        client = self._clients[remote.server]
        async with self._semaphore:
            outcome = await client.call(tool, arguments, tenant_id=tenant_id, trace_id=trace_id)

        self.calls += 1
        if not outcome.ok:
            self.failures += 1
            log.info("tool call failed", tool=tool, server=remote.server, error=outcome.error)
            return outcome.for_model()

        if self._cache is not None and remote.deterministic:
            await self._cache.set(
                tool=tool, arguments=arguments, tenant_id=tenant_id, result=outcome.content
            )

        return outcome.content

    async def health(self) -> dict[str, bool]:
        """Healthcheck every configured server, for the admin dashboard."""
        names = list(self._clients)
        results = await asyncio.gather(*(self._clients[n].healthy() for n in names))
        return dict(zip(names, results, strict=True))

    async def aclose(self) -> None:
        """Close every client."""
        await asyncio.gather(*(c.aclose() for c in self._clients.values()))


def build_registry(
    servers: dict[str, str], *, cache: ToolResultCache | None = None
) -> ToolRegistry:
    """Build a registry from a name-to-URL map.

    Example:
        >>> registry = build_registry({"calculator": "http://mcp-calculator:8080"})
        >>> registry.names()
        ()
    """
    clients = [MCPClient(name=name, base_url=url) for name, url in sorted(servers.items())]
    return ToolRegistry(clients=clients, cache=cache)


#: Default wiring for the local stack, matching the compose file's service names.
DEFAULT_SERVERS: dict[str, str] = {
    "docs-search": "http://mcp-docs-search:8080",
    "sql-analytics": "http://mcp-sql-analytics:8080",
    "web-fetch": "http://mcp-web-fetch:8080",
    "calculator": "http://mcp-calculator:8080",
    "kg-query": "http://mcp-kg-query:8080",
    "code-exec": "http://mcp-code-exec:8080",
}
