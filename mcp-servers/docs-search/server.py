"""docs-search-mcp — tenant-scoped search over a workspace's documents.

The tool the agent reaches for most. It exposes the same hybrid retrieval the
main graph uses, but as a *tool* the model can call deliberately — which matters
for multi-hop questions, where the model needs to search several times with
different terms rather than once with whatever the user happened to type.

Tenant scoping is enforced here rather than trusted from the caller. This server
runs as a separate process and could in principle be reached by anything on the
network, so it takes the tenant from the request and applies it itself. The
alternative — trusting a caller-supplied filter — makes isolation depend on
every future caller being correct.

Filters are typed and validated. A model asked to search "documents from last
quarter tagged finance" will produce dates in half a dozen formats, so the tool
normalises them and reports what it actually applied, rather than silently
ignoring a filter it could not parse.

Example:
    >>> normalise_date("2024-03-15")
    '2024-03-15'
    >>> normalise_date("not a date") is None
    True
"""

from __future__ import annotations

import os
from typing import Any

from mcp_common.server import ToolResult, build_app, build_table, require, tool

VERSION = "1.0.0"

MAX_RESULTS = 20
DEFAULT_RESULTS = 5
MAX_SNIPPET_CHARS = 1200


def normalise_date(value: str | None) -> str | None:
    """Parse a date in whatever format the model produced into ISO-8601.

    Returns None for anything unparseable, and the caller reports that rather
    than applying a filter it guessed at.

    Example:
        >>> normalise_date("15 March 2024")
        '2024-03-15'
        >>> normalise_date(None) is None
        True
    """
    if not value:
        return None
    try:
        from dateutil import parser

        return parser.parse(value).date().isoformat()
    except Exception:  # noqa: BLE001 - an unparseable date is not an error
        return None


class SearchBackend:
    """Calls the main API's internal search endpoint.

    The retrieval stack lives in the API process, where the database sessions,
    embedder and reranker already are. Duplicating it here would mean two
    implementations of hybrid retrieval that could drift, and the eval numbers
    would only cover one of them.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        """Configure the backend from the environment."""
        self._base_url = (base_url or os.getenv("AGRAG_API_URL", "http://api:8000")).rstrip("/")
        self._token = token or os.getenv("AGRAG_INTERNAL_TOKEN", "")

    async def search(self, payload: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """Run a search against the API.

        Raises:
            RuntimeError: when the API is unreachable, so the caller can turn it
                into a failed tool result rather than an empty one. An empty
                result would read to the model as "there is nothing", which is a
                different and worse answer than "search is unavailable".
        """
        import httpx

        headers = {"X-Tenant-Id": tenant_id}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{self._base_url}/internal/search", json=payload, headers=headers
            )
            if response.status_code >= 400:
                msg = f"search backend returned {response.status_code}: {response.text[:200]}"
                raise RuntimeError(msg)
            return dict(response.json())


BACKEND = SearchBackend()


def format_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Shape one result for the model.

    Truncates the snippet: the model receives several results and does not need
    the full chunk of each, and sending them all is how a tool call quietly
    consumes the context window the answer needs.

    Example:
        >>> format_hit({"content": "x" * 5000, "chunk_id": "c1"})["snippet"].endswith("...")
        True
    """
    content = hit.get("content", "")
    snippet = content[:MAX_SNIPPET_CHARS] + ("..." if len(content) > MAX_SNIPPET_CHARS else "")
    return {
        "chunk_id": hit.get("chunk_id"),
        "document_id": hit.get("document_id"),
        "document_title": hit.get("document_title"),
        "page_number": hit.get("page_number"),
        "section": " > ".join(hit.get("section_path") or []) or None,
        "score": round(float(hit.get("score") or 0.0), 4),
        "snippet": snippet,
    }


@tool(
    "search_documents",
    "Search this workspace's documents. Use several searches with different "
    "wording for a question that needs more than one fact.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict the search to specific documents.",
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "date_from": {"type": "string", "description": "Only documents dated on or after."},
            "date_to": {"type": "string", "description": "Only documents dated on or before."},
        },
        "required": ["query"],
    },
    deterministic=False,
    read_only=True,
)
async def search_documents_handler(arguments: dict[str, Any], tenant_id: str) -> ToolResult:
    """Search the tenant's documents."""
    query = require(arguments, "query")
    limit = min(int(arguments.get("limit") or DEFAULT_RESULTS), MAX_RESULTS)

    date_from = normalise_date(arguments.get("date_from"))
    date_to = normalise_date(arguments.get("date_to"))
    ignored = [
        name
        for name, raw, parsed in (
            ("date_from", arguments.get("date_from"), date_from),
            ("date_to", arguments.get("date_to"), date_to),
        )
        if raw and not parsed
    ]

    payload = {
        "query": query,
        "top_k": limit,
        "document_ids": arguments.get("document_ids") or [],
        "tags": arguments.get("tags") or [],
        "date_from": date_from,
        "date_to": date_to,
    }

    try:
        response = await BACKEND.search(payload, tenant_id)
    except Exception as exc:  # noqa: BLE001 - "unavailable" is not "nothing found"
        return ToolResult.failure(f"document search is unavailable: {exc}")

    hits = [format_hit(h) for h in response.get("results", [])]
    summary = f"{len(hits)} result{'s' if len(hits) != 1 else ''} for {query!r}"
    if ignored:
        # Reporting an ignored filter matters: silently dropping it would give
        # the model a result set it believes is narrower than it is.
        summary += f"; could not parse {', '.join(ignored)} and searched without it"

    return ToolResult.success(
        {"results": hits, "query": query, "ignored_filters": ignored},
        summary=summary,
        metadata={"strategy": response.get("strategy")},
    )


@tool(
    "list_documents",
    "List the documents available in this workspace, most recent first.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "tag": {"type": "string"},
        },
    },
    deterministic=False,
    read_only=True,
)
async def list_documents_handler(arguments: dict[str, Any], tenant_id: str) -> ToolResult:
    """List the tenant's documents."""
    payload = {
        "list_only": True,
        "limit": min(int(arguments.get("limit") or 25), 100),
        "tags": [arguments["tag"]] if arguments.get("tag") else [],
    }
    try:
        response = await BACKEND.search(payload, tenant_id)
    except Exception as exc:  # noqa: BLE001 - surfaced as a failed tool result
        return ToolResult.failure(f"document listing is unavailable: {exc}")

    documents = response.get("documents", [])
    return ToolResult.success(
        {"documents": documents},
        summary=f"{len(documents)} document{'s' if len(documents) != 1 else ''}",
    )


TOOLS = build_table(search_documents_handler, list_documents_handler)


async def check_backend() -> list[str]:
    """Report the search backend as degraded when it cannot be reached."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{BACKEND._base_url}/healthz")
            return [] if response.status_code == 200 else ["search-backend"]
    except Exception:  # noqa: BLE001 - a degraded dependency, not a crash
        return ["search-backend"]


app = build_app(
    name="docs-search",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
    health_check=check_backend,
)
