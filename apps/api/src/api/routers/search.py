"""Search endpoints.

Two surfaces over the same retriever: ``/api/search`` for the UI and the
playground, and ``/internal/search`` for the docs-search MCP server. They are
separate routes because they authenticate differently — the MCP server presents
a service token and names its tenant, while a browser presents a session — and
conflating them would mean either the UI could impersonate a tenant or the MCP
server could not name one.

The playground can override the workspace's strategies for a single search,
which is how "does HyDE actually help our corpus" gets answered with a
side-by-side rather than an opinion.

Example:
    >>> from src.api.routers.search import router
    >>> router.prefix
    ''
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request

from src.api.auth import Principal, get_principal
from src.api.dependencies import TenantRuntime, get_tenant_runtime
from src.core.errors import AuthenticationError
from src.core.logging import get_logger
from src.retrieval.types import RetrievalRequest
from src.schemas.documents import SearchHit, SearchRequest, SearchResponse

log = get_logger(__name__)

router = APIRouter(tags=["search"])


@router.post("/api/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
) -> SearchResponse:
    """Search the workspace's documents."""
    return await _run_search(request, runtime=runtime, tenant_id=principal.tenant_id)


@router.post("/internal/search", response_model=SearchResponse, include_in_schema=False)
async def internal_search(
    request: Request,
    payload: dict,
    x_tenant_id: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> SearchResponse:
    """Search on behalf of the docs-search MCP server.

    Excluded from the public schema: it is an internal contract between two of
    our own processes, and publishing it would invite clients to depend on a
    shape that exists only to serve the MCP server.

    Raises:
        AuthenticationError: when the service token is wrong or the tenant
            header is missing. The MCP server always sends both.
    """
    import os

    expected = os.getenv("AGRAG_INTERNAL_TOKEN", "")
    presented = (authorization or "").removeprefix("Bearer ").strip()
    if expected and presented != expected:
        raise AuthenticationError("Invalid internal service token.")
    if not x_tenant_id:
        raise AuthenticationError("Internal search requires an X-Tenant-Id header.")

    from src.api.dependencies import get_services, get_tenant_runtime

    # The MCP server names the tenant; the runtime is built for that tenant, and
    # retrieval is scoped by it exactly as it would be for a browser request.
    request.state.principal = _ServicePrincipal(x_tenant_id)
    services = get_services(request)
    runtime = await get_tenant_runtime(request, services, services.settings)

    if payload.get("list_only"):
        return await _list_documents(tenant_id=x_tenant_id, limit=int(payload.get("limit", 25)))

    return await _run_search(
        SearchRequest(
            query=payload.get("query") or "",
            top_k=int(payload.get("top_k") or 5),
            document_ids=tuple(payload.get("document_ids") or ()),
            tags=tuple(payload.get("tags") or ()),
            date_from=payload.get("date_from"),
            date_to=payload.get("date_to"),
        ),
        runtime=runtime,
        tenant_id=x_tenant_id,
    )


async def _run_search(
    request: SearchRequest, *, runtime: TenantRuntime, tenant_id: str
) -> SearchResponse:
    """Execute one search and record its telemetry."""
    from src.observability.metrics import record_retrieval
    from src.repositories.retrieval_logs import record_retrieval_log

    started = time.perf_counter()
    retriever = runtime.retriever
    if retriever is None:
        return SearchResponse(results=(), strategy="unavailable")

    if request.strategies:
        # The playground compares strategies, so it needs to override the
        # workspace's configuration for one search without persisting anything.
        from src.retrieval.hybrid import HybridConfig

        retriever._config = HybridConfig.from_strategies(
            list(request.strategies), top_k=request.top_k
        )

    result = await retriever.retrieve(
        RetrievalRequest(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
            tags=request.tags,
            kinds=request.kinds,
            date_from=request.date_from,
            date_to=request.date_to,
            include_stale=request.include_stale,
        ),
        tenant_id=tenant_id,
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    record_retrieval(strategy=result.strategy, duration_ms=latency_ms)

    try:
        await record_retrieval_log(tenant_id=tenant_id, request=request, result=result)
    except Exception as exc:  # noqa: BLE001 - telemetry must not fail a search
        log.warning("could not record the retrieval log", reason=str(exc))

    return SearchResponse(
        results=tuple(
            SearchHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                content=chunk.content,
                score=chunk.effective_score,
                rerank_score=chunk.rerank_score,
                page_number=chunk.page_number,
                section_path=chunk.section_path,
                contributing_ranks=chunk.contributing_ranks,
            )
            for chunk in result.chunks
        ),
        strategy=result.strategy,
        latency_ms=latency_ms,
        source_latencies_ms=result.source_latencies_ms,
        expanded=result.expanded,
        crag_verdict=result.crag_verdict.value if result.crag_verdict else None,
        web_fallback_used=result.web_fallback_used,
    )


async def _list_documents(*, tenant_id: str, limit: int) -> SearchResponse:
    """List documents for the MCP server's ``list_documents`` tool."""
    from src.core.db import session_scope
    from src.repositories.documents import list_documents

    async with session_scope() as session:
        documents = await list_documents(session, limit=limit)

    return SearchResponse(
        results=(),
        strategy="list",
        documents=tuple(
            {
                "id": d.id,
                "title": d.title,
                "status": d.status.value,
                "chunks": d.chunk_count,
                "tags": list(d.tags),
                "created_at": d.created_at.isoformat(),
            }
            for d in documents
        ),
    )


class _ServicePrincipal:
    """A minimal principal for internal service calls.

    Carries admin scope because the MCP server acts on the tenant's behalf, and
    carries no user id because there is no user behind the call — which keeps
    audit entries honest about what actually happened.
    """

    def __init__(self, tenant_id: str) -> None:
        """Bind the principal to a tenant."""
        self.tenant_id = tenant_id
        self.user_id = None
        self.api_key_id = None
        self.is_admin = True
        self.method = "internal"
        self.scopes = frozenset({"read", "write", "admin"})

    def has_scope(self, _scope: object) -> bool:
        """Internal calls hold every scope."""
        return True

    def require(self, _scope: object) -> None:
        """Internal calls are always permitted."""
        return
