"""Dense retrieval over pgvector.

Uses the HNSW index with cosine distance. Three details matter more than the
query itself:

* **`ef_search` is set per session, not globally.** It is the recall/latency
  dial, and the right value depends on ``top_k``: asking for 20 results with the
  default ``ef_search`` silently returns a worse 20 than asking for 5 does.
* **The tenant filter is inside the SQL, not applied afterwards.** Post-filtering
  an approximate index is how you get a query that legitimately returns fewer
  results than asked for, or none at all, while the index reports success.
* **Expansions are searched and fused, not concatenated.** Averaging the query
  vectors of a multi-query expansion produces a vector that means nothing; each
  variant is searched separately and the ranked lists are fused.

Example:
    >>> DenseRetriever.name
    'pgvector'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import String, bindparam, text

from src.core.errors import RetrievalError
from src.core.logging import get_logger
from src.retrieval.base import Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.types import RetrievalRequest, RetrievalSource, RetrievedChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.ingestion.embedders.base import Embedder

log = get_logger(__name__)

#: HNSW candidate-list size. Scaled from top_k because a fixed value under-serves
#: large k and over-serves small k. Clamped so a hostile top_k cannot make the
#: database scan the whole index.
EF_SEARCH_MULTIPLIER = 8
EF_SEARCH_MIN = 40
EF_SEARCH_MAX = 400

_SEARCH_SQL = """
SELECT
    c.id                          AS chunk_id,
    c.content                     AS content,
    c.document_id                 AS document_id,
    c.ordinal                     AS ordinal,
    c.kind                        AS kind,
    c.page_number                 AS page_number,
    c.section_path                AS section_path,
    d.title                       AS document_title,
    d.effective_date              AS effective_date,
    1 - (c.embedding <=> CAST(:query_vector AS vector)) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = :tenant_id
  AND c.embedding IS NOT NULL
  AND d.deleted_at IS NULL
  {stale_clause}
  {document_clause}
  {kind_clause}
  {date_from_clause}
  {date_to_clause}
ORDER BY c.embedding <=> CAST(:query_vector AS vector)
LIMIT :limit
"""


class DenseRetriever(Retriever):
    """Vector similarity search over the tenant's chunks."""

    name = "pgvector"
    source = RetrievalSource.DENSE

    def __init__(self, *, session: AsyncSession, embedder: Embedder) -> None:
        """Create the retriever.

        Args:
            session: Tenant-scoped async session. Raw SQL bypasses the ORM tenant
                guard, so this backend applies the tenant filter itself.
            embedder: Used to embed the query with the model's query prefix.
        """
        self._session = session
        self._embedder = embedder

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Search each query variant and fuse the ranked lists."""
        queries = request.all_queries
        per_query: list[list[RetrievedChunk]] = []

        for query in queries:
            vector = await self._embedder.embed_query(query)
            per_query.append(await self._search(vector, request, tenant_id=tenant_id))

        if len(per_query) == 1:
            return per_query[0][: request.top_k]

        # Multiple variants: fuse by rank rather than averaging their vectors.
        return reciprocal_rank_fusion(per_query, top_k=request.top_k)

    async def _search(
        self, vector: Sequence[float], request: RetrievalRequest, *, tenant_id: str
    ) -> list[RetrievedChunk]:
        """Run one vector search."""
        sql, params = self._build_query(vector, request, tenant_id=tenant_id)
        try:
            # PostgreSQL does not accept a bind parameter in SET, so the value
            # is interpolated. It is an int that `_ef_search_for` has already
            # clamped to a fixed range, so there is nothing here a caller can
            # influence - which is what makes the interpolation safe.
            ef_search = _ef_search_for(request.top_k)
            await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search:d}"))
            result = await self._session.execute(sql, params)
            rows = result.mappings().all()
        except Exception as exc:
            log.error("dense retrieval failed", reason=str(exc))
            raise RetrievalError(f"vector search failed: {exc}") from exc

        return [self._to_chunk(row) for row in rows]

    def _build_query(
        self, vector: Sequence[float], request: RetrievalRequest, *, tenant_id: str
    ) -> tuple[Any, dict[str, Any]]:
        """Assemble the SQL and bound parameters for one search."""
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "query_vector": _vector_literal(vector),
            # Over-fetch so post-fusion truncation still has candidates to work
            # with, and so per-document capping does not empty the result.
            "limit": min(request.top_k * 3, 200),
        }
        clauses = {
            "stale_clause": "" if request.include_stale else "AND c.is_stale = FALSE",
            "document_clause": "",
            "kind_clause": "",
            "date_from_clause": "",
            "date_to_clause": "",
        }

        if request.document_ids:
            clauses["document_clause"] = "AND c.document_id = ANY(:document_ids)"
            params["document_ids"] = list(request.document_ids)
        if request.kinds:
            clauses["kind_clause"] = "AND c.kind = ANY(:kinds)"
            params["kinds"] = [k.value for k in request.kinds]
        if request.date_from:
            clauses["date_from_clause"] = "AND d.effective_date >= :date_from"
            params["date_from"] = request.date_from
        if request.date_to:
            clauses["date_to_clause"] = "AND d.effective_date <= :date_to"
            params["date_to"] = request.date_to

        statement = text(_SEARCH_SQL.format(**clauses)).bindparams(
            bindparam("query_vector", type_=String),
            bindparam("tenant_id", type_=String),
        )
        return statement, params

    @staticmethod
    def _to_chunk(row: Any) -> RetrievedChunk:
        """Map a result row onto the shared retrieval contract."""
        from src.models.document import ChunkKind

        section = row["section_path"] or ""
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            score=float(row["score"] or 0.0),
            source=RetrievalSource.DENSE,
            document_id=row["document_id"],
            document_title=row["document_title"],
            kind=ChunkKind(row["kind"]) if row["kind"] else ChunkKind.PROSE,
            ordinal=row["ordinal"],
            page_number=row["page_number"],
            section_path=tuple(section.split(" > ")) if section else (),
            metadata={"effective_date": _isoformat(row["effective_date"])},
        )


def _ef_search_for(top_k: int) -> int:
    """Scale the HNSW candidate list to the requested result count.

    Example:
        >>> _ef_search_for(5)
        40
        >>> _ef_search_for(20)
        160
        >>> _ef_search_for(1000)
        400
    """
    return min(max(top_k * EF_SEARCH_MULTIPLIER, EF_SEARCH_MIN), EF_SEARCH_MAX)


def _vector_literal(vector: Sequence[float]) -> str:
    """Render a vector in pgvector's literal syntax.

    Passing the literal rather than an adapted array keeps this backend working
    whether or not the pgvector SQLAlchemy adapter is registered on the session.
    The SQL casts it explicitly, because a bound string arrives as ``varchar``
    and the ``<=>`` operator has no ``vector <=> varchar`` overload.

    Example:
        >>> _vector_literal([0.5, -1.0])
        '[0.5,-1.0]'
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _isoformat(value: Any) -> str | None:
    """ISO-8601 string for a datetime column, or None."""
    return value.isoformat() if value is not None else None
