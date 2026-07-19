"""Sparse retrieval over OpenSearch BM25.

BM25 is not a legacy fallback here — it is the half of hybrid retrieval that
actually finds exact identifiers, product codes, error strings and rare proper
nouns, all of which dense embeddings blur into their nearest common neighbour.
A user searching for ``ERR_4021`` wants the document containing that exact token,
and only lexical search reliably delivers it.

The index is per tenant (``chunks-{tenant_id}``) rather than one shared index
with a filter. That is a deliberate isolation choice: a filter bug leaks data,
whereas a wrong index name simply finds nothing, and per-tenant indices also make
GDPR erasure a single delete call.

Example:
    >>> index_name_for("ten_acme")
    'agrag-chunks-ten_acme'
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.core.errors import RetrievalError
from src.core.logging import get_logger
from src.retrieval.base import Retriever
from src.retrieval.types import RetrievalRequest, RetrievalSource, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

log = get_logger(__name__)

INDEX_PREFIX = "agrag-chunks"

#: Field weights. Titles and section headings are short and high-signal, so a
#: match there is worth more than the same match buried in body text.
FIELD_WEIGHTS = ("content^1.0", "document_title^2.0", "section_path^1.5")

#: Index settings tuned for retrieval rather than for logging workloads.
INDEX_SETTINGS: dict[str, Any] = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "5s"},
        "analysis": {
            "analyzer": {
                "agrag_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    # `asciifolding` before the stemmer so "café" and "cafe" match;
                    # keyword_repeat + unique keep the unstemmed form searchable so
                    # exact identifiers survive stemming.
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "keyword_repeat",
                        "porter_stem",
                        "unique",
                    ],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "tenant_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "agrag_text"},
            "document_title": {"type": "text", "analyzer": "agrag_text"},
            "section_path": {"type": "text", "analyzer": "agrag_text"},
            "kind": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "page_number": {"type": "integer"},
            "is_stale": {"type": "boolean"},
            "effective_date": {"type": "date"},
        }
    },
}


def index_name_for(tenant_id: str) -> str:
    """Index name for a tenant, sanitised for OpenSearch's naming rules.

    Example:
        >>> index_name_for("ten_ABC")
        'agrag-chunks-ten_abc'
    """
    safe = re.sub(r"[^a-z0-9_\-]", "-", tenant_id.lower())
    return f"{INDEX_PREFIX}-{safe}"


class SparseRetriever(Retriever):
    """BM25 search over the tenant's chunk index."""

    name = "opensearch-bm25"
    source = RetrievalSource.SPARSE

    def __init__(self, *, client: Any) -> None:
        """Create the retriever around an ``AsyncOpenSearch`` client."""
        self._client = client

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Run a multi-match BM25 query, folding in any expansions."""
        index = index_name_for(tenant_id)
        body = self._build_query(request, tenant_id=tenant_id)

        try:
            response = await self._client.search(
                index=index, body=body, size=min(request.top_k * 3, 200)
            )
        except Exception as exc:
            if _is_missing_index(exc):
                # A tenant that has never ingested anything is not an error.
                log.info("sparse index does not exist yet", index=index)
                return []
            log.error("sparse retrieval failed", index=index, reason=str(exc))
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

        return [self._to_chunk(hit) for hit in response.get("hits", {}).get("hits", [])]

    def _build_query(self, request: RetrievalRequest, *, tenant_id: str) -> dict[str, Any]:
        """Build the OpenSearch query body, including filters.

        Expansions become additional ``should`` clauses rather than one long
        concatenated string: concatenating dilutes every term's IDF and makes a
        five-variant expansion score worse than the original query alone.
        """
        should = [
            {
                "multi_match": {
                    "query": query,
                    "fields": list(FIELD_WEIGHTS),
                    "type": "best_fields",
                    "operator": "or",
                    # One typo should not empty the result set, but two usually
                    # means a different word.
                    "fuzziness": "AUTO",
                }
            }
            for query in request.all_queries
        ]

        # The tenant filter is redundant with the per-tenant index, and kept
        # deliberately: defence in depth against a future shared-index change.
        filters: list[dict[str, Any]] = [{"term": {"tenant_id": tenant_id}}]
        if not request.include_stale:
            filters.append({"term": {"is_stale": False}})
        if request.document_ids:
            filters.append({"terms": {"document_id": list(request.document_ids)}})
        if request.tags:
            filters.append({"terms": {"tags": list(request.tags)}})
        if request.kinds:
            filters.append({"terms": {"kind": [k.value for k in request.kinds]}})
        if request.date_from or request.date_to:
            date_range: dict[str, str] = {}
            if request.date_from:
                date_range["gte"] = request.date_from
            if request.date_to:
                date_range["lte"] = request.date_to
            filters.append({"range": {"effective_date": date_range}})

        query: dict[str, Any] = {
            "query": {"bool": {"should": should, "minimum_should_match": 1, "filter": filters}},
            "_source": [
                "chunk_id",
                "content",
                "document_id",
                "document_title",
                "kind",
                "ordinal",
                "page_number",
                "section_path",
                "effective_date",
            ],
        }

        if request.recency_boost > 0:
            query["query"] = _with_recency_boost(query["query"], request.recency_boost)
        return query

    @staticmethod
    def _to_chunk(hit: dict[str, Any]) -> RetrievedChunk:
        """Map an OpenSearch hit onto the shared retrieval contract."""
        from src.models.document import ChunkKind

        source = hit.get("_source", {})
        section = source.get("section_path") or ""
        return RetrievedChunk(
            chunk_id=source.get("chunk_id") or hit["_id"],
            content=source.get("content", ""),
            score=float(hit.get("_score") or 0.0),
            source=RetrievalSource.SPARSE,
            document_id=source.get("document_id"),
            document_title=source.get("document_title"),
            kind=ChunkKind(source["kind"]) if source.get("kind") else ChunkKind.PROSE,
            ordinal=source.get("ordinal"),
            page_number=source.get("page_number"),
            section_path=tuple(section.split(" > ")) if section else (),
            metadata={"effective_date": source.get("effective_date")},
        )

    async def ensure_index(self, tenant_id: str) -> None:
        """Create the tenant's index with the tuned mapping if it is missing."""
        index = index_name_for(tenant_id)
        if await self._client.indices.exists(index=index):
            return
        await self._client.indices.create(index=index, body=INDEX_SETTINGS)
        log.info("created sparse index", index=index)

    async def index_chunks(self, tenant_id: str, documents: Sequence[dict[str, Any]]) -> int:
        """Bulk-index chunk documents, returning the number indexed."""
        if not documents:
            return 0
        await self.ensure_index(tenant_id)
        index = index_name_for(tenant_id)

        actions: list[dict[str, Any]] = []
        for document in documents:
            actions.append({"index": {"_index": index, "_id": document["chunk_id"]}})
            actions.append({**document, "tenant_id": tenant_id})

        response = await self._client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            failures = [
                item["index"]["error"]
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            log.error("bulk index reported failures", count=len(failures), first=failures[:1])
            raise RetrievalError(f"{len(failures)} chunks failed to index")
        return len(documents)

    async def delete_tenant(self, tenant_id: str) -> None:
        """Drop a tenant's entire index. Used by the GDPR erasure cascade."""
        index = index_name_for(tenant_id)
        try:
            await self._client.indices.delete(index=index, ignore_unavailable=True)
        except Exception as exc:
            log.error("failed to delete sparse index", index=index, reason=str(exc))
            raise RetrievalError(f"could not delete index {index}: {exc}") from exc
        log.warning("deleted tenant sparse index", index=index)

    async def aclose(self) -> None:
        """Close the OpenSearch client."""
        await self._client.close()


def _with_recency_boost(query: dict[str, Any], strength: float) -> dict[str, Any]:
    """Wrap a query so recent documents score higher.

    Uses a Gaussian decay over ``effective_date`` rather than a hard filter,
    because a time-sensitive query usually wants *preferentially* recent results,
    not exclusively recent ones — filtering would hide the definitive older
    source entirely.

    Example:
        >>> boosted = _with_recency_boost({"match_all": {}}, 0.5)
        >>> boosted["function_score"]["boost_mode"]
        'multiply'
    """
    return {
        "function_score": {
            "query": query,
            "functions": [
                {
                    "gauss": {"effective_date": {"origin": "now", "scale": "365d", "decay": 0.5}},
                    "weight": max(strength, 0.01),
                }
            ],
            "score_mode": "sum",
            "boost_mode": "multiply",
        }
    }


def _is_missing_index(exc: Exception) -> bool:
    """Whether an OpenSearch error means "index not found".

    Example:
        >>> _is_missing_index(Exception("index_not_found_exception"))
        True
        >>> _is_missing_index(Exception("connection refused"))
        False
    """
    return "index_not_found" in str(exc).lower() or getattr(exc, "status_code", None) == 404
