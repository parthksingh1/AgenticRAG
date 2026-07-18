"""GraphRAG: retrieval over an extracted knowledge graph.

Vector search answers "which passage is similar to this question". It cannot
answer "which of our vendors are also customers" or "what changed between the
two policies that both mention indemnity", because those require *traversal*,
and similarity has no notion of a path.

So entities and relations are extracted at ingestion time into Neo4j, and at
query time the entities in the question are matched to graph nodes, the
neighbourhood is expanded a bounded number of hops, and the chunks attached to
those nodes are returned. The result is context that is topologically related to
the question rather than lexically similar to it — which is exactly the context
multi-hop questions need and dense retrieval never surfaces.

Cypher is never assembled from user text. Entity names are bound as parameters,
because a graph query built by string interpolation is an injection surface with
read access to every relationship in the tenant's graph.

Example:
    >>> GraphRetriever.name
    'graphrag'
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.errors import RetrievalError
from src.core.logging import get_logger
from src.retrieval.base import Retriever
from src.retrieval.types import RetrievalRequest, RetrievalSource, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.llm.router import LLMRouter

log = get_logger(__name__)

#: Hop ceiling. Past two hops on a dense graph the neighbourhood becomes "most of
#: the graph", which is both slow and useless as context.
MAX_HOPS = 2

_EXTRACT_SYSTEM = (
    "Extract the named entities from the question: people, organisations, products, "
    "systems, locations and defined terms. Output one entity per line, exactly as "
    "written in the question, with no numbering or commentary. If there are none, "
    "output nothing."
)

#: Parameterised throughout. `apoc.path` is avoided so the query works on a
#: stock Neo4j without plugins.
_TRAVERSE_CYPHER = """
MATCH (e:Entity)
WHERE e.tenant_id = $tenant_id AND toLower(e.name) IN $entity_names
CALL {
    WITH e
    MATCH path = (e)-[r*1..%(hops)d]-(related:Entity)
    WHERE ALL(rel IN r WHERE rel.tenant_id = $tenant_id)
      AND related.tenant_id = $tenant_id
    RETURN related, length(path) AS distance
    UNION
    WITH e
    RETURN e AS related, 0 AS distance
}
MATCH (related)-[:MENTIONED_IN]->(c:Chunk)
WHERE c.tenant_id = $tenant_id
RETURN DISTINCT
    c.chunk_id       AS chunk_id,
    c.content        AS content,
    c.document_id    AS document_id,
    c.document_title AS document_title,
    c.page_number    AS page_number,
    collect(DISTINCT related.name)[..5] AS via_entities,
    min(distance)    AS distance
ORDER BY distance ASC
LIMIT $limit
"""


class GraphEntity(BaseModel):
    """An entity node extracted from a document."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str = "Entity"
    description: str | None = None
    chunk_ids: tuple[str, ...] = ()


class GraphRelation(BaseModel):
    """A directed relation between two entities."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    #: Upper-snake-case, e.g. ``ACQUIRED``, ``REPORTS_TO``. Normalised on write so
    #: the graph does not accumulate a dozen spellings of the same relationship.
    relation: str
    description: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphRetriever(Retriever):
    """Retrieves chunks by traversing the tenant's knowledge graph."""

    name = "graphrag"
    source = RetrievalSource.GRAPH

    def __init__(
        self,
        *,
        driver: Any,
        router: LLMRouter | None = None,
        model: str | None = None,
        hops: int = 1,
    ) -> None:
        """Create the retriever.

        Args:
            driver: An async Neo4j driver.
            router: Used to extract entities from the query. Without it, entity
                extraction falls back to a capitalisation heuristic.
            model: Extraction model; the cheap model is sufficient.
            hops: Traversal depth, clamped to :data:`MAX_HOPS`.
        """
        self._driver = driver
        self._router = router
        self._model = model
        self._hops = min(max(hops, 1), MAX_HOPS)

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Find query entities, expand their neighbourhood, return attached chunks."""
        entities = await self.extract_entities(request.query)
        if not entities:
            log.debug("no entities in query; graph retrieval yields nothing")
            return []

        cypher = _TRAVERSE_CYPHER % {"hops": self._hops}
        params = {
            "tenant_id": tenant_id,
            "entity_names": [e.lower() for e in entities],
            "limit": min(request.top_k * 3, 100),
        }

        try:
            async with self._driver.session() as session:
                result = await session.run(cypher, params)
                records = [record.data() async for record in result]
        except Exception as exc:
            log.error("graph retrieval failed", reason=str(exc))
            raise RetrievalError(f"graph traversal failed: {exc}") from exc

        return [self._to_chunk(record) for record in records][: request.top_k]

    async def extract_entities(self, query: str) -> list[str]:
        """Extract entity mentions from a query.

        Falls back to a capitalisation heuristic when no router is configured or
        the call fails, because returning nothing would disable graph retrieval
        entirely on a transient error.
        """
        if self._router is None:
            return heuristic_entities(query)

        from src.services.llm.types import CompletionRequest, Message

        request = CompletionRequest(
            messages=(Message.system(_EXTRACT_SYSTEM), Message.user(query)),
            model=self._model or "",
            max_tokens=120,
            temperature=0.0,
            node="graph_entity_extraction",
        )
        try:
            completion = await self._router.complete(
                request.model_copy(update={"model": self._model or request.model}),
                allow_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the heuristic
            log.warning("entity extraction failed; using heuristic", reason=str(exc))
            return heuristic_entities(query)

        entities = [
            line.strip().strip('"-*• ') for line in completion.content.splitlines() if line.strip()
        ]
        return [e for e in entities if e][:10] or heuristic_entities(query)

    @staticmethod
    def _to_chunk(record: dict[str, Any]) -> RetrievedChunk:
        """Map a Cypher record onto the shared retrieval contract.

        Distance becomes the score, inverted: a directly-matched entity is more
        relevant than one reached in two hops.
        """
        distance = int(record.get("distance") or 0)
        return RetrievedChunk(
            chunk_id=record["chunk_id"],
            content=record.get("content", ""),
            score=1.0 / (1.0 + distance),
            source=RetrievalSource.GRAPH,
            document_id=record.get("document_id"),
            document_title=record.get("document_title"),
            page_number=record.get("page_number"),
            metadata={
                "via_entities": record.get("via_entities", []),
                "graph_distance": distance,
            },
        )

    async def upsert(
        self,
        *,
        tenant_id: str,
        entities: Sequence[GraphEntity],
        relations: Sequence[GraphRelation],
        chunk_lookup: dict[str, dict[str, Any]],
    ) -> None:
        """Write extracted entities and relations for one document.

        Everything is MERGEd on ``(tenant_id, name)`` so re-ingesting a document
        version updates the graph rather than duplicating it.
        """
        if not entities:
            return

        entity_rows = [
            {
                "name": e.name,
                "lower_name": e.name.lower(),
                "type": e.type,
                "description": e.description,
                "chunk_ids": list(e.chunk_ids),
            }
            for e in entities
        ]
        relation_rows = [
            {
                "source": r.source.lower(),
                "target": r.target.lower(),
                "relation": normalise_relation(r.relation),
                "description": r.description,
                "confidence": r.confidence,
            }
            for r in relations
        ]

        try:
            async with self._driver.session() as session:
                await session.run(
                    """
                    UNWIND $entities AS row
                    MERGE (e:Entity {tenant_id: $tenant_id, lower_name: row.lower_name})
                    SET e.name = row.name,
                        e.type = row.type,
                        e.description = coalesce(row.description, e.description)
                    WITH e, row
                    UNWIND row.chunk_ids AS chunk_id
                    MATCH (c:Chunk {tenant_id: $tenant_id, chunk_id: chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    """,
                    {"tenant_id": tenant_id, "entities": entity_rows},
                )
                if relation_rows:
                    await session.run(
                        """
                        UNWIND $relations AS row
                        MATCH (a:Entity {tenant_id: $tenant_id, lower_name: row.source})
                        MATCH (b:Entity {tenant_id: $tenant_id, lower_name: row.target})
                        MERGE (a)-[r:RELATED {type: row.relation, tenant_id: $tenant_id}]->(b)
                        SET r.description = row.description, r.confidence = row.confidence
                        """,
                        {"tenant_id": tenant_id, "relations": relation_rows},
                    )
                await session.run(
                    """
                    UNWIND $chunks AS row
                    MERGE (c:Chunk {tenant_id: $tenant_id, chunk_id: row.chunk_id})
                    SET c.content = row.content,
                        c.document_id = row.document_id,
                        c.document_title = row.document_title,
                        c.page_number = row.page_number
                    """,
                    {"tenant_id": tenant_id, "chunks": list(chunk_lookup.values())},
                )
        except Exception as exc:
            log.error("graph upsert failed", reason=str(exc))
            raise RetrievalError(f"graph write failed: {exc}") from exc

    async def delete_tenant(self, tenant_id: str) -> None:
        """Delete every node and relationship for a tenant. Used by GDPR erasure."""
        async with self._driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
                {"tenant_id": tenant_id},
            )
        log.warning("deleted tenant knowledge graph", tenant_id=tenant_id)

    async def aclose(self) -> None:
        """Close the Neo4j driver."""
        await self._driver.close()


def normalise_relation(relation: str) -> str:
    """Normalise a relation label to upper snake case.

    Without this the graph accumulates ``acquired``, ``Acquired`` and
    ``has acquired`` as three distinct relationship types, and traversal silently
    misses two of them.

    Example:
        >>> normalise_relation("has acquired")
        'HAS_ACQUIRED'
        >>> normalise_relation("reports-to")
        'REPORTS_TO'
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", relation.strip()).strip("_")
    return cleaned.upper() or "RELATED_TO"


def heuristic_entities(query: str) -> list[str]:
    """Extract probable entities by capitalisation, as an extraction fallback.

    Crude, and deliberately so: it exists to keep graph retrieval working when
    the extraction model is unavailable, not to replace it.

    Example:
        >>> heuristic_entities("Did ACME acquire Globex last year?")
        ['ACME', 'Globex']
        >>> heuristic_entities("what is our revenue?")
        []
    """
    words = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]{1,}\b", query)
    # Drop a leading sentence-initial capital, which is grammar rather than a name.
    if words and query.strip().startswith(words[0]):
        words = words[1:]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:10]
