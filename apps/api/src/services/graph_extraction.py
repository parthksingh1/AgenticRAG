"""Knowledge-graph extraction.

Chunks answer "what does the corpus say about X". A graph answers "how is X
connected to Y", which retrieval over independent chunks cannot do at all: the
two hops of the answer live in different documents and neither one ranks for the
question.

Entities and relations are extracted with a structured LLM call, normalised, and
merged into Neo4j. Three decisions shape everything here.

**The model never writes Cypher.** It returns entities and relations as JSON; the
writes use fixed, parameterised query shapes. A model that emits query text is a
model that can be prompt-injected into emitting ``MATCH (n) DETACH DELETE n``,
and no amount of output validation makes that safe.

**Relation types come from a closed vocabulary.** Free-text predicates produce a
graph where ``WORKS_FOR``, ``works for``, ``employed_by`` and ``is employed by``
are four different edges, which is a graph nothing can query.

**Merges are idempotent.** Re-ingesting a document must not double every edge, so
nodes and relations are ``MERGE``d on a normalised key rather than created.

Example:
    >>> normalise_entity("  Acme   Corporation ")
    'acme corporation'
    >>> normalise_relation("works for")
    'WORKS_FOR'
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: The closed relation vocabulary. Anything the model returns is mapped into
#: this set or dropped; an open vocabulary makes the graph unqueryable.
RELATION_TYPES = frozenset(
    {
        "WORKS_FOR",
        "PART_OF",
        "LOCATED_IN",
        "FOUNDED_BY",
        "ACQUIRED",
        "COMPETES_WITH",
        "PARTNERED_WITH",
        "AUTHORED",
        "CITES",
        "USES",
        "PRODUCES",
        "REPORTS_TO",
        "MEMBER_OF",
        "RELATED_TO",
    }
)

#: Entity labels. Same reasoning as the relation vocabulary.
ENTITY_TYPES = frozenset(
    {"PERSON", "ORGANISATION", "PRODUCT", "LOCATION", "TECHNOLOGY", "EVENT", "CONCEPT", "METRIC"}
)

#: Relations below this confidence are dropped. An LLM will produce a relation
#: for any pair of entities that appear near each other; keeping the low-confidence
#: ones fills the graph with edges that are technically present in the text and
#: useless to reason over.
MIN_CONFIDENCE = 0.6

#: Characters of a chunk sent for extraction. Longer inputs raise both cost and
#: the rate at which the model invents relations spanning unrelated paragraphs.
MAX_EXTRACT_CHARS = 4000

#: Entities per chunk. A chunk that yields fifty entities has been misread as a
#: list of names rather than prose, and the result is noise either way.
MAX_ENTITIES_PER_CHUNK = 20

#: JSON Schema the provider is asked to conform to. Structured output is not
#: a substitute for validating the result — a provider that does not support it
#: silently ignores it — but where it is supported it removes most of the ways
#: an extraction comes back unusable.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(ENTITY_TYPES)},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(RELATION_TYPES)},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["source", "target", "type", "confidence"],
            },
        },
    },
    "required": ["entities", "relations"],
}

_WHITESPACE = re.compile(r"\s+")
_NON_RELATION = re.compile(r"[^A-Z_]")


@dataclass(frozen=True, slots=True)
class Entity:
    """One extracted entity."""

    name: str
    type: str
    normalised: str

    @classmethod
    def of(cls, name: str, type_: str) -> Entity | None:
        """Build an entity, or None when it is unusable.

        Example:
            >>> Entity.of("Acme", "ORGANISATION").normalised
            'acme'
            >>> Entity.of("", "PERSON") is None
            True
            >>> Entity.of("Acme", "VEGETABLE").type
            'CONCEPT'
        """
        cleaned = _WHITESPACE.sub(" ", (name or "")).strip()
        if not cleaned or len(cleaned) > 200:
            return None
        label = (type_ or "").strip().upper()
        return cls(
            name=cleaned,
            type=label if label in ENTITY_TYPES else "CONCEPT",
            normalised=normalise_entity(cleaned),
        )


@dataclass(frozen=True, slots=True)
class Relation:
    """One extracted relation between two entities."""

    source: str
    target: str
    type: str
    confidence: float
    evidence: str


@dataclass(frozen=True, slots=True)
class Extraction:
    """Everything pulled out of one chunk."""

    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]

    @property
    def is_empty(self) -> bool:
        """Whether nothing usable was found.

        Example:
            >>> Extraction((), ()).is_empty
            True
        """
        return not self.entities and not self.relations


def normalise_entity(name: str) -> str:
    """Normalise an entity name into a merge key.

    Casefolded and whitespace-collapsed, with a leading article dropped, so
    "The Acme Corporation" and "the  acme corporation" become one node instead of
    two. Deliberately conservative: aggressive normalisation merges genuinely
    different entities, and a wrongly merged node is much harder to notice than a
    duplicated one.

    Example:
        >>> normalise_entity("The Acme Corporation")
        'acme corporation'
        >>> normalise_entity("A1")
        'a1'
    """
    cleaned = _WHITESPACE.sub(" ", (name or "")).strip().casefold()
    for article in ("the ", "a ", "an "):
        if cleaned.startswith(article):
            return cleaned[len(article) :].strip()
    return cleaned


def normalise_relation(predicate: str) -> str | None:
    """Map a predicate onto the closed vocabulary, or None if it does not fit.

    Example:
        >>> normalise_relation("acquired")
        'ACQUIRED'
        >>> normalise_relation("is located in")
        'LOCATED_IN'
        >>> normalise_relation("smells like") is None
        True
    """
    raw = _WHITESPACE.sub("_", (predicate or "").strip().upper())
    candidate = _NON_RELATION.sub("", raw).strip("_")
    if candidate in RELATION_TYPES:
        return candidate

    # A model asked for WORKS_FOR often answers "is employed by". Rather than
    # dropping those, the obvious phrasings are folded in; anything else is
    # dropped, because guessing is how an open vocabulary sneaks back in.
    aliases = {
        "IS_LOCATED_IN": "LOCATED_IN",
        "IN": "LOCATED_IN",
        "EMPLOYED_BY": "WORKS_FOR",
        "IS_EMPLOYED_BY": "WORKS_FOR",
        "WORKS_AT": "WORKS_FOR",
        "SUBSIDIARY_OF": "PART_OF",
        "BELONGS_TO": "PART_OF",
        "IS_PART_OF": "PART_OF",
        "CREATED_BY": "FOUNDED_BY",
        "WROTE": "AUTHORED",
        "AUTHOR_OF": "AUTHORED",
        "BOUGHT": "ACQUIRED",
        "PARTNERS_WITH": "PARTNERED_WITH",
        "MANAGES": "REPORTS_TO",
    }
    return aliases.get(candidate)


def parse_extraction(payload: str | dict[str, Any]) -> Extraction:
    """Parse a model's extraction response into validated structures.

    Malformed output yields an empty extraction rather than raising: a model
    returning prose instead of JSON is a quality problem for one chunk, not a
    reason to fail an ingestion that is otherwise fine.

    Example:
        >>> e = parse_extraction('{"entities": [{"name": "Acme", "type": "ORGANISATION"}]}')
        >>> e.entities[0].name
        'Acme'
        >>> parse_extraction("not json").is_empty
        True
    """
    if isinstance(payload, str):
        try:
            data = json.loads(_strip_fences(payload))
        except (ValueError, TypeError):
            log.debug("graph extraction returned unparseable output")
            return Extraction((), ())
    else:
        data = payload

    if not isinstance(data, dict):
        return Extraction((), ())

    entities: list[Entity] = []
    seen: set[str] = set()
    for raw in data.get("entities") or ():
        if not isinstance(raw, dict):
            continue
        entity = Entity.of(str(raw.get("name", "")), str(raw.get("type", "")))
        if entity is None or entity.normalised in seen:
            continue
        seen.add(entity.normalised)
        entities.append(entity)
        if len(entities) >= MAX_ENTITIES_PER_CHUNK:
            break

    relations: list[Relation] = []
    for raw in data.get("relations") or ():
        if not isinstance(raw, dict):
            continue
        relation_type = normalise_relation(str(raw.get("type", "")))
        source = normalise_entity(str(raw.get("source", "")))
        target = normalise_entity(str(raw.get("target", "")))
        confidence = _confidence(raw.get("confidence"))

        # Both endpoints must be entities we actually extracted: a relation to a
        # name that never appeared is the model inventing a node.
        if (
            relation_type is None
            or not source
            or not target
            or source == target
            or source not in seen
            or target not in seen
            or confidence < MIN_CONFIDENCE
        ):
            continue

        relations.append(
            Relation(
                source=source,
                target=target,
                type=relation_type,
                confidence=confidence,
                evidence=str(raw.get("evidence", ""))[:500],
            )
        )

    return Extraction(tuple(entities), tuple(relations))


async def extract_from_chunk(
    text: str, *, router: Any, model: str, prompts: Any = None
) -> Extraction:
    """Extract entities and relations from one chunk.

    Uses the cheap model: extraction is a structured-output task run over every
    chunk of every document, and the frontier model's marginal accuracy does not
    survive contact with that bill.
    """
    if not text.strip():
        return Extraction((), ())

    from src.services.llm.types import CompletionRequest, Message

    instruction = (
        prompts.render("graph_extraction", text=text[:MAX_EXTRACT_CHARS])
        if prompts is not None
        else _fallback_prompt(text[:MAX_EXTRACT_CHARS])
    )

    try:
        completion = await router.complete(
            CompletionRequest(
                messages=(Message(role="user", content=instruction),),
                model=model,
                temperature=0.0,
                max_tokens=1500,
                response_schema=EXTRACTION_SCHEMA,
                node="graph_extraction",
            )
        )
    except Exception as exc:  # noqa: BLE001 - extraction is optional enrichment
        log.warning("graph extraction call failed", reason=str(exc))
        return Extraction((), ())

    return parse_extraction(completion.content)


async def merge_into_graph(
    driver: Any,
    extraction: Extraction,
    *,
    tenant_id: str,
    document_id: str,
    chunk_id: str,
) -> dict[str, int]:
    """Merge an extraction into Neo4j.

    Every node carries its ``tenant_id`` and every query filters on it, which is
    how tenant isolation reaches a store that has no row-level security of its
    own. Relations record which chunk they came from, so a graph answer can cite
    its source rather than asserting a fact from nowhere.
    """
    if extraction.is_empty:
        return {"entities": 0, "relations": 0}

    async with driver.session() as session:
        await session.run(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {tenant_id: $tenant_id, key: entity.key})
            ON CREATE SET e.name = entity.name, e.type = entity.type, e.created_at = timestamp()
            SET e.last_seen_at = timestamp()
            MERGE (d:Document {tenant_id: $tenant_id, id: $document_id})
            MERGE (e)-[m:MENTIONED_IN]->(d)
            SET m.chunk_id = $chunk_id
            """,
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=chunk_id,
            entities=[
                {"key": e.normalised, "name": e.name, "type": e.type} for e in extraction.entities
            ],
        )

        for relation in extraction.relations:
            # The relation type is interpolated because Cypher cannot bind a
            # relationship type as a parameter. It is safe only because the value
            # is a member of RELATION_TYPES, which the model cannot extend —
            # normalise_relation returns None for anything else.
            if relation.type not in RELATION_TYPES:  # pragma: no cover - defensive
                continue
            await session.run(
                f"""
                MATCH (a:Entity {{tenant_id: $tenant_id, key: $source}})
                MATCH (b:Entity {{tenant_id: $tenant_id, key: $target}})
                MERGE (a)-[r:{relation.type}]->(b)
                ON CREATE SET r.created_at = timestamp()
                SET r.confidence = $confidence,
                    r.evidence = $evidence,
                    r.chunk_id = $chunk_id,
                    r.document_id = $document_id
                """,
                tenant_id=tenant_id,
                source=relation.source,
                target=relation.target,
                confidence=relation.confidence,
                evidence=relation.evidence,
                chunk_id=chunk_id,
                document_id=document_id,
            )

    return {"entities": len(extraction.entities), "relations": len(extraction.relations)}


async def extract_document(
    chunks: Sequence[Any],
    *,
    driver: Any,
    router: Any,
    model: str,
    tenant_id: str,
    document_id: str,
) -> dict[str, int]:
    """Extract and merge a whole document's graph.

    Chunks are processed sequentially rather than concurrently: extraction runs
    against the same provider as the user-facing traffic, and fanning out over a
    500-chunk document would consume the rate limit that interactive answers need.
    """
    totals = {"entities": 0, "relations": 0, "chunks": 0}
    for chunk in chunks:
        extraction = await extract_from_chunk(chunk.content, router=router, model=model)
        merged = await merge_into_graph(
            driver,
            extraction,
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=chunk.id,
        )
        totals["entities"] += merged["entities"]
        totals["relations"] += merged["relations"]
        totals["chunks"] += 1

    log.info("extracted a document graph", document_id=document_id, **totals)
    return totals


def _confidence(raw: Any) -> float:
    """Coerce a confidence into [0, 1], defaulting low.

    A missing confidence defaults below the threshold rather than above it: a
    model that did not say how sure it is has not earned the benefit of the
    doubt.

    Example:
        >>> _confidence(0.9), _confidence("high"), _confidence(None)
        (0.9, 0.0, 0.0)
        >>> _confidence(5)
        1.0
    """
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def _strip_fences(text: str) -> str:
    """Remove markdown code fences a model wrapped its JSON in.

    Example:
        >>> fenced = chr(10).join(["```json", '{"a": 1}', "```"])
        >>> _strip_fences(fenced)
        '{"a": 1}'
        >>> _strip_fences('{"a": 1}')
        '{"a": 1}'
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1]
    return body.rsplit("```", 1)[0].strip()


def _fallback_prompt(text: str) -> str:
    """The extraction instruction used when the prompt registry is unavailable.

    Kept in code so extraction still works in a worker that could not load the
    prompt directory, rather than silently producing nothing.
    """
    return (
        "Extract entities and relations from the text below.\n\n"
        f"Entity types: {', '.join(sorted(ENTITY_TYPES))}.\n"
        f"Relation types: {', '.join(sorted(RELATION_TYPES))}.\n\n"
        "Return JSON only, shaped as:\n"
        '{"entities": [{"name": str, "type": str}], '
        '"relations": [{"source": str, "target": str, "type": str, '
        '"confidence": float, "evidence": str}]}\n\n'
        "Use only the listed types. Use the entity names exactly as they appear "
        "in your entities list for relation endpoints. Give confidence as your "
        "certainty the text states the relation, not that it is true in general. "
        "Extract nothing if the text states no relations.\n\n"
        f"Text:\n{text}"
    )
