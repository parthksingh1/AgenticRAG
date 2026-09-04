"""The ingestion pipeline.

Runs parse → chunk → enrich → embed → index for one document, reporting progress
as it goes. It is the only place that writes chunks, which is what lets the
retrieval side assume every chunk has an embedding and a document.

Three properties matter more than the sequence itself:

**Idempotent.** Re-running a job on the same version is safe: chunks are keyed by
document version and replaced wholesale, so a Celery retry after a partial
failure produces the same result as a clean run rather than a duplicated index.

**Versioned, not overwritten.** Re-uploading a file adds a version and marks the
previous version's chunks stale. Old citations still resolve, which is why a
conversation from last month does not turn into a wall of dead links.

**Progress is published as it happens.** Ingestion of a large PDF takes minutes;
a UI that shows nothing until it finishes looks broken, so each stage publishes
to Redis and the SSE endpoint relays it.

Example:
    >>> from src.services.ingestion import Stage
    >>> Stage.EMBEDDING.value
    'embedding'
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.core.errors import IngestionError
from src.core.logging import get_logger
from src.ingestion.chunkers.base import get_chunker
from src.ingestion.parsers.base import parse_bytes
from src.ingestion.types import ChunkDraft, ChunkingConfig, ParsedDocument
from src.models.document import Chunk, Document, DocumentStatus, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.ingestion.embedders.base import Embedder

log = get_logger(__name__)

#: Chunks embedded per batch. The embedder sizes its own batches by text length;
#: this bounds how many drafts are held in memory at once.
EMBED_BATCH = 128

#: Near-duplicate threshold for MinHash LSH. High, because collapsing two
#: genuinely different chunks loses content, while keeping a near-duplicate only
#: costs an embedding.
DEDUPE_THRESHOLD = 0.92


class Stage(StrEnum):
    """Pipeline stages, reported as progress."""

    PARSING = "parsing"
    CHUNKING = "chunking"
    ENRICHING = "enriching"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    GRAPH = "graph"
    DONE = "done"


#: Fraction complete at the end of each stage. Rough, and honest about being
#: rough: a progress bar that jumps from 30% to 100% is worse than one that
#: moves steadily even if the intervals are uneven.
STAGE_PROGRESS: dict[Stage, float] = {
    Stage.PARSING: 0.15,
    Stage.CHUNKING: 0.30,
    Stage.ENRICHING: 0.50,
    Stage.EMBEDDING: 0.80,
    Stage.INDEXING: 0.95,
    Stage.GRAPH: 0.99,
    Stage.DONE: 1.0,
}


@dataclass(slots=True)
class IngestionResult:
    """What one ingestion produced."""

    document_id: str
    version: int
    chunks_created: int = 0
    chunks_deduplicated: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    parser: str | None = None
    page_count: int | None = None
    language: str | None = None
    entities_extracted: int = 0
    warnings: list[str] = field(default_factory=list)


class ProgressReporter:
    """Publishes stage progress to Redis for the SSE stream.

    Failures are swallowed: a document that ingests successfully but whose
    progress bar stalled is a cosmetic problem, and failing the ingestion over it
    would be a real one.
    """

    def __init__(self, redis: Any = None, *, channel_prefix: str = "ingestion") -> None:
        """Create a reporter, optionally without a Redis client."""
        self._redis = redis
        self._prefix = channel_prefix

    async def publish(
        self,
        *,
        tenant_id: str,
        document_id: str,
        stage: Stage,
        chunks: int = 0,
        error: str | None = None,
    ) -> None:
        """Publish one progress update."""
        if self._redis is None:
            return

        import json

        payload = json.dumps(
            {
                "document_id": document_id,
                "stage": stage.value,
                "progress": STAGE_PROGRESS.get(stage, 0.0),
                "chunks_created": chunks,
                "error": error,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        try:
            await self._redis.publish(f"{self._prefix}:{tenant_id}", payload)
        except Exception as exc:  # noqa: BLE001 - progress is cosmetic
            log.debug("could not publish ingestion progress", reason=str(exc))


class IngestionPipeline:
    """Turns a stored document into indexed, embedded chunks."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        storage: Any,
        sparse: Any = None,
        graph: Any = None,
        graph_router: Any = None,
        graph_model: str = "",
        enricher: Any = None,
        progress: ProgressReporter | None = None,
        max_bytes: int = 200 * 1024 * 1024,
    ) -> None:
        """Wire the pipeline to its collaborators.

        Graph extraction needs both a Neo4j driver (``graph``) and an LLM
        router (``graph_router``): extraction is a model call, not a database
        write. Passing one without the other leaves the feature gated off
        rather than half-configured — see ``_extract_graph``.
        """
        self._embedder = embedder
        self._storage = storage
        self._sparse = sparse
        self._graph = graph
        self._graph_router = graph_router
        self._graph_model = graph_model
        self._enricher = enricher
        self._progress = progress or ProgressReporter()
        self._max_bytes = max_bytes

    async def run(
        self, session: AsyncSession, *, document_id: str, tenant_id: str
    ) -> IngestionResult:
        """Ingest one document end to end.

        Raises:
            IngestionError: when the document cannot be read or parsed. The
                caller records the failure on the job and decides whether to
                retry, so this raises rather than returning a partial result.
        """
        started = time.perf_counter()
        document = await self._load(session, document_id=document_id)
        version = await self._current_version(session, document=document)

        result = IngestionResult(document_id=document_id, version=version.version)

        await self._report(tenant_id, document_id, Stage.PARSING)
        parsed = await self._parse(document=document, version=version)
        result.parser = parsed.parser
        result.page_count = parsed.page_count
        result.language = parsed.language

        version.extracted_text = parsed.full_text[:2_000_000]
        version.parser = parsed.parser
        version.parse_metadata = dict(parsed.metadata)

        await self._report(tenant_id, document_id, Stage.CHUNKING)
        drafts = self._chunk(parsed, document=document)
        if not drafts:
            # An empty document is not an error — a scanned page with no OCR
            # output is a real thing — but it must not silently look ingested.
            result.warnings.append("no chunks were produced; the document appears to be empty")
            await self._finalise(session, document=document, result=result, started=started)
            return result

        drafts, duplicates = deduplicate_chunks(drafts)
        result.chunks_deduplicated = duplicates

        if self._enricher is not None and document.tenant_id and _contextual_enabled(document):
            await self._report(tenant_id, document_id, Stage.ENRICHING)
            drafts = await self._enricher.enrich(
                drafts, document_text=parsed.full_text, document_title=document.title
            )

        await self._report(tenant_id, document_id, Stage.EMBEDDING, chunks=len(drafts))
        embeddings = await self._embed(drafts)

        await self._report(tenant_id, document_id, Stage.INDEXING, chunks=len(drafts))
        chunks = await self._write_chunks(
            session,
            document=document,
            version=version,
            drafts=drafts,
            embeddings=embeddings,
        )
        result.chunks_created = len(chunks)
        result.embedding_tokens = sum(c.token_count or 0 for c in chunks)

        await self._index_sparse(document=document, chunks=chunks)

        if (
            self._graph is not None
            and self._graph_router is not None
            and document.tenant_id
            and _graph_enabled(document)
        ):
            await self._report(tenant_id, document_id, Stage.GRAPH)
            result.entities_extracted = await self._extract_graph(document=document, chunks=chunks)

        await self._finalise(session, document=document, result=result, started=started)
        await self._report(tenant_id, document_id, Stage.DONE, chunks=result.chunks_created)
        return result

    # ── stages ───────────────────────────────────────────────────────────────

    async def _load(self, session: AsyncSession, *, document_id: str) -> Document:
        """Load the document row."""
        from sqlalchemy import select

        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document is None:
            msg = f"document {document_id} not found"
            raise IngestionError(msg)
        return document

    async def _current_version(
        self, session: AsyncSession, *, document: Document
    ) -> DocumentVersion:
        """Load the version being ingested."""
        from sqlalchemy import select

        result = await session.execute(
            select(DocumentVersion)
            .where(
                DocumentVersion.document_id == document.id,
                DocumentVersion.version == document.current_version,
            )
            .limit(1)
        )
        version = result.scalar_one_or_none()
        if version is None:
            msg = f"document {document.id} has no version {document.current_version}"
            raise IngestionError(msg)
        return version

    async def _parse(self, *, document: Document, version: DocumentVersion) -> ParsedDocument:
        """Fetch the bytes and parse them."""
        import asyncio

        if self._storage is None:
            msg = "object storage is not configured"
            raise IngestionError(msg)

        data = await asyncio.to_thread(
            self._storage.download, version.storage_key, max_bytes=self._max_bytes
        )
        digest = hashlib.sha256(data).hexdigest()
        if version.content_hash and version.content_hash != digest:
            log.warning(
                "stored content hash does not match the object; using the object",
                document_id=document.id,
            )
        version.content_hash = digest
        document.content_hash = digest
        document.byte_size = len(data)

        return await asyncio.to_thread(parse_bytes, data, filename=document.filename)

    def _chunk(self, parsed: ParsedDocument, *, document: Document) -> list[ChunkDraft]:
        """Split the parsed document using the tenant's strategy.

        Falls back to the default strategy when the configured one is not
        registered — which happens when a tenant selects ``semantic`` but no
        embedder was bound. Failing the upload instead would be a worse trade.
        """
        strategy = str(document.doc_metadata.get("chunking_strategy") or "layout_aware")
        try:
            chunker = get_chunker(strategy)
        except KeyError:
            log.warning(
                "chunking strategy unavailable; falling back to layout_aware",
                requested=strategy,
            )
            chunker = get_chunker("layout_aware")
        return chunker.chunk(parsed, ChunkingConfig())

    async def _embed(self, drafts: list[ChunkDraft]) -> list[list[float]]:
        """Embed every chunk, in batches."""
        vectors: list[list[float]] = []
        for start in range(0, len(drafts), EMBED_BATCH):
            window = drafts[start : start + EMBED_BATCH]
            vectors.extend(await self._embedder.embed([d.embedded_text for d in window]))
        return vectors

    async def _write_chunks(
        self,
        session: AsyncSession,
        *,
        document: Document,
        version: DocumentVersion,
        drafts: list[ChunkDraft],
        embeddings: list[list[float]],
    ) -> list[Chunk]:
        """Replace this version's chunks, marking older versions stale."""
        from sqlalchemy import delete, update

        # Replace rather than append: a retried job must not double the index.
        await session.execute(delete(Chunk).where(Chunk.document_version_id == version.id))
        # Older versions stay resolvable for historical citations, but are hidden
        # from new retrieval.
        await session.execute(
            update(Chunk)
            .where(Chunk.document_id == document.id, Chunk.document_version_id != version.id)
            .values(is_stale=True)
        )

        chunks: list[Chunk] = []
        for draft, vector in zip(drafts, embeddings, strict=True):
            chunk = Chunk(
                tenant_id=document.tenant_id,
                document_id=document.id,
                document_version_id=version.id,
                ordinal=draft.ordinal,
                kind=draft.kind,
                content=draft.content,
                embedded_text=draft.embedded_text,
                context_preamble=draft.context_preamble,
                embedding=vector,
                embedding_model=self._embedder.model_name,
                token_count=max(len(draft.embedded_text) // 4, 1),
                char_start=draft.char_start,
                char_end=draft.char_end,
                page_number=draft.page_number,
                section_path=" > ".join(draft.section_path) or None,
                bbox=draft.bbox.model_dump() if draft.bbox else None,
                minhash_signature=minhash_signature(draft.content),
                chunk_metadata=dict(draft.metadata),
            )
            session.add(chunk)
            chunks.append(chunk)

        await session.flush()
        return chunks

    async def _index_sparse(self, *, document: Document, chunks: list[Chunk]) -> None:
        """Mirror the chunks into OpenSearch for BM25.

        A sparse indexing failure degrades retrieval to dense-only rather than
        failing the ingestion: the document is still searchable, just less well,
        and a reindex can repair it later.
        """
        if self._sparse is None or not chunks:
            return

        documents = [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "document_title": document.title,
                "section_path": chunk.section_path or "",
                "kind": chunk.kind.value,
                "tags": list(document.tags or []),
                "ordinal": chunk.ordinal,
                "page_number": chunk.page_number,
                "is_stale": False,
                "effective_date": (
                    document.effective_date.isoformat() if document.effective_date else None
                ),
            }
            for chunk in chunks
        ]
        try:
            await self._sparse.index_chunks(document.tenant_id, documents)
        except Exception as exc:  # noqa: BLE001 - degrade to dense-only
            log.error(
                "sparse indexing failed; the document is searchable but dense-only",
                document_id=document.id,
                reason=str(exc),
            )

    async def _extract_graph(self, *, document: Document, chunks: list[Chunk]) -> int:
        """Extract entities and relations into the knowledge graph.

        Best effort for the same reason as sparse indexing: graph retrieval is an
        enhancement, and losing it must not cost the document.
        """
        try:
            from src.services.graph_extraction import extract_document

            totals = await extract_document(
                chunks,
                driver=self._graph,
                router=self._graph_router,
                model=self._graph_model,
                tenant_id=document.tenant_id,
                document_id=document.id,
            )
            return totals["entities"]
        except Exception as exc:  # noqa: BLE001 - graph is an enhancement
            log.warning("graph extraction failed", document_id=document.id, reason=str(exc))
            return 0

    async def _finalise(
        self,
        session: AsyncSession,
        *,
        document: Document,
        result: IngestionResult,
        started: float,
    ) -> None:
        """Mark the document indexed and record its statistics."""
        from src.observability.metrics import record_ingestion

        result.duration_seconds = round(time.perf_counter() - started, 2)
        document.status = DocumentStatus.INDEXED
        document.indexed_at = datetime.now(UTC)
        document.page_count = result.page_count or document.page_count
        document.language = result.language or document.language
        document.error_message = None

        record_ingestion(
            status=DocumentStatus.INDEXED.value,
            chunks=result.chunks_created,
            duration_seconds=result.duration_seconds,
        )
        log.info(
            "document ingested",
            document_id=document.id,
            chunks=result.chunks_created,
            deduplicated=result.chunks_deduplicated,
            seconds=result.duration_seconds,
        )

    async def _report(
        self, tenant_id: str, document_id: str, stage: Stage, *, chunks: int = 0
    ) -> None:
        """Publish a progress update."""
        await self._progress.publish(
            tenant_id=tenant_id, document_id=document_id, stage=stage, chunks=chunks
        )


# ── Deduplication ────────────────────────────────────────────────────────────


def minhash_signature(text: str, *, shingle_size: int = 5, permutations: int = 64) -> str:
    """A compact signature for near-duplicate detection.

    Uses datasketch when available and falls back to a shingle hash otherwise.
    The fallback catches exact and near-exact duplicates but not paraphrases,
    which is the common case — boilerplate repeated verbatim across documents.

    Example:
        >>> a = minhash_signature("the quick brown fox jumps over the lazy dog")
        >>> b = minhash_signature("the quick brown fox jumps over the lazy dog")
        >>> a == b
        True
    """
    normalised = " ".join(text.lower().split())
    if not normalised:
        return ""

    try:
        from datasketch import MinHash

        minhash = MinHash(num_perm=permutations)
        for index in range(max(len(normalised) - shingle_size + 1, 1)):
            minhash.update(normalised[index : index + shingle_size].encode())
        return hashlib.sha256(minhash.digest().tobytes()).hexdigest()[:32]
    except ImportError:
        return hashlib.sha256(normalised.encode()).hexdigest()[:32]


def deduplicate_chunks(drafts: list[ChunkDraft]) -> tuple[list[ChunkDraft], int]:
    """Drop chunks that duplicate an earlier one.

    Ordinals are renumbered afterwards, because the citation binder resolves
    markers by position and a gap would misattribute a citation.

    Example:
        >>> from src.ingestion.types import ChunkDraft
        >>> drafts = [
        ...     ChunkDraft(content="same text here", ordinal=0),
        ...     ChunkDraft(content="same text here", ordinal=1),
        ...     ChunkDraft(content="different text", ordinal=2),
        ... ]
        >>> kept, removed = deduplicate_chunks(drafts)
        >>> len(kept), removed, [c.ordinal for c in kept]
        (2, 1, [0, 1])
    """
    seen: set[str] = set()
    kept: list[ChunkDraft] = []
    removed = 0

    for draft in drafts:
        signature = minhash_signature(draft.content)
        if signature and signature in seen:
            removed += 1
            continue
        seen.add(signature)
        kept.append(draft)

    return [d.model_copy(update={"ordinal": i}) for i, d in enumerate(kept)], removed


def _contextual_enabled(document: Document) -> bool:
    """Whether contextual retrieval is on for this document's tenant."""
    return bool(document.doc_metadata.get("contextual_retrieval", True))


def _graph_enabled(document: Document) -> bool:
    """Whether graph extraction is on for this document's tenant."""
    return bool(document.doc_metadata.get("graph_extraction", False))
