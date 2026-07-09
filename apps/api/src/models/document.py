"""Document, version, chunk and ingestion-job models.

Documents are versioned: re-uploading a file creates a new
:class:`DocumentVersion` rather than mutating the old one, and chunks belonging
to superseded versions are marked stale instead of deleted so that citations in
historical conversations still resolve.

Example:
    >>> from src.models.document import Chunk
    >>> Chunk.__tablename__
    'chunks'
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, JSONColumn, SoftDeleteMixin, TenantScoped, TimestampMixin, new_id

#: Dimension of the default embedding model (BAAI/bge-large-en-v1.5).
#: Tenants on a different model store into `chunks_alt` — see
#: docs/adr/0005-mixed-embedding-dimensions.md.
EMBEDDING_DIM = 1024


class DocumentStatus(StrEnum):
    """Lifecycle of a document as it moves through the ingestion pipeline."""

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class SourceType(StrEnum):
    """Where a document came from."""

    UPLOAD = "upload"
    URL = "url"
    CONNECTOR = "connector"
    SEED = "seed"


class ChunkKind(StrEnum):
    """Structural role of a chunk, preserved by the layout-aware chunker.

    Kept as a first-class field because retrieval treats them differently: a
    ``TABLE`` chunk is never split further, and ``CODE`` chunks are excluded from
    the semantic-similarity dedupe pass.
    """

    PROSE = "prose"
    TABLE = "table"
    CODE = "code"
    HEADING = "heading"
    LIST = "list"
    FOOTNOTE = "footnote"
    CAPTION = "caption"


class Document(Base, TenantScoped, TimestampMixin, SoftDeleteMixin):
    """A logical document owned by a tenant, pointing at its current version."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_tenant_id_status", "tenant_id", "status"),
        Index("ix_documents_tenant_id_content_hash", "tenant_id", "content_hash"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("doc"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(500))
    mime_type: Mapped[str | None] = mapped_column(String(150))
    source_type: Mapped[SourceType] = mapped_column(
        String(20), default=SourceType.UPLOAD, nullable=False
    )
    source_uri: Mapped[str | None] = mapped_column(
        Text, comment="Original URL, or the s3:// key in MinIO."
    )

    status: Mapped[DocumentStatus] = mapped_column(
        String(20), default=DocumentStatus.QUEUED, nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    current_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(
        String(64), comment="SHA-256 of the raw bytes; used for exact dedupe."
    )
    byte_size: Mapped[int | None] = mapped_column(BigInteger)
    page_count: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String(16))

    tags: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)

    # Recency signal for temporal RAG; distinct from created_at because a 2019
    # policy PDF uploaded today should be treated as a 2019 document.
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    uploaded_by_user_id: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentVersion.version"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    @property
    def is_ready(self) -> bool:
        """True once the document is searchable.

        Example:
            >>> Document(title="x", status=DocumentStatus.INDEXED).is_ready
            True
        """
        return self.status == DocumentStatus.INDEXED


class DocumentVersion(Base, TenantScoped, TimestampMixin):
    """An immutable snapshot of a document's extracted content.

    Storing the extracted text (not just the blob) makes the version diff view
    cheap and lets us re-chunk without re-parsing when only the strategy changes.
    """

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_versions_document_id_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("dver"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    parser: Mapped[str | None] = mapped_column(String(64))
    parse_metadata: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_by_user_id: Mapped[str | None] = mapped_column(String(64))

    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class Chunk(Base, TenantScoped, TimestampMixin):
    """A retrievable unit of text with its dense embedding.

    ``content`` is what is shown to the user as a citation. ``embedded_text`` is
    what was actually vectorised: for Contextual Retrieval it carries the
    LLM-generated situating preamble, which we deliberately do not display.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_tenant_id_document_id", "tenant_id", "document_id"),
        Index("ix_chunks_tenant_id_is_stale", "tenant_id", "is_stale"),
        Index("ix_chunks_minhash_signature", "minhash_signature"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("chk"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("document_versions.id", ondelete="CASCADE")
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[ChunkKind] = mapped_column(String(20), default=ChunkKind.PROSE, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedded_text: Mapped[str | None] = mapped_column(
        Text, comment="Text actually embedded, including any contextual preamble."
    )
    context_preamble: Mapped[str | None] = mapped_column(
        Text, comment="LLM-generated situating context (Anthropic Contextual Retrieval)."
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(200))

    token_count: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(
        String(1000), comment="Breadcrumb of enclosing headings, e.g. '2 Method > 2.1 Setup'."
    )
    bbox: Mapped[dict[str, Any] | None] = mapped_column(
        JSONColumn, comment="Page-space bounding box, so the viewer can highlight the source."
    )

    minhash_signature: Mapped[str | None] = mapped_column(
        String(64), comment="LSH bucket key for near-duplicate detection."
    )
    is_stale: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True when a newer document version supersedes this chunk.",
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")
    # Declared so the unit of work knows a chunk depends on its version and
    # orders the inserts accordingly. With only the foreign key column,
    # SQLAlchemy flushes in mapper order and can emit the chunk first.
    version: Mapped[DocumentVersion | None] = relationship(back_populates="chunks")

    def citation_label(self) -> str:
        """Short human-readable source label used in citation UI.

        Example:
            >>> Chunk(ordinal=0, content="x", page_number=7).citation_label()
            'p. 7'
        """
        if self.page_number is not None:
            return f"p. {self.page_number}"
        if self.section_path:
            return self.section_path.split(" > ")[-1]
        return f"chunk {self.ordinal}"


class IngestionJob(Base, TenantScoped, TimestampMixin):
    """Tracks one document through the async Celery pipeline.

    Progress is mirrored to Redis pub/sub for the SSE progress stream; this table
    is the durable record used for retries, the dead-letter queue and the
    ingestion metrics on the admin dashboard.
    """

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_tenant_id_created_at", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("job"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    celery_task_id: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        String(20), default=DocumentStatus.QUEUED, nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(50))
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    dead_lettered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    @property
    def can_retry(self) -> bool:
        """True while the job still has retry budget left."""
        return self.attempts < self.max_attempts and self.dead_lettered_at is None
