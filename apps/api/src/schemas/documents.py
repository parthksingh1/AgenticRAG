"""Document, upload and search schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from src.models.document import ChunkKind, DocumentStatus, SourceType


class PresignedUpload(BaseModel):
    """A presigned URL the browser uploads to directly.

    Uploads bypass the API process entirely. Streaming a 200MB PDF through a
    FastAPI worker ties that worker up for the duration and puts the file in the
    API's memory; a presigned PUT goes straight to object storage, and the API
    only hears about it when the browser confirms.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    upload_url: str
    storage_key: str
    expires_in_seconds: int
    #: Headers the browser must send with the PUT, exactly as given.
    required_headers: dict[str, str] = Field(default_factory=dict)


class UploadRequest(BaseModel):
    """Ask for an upload slot."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1, max_length=500)
    content_type: str | None = None
    size_bytes: int = Field(ge=1)
    title: str | None = None
    tags: tuple[str, ...] = ()
    #: The document's own date, distinct from when it was uploaded: a 2019 policy
    #: uploaded today should be treated as a 2019 document by temporal retrieval.
    effective_date: datetime | None = None


class UrlIngestRequest(BaseModel):
    """Ingest a web page rather than an uploaded file."""

    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    title: str | None = None
    tags: tuple[str, ...] = ()


class DocumentOut(BaseModel):
    """A document as the library view renders it."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    filename: str | None = None
    mime_type: str | None = None
    source_type: SourceType
    status: DocumentStatus
    #: Present when status is FAILED; shown to the user so they can fix and retry.
    error_message: str | None = None
    created_at: datetime
    indexed_at: datetime | None = None
    effective_date: datetime | None = None
    byte_size: int | None = None
    page_count: int | None = None
    chunk_count: int = 0
    current_version: int = 1
    tags: tuple[str, ...] = ()
    language: str | None = None


class DocumentVersionOut(BaseModel):
    """One version of a document, for the version history and diff view."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: int
    created_at: datetime
    parser: str | None = None
    content_hash: str
    is_current: bool
    byte_size: int | None = None
    created_by_user_id: str | None = None


class IngestionProgress(BaseModel):
    """Live ingestion progress, pushed over SSE.

    Reported per document rather than per batch: a user watching ten uploads
    wants to know which one is stuck, and a single aggregate percentage hides
    exactly that.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    status: DocumentStatus
    stage: str | None = None
    progress: float = Field(ge=0.0, le=1.0)
    chunks_created: int = 0
    error: str | None = None
    attempt: int = 1


class ChunkOut(BaseModel):
    """A chunk, as shown in the citation sheet and the document inspector."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    ordinal: int
    kind: ChunkKind
    content: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    #: Normalised 0-1 box, so the PDF viewer can highlight the source region.
    bbox: dict[str, float] | None = None
    token_count: int | None = None
    is_stale: bool = False


class SearchRequest(BaseModel):
    """A direct search, used by the search UI and the docs-search MCP server."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=100)
    document_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    kinds: tuple[ChunkKind, ...] = ()
    date_from: str | None = None
    date_to: str | None = None
    include_stale: bool = False
    #: Overrides the workspace's configured strategies for this search only,
    #: which is how the playground compares strategies side by side.
    strategies: tuple[str, ...] = ()


class SearchHit(BaseModel):
    """One search result."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str | None = None
    document_title: str | None = None
    content: str
    score: float
    rerank_score: float | None = None
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    #: Which backends found this chunk and at what rank, so a surprising result
    #: can be explained rather than guessed at.
    contributing_ranks: dict[str, int] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """A search result set with the telemetry that produced it."""

    model_config = ConfigDict(frozen=True)

    results: tuple[SearchHit, ...] = ()
    strategy: str
    latency_ms: int = 0
    source_latencies_ms: dict[str, int] = Field(default_factory=dict)
    expanded: bool = False
    crag_verdict: str | None = None
    web_fallback_used: bool = False
    #: Populated on the list-only path used by the MCP server.
    documents: tuple[dict[str, Any], ...] = ()
