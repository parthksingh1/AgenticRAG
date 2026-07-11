"""Document, version and chunk persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.models.document import (
    Chunk,
    Document,
    DocumentStatus,
    DocumentVersion,
    IngestionJob,
    SourceType,
)
from src.schemas.documents import ChunkOut, DocumentOut, DocumentVersionOut

log = get_logger(__name__)


async def create_document(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str | None,
    filename: str,
    title: str,
    mime_type: str | None,
    size_bytes: int,
    tags: tuple[str, ...] = (),
    effective_date: datetime | None = None,
) -> Document:
    """Create a document row awaiting its bytes."""
    document = Document(
        tenant_id=tenant_id,
        title=title,
        filename=filename,
        mime_type=mime_type,
        byte_size=size_bytes,
        source_type=SourceType.UPLOAD,
        status=DocumentStatus.QUEUED,
        tags=list(tags),
        effective_date=effective_date,
        uploaded_by_user_id=user_id,
    )
    session.add(document)
    await session.flush()
    return document


async def create_url_document(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str | None,
    url: str,
    title: str | None,
    tags: tuple[str, ...] = (),
) -> DocumentOut:
    """Create a document that will be fetched from a URL."""
    document = Document(
        tenant_id=tenant_id,
        title=title or url,
        source_type=SourceType.URL,
        source_uri=url,
        status=DocumentStatus.QUEUED,
        tags=list(tags),
        uploaded_by_user_id=user_id,
    )
    session.add(document)
    await session.flush()
    return _to_out(document, chunk_count=0)


async def attach_version(
    session: AsyncSession, *, document_id: str, storage_key: str
) -> DocumentVersion:
    """Create the next version row for a document.

    The version number comes from the document's counter rather than a
    ``max(version) + 1`` query, so two concurrent uploads cannot both compute the
    same next number.
    """
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one()

    existing = await session.execute(
        select(func.count(DocumentVersion.id)).where(DocumentVersion.document_id == document_id)
    )
    count = int(existing.scalar_one() or 0)
    version_number = count + 1
    document.current_version = version_number

    await session.execute(
        update(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .values(is_current=False)
    )

    version = DocumentVersion(
        tenant_id=document.tenant_id,
        document_id=document_id,
        version=version_number,
        storage_key=storage_key,
        content_hash="",
        is_current=True,
    )
    session.add(version)
    await session.flush()
    return version


async def get_document(session: AsyncSession, *, document_id: str) -> Any | None:
    """Load a document with its current storage key.

    Returns a lightweight object rather than the ORM row, because the caller
    uses it after the session closes and a lazy attribute would raise there.
    """
    from types import SimpleNamespace

    result = await session.execute(
        select(Document, DocumentVersion.storage_key)
        .join(
            DocumentVersion,
            (DocumentVersion.document_id == Document.id)
            & (DocumentVersion.version == Document.current_version),
            isouter=True,
        )
        .where(Document.id == document_id, Document.deleted_at.is_(None))
    )
    row = result.first()
    if row is None:
        return None

    document, storage_key = row
    return SimpleNamespace(
        id=document.id,
        title=document.title,
        storage_key=storage_key or "",
        status=document.status,
        tenant_id=document.tenant_id,
    )


async def get_document_out(session: AsyncSession, *, document_id: str) -> DocumentOut | None:
    """Load a document in its API shape."""
    chunk_count = (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == Document.id, Chunk.is_stale.is_(False))
        .correlate(Document)
        .scalar_subquery()
    )
    result = await session.execute(
        select(Document, chunk_count.label("chunk_count")).where(
            Document.id == document_id, Document.deleted_at.is_(None)
        )
    )
    row = result.first()
    return _to_out(row[0], chunk_count=int(row[1] or 0)) if row else None


async def list_documents(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    tag: str | None = None,
) -> list[DocumentOut]:
    """List documents with their live chunk counts."""
    chunk_count = (
        select(func.count(Chunk.id))
        .where(Chunk.document_id == Document.id, Chunk.is_stale.is_(False))
        .correlate(Document)
        .scalar_subquery()
    )
    statement = (
        select(Document, chunk_count.label("chunk_count"))
        .where(Document.deleted_at.is_(None))
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status:
        statement = statement.where(Document.status == status)
    if tag:
        statement = statement.where(Document.tags.contains([tag]))

    result = await session.execute(statement)
    return [_to_out(document, chunk_count=int(count or 0)) for document, count in result.all()]


async def list_chunks(
    session: AsyncSession, *, document_id: str, limit: int = 100, offset: int = 0
) -> list[ChunkOut]:
    """List a document's live chunks in reading order."""
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.is_stale.is_(False))
        .order_by(Chunk.ordinal)
        .limit(limit)
        .offset(offset)
    )
    return [
        ChunkOut(
            id=chunk.id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            kind=chunk.kind,
            content=chunk.content,
            page_number=chunk.page_number,
            section_path=tuple((chunk.section_path or "").split(" > "))
            if chunk.section_path
            else (),
            bbox=chunk.bbox,
            token_count=chunk.token_count,
            is_stale=chunk.is_stale,
        )
        for chunk in result.scalars().all()
    ]


async def list_versions(session: AsyncSession, *, document_id: str) -> list[DocumentVersionOut]:
    """List a document's versions, newest first."""
    result = await session.execute(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version.desc())
    )
    return [
        DocumentVersionOut(
            id=version.id,
            version=version.version,
            created_at=version.created_at,
            parser=version.parser,
            content_hash=version.content_hash,
            is_current=version.is_current,
            created_by_user_id=version.created_by_user_id,
        )
        for version in result.scalars().all()
    ]


async def mark_queued(session: AsyncSession, *, document_id: str) -> DocumentOut:
    """Mark a document queued and open an ingestion job for it."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one()
    document.status = DocumentStatus.QUEUED
    document.error_message = None

    session.add(
        IngestionJob(
            tenant_id=document.tenant_id,
            document_id=document_id,
            status=DocumentStatus.QUEUED,
            started_at=datetime.now(UTC),
        )
    )
    await session.flush()
    return _to_out(document, chunk_count=0)


async def soft_delete_document(session: AsyncSession, *, document_id: str) -> bool:
    """Soft-delete a document and hide its chunks from retrieval."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        return False

    document.deleted_at = datetime.now(UTC)
    # Marked stale, not deleted: existing citations must still resolve.
    await session.execute(
        update(Chunk).where(Chunk.document_id == document_id).values(is_stale=True)
    )
    return True


def _to_out(document: Document, *, chunk_count: int) -> DocumentOut:
    """Map an ORM document onto its API shape."""
    return DocumentOut(
        id=document.id,
        title=document.title,
        filename=document.filename,
        mime_type=document.mime_type,
        source_type=document.source_type,
        status=document.status,
        error_message=document.error_message,
        created_at=document.created_at,
        indexed_at=document.indexed_at,
        effective_date=document.effective_date,
        byte_size=document.byte_size,
        page_count=document.page_count,
        chunk_count=chunk_count,
        current_version=document.current_version,
        tags=tuple(document.tags or ()),
        language=document.language,
    )
