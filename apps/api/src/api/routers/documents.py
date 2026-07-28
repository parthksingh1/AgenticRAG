"""Document upload, listing, versions and ingestion progress.

Upload is a three-step handshake rather than a single POST:

1. The client asks for a slot. The API creates the document row and returns a
   presigned PUT.
2. The client uploads straight to object storage.
3. The client confirms. The API verifies the object actually exists and only
   then enqueues ingestion.

The verification in step 3 is the reason for the handshake. Without it a client
could confirm an upload that never happened, and the worker would spend its
retries failing to fetch a missing object before dead-lettering a document the
user believes is being processed.

Example:
    >>> from src.api.routers.documents import router
    >>> router.prefix
    '/api/documents'
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse

from src.api.auth import Principal, get_principal
from src.api.dependencies import AppServices, get_services
from src.core.errors import ConflictError, IngestionError, NotFoundError, PayloadTooLargeError
from src.core.logging import get_logger
from src.models.tenant import ApiKeyScope
from src.schemas.documents import (
    ChunkOut,
    DocumentOut,
    DocumentVersionOut,
    PresignedUpload,
    UploadRequest,
    UrlIngestRequest,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=PresignedUpload, status_code=201)
async def request_upload(
    request: UploadRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> PresignedUpload:
    """Create a document and return a presigned URL to upload its bytes to.

    Raises:
        PayloadTooLargeError: when the declared size exceeds the limit. Checked
            here rather than after the upload, so the user is told before
            spending their bandwidth.
    """
    principal.require(ApiKeyScope.WRITE)

    if request.size_bytes > services.settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"{request.filename} is {request.size_bytes} bytes; the limit is "
            f"{services.settings.max_upload_bytes}"
        )
    if services.storage is None:
        raise IngestionError("Object storage is not configured on this deployment.")

    from src.core.db import session_scope
    from src.repositories.documents import create_document

    async with session_scope() as session:
        document = await create_document(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            filename=request.filename,
            title=request.title or request.filename,
            mime_type=request.content_type,
            size_bytes=request.size_bytes,
            tags=request.tags,
            effective_date=request.effective_date,
        )
        document_id = document.id

    target = await asyncio.to_thread(
        services.storage.presign_upload,
        tenant_id=principal.tenant_id,
        document_id=document_id,
        filename=request.filename,
        content_type=request.content_type,
    )

    from src.core.db import session_scope as scope

    async with scope() as session:
        from src.repositories.documents import attach_version

        await attach_version(session, document_id=document_id, storage_key=target.key)

    return PresignedUpload(
        document_id=document_id,
        upload_url=target.url,
        storage_key=target.key,
        expires_in_seconds=target.expires_in_seconds,
        required_headers=target.headers,
    )


@router.post("/{document_id}/confirm", response_model=DocumentOut)
async def confirm_upload(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> DocumentOut:
    """Confirm the bytes arrived and queue ingestion.

    Raises:
        NotFoundError: when the document does not exist for this workspace.
        ConflictError: when the object is not in storage. The client believes it
            uploaded; queueing anyway would burn the worker's retries and end in
            a dead letter the user cannot explain.
    """
    principal.require(ApiKeyScope.WRITE)

    from src.core.db import session_scope
    from src.repositories.documents import get_document, mark_queued

    async with session_scope() as session:
        document = await get_document(session, document_id=document_id)
        if document is None:
            raise NotFoundError("Document not found.")
        storage_key = document.storage_key

    if services.storage is not None:
        info = await asyncio.to_thread(services.storage.stat, storage_key)
        if info is None:
            raise ConflictError(
                "The upload has not arrived in storage yet. Retry the PUT, then confirm again."
            )

    async with session_scope() as session:
        updated = await mark_queued(session, document_id=document_id)

    from src.workers.celery_app import ingest_document

    ingest_document.delay(document_id=document_id, tenant_id=principal.tenant_id)
    log.info("queued ingestion", document_id=document_id)
    return updated


@router.post("/url", response_model=DocumentOut, status_code=201)
async def ingest_url(
    request: UrlIngestRequest,
    principal: Annotated[Principal, Depends(get_principal)],
) -> DocumentOut:
    """Ingest a web page.

    The fetch happens in the worker, not here: a slow page would otherwise hold
    an API connection open, and the same SSRF checks the web-fetch MCP server
    applies are needed either way.
    """
    principal.require(ApiKeyScope.WRITE)

    from src.core.db import session_scope
    from src.repositories.documents import create_url_document

    async with session_scope() as session:
        document = await create_url_document(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            url=str(request.url),
            title=request.title,
            tags=request.tags,
        )
        document_id = document.id
        response = document

    from src.workers.celery_app import ingest_document

    ingest_document.delay(document_id=document_id, tenant_id=principal.tenant_id)
    return response


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    principal: Annotated[Principal, Depends(get_principal)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
) -> list[DocumentOut]:
    """List the workspace's documents."""
    from src.core.db import session_scope
    from src.repositories.documents import list_documents as query

    async with session_scope() as session:
        return await query(session, limit=limit, offset=offset, status=status, tag=tag)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document_detail(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> DocumentOut:
    """Load one document."""
    from src.core.db import session_scope
    from src.repositories.documents import get_document_out

    async with session_scope() as session:
        document = await get_document_out(session, document_id=document_id)

    if document is None:
        raise NotFoundError("Document not found.")
    return document


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
async def list_chunks(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ChunkOut]:
    """List a document's chunks, for the inspector and the citation sheet."""
    from src.core.db import session_scope
    from src.repositories.documents import list_chunks as query

    async with session_scope() as session:
        return await query(session, document_id=document_id, limit=limit, offset=offset)


@router.get("/{document_id}/versions", response_model=list[DocumentVersionOut])
async def list_versions(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> list[DocumentVersionOut]:
    """List a document's versions, newest first."""
    from src.core.db import session_scope
    from src.repositories.documents import list_versions as query

    async with session_scope() as session:
        return await query(session, document_id=document_id)


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> dict[str, str]:
    """Return a short-lived URL for the document viewer.

    A redirect rather than a proxy: streaming the PDF through the API to render
    a citation would put every page view through a worker for no benefit.
    """
    from src.core.db import session_scope
    from src.repositories.documents import get_document

    async with session_scope() as session:
        document = await get_document(session, document_id=document_id)

    if document is None:
        raise NotFoundError("Document not found.")
    if services.storage is None:
        raise IngestionError("Object storage is not configured on this deployment.")

    url = await asyncio.to_thread(services.storage.presign_download, document.storage_key)
    return {"url": url}


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Soft-delete a document and hide its chunks from retrieval.

    The chunks are marked stale rather than deleted so citations in existing
    conversations still resolve. Permanent removal is the GDPR path.
    """
    principal.require(ApiKeyScope.WRITE)

    from src.core.db import session_scope
    from src.repositories.documents import soft_delete_document

    async with session_scope() as session:
        deleted = await soft_delete_document(session, document_id=document_id)

    if not deleted:
        raise NotFoundError("Document not found.")


@router.get("/progress/stream")
async def stream_progress(
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> EventSourceResponse:
    """Stream ingestion progress for the workspace.

    One subscription for the whole workspace rather than one per document: a
    user dropping thirty files would otherwise open thirty SSE connections, and
    browsers cap concurrent connections per origin well below that.
    """

    async def events() -> AsyncIterator[str]:
        """Relay Redis pub/sub messages as SSE frames."""
        import json

        if services.redis is None:
            yield f"event: error\ndata: {json.dumps({'error': 'progress unavailable'})}\n\n"
            return

        pubsub = services.redis.pubsub()
        channel = f"ingestion:{principal.tenant_id}"
        await pubsub.subscribe(channel)
        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if message is None:
                    # A comment frame keeps proxies from closing an idle
                    # connection, which would otherwise look like a stalled
                    # upload to the user.
                    yield ": keepalive\n\n"
                    continue
                data = message["data"]
                payload = data.decode() if isinstance(data, bytes) else str(data)
                yield f"event: progress\ndata: {payload}\n\n"
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return EventSourceResponse(
        events(), headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )
