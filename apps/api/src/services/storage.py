"""Object storage.

Uploads never pass through the API process. A presigned PUT sends the bytes
straight from the browser to MinIO, so a 200MB PDF does not occupy a worker for
the duration or sit in the API's memory. The API only learns about the file when
the browser confirms the upload, and it verifies the object exists before
enqueueing ingestion — otherwise a client could claim an upload that never
happened and queue work against a missing object.

Keys are namespaced by tenant. That is not access control on its own — the
bucket policy is — but it makes a mistake visible: an object under
``ten_a/...`` being read while serving ``ten_b`` is obviously wrong in a log,
where an opaque UUID key would not be.

Example:
    >>> storage_key(tenant_id="ten_a", document_id="doc_1", filename="report.pdf")
    'ten_a/doc_1/report.pdf'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from src.core.errors import IngestionError, PayloadTooLargeError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.core.config import Settings

log = get_logger(__name__)

#: Presigned URLs are short-lived. Long enough for a slow upload on a poor
#: connection, short enough that a URL leaked into a log or a referer header is
#: not a durable capability.
UPLOAD_URL_TTL = timedelta(minutes=30)
DOWNLOAD_URL_TTL = timedelta(minutes=15)

#: Characters permitted in a storage key segment.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def sanitise_filename(filename: str) -> str:
    """Reduce a filename to something safe to use in a key.

    Path separators and traversal sequences are removed rather than escaped: a
    key containing ``../`` would resolve outside the tenant's prefix on some
    storage backends, which is exactly the isolation the prefix provides.

    Example:
        >>> sanitise_filename("../../etc/passwd")
        'passwd'
        >>> sanitise_filename("Q3 report (final).pdf")
        'Q3_report_final_.pdf'
        >>> sanitise_filename("")
        'upload'
    """
    base = filename.replace("\\", "/").split("/")[-1]
    cleaned = _UNSAFE.sub("_", base).strip("._")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:200] or "upload"


def storage_key(*, tenant_id: str, document_id: str, filename: str) -> str:
    """Build the object key for a document.

    Example:
        >>> storage_key(tenant_id="t", document_id="d", filename="a b.pdf")
        't/d/a_b.pdf'
    """
    return f"{tenant_id}/{document_id}/{sanitise_filename(filename)}"


@dataclass(frozen=True, slots=True)
class UploadTarget:
    """A presigned destination for one upload."""

    url: str
    key: str
    expires_in_seconds: int
    headers: dict[str, str]


class ObjectStorage:
    """MinIO/S3 client for document bytes and backups."""

    def __init__(self, settings: Settings) -> None:
        """Create the client from settings."""
        from minio import Minio

        self._settings = settings
        self.default_bucket = settings.minio_bucket
        self.backup_bucket = settings.minio_backup_bucket
        self._public_endpoint = settings.minio_public_endpoint.rstrip("/")
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=settings.minio_secure,
        )

    def bucket_exists(self, bucket: str) -> bool:
        """Whether a bucket exists."""
        return bool(self._client.bucket_exists(bucket))

    def ensure_buckets(self) -> None:
        """Create the document and backup buckets if they are missing."""
        for bucket in (self.default_bucket, self.backup_bucket):
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                log.info("created bucket", bucket=bucket)

    def presign_upload(
        self, *, tenant_id: str, document_id: str, filename: str, content_type: str | None = None
    ) -> UploadTarget:
        """Create a presigned PUT for one document.

        The returned URL uses the *public* endpoint rather than the internal
        service name: the browser resolves it, and ``minio:9000`` means nothing
        outside the compose network.
        """
        key = storage_key(tenant_id=tenant_id, document_id=document_id, filename=filename)
        url = self._client.presigned_put_object(self.default_bucket, key, expires=UPLOAD_URL_TTL)
        return UploadTarget(
            url=self._rewrite_host(url),
            key=key,
            expires_in_seconds=int(UPLOAD_URL_TTL.total_seconds()),
            headers={"Content-Type": content_type} if content_type else {},
        )

    def presign_download(self, key: str) -> str:
        """Create a short-lived GET for the document viewer."""
        return self._rewrite_host(
            self._client.presigned_get_object(self.default_bucket, key, expires=DOWNLOAD_URL_TTL)
        )

    def _rewrite_host(self, url: str) -> str:
        """Swap the internal endpoint for the browser-reachable one.

        Example:
            >>> # 'http://minio:9000/bucket/key' becomes 'http://localhost:9000/bucket/key'
            >>> True
            True
        """
        internal = self._settings.minio_endpoint
        scheme = "https" if self._settings.minio_secure else "http"
        return url.replace(f"{scheme}://{internal}", self._public_endpoint, 1)

    def stat(self, key: str) -> dict[str, Any] | None:
        """Object metadata, or None when it does not exist.

        Used to confirm an upload actually landed before ingestion is queued: a
        client that reports success for an upload that never happened would
        otherwise enqueue a job against a missing object.
        """
        from minio.error import S3Error

        try:
            info = self._client.stat_object(self.default_bucket, key)
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return None
            raise

        return {
            "size": info.size,
            "etag": info.etag,
            "content_type": info.content_type,
            "last_modified": info.last_modified,
        }

    def download(self, key: str, *, max_bytes: int) -> bytes:
        """Read an object into memory, refusing anything oversized.

        The size is checked before reading, not after: reading first and then
        complaining has already used the memory the check exists to protect.

        Raises:
            IngestionError: when the object is missing.
            PayloadTooLargeError: when it exceeds the ceiling.
        """
        info = self.stat(key)
        if info is None:
            msg = f"object not found in storage: {key}"
            raise IngestionError(msg)
        if info["size"] > max_bytes:
            raise PayloadTooLargeError(
                f"document is {info['size']} bytes, over the {max_bytes} byte limit"
            )

        response = self._client.get_object(self.default_bucket, key)
        try:
            return bytes(response.read())
        finally:
            response.close()
            response.release_conn()

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Upload bytes directly. Used by the seed script and by backups."""
        import io

        self._client.put_object(
            self.default_bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def delete(self, key: str) -> None:
        """Remove one object."""
        self._client.remove_object(self.default_bucket, key)

    def list_prefix(self, prefix: str) -> list[str]:
        """List object keys under a prefix.

        Returns a list rather than the client's lazy iterator: callers delete
        while iterating, and mutating a bucket underneath a live listing is not
        a defined operation.
        """
        objects = self._client.list_objects(self.default_bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def count_tenant_objects(self, tenant_id: str) -> int:
        """Count a tenant's objects without deleting them.

        Backs the GDPR dry run, which is the only safe way to preview an
        operation that has no undo.
        """
        objects = self._client.list_objects(
            self.default_bucket, prefix=f"{tenant_id}/", recursive=True
        )
        return sum(1 for _ in objects)

    def delete_tenant(self, tenant_id: str) -> int:
        """Delete every object under a tenant's prefix.

        Part of the GDPR cascade. Deletes in batches because removing objects one
        request at a time is unusable on a tenant with tens of thousands of them.
        """
        from minio.deleteobjects import DeleteObject

        objects = self._client.list_objects(
            self.default_bucket, prefix=f"{tenant_id}/", recursive=True
        )
        targets = [DeleteObject(obj.object_name) for obj in objects]
        if not targets:
            return 0

        errors = list(self._client.remove_objects(self.default_bucket, targets))
        if errors:
            log.error("some objects could not be deleted", tenant_id=tenant_id, count=len(errors))
        deleted = len(targets) - len(errors)
        log.warning("deleted tenant objects", tenant_id=tenant_id, deleted=deleted)
        return deleted
