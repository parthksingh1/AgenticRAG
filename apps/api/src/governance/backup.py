"""Nightly backups.

A backup that has never been restored is a hypothesis, not a backup. So this
module does three things rather than one: it dumps, it records what it dumped,
and it **verifies the dump is readable** before reporting success. A truncated
``pg_dump`` — a disk that filled halfway through — exits zero often enough that
"the job succeeded" is not evidence of anything.

Backups are written to object storage under a date-partitioned prefix and pruned
past the retention window. They are not encrypted here: MinIO and every managed
equivalent encrypt at rest, and rolling our own key management on top of that
would add a way to lose the data without adding protection.

Example:
    >>> backup_key(datetime(2026, 3, 4, 3, 0, tzinfo=UTC), "postgres")
    'backups/2026/03/04/postgres-20260304T030000Z.sql.gz'
"""

from __future__ import annotations

import gzip
import os
import shutil

# pg_dump is invoked below with a fixed argument vector and shell=False.
import subprocess  # nosec B404
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: How long backups are kept. Long enough to notice corruption that only shows
#: up weeks later, short enough that storage stays bounded.
RETENTION_DAYS = 30

#: A dump smaller than this is treated as failed. An empty or near-empty dump
#: from a healthy database means the dump broke, not that the data vanished.
MIN_PLAUSIBLE_BYTES = 4096

DUMP_TIMEOUT_SECONDS = 3600


def backup_key(when: datetime, component: str) -> str:
    """Object-storage key for one backup.

    Date-partitioned so listing a day's backups does not scan the whole prefix,
    and so lifecycle rules can expire by path.

    Example:
        >>> backup_key(datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC), "neo4j")
        'backups/2026/12/31/neo4j-20261231T235959Z.sql.gz'
    """
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return f"backups/{when:%Y/%m/%d}/{component}-{stamp}.sql.gz"


async def run_backup() -> dict[str, Any]:
    """Dump Postgres, verify it, upload it and prune old backups.

    Returns:
        A report naming the key written and the bytes uploaded. A failure is
        returned rather than raised so the beat schedule records it and the
        alerting rule fires on the metric instead of on a stack trace.
    """
    from src.core.config import get_settings
    from src.observability.metrics import record_backup

    settings = get_settings()
    when = datetime.now(UTC)
    report: dict[str, Any] = {"started_at": when.isoformat(), "components": {}}

    try:
        path = _dump_postgres(settings.database_url_sync, when=when)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        log.error("postgres backup failed", error=str(exc))
        record_backup(component="postgres", success=False)
        report["components"]["postgres"] = {"ok": False, "error": str(exc)[:500]}
        return report

    try:
        size = path.stat().st_size
        _verify(path)
        key = backup_key(when, "postgres")
        _upload(settings, key=key, path=path)
        report["components"]["postgres"] = {"ok": True, "key": key, "bytes": size}
        record_backup(component="postgres", success=True, size_bytes=size)
        log.info("backup uploaded", key=key, bytes=size)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        log.error("backup verification or upload failed", error=str(exc))
        record_backup(component="postgres", success=False)
        report["components"]["postgres"] = {"ok": False, "error": str(exc)[:500]}
    finally:
        path.unlink(missing_ok=True)

    report["pruned"] = _prune(settings, before=when - timedelta(days=RETENTION_DAYS))
    report["finished_at"] = datetime.now(UTC).isoformat()
    return report


def _dump_postgres(database_url: str, *, when: datetime) -> Path:
    """Run ``pg_dump`` into a gzipped temporary file.

    The password is passed through the environment, never on the command line:
    an argv is visible to every process on the host via ``ps``.

    Raises:
        RuntimeError: when ``pg_dump`` is missing or exits non-zero.
    """
    executable = shutil.which("pg_dump")
    if executable is None:
        msg = "pg_dump is not installed in this image"
        raise RuntimeError(msg)

    target = Path(tempfile.gettempdir()) / f"agrag-{when:%Y%m%dT%H%M%SZ}.sql.gz"
    environment = {**os.environ, "PGCONNECT_TIMEOUT": "30"}

    # A fixed argument vector with shell=False: nothing here is interpolated
    # from user input, and there is no shell to inject into.
    argv = [executable, "--no-owner", "--no-privileges", "--format=plain", database_url]
    with gzip.open(target, "wb") as sink:
        # Fixed argv, shell=False: nothing here is interpolated from user input.
        process = subprocess.run(  # noqa: S603 # nosec B603
            argv,
            capture_output=True,
            env=environment,
            timeout=DUMP_TIMEOUT_SECONDS,
            check=False,
        )
        sink.write(process.stdout)

    if process.returncode != 0:
        target.unlink(missing_ok=True)
        detail = process.stderr.decode(errors="replace").strip()[:500]
        msg = f"pg_dump exited {process.returncode}: {detail}"
        raise RuntimeError(msg)

    return target


def _verify(path: Path) -> None:
    """Check the dump is a readable gzip stream of plausible size.

    Decompressing the whole file is the point: a truncated gzip only fails at
    the end, so a check that reads the header alone would pass on exactly the
    corruption worth catching.

    Raises:
        RuntimeError: when the dump is implausibly small or will not decompress.
    """
    size = path.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        msg = f"the dump is only {size} bytes, which is too small to be a real backup"
        raise RuntimeError(msg)

    decompressed = 0
    try:
        with gzip.open(path, "rb") as source:
            while chunk := source.read(1 << 20):
                decompressed += len(chunk)
    except OSError as exc:
        msg = f"the dump does not decompress cleanly: {exc}"
        raise RuntimeError(msg) from exc

    if decompressed == 0:
        msg = "the dump decompressed to nothing"
        raise RuntimeError(msg)


def _upload(settings: Any, *, key: str, path: Path) -> None:
    """Upload a verified dump to object storage."""
    from src.services.storage import ObjectStorage

    storage = ObjectStorage(settings)
    storage.put(key, path.read_bytes(), content_type="application/gzip")


def _prune(settings: Any, *, before: datetime) -> int:
    """Delete backups older than the retention window.

    Returns:
        How many objects were removed.
    """
    from src.services.storage import ObjectStorage

    storage = ObjectStorage(settings)
    removed = 0
    for key in storage.list_prefix("backups/"):
        stamp = _stamp_of(key)
        if stamp is not None and stamp < before:
            storage.delete(key)
            removed += 1

    if removed:
        log.info("pruned expired backups", removed=removed, before=before.isoformat())
    return removed


def _stamp_of(key: str) -> datetime | None:
    """Parse the timestamp out of a backup key.

    Returns None for anything that does not match, so an unexpected object under
    the prefix is left alone rather than deleted on a guess.

    Example:
        >>> _stamp_of("backups/2026/03/04/postgres-20260304T030000Z.sql.gz").hour
        3
        >>> _stamp_of("backups/readme.txt") is None
        True
    """
    stem = key.rsplit("-", 1)[-1]
    stamp = stem.removesuffix(".sql.gz")
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
