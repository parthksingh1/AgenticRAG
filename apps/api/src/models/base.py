"""Declarative base and shared column mixins.

The important type here is :class:`TenantScoped`. Any model that inherits it
gets a non-null ``tenant_id`` column *and* is automatically filtered by the
active tenant on every ORM query (see :mod:`src.core.db`). Isolation is
therefore a property of the mapping, not of each individual query — a developer
who forgets a ``WHERE tenant_id = ...`` clause still cannot read another
tenant's rows.

Example:
    >>> from src.models.base import TenantScoped, TimestampMixin
    >>> issubclass(TenantScoped, object)
    True
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, MetaData, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Postgres is the production database, but the unit-test suite runs on SQLite so
#: it needs no Docker. Declaring JSONB with a SQLite variant means one column
#: definition serves both without any test-only branching in the models.
JSONColumn = JSONB().with_variant(JSON(), "sqlite")

# Explicit naming convention so Alembic autogenerate produces stable, reviewable
# migration names instead of database-assigned ones.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def new_id(prefix: str) -> str:
    """Generate a sortable, human-readable, prefixed identifier.

    Prefixed ids make logs and support conversations unambiguous: you can tell a
    ``doc_`` from a ``chunk_`` at a glance, and a leaked id reveals its type.

    Example:
        >>> new_id("doc").startswith("doc_")
        True
        >>> len(new_id("doc"))
        30
    """
    return f"{prefix}_{uuid.uuid4().hex[:26]}"


class Base(DeclarativeBase):
    """Root declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy reads this as a plain dict
        dict[str, Any]: JSONColumn,
        list[str]: JSONColumn,
    }

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view of the mapped columns.

        Intended for logging and test assertions, not for API responses — those
        go through explicit Pydantic schemas.
        """
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self) -> str:
        """Render as ``<Model id=...>`` without dumping every column."""
        identifier = getattr(self, "id", None)
        return f"<{type(self).__name__} id={identifier!r}>"


class TimestampMixin:
    """Adds server-side ``created_at`` / ``updated_at`` columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Adds ``deleted_at`` for reversible deletes.

    GDPR erasure is a *hard* delete and does not use this — see
    :mod:`src.governance.gdpr`. Soft delete exists so a user who removes a
    document from the UI can be undone within the retention window.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    @property
    def is_deleted(self) -> bool:
        """True when the row has been soft-deleted."""
        return self.deleted_at is not None


class TenantScoped:
    """Marks a model as belonging to exactly one tenant.

    Inheriting this has two effects:

    1. A non-null, indexed ``tenant_id`` column is added.
    2. The session-level guard in :mod:`src.core.db` transparently appends
       ``WHERE tenant_id = :current_tenant`` to every SELECT against the model.

    Never query a ``TenantScoped`` model through a raw connection: that bypasses
    the guard. Use the ORM session from ``Depends(get_session)``.
    """

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Owning tenant. Enforced by the session-level tenant guard.",
    )


class VectorColumnMixin:
    """Shared helper for models carrying a pgvector embedding."""

    @staticmethod
    def hnsw_index_options() -> dict[str, Any]:
        """Return the HNSW index parameters used across the schema.

        ``m=16`` / ``ef_construction=200`` is the configuration benchmarked in
        docs/adr/0004-vector-index-parameters.md: it costs ~2x build time over
        the defaults and buys ~4 points of recall@10 on the golden set.
        """
        return {
            "postgresql_using": "hnsw",
            "postgresql_with": {"m": 16, "ef_construction": 200},
            "postgresql_ops": {"embedding": "vector_cosine_ops"},
        }


UTC_NOW = text("timezone('utc', now())")
