"""Async database engine, session factory and the tenant isolation guard.

The guard is the security-critical part of this module. It hooks SQLAlchemy's
``do_orm_execute`` event and appends ``tenant_id = :current_tenant`` to every
SELECT that touches a :class:`~src.models.base.TenantScoped` entity, using
:func:`sqlalchemy.orm.with_loader_criteria` so the predicate also reaches
eagerly-loaded relationships and subqueries.

Two escape hatches exist, both deliberately loud:

* :func:`system_session` — for migrations, backups and the eval harness, which
  legitimately span tenants. It logs at WARNING every time it is opened.
* ``session.execute(stmt, execution_options={"skip_tenant_guard": True})`` —
  per-statement, used only by :mod:`src.governance.gdpr` when cascading a
  tenant deletion.

Example:
    >>> from src.core.db import get_session
    >>> # in a router:  session: AsyncSession = Depends(get_session)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import ORMExecuteState, with_loader_criteria
from sqlalchemy.pool import NullPool

from src.core.config import Settings, get_settings
from src.core.context import current_tenant_id, require_tenant_id
from src.core.errors import TenantIsolationError
from src.core.logging import get_logger
from src.models.base import TenantScoped

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

#: Sessions and statements may opt out of the guard with this key, set either in
#: ``Session.info`` (whole session) or in per-statement execution options.
SKIP_TENANT_GUARD = "skip_tenant_guard"


def _build_engine(settings: Settings) -> AsyncEngine:
    """Create the async engine with pooling appropriate to the environment."""
    # Tests run against short-lived containers where pooling causes teardown
    # hangs; production wants a real pool.
    pool_kwargs: dict[str, Any] = (
        {"poolclass": NullPool}
        if settings.app_env == "ci"
        else {
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }
    )
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        future=True,
        **pool_kwargs,
    )


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = _build_engine(get_settings())
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


def _apply_tenant_criteria(state: ORMExecuteState) -> None:
    """Append the tenant predicate to a SELECT, or fail closed."""
    if not state.is_select or state.is_column_load or state.is_relationship_load:
        return
    if state.execution_options.get(SKIP_TENANT_GUARD, False):
        return
    if state.session.info.get(SKIP_TENANT_GUARD, False):
        return

    tenant_id = current_tenant_id()
    if tenant_id is None:
        # No tenant bound: this is either a system task that forgot to use
        # `system_session`, or a genuine leak. Fail closed.
        require_tenant_id()
        return

    state.statement = state.statement.options(
        with_loader_criteria(
            TenantScoped,
            lambda cls: cls.tenant_id == tenant_id,
            include_aliases=True,
            propagate_to_loaders=True,
        )
    )


@event.listens_for(AsyncSession.sync_session_class, "do_orm_execute")
def _tenant_guard_listener(state: ORMExecuteState) -> None:
    """Session-class-wide hook that enforces tenant scoping on ORM reads."""
    _apply_tenant_criteria(state)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Yield a tenant-scoped session, committing on success and rolling back on error.

    Example:
        >>> async with session_scope() as session:  # doctest: +SKIP
        ...     doc = await session.get(Document, "doc_123")
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped, tenant-filtered session."""
    async with session_scope() as session:
        yield session


@asynccontextmanager
async def system_session(*, reason: str) -> AsyncIterator[AsyncSession]:
    """Yield a session with the tenant guard disabled.

    Only for genuinely cross-tenant work: migrations, nightly backups, the eval
    harness and GDPR cascades. Every use is logged at WARNING with the stated
    reason so an audit can enumerate them.

    Args:
        reason: Why cross-tenant access is legitimate here. Appears in the log.

    Example:
        >>> async with system_session(reason="nightly backup") as s:  # doctest: +SKIP
        ...     ...
    """
    log.warning("opening cross-tenant system session", reason=reason)
    factory = get_session_factory()
    async with factory(info={SKIP_TENANT_GUARD: True}) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def assert_belongs_to_tenant(entity: object, *, resource: str) -> None:
    """Defence in depth: verify a loaded entity really belongs to the active tenant.

    The loader criteria should make this unreachable. It exists so that a future
    refactor which bypasses the ORM surfaces as a clear error rather than a
    silent leak, and it is exercised directly by the isolation test suite.

    Raises:
        TenantIsolationError: if the entity belongs to a different tenant.
    """
    entity_tenant = getattr(entity, "tenant_id", None)
    if entity_tenant is None:
        return
    active = require_tenant_id()
    if entity_tenant != active:
        log.error(
            "tenant isolation violation",
            active_tenant=active,
            entity_tenant=entity_tenant,
            resource=resource,
        )
        raise TenantIsolationError(tenant_id=active, resource=resource)


async def dispose_engine() -> None:
    """Close pooled connections. Called on application shutdown and in tests."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
