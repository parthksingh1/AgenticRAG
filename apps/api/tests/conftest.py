"""Shared pytest fixtures.

Unit tests run against an in-memory SQLite database so the suite stays fast and
needs no Docker. Anything that genuinely depends on Postgres semantics
(pgvector, JSONB operators, HNSW) lives in ``tests/integration`` and uses
testcontainers instead.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncEngine

os.environ.setdefault("APP_ENV", "ci")
os.environ.setdefault("AUTH_DEV_MODE", "true")
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-000000000")

from src.core import db as db_module
from src.core.config import get_settings
from src.models.base import Base
from src.models.conversation import Conversation, Message
from src.models.document import Document
from src.models.tenant import ApiKey, Tenant, User

#: Tables safe to create on SQLite. `chunks` carries a pgvector column and
#: `document_versions` depends on it transitively, so those are exercised in the
#: integration suite against real Postgres.
SQLITE_SAFE_MODELS = (Tenant, User, ApiKey, Document, Conversation, Message)


@pytest.fixture(scope="session", autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Ensure the settings singleton reflects the test environment."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """An in-memory SQLite engine with the SQLite-safe tables created."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    tables: list[Table] = [model.__table__ for model in SQLITE_SAFE_MODELS]
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=tables)
    try:
        yield test_engine
    finally:
        await test_engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Point ``src.core.db`` at the test engine so the tenant guard is exercised.

    The guard is registered on ``AsyncSession.sync_session_class`` at import
    time, so simply binding a session factory to the test engine gives real
    coverage of production behaviour rather than a stubbed approximation.
    """
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_module, "_engine", engine, raising=False)
    monkeypatch.setattr(db_module, "_session_factory", factory, raising=False)
    yield factory


@pytest_asyncio.fixture
async def seeded_tenants(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[str, str]:
    """Create two tenants, each owning one document and one conversation.

    Returns:
        The ids of tenant A and tenant B.
    """
    tenant_a, tenant_b = "ten_aaaa", "ten_bbbb"
    # INSERTs are not filtered by the guard, so seeding needs no escape hatch;
    # the session is opened tenant-free on purpose to prove that.
    async with session_factory() as session:
        session.add_all(
            [
                Tenant(id=tenant_a, name="Acme", slug="acme"),
                Tenant(id=tenant_b, name="Globex", slug="globex"),
                Document(id="doc_a", tenant_id=tenant_a, title="Acme handbook"),
                Document(id="doc_b", tenant_id=tenant_b, title="Globex handbook"),
                Conversation(id="cnv_a", tenant_id=tenant_a, title="Acme chat"),
                Conversation(id="cnv_b", tenant_id=tenant_b, title="Globex chat"),
            ]
        )
        await session.commit()
    return tenant_a, tenant_b
