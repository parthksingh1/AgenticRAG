"""Tenant isolation tests.

This is the highest-stakes test file in the repository. The premise is that a
developer *will* eventually write a query that forgets its tenant filter, so
isolation must be enforced by the session, not by discipline. Each test below
issues a deliberately unscoped or actively hostile query and asserts it still
cannot reach another tenant's rows.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from src.core import db as db_module
from src.core.context import request_context
from src.core.db import assert_belongs_to_tenant
from src.core.errors import AuthenticationError, TenantIsolationError
from src.models.conversation import Conversation
from src.models.document import Document

pytestmark = pytest.mark.unit


async def test_unscoped_select_returns_only_active_tenant_rows(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """A SELECT with no WHERE clause still sees only the active tenant."""
    tenant_a, _ = seeded_tenants
    with request_context(tenant_id=tenant_a):
        async with session_factory() as session:
            docs = (await session.execute(select(Document))).scalars().all()

    assert [d.id for d in docs] == ["doc_a"]


async def test_switching_tenant_switches_visible_rows(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """The same query returns disjoint results under two different tenants."""
    tenant_a, tenant_b = seeded_tenants

    async with session_factory() as session:
        with request_context(tenant_id=tenant_a):
            a_docs = (await session.execute(select(Document))).scalars().all()
        session.expunge_all()
        with request_context(tenant_id=tenant_b):
            b_docs = (await session.execute(select(Document))).scalars().all()

    assert {d.id for d in a_docs}.isdisjoint({d.id for d in b_docs})
    assert {d.id for d in b_docs} == {"doc_b"}


async def test_direct_get_by_foreign_id_returns_nothing(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """Guessing another tenant's primary key does not leak the row.

    This is the realistic attack: an id harvested from a shared link or a log.
    """
    tenant_a, _ = seeded_tenants
    with request_context(tenant_id=tenant_a):
        async with session_factory() as session:
            stolen = (
                await session.execute(select(Document).where(Document.id == "doc_b"))
            ).scalar_one_or_none()

    assert stolen is None


async def test_explicit_wrong_tenant_filter_cannot_widen_access(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """An explicit foreign ``tenant_id`` predicate ANDs with the guard, yielding nothing.

    A compromised or buggy call site cannot opt itself into another tenant.
    """
    tenant_a, tenant_b = seeded_tenants
    with request_context(tenant_id=tenant_a):
        async with session_factory() as session:
            result = await session.execute(select(Document).where(Document.tenant_id == tenant_b))
            rows = result.scalars().all()

    assert rows == []


@pytest.mark.parametrize("model", [Document, Conversation])
async def test_guard_applies_to_every_tenant_scoped_model(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
    model: type[Document] | type[Conversation],
) -> None:
    """The guard keys off the TenantScoped mixin, so new models are covered automatically."""
    tenant_a, _ = seeded_tenants
    with request_context(tenant_id=tenant_a):
        async with session_factory() as session:
            rows = (await session.execute(select(model))).scalars().all()

    assert rows, "expected at least one row for the active tenant"
    assert all(row.tenant_id == tenant_a for row in rows)


async def test_query_without_a_bound_tenant_fails_closed(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """With no tenant in context the query raises rather than returning everything.

    Failing closed is the whole point: a background task that forgets to bind a
    tenant must break loudly in CI, not quietly serve cross-tenant data.
    """
    async with session_factory() as session:
        with pytest.raises(AuthenticationError):
            await session.execute(select(Document))


async def test_system_session_escape_hatch_sees_all_tenants(
    session_factory: async_sessionmaker[AsyncSession],
    seeded_tenants: tuple[str, str],
) -> None:
    """``system_session`` deliberately spans tenants for backups and GDPR cascades."""
    async with db_module.system_session(reason="isolation test") as session:
        docs = (await session.execute(select(Document))).scalars().all()

    assert {d.id for d in docs} == {"doc_a", "doc_b"}


def test_assert_belongs_to_tenant_rejects_foreign_entity() -> None:
    """The defence-in-depth assertion raises on a mismatched entity."""
    foreign = Document(id="doc_b", tenant_id="ten_bbbb", title="Globex handbook")
    with request_context(tenant_id="ten_aaaa"), pytest.raises(TenantIsolationError) as excinfo:
        assert_belongs_to_tenant(foreign, resource="document:doc_b")

    assert excinfo.value.status_code == 403
    assert excinfo.value.details["tenant_id"] == "ten_aaaa"


def test_assert_belongs_to_tenant_accepts_own_entity() -> None:
    """The same assertion is a no-op for a correctly scoped entity."""
    own = Document(id="doc_a", tenant_id="ten_aaaa", title="Acme handbook")
    with request_context(tenant_id="ten_aaaa"):
        assert_belongs_to_tenant(own, resource="document:doc_a")
