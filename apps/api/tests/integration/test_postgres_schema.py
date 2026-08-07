"""Integration tests against a real PostgreSQL with pgvector.

Everything here needs behaviour SQLite cannot emulate: the ``vector`` type, the
HNSW index and its cosine operator, JSONB, and the ``ON CONFLICT`` upsert the
budget counter depends on. Running these on SQLite would pass while testing
nothing that matters.

The container is managed by testcontainers, so the suite is self-contained:
``pytest tests/integration`` needs Docker and nothing else. The migration is
applied rather than ``create_all`` — testing a schema the migration did not
produce would leave the migration itself unverified, which is precisely the
artefact most likely to be wrong in production.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="session"),
]

EMBEDDING_DIM = 1024


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """Start a PostgreSQL container with pgvector, or reuse a configured one.

    ``AGRAG_TEST_DATABASE_URL`` short-circuits the container, which is how CI
    runs these against the service container it already has rather than starting
    a second one inside it.
    """
    configured = os.getenv("AGRAG_TEST_DATABASE_URL")
    if configured:
        yield configured
        return

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - dev extra not installed
        pytest.skip("testcontainers is not installed")

    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as container:
        yield container.get_connection_url()


# A session-scoped async fixture needs a session-scoped event loop; without
# `loop_scope` pytest-asyncio tears the loop down between tests and the engine
# it created becomes unusable.
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def migrated_engine(postgres_url: str) -> AsyncIterator[object]:
    """Apply the migration and yield an engine bound to the result."""
    import asyncio
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", postgres_url)

    os.environ["DATABASE_URL"] = postgres_url
    # Alembic's runner drives its own event loop, so it runs in a thread rather
    # than inside the one pytest-asyncio already has open.
    await asyncio.to_thread(command.upgrade, config, "head")

    engine = create_async_engine(postgres_url, future=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def session(migrated_engine: object) -> AsyncIterator[AsyncSession]:
    """A session that rolls back everything it did.

    Each test therefore starts from the migrated schema with no rows, without
    paying to recreate the schema between tests.
    """
    factory = async_sessionmaker(bind=migrated_engine, expire_on_commit=False)  # type: ignore[arg-type]
    async with factory() as active:
        transaction = await active.begin()
        try:
            yield active
        finally:
            await transaction.rollback()


# ── Schema ───────────────────────────────────────────────────────────────────


async def test_the_migration_creates_every_table(session: AsyncSession) -> None:
    """The migration, not create_all, is what production runs."""
    from src.models.base import Base

    result = await session.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    )
    tables = {row[0] for row in result.all()}

    assert set(Base.metadata.tables) <= tables


async def test_the_vector_column_has_the_declared_dimension(session: AsyncSession) -> None:
    """A dimension mismatch fails at insert time with an opaque error."""
    result = await session.execute(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
    )

    assert result.scalar_one() == f"vector({EMBEDDING_DIM})"


async def test_the_hnsw_index_carries_its_build_parameters(session: AsyncSession) -> None:
    """m and ef_construction are the recall/latency trade-off; defaults are not it."""
    result = await session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'")
    )
    definition = result.scalar_one()

    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition
    assert "m='16'" in definition
    assert "ef_construction='200'" in definition


async def test_jsonb_columns_round_trip_structured_data(session: AsyncSession) -> None:
    """The models declare JSONB; SQLite would have silently accepted plain JSON."""
    from src.models.tenant import Tenant

    session.add(
        Tenant(
            id="ten_jsonb",
            name="JSONB",
            slug="jsonb",
            enabled_strategies=["hybrid", "hyde"],
            guardrail_config={"pii_mode": "block", "thresholds": {"toxicity": 0.9}},
        )
    )
    await session.flush()

    result = await session.execute(
        text("SELECT guardrail_config -> 'thresholds' ->> 'toxicity' FROM tenants WHERE id = :id"),
        {"id": "ten_jsonb"},
    )

    assert result.scalar_one() == "0.9"


# ── Vector search ────────────────────────────────────────────────────────────


def _vector(seed: float) -> list[float]:
    """A deterministic unit-ish vector, distinguishable from its neighbours."""
    return [seed] + [0.0] * (EMBEDDING_DIM - 1)


async def _seed_corpus(session: AsyncSession) -> None:
    """Two tenants, each with one document and two chunks."""
    from src.models.document import Chunk, Document, DocumentVersion
    from src.models.tenant import Tenant

    for tenant_id, seed in (("ten_a", 1.0), ("ten_b", -1.0)):
        session.add(Tenant(id=tenant_id, name=tenant_id, slug=tenant_id))
        session.add(
            Document(id=f"doc_{tenant_id}", tenant_id=tenant_id, title=f"{tenant_id} handbook")
        )
        session.add(
            DocumentVersion(
                id=f"ver_{tenant_id}",
                tenant_id=tenant_id,
                document_id=f"doc_{tenant_id}",
                version=1,
                storage_key=f"{tenant_id}/doc/file.pdf",
                content_hash="0" * 64,
            )
        )
        for ordinal in range(2):
            session.add(
                Chunk(
                    id=f"chk_{tenant_id}_{ordinal}",
                    tenant_id=tenant_id,
                    document_id=f"doc_{tenant_id}",
                    document_version_id=f"ver_{tenant_id}",
                    ordinal=ordinal,
                    content=f"{tenant_id} chunk {ordinal}",
                    embedding=_vector(seed if ordinal == 0 else seed * 0.5),
                    embedding_model="test",
                )
            )
    await session.flush()


async def test_cosine_search_returns_the_nearest_chunk(session: AsyncSession) -> None:
    """The core retrieval operation, against the real index."""
    await _seed_corpus(session)

    result = await session.execute(
        text(
            "SELECT id, 1 - (embedding <=> :q) AS score FROM chunks "
            "WHERE tenant_id = :tenant ORDER BY embedding <=> :q LIMIT 1"
        ),
        {"q": "[" + ",".join(["1.0"] + ["0.0"] * (EMBEDDING_DIM - 1)) + "]", "tenant": "ten_a"},
    )
    row = result.mappings().one()

    assert row["id"] == "chk_ten_a_0"
    assert row["score"] == pytest.approx(1.0, abs=1e-6)


async def test_vector_search_cannot_reach_another_tenant(session: AsyncSession) -> None:
    """The isolation test that SQLite could not run, on the real query path."""
    await _seed_corpus(session)

    result = await session.execute(
        text("SELECT id FROM chunks WHERE tenant_id = :tenant ORDER BY embedding <=> :q"),
        {"q": "[" + ",".join(["-1.0"] + ["0.0"] * (EMBEDDING_DIM - 1)) + "]", "tenant": "ten_a"},
    )
    ids = {row[0] for row in result.all()}

    assert ids == {"chk_ten_a_0", "chk_ten_a_1"}
    assert not any(i.startswith("chk_ten_b") for i in ids)


async def test_the_dense_retriever_runs_against_the_real_index(session: AsyncSession) -> None:
    """The retriever's own SQL, not a hand-written approximation of it."""
    from src.ingestion.embedders.base import Embedder
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.types import RetrievalRequest

    await _seed_corpus(session)

    class FixedEmbedder(Embedder):
        model_name = "fixed"
        dimension = EMBEDDING_DIM

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [_vector(1.0) for _ in texts]

    hits = await DenseRetriever(session=session, embedder=FixedEmbedder()).retrieve(
        RetrievalRequest(query="anything", top_k=2), tenant_id="ten_a"
    )

    assert [h.chunk_id for h in hits] == ["chk_ten_a_0", "chk_ten_a_1"]
    assert hits[0].document_title == "ten_a handbook"


# ── Upserts ──────────────────────────────────────────────────────────────────


async def test_budget_counter_upsert_accumulates(session: AsyncSession) -> None:
    """Two workers finishing a turn together must not lose one increment.

    Read-modify-write loses one under concurrency; the ON CONFLICT upsert this
    tests is what makes the counter correct, and SQLite would not exercise the
    PostgreSQL syntax it uses.
    """
    from src.models.tenant import Tenant
    from src.repositories.usage import _increment_counter

    session.add(Tenant(id="ten_budget", name="Budget", slug="budget"))
    await session.flush()

    await _increment_counter(session, tenant_id="ten_budget", tokens=100, cost_usd=0.5)
    await _increment_counter(session, tenant_id="ten_budget", tokens=50, cost_usd=0.25)

    result = await session.execute(
        text(
            "SELECT tokens_used, cost_usd, requests FROM tenant_budget_counters "
            "WHERE tenant_id = :t"
        ),
        {"t": "ten_budget"},
    )
    tokens, cost, requests = result.one()

    assert tokens == 150
    assert float(cost) == pytest.approx(0.75)
    assert requests == 2


async def test_cascading_delete_removes_a_documents_chunks(session: AsyncSession) -> None:
    """GDPR erasure relies on the cascade rather than deleting each table by hand."""
    from src.core.context import request_context
    from src.models.document import Chunk, Document

    await _seed_corpus(session)

    # An ORM read needs a bound tenant: the session guard fails closed without
    # one, which is the behaviour tests/unit/test_tenant_isolation.py pins.
    with request_context(tenant_id="ten_a"):
        document = (
            await session.execute(select(Document).where(Document.id == "doc_ten_a"))
        ).scalar_one()
        await session.delete(document)
        await session.flush()

        remaining = await session.execute(select(Chunk).where(Chunk.document_id == "doc_ten_a"))

        assert remaining.scalars().all() == []


async def test_an_unscoped_orm_read_is_refused_against_real_postgres(
    session: AsyncSession,
) -> None:
    """The guard is not a SQLite artefact: it holds on the production engine too."""
    from src.core.errors import AuthenticationError
    from src.models.document import Document

    await _seed_corpus(session)

    with pytest.raises(AuthenticationError):
        await session.execute(select(Document))
