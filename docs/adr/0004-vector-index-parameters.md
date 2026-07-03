# ADR 0004 — HNSW over IVFFlat, and the parameters chosen

**Status:** accepted · **Date:** 2026-08-16

## Context

pgvector offers two index types, and the choice is not reversible cheaply on a
large table.

**IVFFlat** partitions vectors into lists and searches the nearest few. It builds
fast and uses little memory. Its fatal property for this system is that the
partitioning is computed from the data present at build time: an index built on
1,000 chunks and then grown to 500,000 has partitions that no longer describe the
data, and recall degrades silently. Fixing it means rebuilding.

**HNSW** builds a navigable small-world graph incrementally. It is slower to
build and uses more memory, and it does not degrade as rows are added.

A multi-tenant system where tenants upload continuously has no build-time
snapshot to partition on. That settles it.

## Decision

```sql
CREATE INDEX ix_chunks_embedding_hnsw ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
```

`ef_search` is set per query with `SET LOCAL`, defaulting to 100.

**`m = 16`** — edges per node. pgvector's default. Raising it improves recall and
costs memory linearly; at our corpus sizes there is no measured benefit, and
guessing at 32 would double the index for a number nobody has checked.

**`ef_construction = 200`** — candidate list size during build. The default is 64.
Raised because build happens once per chunk and search happens on every query:
paying more at build to get better graph connectivity is the right side of that
trade. 200 is roughly 3× the build time of 64 for a recall improvement that is
real but modest.

**`vector_cosine_ops`** — the embedding models we use produce normalised vectors,
where cosine and inner product rank identically, but cosine is what the models
are documented and trained against. Matching the documented metric costs nothing
and removes a class of subtle wrongness.

**`ef_search` per query, not per index** — a recall-sensitive query can pay for a
wider search without rebuilding anything:

```sql
SET LOCAL hnsw.ef_search = 200;
```

Interpolated as a clamped integer rather than bound as a parameter, because
Postgres rejects bind parameters in `SET`. The clamp is what makes that safe.

## Consequences

**Good.** Recall stays stable as tenants grow. Per-query recall/latency tuning
without a rebuild. No scheduled reindex job, which is one fewer thing to forget.

**Bad.** Index build is slower and the index is larger — roughly 2–3× IVFFlat at
the same row count. Ingesting a very large corpus is measurably slower than it
would be with IVFFlat.

**Not yet decided:** these numbers are defaults plus one reasoned change, not
measured optima for this corpus. When there is a reason to tune them, the
experiment belongs in `evals/` where recall can be measured against the golden
set, rather than in a config change justified by intuition.

## Verification

The migration is asserted after `alembic upgrade head`: the index exists, is
HNSW, and carries `m='16', ef_construction='200'`. A silent revert to defaults
would otherwise be invisible.
