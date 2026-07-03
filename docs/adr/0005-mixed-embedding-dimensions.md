# ADR 0005 — One embedding column, one dimension, reindex to change it

**Status:** accepted · **Date:** 2026-08-16

## Context

Tenants can choose their embedding model. BGE-large is 1024 dimensions,
multilingual-e5-large is 1024, OpenAI's `text-embedding-3-small` is 1536,
`-3-large` is 3072. A `vector(n)` column in Postgres has one fixed `n`.

Three ways to allow more than one:

1. **One column per dimension** — `embedding_1024`, `embedding_1536`, each with
   its own HNSW index. Every query needs to know which column to read, and every
   index costs memory whether or not any tenant uses it.
2. **A table per dimension**, routed at query time. Correct, and it doubles the
   migration surface and the number of places a tenant filter can be forgotten —
   which, given ADR 0001, is the thing most worth avoiding.
3. **One column at one dimension**, with a model change requiring a reindex.

There is a deeper reason none of the first two are as attractive as they look:
**vectors from different models are not comparable at all.** Two 1024-dimension
embeddings from BGE and from e5 occupy unrelated spaces. A cosine similarity
between them is a real number with no meaning. So even with per-dimension
columns, a tenant switching between two 1024-dimension models still needs a full
reindex — the dimension matching is a coincidence, not compatibility.

Once that is understood, the multi-column design buys nothing except the illusion
that switching is cheap.

## Decision

One `embedding vector(1024)` column, one HNSW index, one dimension configured per
deployment (`AGRAG_EMBEDDING_DIM`).

Changing a tenant's embedding model does not re-embed anything automatically.
The admin API returns a warning saying a reindex is required, and
`agrag.reindex_tenant` re-ingests every document for that tenant.

## Consequences

**Good.** One column, one index, one query path. No routing logic that can send a
query to the wrong space. The expensive operation is explicit and initiated
deliberately, rather than happening silently at a scale nobody sized.

**Bad.** A model change costs a full re-ingestion of the tenant's corpus. For a
large tenant that is hours of worker time and a real provider bill. There is no
way to make it cheap, but there is a way to make it a surprise, and this is not
it.

**Also bad:** a deployment is locked to one dimension. Serving a 1536-dimension
tenant means a second deployment. Acceptable for a system with one demo tenant;
it is the first thing that would need revisiting under real multi-model demand,
at which point a table per dimension is the answer — with the routing logic
written once, tested against the isolation suite, and not before.

## Consequences for correctness

The `Chunk` model records `embedding_model` alongside the vector. Retrieval does
**not** currently filter on it, because within one deployment every chunk shares
a model. If per-dimension tables ever arrive, that filter becomes mandatory
before anything else, or a half-reindexed tenant will get confident nonsense
ranked by a meaningless similarity.

## Verification

`apps/api/tests/unit/test_parsers_and_embedders.py` asserts that an embedder
whose output dimension does not match the configured one raises at ingestion
rather than writing a row Postgres will reject later with an error that names a
column instead of a model.
