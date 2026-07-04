# Architecture

## The shape of a request

```mermaid
sequenceDiagram
  participant B as Browser
  participant A as FastAPI
  participant G as Guardrails
  participant L as LangGraph
  participant R as Retrieval
  participant M as Model provider

  B->>A: POST /api/chat/stream
  A->>A: Resolve principal, set tenant context
  A->>G: Input checks (injection, PII, off-topic)
  G-->>A: allow / redact / block
  A->>L: Run the graph
  L->>R: Retrieve (dense + sparse, fused, reranked)
  R-->>L: Ranked chunks
  L->>L: Evaluate relevance; rewrite or fall back if weak
  L->>M: Generate with the retrieved context
  M-->>L: Tokens
  L->>G: Output checks (groundedness, PII, moderation)
  L->>L: Bind citations, verify each one resolves
  L-->>A: Tokens, then citations, then done
  A-->>B: SSE frames
```

## Why each store is there

| Store | Holds | What breaks without it |
|---|---|---|
| **Postgres + pgvector** | Everything relational, and the embeddings | Nothing works. This is the only hard dependency. |
| **Redis** | Semantic and exact caches, rate limits, budget counters, ingestion progress | Caching, rate limiting and live progress. Answers still work, slower and dearer. Budget counters fall back to the durable Postgres counter, so a cache flush cannot hand every tenant an unlimited budget. |
| **OpenSearch** | BM25 index | Hybrid retrieval degrades to dense-only. Exact-term queries — a product code, an error string — get noticeably worse, because that is precisely what BM25 is good at and embeddings are not. |
| **MinIO / S3** | Original document bytes | Uploads and the document viewer. Existing chunks still answer. |
| **Neo4j** | The extracted knowledge graph | GraphRAG. Multi-hop questions fall back to chunk retrieval, which is worse at them but not useless. |

Every optional store is probed at startup and its absence is reported by
`/readyz` as a named missing capability rather than as a failure. That is a
deliberate design choice: a portfolio project that will not boot without five
managed services is a project nobody runs.

## Retrieval

Two retrievers run concurrently and their rankings are fused.

**Dense** is pgvector with an HNSW index (`vector_cosine_ops`, `m=16`,
`ef_construction=200`). `ef_search` is set per query with `SET LOCAL`, so a
recall-sensitive query can pay for more search without changing the index.

**Sparse** is OpenSearch BM25.

They are combined with **reciprocal rank fusion**:

```
score(d) = Σ 1 / (k + rank_i(d))
```

RRF rather than a weighted sum of scores, because the two scores are not on the
same scale and never will be — a cosine similarity of 0.83 and a BM25 score of
11.4 have no defensible exchange rate. RRF only uses ranks, so it needs no
normalisation and no tuning constant beyond `k`.

`k=60` is the value from the original RRF paper, kept because we have no
evidence for a better one. The moment we do, it belongs in the eval suite rather
than in a config file.

Then, in order and each toggleable per tenant:

- **Reranking** with a cross-encoder over the top 50. Bi-encoder retrieval scores
  a query and a document independently; a cross-encoder reads them together,
  which is much better and far too slow to run over the whole corpus.
- **Query rewriting** — HyDE (embed a hypothetical answer rather than the
  question, because answers live closer to answers in embedding space than
  questions do), multi-query expansion, and decomposition for multi-hop.
- **Corrective RAG** — if the retrieved set scores poorly, retrieve again with a
  rewritten query, and fall back to web search if it still does.
- **Adaptive k** — a specific question needs three chunks; a summarisation
  question needs twenty.

## The agent

A LangGraph state machine with explicit nodes and conditional edges:

```
intent_router → query_rewriter → retriever → retrieval_evaluator
                                                     ↓
                          (weak) ← corrective retrieval
                                                     ↓
             reranker → planner → tool_executor → generator
                                                     ↓
                       citation_binder → self_critic → formatter
```

Two properties matter more than the node list.

**Budgets are explicit.** Token and tool-call budgets live in the graph state,
and a node that would exceed one stops rather than being stopped. An agent that
can loop is an agent that will, and discovering it on the invoice is the usual
way.

**Streaming is real.** `astream_events(version="v2")` surfaces tokens as the
model produces them, plus node transitions for the reasoning panel. Buffering the
answer and revealing it at the end doubles perceived latency for no benefit.

## Tenant isolation

See the README. The short version: a `do_orm_execute` listener appends the tenant
predicate to every query against a tenant-scoped model, and a query with no
tenant in context raises rather than returning everything.

The four legitimate escapes each call `system_session(reason=...)`, which logs at
WARNING. They are the GDPR cascade, the nightly backup, the drift sweep and the
eval harness.

Isolation for the stores that have no row-level security of their own —
OpenSearch, Neo4j, Redis, MinIO — is enforced by putting the tenant id in the
key, the filter and the node property, and by testing that a query without it
returns nothing.

## Ingestion

```
upload → parse → chunk → contextualise → embed → dedupe → index
```

- **Parsing** is format-specific with content sniffing, not extension trust. A
  `.pdf` that is actually HTML is common enough to matter.
- **Chunking** is layout-aware by default: headings become breadcrumbs on their
  chunks rather than chunks of their own, tables stay whole, and overlap is
  suppressed across atomic blocks so table markup cannot bleed into the prose
  after it.
- **Contextual retrieval** prepends a short, document-aware summary to each chunk
  before embedding. A chunk that says "it increased by 12%" is unretrievable on
  its own; the same chunk prefixed with what "it" is, is not.
- **Deduplication** uses MinHash LSH. Near-duplicate chunks — the same policy
  paragraph in four documents — otherwise fill the top-k with one fact.
- **Versioning** marks old chunks stale rather than deleting them, so citations in
  existing conversations still resolve.

Work happens in Celery with `task_acks_late` and `task_reject_on_worker_lost`, so
a worker killed mid-task loses nothing. A task that exhausts its retries is
dead-lettered with the error recorded on the document — a silently disappearing
upload leaves the user watching a spinner that will never resolve.

## Guardrails

Input and output pipelines, each a sequence of checks that can allow, flag,
redact or block.

**Input:** prompt-injection detection (heuristics, then a classifier, then an LLM
judge, in ascending order of cost so most requests never reach the expensive
check), PII detection, and off-topic detection via retrieval score.

**Output:** PII redaction, moderation, NLI-based groundedness checking, and
citation verification — every inline marker must resolve to a chunk that was
actually retrieved.

Every decision is written twice: as an OpenTelemetry span attribute, so it appears
on the trace next to the request that caused it, and as a `guardrail_events` row,
so it can be aggregated.

This module is held at **100% line and branch coverage** by CI. It is the one
place where an untested branch is a security bug rather than a latent one.

## Observability

- **Traces** — OpenTelemetry across FastAPI, SQLAlchemy, Redis, HTTPX and Celery,
  with a span per graph node. Context propagates into the workers, so an
  ingestion failure is on the same trace as the upload that caused it.
- **Metrics** — Prometheus on a private registry with route templates as labels,
  never raw paths. `/api/documents/{id}` as a label is one series;
  `/api/documents/abc123` is one series per document and will take the metrics
  backend down.
- **Logs** — structured JSON via loguru, enriched with request, tenant and trace
  ids, with credential-shaped keys redacted at the sink rather than at each call
  site.
- **Alerts** — p95 TTFT over 3s, error rate over 1%, cost more than 3σ above the
  mean, hallucination rate over 5%, and no successful backup in 36 hours. The
  last one catches a backup job that stopped running, which a failure-rate alert
  cannot.

## Decisions worth reading

The non-obvious ones are written up as ADRs in [`adr/`](adr/):

- [0001](adr/0001-session-level-tenant-isolation.md) — session-level tenant isolation
- [0002](adr/0002-rrf-over-weighted-fusion.md) — RRF over weighted score fusion
- [0003](adr/0003-two-judge-calibrated-panel.md) — a calibrated two-judge panel
- [0004](adr/0004-vector-index-parameters.md) — HNSW parameters
- [0005](adr/0005-mixed-embedding-dimensions.md) — one dimension per tenant
- [0006](adr/0006-subprocess-sandbox.md) — a subprocess sandbox for code execution
