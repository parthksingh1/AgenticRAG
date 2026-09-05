<div align="center">

# AgenticRAG

**A multi-tenant, agentic RAG platform with an evals suite that gates its own deployments.**

[![CI](https://github.com/parthksingh1/AgenticRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/parthksingh1/AgenticRAG/actions/workflows/ci.yml)
[![Security](https://github.com/parthksingh1/AgenticRAG/actions/workflows/security.yml/badge.svg)](https://github.com/parthksingh1/AgenticRAG/actions/workflows/security.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/next.js-15-black.svg)](https://nextjs.org/)

[Architecture](docs/ARCHITECTURE.md) · [Evals](docs/EVALS.md) · [Security](docs/SECURITY.md) · [Testing](docs/TESTING.md) · [Demo](docs/DEMO.md) · [Deploying the demo](docs/DEPLOYING-THE-DEMO.md)

</div>

---

Most RAG demos answer a question and stop. The interesting problems start after
that: proving the answer was grounded, keeping one customer's documents away from
another's, noticing three weeks later that quality has slipped, and refusing to
ship a change that makes any of it worse.

This is a system built around those problems.

```bash
git clone https://github.com/parthksingh1/AgenticRAG.git && cd AgenticRAG
cp .env.example .env          # works with no API keys; add them for real answers
docker compose up             # ~4 minutes on a cold machine
open http://localhost:3000
```

If a port is already taken — most developers already run Postgres on 5432 —
override it rather than stopping the other service:

```bash
AGRAG_PORT_POSTGRES=55432 AGRAG_PORT_WEB=3010 docker compose up
```

No API key is required to boot. Every optional dependency degrades rather than
fails — no Redis means no caching, no OpenSearch means dense-only retrieval,
no provider key means a hashing embedder and a clear error on generation — and
`/readyz` reports exactly which capabilities are live.

---

## What it looks like

These are captured from the demo build described in
[docs/DEPLOYING-THE-DEMO.md](docs/DEPLOYING-THE-DEMO.md) — the real interface on
fixture data, which is why each one carries a banner saying so. The banner is
the point: a screenshot of a demo that does not admit to being one is a claim
about behaviour that nobody can check.

| | |
|---|---|
| ![An answer with an inline citation](docs/screenshots/01-answer-with-citations.png) | ![A question needing two sections](docs/screenshots/02-multi-hop.png) |
| **A grounded answer.** The superscript marker opens the passage that supports the claim, not the whole document. | **Two hops.** Neither the approval tier nor the 45-day deadline ranks well for the question as asked; the answer applies the rule rather than quoting it. |
| ![Declining a question the corpus cannot answer](docs/screenshots/03-refusal.png) | ![A prompt injection being blocked](docs/screenshots/05-injection-blocked.png) |
| **Knowing when to refuse.** The handbook has no insurance provider in it. A confident answer here would be wrong in a way the reader cannot detect. | **An attack that fails.** Blocked at the input guardrail before a model is called, which is why the cost is zero. |
| ![Groundedness by run against the gate floor](docs/screenshots/08-admin-evals.png) | ![Reliability diagrams for two judges](docs/screenshots/09-admin-calibration.png) |
| **The gate, drawn.** The dashed line is the absolute floor. The run that dips under it is the one marked `fail`. | **Judges, checked.** Stated confidence against observed accuracy. The dashed diagonal is perfect calibration; distance from it is the ECE that weights each judge's vote. |

More: [retrieval playground](docs/screenshots/07-playground.png) ·
[documents](docs/screenshots/06-documents.png) ·
[cost](docs/screenshots/10-admin-cost.png) ·
[graph](docs/screenshots/11-graph.png) ·
[tool call](docs/screenshots/04-tool-call.png)

---

## What is actually here

| | |
|---|---|
| **Ingestion** | PDF/DOCX/PPTX/HTML/MD/CSV/XLSX/images, layout-aware and semantic chunking, contextual retrieval, MinHash dedupe, document versioning with stale-chunk marking, Celery with retry, backoff and a dead-letter path |
| **Retrieval** | pgvector HNSW + OpenSearch BM25, reciprocal-rank fusion, cross-encoder reranking, HyDE and multi-query rewriting, Corrective RAG with a web fallback, GraphRAG over Neo4j — all toggleable per tenant |
| **Agent** | A LangGraph state machine with thirteen nodes, explicit token and tool budgets, and token-level streaming over SSE |
| **Tools** | Six MCP servers: document search, SQL analytics, web fetch, calculator, graph query, sandboxed code execution |
| **Guardrails** | Prompt-injection detection, PII detection and redaction, an NLI-based hallucination check, citation verification, cost caps and token-bucket rate limits — every decision logged as a span attribute and a database row |
| **Multi-tenancy** | Row-level isolation enforced at the SQLAlchemy session level, which fails closed |
| **Evals** | 198 golden, 410 regression and 120 adversarial cases; a two-judge panel calibrated against human labels; a CI gate that blocks a merge on a groundedness drop |
| **Observability** | OpenTelemetry end to end, Prometheus metrics, committed Grafana dashboards, Langfuse traces, structured JSON logs |

---

## The parts worth reading

If you are evaluating this as a portfolio piece, these are the four places where
the engineering is not obvious.

### 1. Tenant isolation that fails closed

Filtering by `tenant_id` in each query is a policy that holds until somebody
forgets. This enforces it one level down, in a SQLAlchemy `do_orm_execute`
listener that appends the tenant predicate to every query against a
tenant-scoped model:

```python
@event.listens_for(AsyncSession.sync_session_class, "do_orm_execute")
def _tenant_guard_listener(state: ORMExecuteState) -> None:
    _apply_tenant_criteria(state)
```

A query with no tenant in context **raises** rather than returning everything.
Escaping it requires `system_session(reason=...)`, which logs at WARNING with a
stated reason, so an audit can enumerate every cross-tenant read in the codebase.
There are four, and they are the GDPR cascade, the nightly backup, the drift
sweep and the eval harness.

[`apps/api/src/core/db.py`](apps/api/src/core/db.py) ·
[isolation tests](apps/api/tests/unit/test_tenant_isolation.py)

### 2. Judges that are checked against people

An LLM scoring another LLM's answers is an opinion dressed as a measurement. This
uses two judges from different providers, surfaces their disagreements as a
human-labelling queue instead of averaging them away, and weights each judge by
its measured expected calibration error:

```
weight = 1 / (1 + ECE)
```

The calibration itself — ECE, Cohen's kappa, a least-squares recalibration
against human labels — is computed weekly and stored, so a judge that drifts is
visible rather than silently authoritative.

[`apps/api/src/services/calibration.py`](apps/api/src/services/calibration.py) ·
[`evals/judges.py`](evals/judges.py) · [docs/EVALS.md](docs/EVALS.md)

### 3. A gate with floors, not just deltas

A delta-only quality gate ratchets downward: ten pull requests each dropping
groundedness by 2.9% all pass, and the system ends up 29% worse with a green
history. So the gate enforces absolute floors as well, fails on any regression
case that flips from passing to failing regardless of the aggregate, and gives
injection resistance zero tolerance.

[`evals/gate.py`](evals/gate.py) · [the workflow](.github/workflows/evals-gate.yml)

### 4. A code sandbox that can actually be killed

The code-execution MCP server runs RestrictedPython in a separate, isolated
process (`python -I worker.py`) communicating over a JSON pipe. The earlier
thread-based version could not interrupt `while True: pass` — it leaked a
spinning core and the timeout never fired. Eight documented escape attempts are
in the test suite, and all of them are blocked.

[`mcp-servers/code-exec/`](mcp-servers/code-exec/)

---

## Numbers

**Every number in this README is produced by a command in this repository.**
Nothing here is illustrative, and nothing was typed in by hand.

| Claim | Produced by |
|---|---|
| Test counts and coverage | `cd apps/api && pytest tests/unit --doctest-modules src --cov=src` |
| Guardrails and fusion at 100% line + branch | `pytest --cov=src/guardrails --cov=src/retrieval/fusion.py --cov-branch --cov-fail-under=100` |
| Eval-set sizes and stratification | `python -m evals.run --set golden --offline --limit 1` |
| Answer-quality metrics | `python -m evals.run --set golden` — needs the stack up and provider keys |
| Judge calibration (ECE, kappa) | `python -m evals.scripts.calibrate_judges` |
| Load-test p95 and error rate | `k6 run scripts/load/chat.js` |

Current test suite, at the commit you are reading:

```
apps/api     938 tests (unit + doctests + contract)
mcp-servers   82 tests
coverage      66% overall; 100% line and branch on guardrails and fusion
```

The 66% is the real number, not a target. `docs/TESTING.md` names the four
modules carrying the shortfall and why they are the ones left: they are the
I/O-heavy paths that need fake backends rather than fixtures. The gate is set
at 65 so it reflects the suite as it stands -- a gate pinned above reality
fails every build and ends up deleted rather than satisfied.

Answer-quality metrics are **not** printed in this README, because they depend on
which provider and model you configure. Run the command above and you will get
your own — that is the point. `docs/EVALS.md` explains how to read them and what
the gate thresholds mean.

---

## What running it actually found

Every layer here was executed, not just written, and each one contained bugs
that reading could not have surfaced. They are worth listing because the
failures are more informative than the features:

| Found by | Bug |
|---|---|
| Compiling the frontend | The API client was generic over `z.ZodType<T>`, which infers from the **input** side — so every field with a `.default()` came back optional and four pages would not type-check |
| Running the seed script | `SentenceTransformerEmbedder.__init__` never imported its dependency; it lazy-loaded inside `embed()`, so every caller's `try/except` fallback was dead and the failure surfaced mid-ingestion, after the user had been told their upload was processing |
| Starting the stack | The app read bare `DATABASE_URL` while CI, Helm and Terraform all set `AGRAG_*` — every deployment artifact was setting variables the application silently ignored |
| Contract tests | `response_model=None` had quietly dropped `OpenAIChatResponse` from the OpenAPI document, so every generated SDK lost the shape |
| Strict type-checking | A custom `_NotGiven` sentinel failed Anthropic's `isinstance` check, so "omitted" parameters were serialised into the request body; and both `stream()` paths dropped `tools`, meaning a streamed tool-calling turn could never invoke a tool |
| CI on a clean machine | `aiosqlite` was used by every test fixture and never declared — it was present locally as a transitive install |

The last one is the general lesson: **a green suite on the machine that wrote
it proves less than it appears to.** `docs/TESTING.md` separates what has been
executed from what has only been written.

---

## Architecture

```mermaid
flowchart LR
  U[Browser] -->|SSE| API[FastAPI]
  API --> G{{Guardrails: input}}
  G --> GR[LangGraph agent]
  GR --> R[Hybrid retrieval]
  R --> PG[(pgvector HNSW)]
  R --> OS[(OpenSearch BM25)]
  R --> NEO[(Neo4j)]
  GR --> MCP[6 MCP servers]
  GR --> LLM[Provider router]
  GR --> G2{{Guardrails: output}}
  G2 --> U
  API --> Q[Celery]
  Q --> ING[Ingestion pipeline]
  ING --> S3[(MinIO)]
  ING --> PG
  API -.-> OTEL[OTel → Prometheus, Jaeger, Langfuse]
```

The full walkthrough, including why each store is there and what happens when
each one is missing, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Running it

```bash
docker compose up                          # core stack
docker compose --profile obs up            # + Prometheus, Grafana, Jaeger, Langfuse
docker compose --profile mcp up            # + the six MCP servers

make seed                                  # demo tenant, corpus and five pinned conversations
make test                                  # the full suite
make evals                                 # golden set against the running stack
```

| Service | URL |
|---|---|
| Web | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Grafana | http://localhost:3001 |
| Jaeger | http://localhost:16686 |
| Langfuse | http://localhost:3002 |

---

## Using the API

It speaks the OpenAI wire format, so any OpenAI SDK works with a changed
`base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="agr_...")
answer = client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "What is the carry-over limit for annual leave?"}],
)
```

Citations ride along in a non-standard `citations` field, because an
OpenAI-shaped RAG response with no citations is indistinguishable from a plain
completion. Clients that ignore unknown fields are unaffected.

---

## Repository layout

```
apps/api          FastAPI backend, LangGraph agent, ingestion, retrieval, guardrails
apps/web          Next.js 15 frontend
mcp-servers       Six MCP servers, each with its own image and tests
evals             Datasets, metrics, judge panel, runner, CI gate, reports
prompts           Versioned, content-hashed YAML + Jinja2 prompts
infra             Prometheus, Grafana, Alertmanager, OTel, Kubernetes, Terraform
scripts           Seeding, load, chaos, canary watch
docs              Architecture, evals, security, testing, ADRs
```

---

## Development

The repository uses a root virtualenv at `.venv-dev/` and two Python lint
configurations kept deliberately in step — `apps/api/pyproject.toml` for the
backend, `ruff.toml` at the root for `evals/`, `scripts/` and `cli/`.

```bash
python -m venv .venv-dev && .venv-dev/bin/pip install -e "apps/api[dev]"

cd apps/api && ruff check . && pytest tests/unit --doctest-modules src
cd ../..    && ruff check evals scripts
python -m evals.run --set golden --offline    # exercises the harness, free
```

`evals/` imports from `src.*`, so it needs `apps/api` on `PYTHONPATH`.

## Status

Built as a portfolio project. The backend, evals harness, guardrails,
observability, CI and frontend are complete and tested; the deployment pipeline
is written but has never been run, and [docs/DEMO.md](docs/DEMO.md) says so
rather than claiming a URL that does not exist.

[docs/adr/](docs/adr/) records the decisions that were not obvious, including
the ones that were rejected and why.
