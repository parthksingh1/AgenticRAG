# Testing

```bash
make test          # everything that does not need Docker
make test-all      # plus integration, against real containers
make evals         # answer quality, against a running stack
```

## What is tested how, and why

| Layer | Tool | What it catches that nothing else does |
|---|---|---|
| Unit | pytest | Logic errors, in milliseconds |
| Property | hypothesis | The input you did not think of |
| Doctest | pytest `--doctest-modules` | Documentation that has quietly become false |
| Integration | testcontainers | Everything the ORM abstracts away until it does not |
| Contract | schemathesis | Endpoints that violate their own OpenAPI schema |
| E2E | Playwright | The wiring between two things that each work |
| Load | k6 | Behaviour under concurrency |
| Soak | k6, 4 hours | Leaks that a 60-second run cannot show |
| Chaos | `scripts/chaos.py` | Whether "degrades gracefully" is true or aspirational |
| Mutation | mutmut | Tests that execute code without asserting anything about it |
| Evals | `evals/` | Answer quality, which no unit test can reach |

## Doctests are load-bearing

238 of them, run in CI. A docstring example that no longer produces its stated
output is a failing test, not a stale comment. This is the cheapest defence
against documentation drift there is, and it means every example in the codebase
is one somebody can trust.

## Property tests

Used where a property is easier to state correctly than a set of examples:

- **Chunkers** — no chunk exceeds `max_chars`; concatenating chunks minus overlap
  reconstructs the source; overlap never splits a word.
- **Fusion** — order-independent; a document ranked first by both retrievers
  always outranks one ranked first by only one; per-document capping cannot
  starve a source.
- **Calibration** — kappa is symmetric; a judge agreeing with the humans always
  scores zero ECE; a combined score never leaves the range of its inputs.
- **Graph extraction** — for *any* model output, a mapped relation is either a
  member of the closed vocabulary or `None`. That property is what makes the one
  unavoidable Cypher interpolation safe, so it is asserted rather than assumed.

## Integration tests found things unit tests could not

Three bugs surfaced only against real Postgres, all invisible on SQLite:

1. `SET LOCAL hnsw.ef_search = :ef` — Postgres rejects bind parameters in `SET`.
2. `operator does not exist: vector <=> character varying` — the parameter needed
   an explicit `CAST(:query_vector AS vector)`.
3. A foreign-key violation on chunk insert, because a missing relationship meant
   the unit of work ordered the inserts wrongly.

None of these are exotic. They are the ordinary cost of testing a database
application against a different database, and the reason `make test-all` exists.

## Fake providers, not mocks of our own code

LLM calls in tests go through a fake provider that returns scripted responses,
including failures: rate limits, timeouts, malformed JSON, empty completions.

Mocking `LLMRouter` itself would test that the mock was called. The fake sits at
the provider boundary, so the retry logic, the fallback chain, the budget
accounting and the error classification are all exercised for real.

This is how the retry classification bug was found: unknown provider errors were
being treated as permanent, so nothing retried. Every test passed, because every
test mocked the layer above it.

## Coverage, and where it is absolute

| Scope | Gate |
|---|---|
| `apps/api` overall | 85% |
| `apps/web` | 75% |
| `src/guardrails/` | **100% line and branch** |
| `src/retrieval/fusion.py` | **100% line and branch** |

The two absolutes are enforced as a separate CI job so a shortfall is a
named failure rather than a line inside a 900-test run.

They are the two places where an untested branch is a security or correctness
bug rather than a latent one. Reaching 100% on the guardrails meant deleting two
genuinely unreachable defensive branches rather than marking them `# pragma: no
cover` — an unreachable branch is dead code, and the honest response is to remove
it.

## Chaos

`scripts/chaos.py` kills dependencies while traffic runs and asserts the system
does what it claims:

| Killed | Expected |
|---|---|
| Redis | Answers still work, uncached; rate limits fall back to the durable counter |
| OpenSearch | Retrieval degrades to dense-only; `/readyz` says so |
| Neo4j | GraphRAG unavailable; other strategies unaffected |
| A provider | Fallback provider takes over; the failure is on the trace |
| A worker mid-task | The task is redelivered, not lost |

"Degrades gracefully" is a claim. This is the test that makes it a fact.

## Mutation testing

`mutmut` on the guardrails and fusion modules. Coverage says a line ran;
mutation testing says an assertion would have noticed if it had been wrong.
100% coverage with a surviving mutant means a test that executes code and checks
nothing about it.

## Load

k6, 50 virtual users, with thresholds that fail the run rather than printing a
summary nobody reads:

```
p95 time-to-first-token  < 3000ms
error rate               < 0.1%
```

The soak run is four hours at lower concurrency, asserting RSS growth under 5%.
A leak in the embedding cache or an unbounded conversation buffer is invisible in
a 60-second run and obvious in four hours.

## Running one thing

```bash
cd apps/api
pytest tests/unit/test_guardrails.py -q                    # one file
pytest tests/unit -k tenant -q                             # by name
pytest --doctest-modules src/retrieval -q                  # doctests only
pytest tests/integration -m integration -q                 # needs Docker
pytest tests/unit --cov=src/guardrails --cov-branch \
       --cov-fail-under=100 --cov-report=term-missing      # the absolute gate
```

The tooling setup — the repo-root `.venv-dev`, the two ruff configs, the
`PYTHONPATH` the eval harness needs — is in the README's development section.
