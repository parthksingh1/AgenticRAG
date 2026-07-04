# Security

## Threat model

This system takes untrusted input from three directions at once, which is what
makes RAG security different from ordinary web-application security.

| Source | What it can attempt |
|---|---|
| **The user's question** | Prompt injection, jailbreaks, PII extraction, tool abuse |
| **An uploaded document** | *Indirect* injection — an instruction planted in a PDF that reaches the model as retrieved context |
| **The model's own output** | Tool calls, SQL, Cypher and code that the model wrote and nobody reviewed |

The third is the one most systems miss. An LLM that emits a tool call is an
untrusted code generator, and the fact that it is *our* model does not make its
output trusted.

## Tenant isolation

Enforced at the SQLAlchemy session level and **fails closed** — a query with no
tenant in context raises rather than returning everything. See
[ADR 0001](adr/0001-session-level-tenant-isolation.md).

Stores with no row-level security of their own carry the tenant in the key:

| Store | How |
|---|---|
| OpenSearch | `tenant_id` term filter on every query, asserted in tests |
| Neo4j | `tenant_id` property on every node, in every `MATCH` |
| Redis | Tenant id in every key prefix |
| MinIO | Objects live under `<tenant_id>/` |

`apps/api/tests/unit/test_tenant_isolation.py` walks every endpoint with two
tenants and asserts nothing crosses, including the case where the context is
missing entirely.

## Prompt injection

Layered, cheapest first, so most requests never reach the expensive check:

1. **Heuristics** — known patterns (`ignore previous instructions`, delimiter
   escapes, role-play framing). Fast, free, and evadable on its own.
2. **A classifier** — DeBERTa fine-tuned for injection detection.
3. **An LLM judge** — only when the first two are ambiguous.

Detection is not the main defence. The main defence is that **the model is never
given an instruction channel it can be talked out of**:

- Retrieved context is presented as data, in delimited blocks, with the system
  prompt stating that content inside them is untrusted.
- Tools take structured arguments, never free text that becomes a command.
- The model cannot emit SQL, Cypher or a URL that is executed as written.

The adversarial eval set holds 120 attacks across 7 kinds. Injection resistance
has **zero tolerance** in the CI gate.

## Tool safety

### SQL analytics

The model never writes SQL. It names a query shape and supplies bound parameters.
Where free SQL is permitted, it is validated **structurally** with `sqlparse`
over the token stream, not with a regex:

- one statement only, and it must be a `SELECT`
- no DDL or DML tokens anywhere in the tree
- no `pg_`/`information_schema` access
- a mandatory `LIMIT`
- a statement timeout

Regex validation fails on the first case where a keyword appears as a string
literal. `SELECT 'delete me'` is a legitimate query; five of eleven documented
attack strings are rejected only by the token-stream check.

### Code execution

RestrictedPython in a separate, killable process. See
[ADR 0006](adr/0006-subprocess-sandbox.md). Eight documented escape attempts are
in the test suite; all are blocked, and the timeout genuinely terminates
`while True: pass`.

### Web fetch

Server-side request forgery is the risk: the API makes requests from inside the
perimeter on behalf of someone outside it.

`src/core/net.py` resolves the hostname and checks **every address it resolves
to**, rather than pattern-matching the URL:

- scheme allowlist (`http`, `https` only — `file://` reads the container's disk)
- `ipaddress.is_global`, which covers loopback, link-local, RFC1918, multicast and
  the reserved ranges in one check, and stays correct for IPv6
- IPv4-mapped IPv6 unwrapped before the check
- blocked ports for common internal services
- **redirects are not followed** — a 302 to an internal address would bypass
  everything above

A name like `evil.example.com` can resolve to `127.0.0.1`, and `http://0x7f.1/`
is loopback written so no string denylist catches it. Both are in the test suite,
along with nine other spellings including the cloud metadata endpoint.

The same guard runs on webhook delivery, and it runs **again at send time**, not
only at registration: a tenant can repoint a hostname after registering it.

### Graph queries

The model returns entities and relations as JSON. Writes use fixed, parameterised
Cypher. The one unavoidable interpolation — Cypher cannot bind a relationship
type as a parameter — is safe only because the type is drawn from a closed
vocabulary the model cannot extend. A hypothesis property test asserts that
invariant over arbitrary model output, including injected Cypher.

## Output safety

- **PII** detected and redacted with Presidio before the answer leaves.
- **Groundedness** checked with an NLI model: claims the retrieved context does
  not support are flagged.
- **Citations verified** — every inline `[n]` must resolve to a chunk that was
  actually retrieved. A dangling marker is a fabricated reference in a PDF export.
- **Moderation** via the OpenAI moderation endpoint.

Every decision is written twice: as an OpenTelemetry span attribute, so it sits
on the trace next to the request that caused it, and as a `guardrail_events` row
so it can be aggregated.

The guardrails module is held at **100% line and branch coverage** by CI. It is
the one place where an untested branch is a security bug rather than a latent
one.

## Secrets

- No secret in code. All configuration via `pydantic-settings`; `.env.example`
  documents the names and holds no values.
- API keys and webhook signing secrets are stored **hashed**. A database dump
  does not yield working credentials. The consequence is that a secret is shown
  once at creation and rotation is the only recovery — which is the correct
  trade.
- Credential-shaped keys are redacted at the log sink, not at each call site, so
  a new call site cannot forget.
- The audit log redacts the same keys before writing, because it is read by more
  people than the database is.
- `gitleaks` runs in CI over the full history: a secret committed and later
  removed is still in the history and still valid until rotated.

## Webhooks

Signed `HMAC-SHA256(secret, "{timestamp}.{body}")`, Stripe-style. The timestamp
is inside the signed input, which is what makes a captured delivery unusable
after five minutes — a signature over the body alone stays valid forever.
Verification uses `hmac.compare_digest`; a naive `==` leaks the position of the
first differing byte through timing.

## Rate limits and cost

- Token-bucket rate limiting in Redis, refilled lazily inside one atomic Lua
  script so two concurrent requests cannot both see a stale count.
- Per-tenant daily token budgets and monthly cost caps, enforced before the call
  is made rather than after the invoice.
- The durable Postgres counter is authoritative, so a Redis flush cannot hand
  every tenant an unlimited budget.

## Supply chain

`pip-audit`, `npm audit`, `bandit`, `semgrep`, `trivy` and `gitleaks` run on every
pull request and weekly. The weekly run matters more: most vulnerabilities appear
in dependencies that have not changed, so a scan that only runs on diffs never
finds them.

## Reporting

This is a portfolio project with no production users. If you find something,
open an issue — there is nothing here worth a private disclosure process, and
saying so is more honest than a `security@` address nobody reads.
