# ADR 0001 — Tenant isolation at the session, not the query

**Status:** accepted · **Date:** 2026-08-14

## Context

Every tenant-scoped table needs `WHERE tenant_id = ...` on every read and write.
Three ways to get it:

1. **Per-query filters.** Correct until somebody forgets, and the failure is
   silent — a leak looks like a working query returning slightly more rows.
2. **Postgres row-level security.** Genuinely enforced by the database, and the
   strongest option. It needs `SET LOCAL` on every checkout, works poorly with a
   pool that multiplexes connections, and pushes the policy into migrations where
   it cannot be unit-tested without a live Postgres.
3. **A session-level ORM listener.** Enforced in one place, testable on SQLite,
   and visible in the code a reviewer is already reading.

## Decision

A `do_orm_execute` listener appends `with_loader_criteria(tenant_id == current)`
to every query touching a `TenantScoped` model. A query with no tenant in context
**raises**. Escaping requires `system_session(reason=...)`, which logs at
WARNING.

## Consequences

**Good.** One place to audit. Adding a tenant-scoped model requires no new
filtering code. The isolation tests run in milliseconds on SQLite and again
against real Postgres. Every legitimate cross-tenant read is enumerable by
grepping for one function — there are four: the GDPR cascade, the nightly backup,
the drift sweep, and the eval harness.

**Bad.** It is application-level. Anyone with direct database access bypasses it
entirely, and raw `text()` SQL is not covered — those queries are individually
reviewed, and the GDPR cascade is the only one that uses them.

**Rejected: RLS.** Not because it is weaker — it is stronger — but because
pgbouncer-style pooling combined with per-request `SET LOCAL` is a well-known
source of leaks when a connection is returned mid-transaction. For a system this
size, an enforcement mechanism that is easy to test and hard to misconfigure
beats a stronger one that is easy to misconfigure. Revisit under a real
compliance requirement, where "the database enforces it" is the answer an auditor
needs to hear.

## Verification

`apps/api/tests/unit/test_tenant_isolation.py` walks every endpoint with two
tenants and asserts nothing crosses, including the failure mode where the tenant
context is missing entirely.
