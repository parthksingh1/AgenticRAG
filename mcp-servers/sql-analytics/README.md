# sql-analytics-mcp

Read-only SQL over a tenant's metadata schema.

## Tools

- **`sql_query`** — Run a read-only SQL SELECT over this workspace's metadata.
- **`describe_schema`** — List the views available to query and their columns.  Deterministic; results are cached.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-sql-analytics
curl localhost:8102/healthz
curl localhost:8102/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:sql-analytics uvicorn server:app --app-dir sql-analytics --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8102/call   -H 'content-type: application/json'   -d '{"name": "sql_query", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest sql-analytics
```

## Design notes

Lets the agent answer questions the documents cannot: "how many documents did we
upload last month", "which sources produce the most citations". The query is
written by a model from a user's question, so it is untrusted, and the whole
design follows from that.

**Why validation is structural, not textual.** The tempting implementation greps
for "DROP" and "DELETE". That fails on ``SELECT 'delete'``, on comments
(``SELECT 1 /*;DROP TABLE x*/``), on casing, on Unicode homoglyphs, and on
stacked statements. Instead the statement is *parsed* with sqlparse and the
resulting token stream is checked:

* exactly one statement, so ``; DROP TABLE`` cannot ride along;
* its type is ``SELECT`` (or a ``WITH`` whose body is a SELECT);
* no DDL or DML keyword appears as a *keyword token* anywhere in the tree;
* every table referenced is on the allowlist.

**Why tenant scoping is not left to the query.** Every allowed view is already
filtered to the calling tenant before the query runs, so a model that forgets a
``WHERE tenant_id = ...`` — or is talked into omitting it — still cannot see
another tenant's rows. Asking the query to enforce isolation would make isolation
depend on the model getting it right.

A row limit is injected, a statement timeout is set, and the connection is opened
with a read-only role. Four independent layers, because each of them has a way to
be wrong.
