"""sql-analytics-mcp — read-only SQL over a tenant's metadata schema.

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

Example:
    >>> validate("SELECT count(*) FROM documents").ok
    True
    >>> validate("DROP TABLE documents").ok
    False
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mcp_common.server import ToolResult, build_app, build_table, require, tool

VERSION = "1.0.0"

#: Views the agent may query. Each is defined in the migration as a
#: tenant-filtered view over the real table, never the table itself.
ALLOWED_TABLES = frozenset(
    {
        "v_documents",
        "v_chunks",
        "v_conversations",
        "v_messages",
        "v_citations",
        "v_feedback",
        "v_usage",
        "v_ingestion_jobs",
        "v_sales",
    }
)

#: Keywords that must never appear as keyword tokens. Checked structurally, so a
#: string literal or an identifier containing one of these is fine.
FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE",
        "GRANT", "REVOKE", "COMMIT", "ROLLBACK", "SAVEPOINT", "VACUUM",
        "COPY", "CALL", "EXECUTE", "PREPARE", "DEALLOCATE", "LISTEN", "NOTIFY",
        "LOCK", "SET", "RESET", "DISCARD", "REINDEX", "CLUSTER", "ANALYZE",
        "MERGE", "UPSERT", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
    }
)  # fmt: skip

#: Functions that read the filesystem, run commands or reach the network.
FORBIDDEN_FUNCTIONS = frozenset(
    {"pg_read_file", "pg_read_binary_file", "pg_ls_dir", "lo_import", "lo_export", "dblink"}
)

MAX_ROWS = 1000
STATEMENT_TIMEOUT_MS = 5000
MAX_QUERY_LENGTH = 4000


@dataclass(frozen=True, slots=True)
class Validation:
    """The outcome of validating a query."""

    ok: bool
    reason: str | None = None
    tables: tuple[str, ...] = ()

    @classmethod
    def reject(cls, reason: str) -> Validation:
        """Build a rejection."""
        return cls(ok=False, reason=reason)


def validate(query: str) -> Validation:
    """Check that a query is a single, read-only SELECT over allowed views.

    Example:
        >>> validate("SELECT count(*) FROM v_documents").ok
        True
        >>> validate("SELECT 1; DROP TABLE v_documents").reason
        'only one statement is allowed per call'
        >>> validate("UPDATE v_documents SET title = 'x'").ok
        False
        >>> validate("SELECT * FROM pg_user").reason
        "query references tables that are not available: pg_user"
    """
    import sqlparse
    from sqlparse import tokens as token_types

    if not query.strip():
        return Validation.reject("query is empty")
    if len(query) > MAX_QUERY_LENGTH:
        return Validation.reject(f"query too long ({len(query)} chars, limit {MAX_QUERY_LENGTH})")

    statements = [s for s in sqlparse.parse(query) if str(s).strip(" \t\n;")]
    if not statements:
        return Validation.reject("query contains no statement")
    if len(statements) > 1:
        return Validation.reject("only one statement is allowed per call")

    statement = statements[0]
    kind = statement.get_type()
    if kind not in ("SELECT", "UNKNOWN"):
        return Validation.reject(f"only SELECT queries are allowed, got {kind}")

    normalised = str(statement).strip().upper()
    if not (normalised.startswith("SELECT") or normalised.startswith("WITH")):
        return Validation.reject("query must begin with SELECT or WITH")

    for token in statement.flatten():
        if token.ttype in token_types.Keyword and token.normalized.upper() in FORBIDDEN_KEYWORDS:
            return Validation.reject(f"{token.normalized.upper()} is not allowed")
        is_statement_keyword = (
            token.ttype in token_types.Keyword.DDL or token.ttype in token_types.Keyword.DML
        )
        if is_statement_keyword and token.normalized.upper() != "SELECT":
            return Validation.reject(f"{token.normalized.upper()} is not allowed")

    lowered = query.lower()
    banned = sorted(fn for fn in FORBIDDEN_FUNCTIONS if fn in lowered)
    if banned:
        return Validation.reject(f"function not allowed: {', '.join(banned)}")

    tables = extract_tables(query)
    unknown = sorted(t for t in tables if t not in ALLOWED_TABLES)
    if unknown:
        return Validation.reject(
            f"query references tables that are not available: {', '.join(unknown)}"
        )
    if not tables:
        return Validation.reject("query does not reference any known view")

    return Validation(ok=True, tables=tuple(sorted(tables)))


def extract_tables(query: str) -> set[str]:
    """Table names referenced after FROM or JOIN.

    Common table expressions are excluded: a ``WITH x AS (...)`` name is not a
    real table, and treating it as one would reject every legitimate CTE.

    Example:
        >>> sorted(extract_tables("SELECT * FROM v_documents d JOIN v_chunks c ON c.d = d.id"))
        ['v_chunks', 'v_documents']
        >>> extract_tables("WITH recent AS (SELECT 1 FROM v_documents) SELECT * FROM recent")
        {'v_documents'}
    """
    cte_names = {m.lower() for m in re.findall(r"\bWITH\s+(\w+)\s+AS\b", query, re.IGNORECASE)}
    cte_names |= {m.lower() for m in re.findall(r",\s*(\w+)\s+AS\s*\(", query, re.IGNORECASE)}

    found = {
        match.lower()
        for match in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", query, re.IGNORECASE)
    }
    return found - cte_names


def apply_row_limit(query: str, limit: int = MAX_ROWS) -> str:
    """Wrap the query so it cannot return more than ``limit`` rows.

    Wrapping rather than appending ``LIMIT``: appending breaks on a query that
    already has a LIMIT, on one ending in a semicolon, and on a UNION where the
    limit would bind to the last branch only.

    Example:
        >>> apply_row_limit("SELECT 1", limit=10)
        'SELECT * FROM (SELECT 1) AS agent_query LIMIT 10'
    """
    inner = query.strip().rstrip(";").strip()
    # The query reaching here has already passed `validate`, which proves it is a
    # single read-only SELECT over allowlisted views; `limit` is an int constant.
    return f"SELECT * FROM ({inner}) AS agent_query LIMIT {limit}"  # noqa: S608


class QueryExecutor:
    """Runs validated queries against a read-only connection.

    Injected rather than constructed here so the tests can drive the whole
    validation and formatting path without a database.
    """

    def __init__(self, dsn: str | None = None) -> None:
        """Store the connection string; connections are opened per query."""
        self._dsn = dsn

    async def run(self, query: str, tenant_id: str) -> list[dict[str, Any]]:
        """Execute a validated query with the tenant bound as a parameter.

        Raises:
            RuntimeError: when no database is configured.
        """
        if not self._dsn:
            msg = "no database configured for the SQL analytics server"
            raise RuntimeError(msg)

        import asyncpg

        connection = await asyncpg.connect(self._dsn)
        try:
            await connection.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
            await connection.execute("SET default_transaction_read_only = on")
            # The views read this setting, which is what scopes them to the
            # tenant without the query having to say so.
            await connection.execute("SELECT set_config('agrag.tenant_id', $1, true)", tenant_id)
            rows = await connection.fetch(apply_row_limit(query))
        finally:
            await connection.close()

        return [dict(row) for row in rows]


EXECUTOR = QueryExecutor()


@tool(
    "sql_query",
    "Run a read-only SQL SELECT over this workspace's metadata. Available views: "
    + ", ".join(sorted(ALLOWED_TABLES))
    + ". Rows are already scoped to the workspace, so no tenant filter is needed.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["query"],
    },
    deterministic=False,
    read_only=True,
)
async def sql_query_handler(arguments: dict[str, Any], tenant_id: str) -> ToolResult:
    """Validate and run a query."""
    query = require(arguments, "query")
    verdict = validate(query)
    if not verdict.ok:
        return ToolResult.failure(f"query rejected: {verdict.reason}")

    try:
        rows = await EXECUTOR.run(query, tenant_id)
    except Exception as exc:  # noqa: BLE001 - a failed query is a result the model can use
        return ToolResult.failure(f"query failed: {exc}")

    truncated = len(rows) >= MAX_ROWS
    return ToolResult.success(
        {"rows": rows, "row_count": len(rows), "truncated": truncated},
        summary=f"{len(rows)} row{'s' if len(rows) != 1 else ''} from "
        f"{', '.join(verdict.tables)}" + (" (truncated)" if truncated else ""),
        metadata={"tables": list(verdict.tables)},
    )


@tool(
    "describe_schema",
    "List the views available to query and their columns.",
    {"type": "object", "properties": {}},
    deterministic=True,
    read_only=True,
)
async def describe_schema_handler(_arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Describe the queryable views."""
    return ToolResult.success(
        {"views": SCHEMA_DESCRIPTION},
        summary=f"{len(SCHEMA_DESCRIPTION)} views available",
    )


#: Documented shape of each view. Kept alongside the allowlist so the two cannot
#: drift: a view the agent is told about but may not query is a confusing dead
#: end, and one it may query but is never told about is unusable.
SCHEMA_DESCRIPTION: dict[str, list[str]] = {
    "v_documents": ["id", "title", "filename", "status", "created_at", "effective_date", "tags"],
    "v_chunks": ["id", "document_id", "ordinal", "kind", "token_count", "page_number"],
    "v_conversations": ["id", "title", "created_at", "total_tokens", "total_cost_usd"],
    "v_messages": ["id", "conversation_id", "role", "created_at", "model", "intent", "ttft_ms"],
    "v_citations": ["id", "message_id", "document_id", "marker", "verified", "entailment_score"],
    "v_feedback": ["id", "message_id", "rating", "failure_mode", "created_at"],
    "v_usage": ["usage_date", "provider", "model", "operation", "prompt_tokens", "cost_usd"],
    "v_ingestion_jobs": ["id", "document_id", "status", "chunks_created", "cost_usd"],
    "v_sales": ["order_id", "region", "product", "quantity", "amount_usd", "ordered_at"],
}


TOOLS = build_table(sql_query_handler, describe_schema_handler)

app = build_app(
    name="sql-analytics",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
)
