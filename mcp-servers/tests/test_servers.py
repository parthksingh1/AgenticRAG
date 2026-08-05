"""MCP server tests.

Weighted towards what each server refuses. These processes accept input written
by a model that may be repeating text from a document an attacker uploaded, so
the interesting behaviour is the boundary, not the happy path.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp_common.server import Metrics, build_table, require, tool
from mcp_common.types import ToolResult

# ── Shared scaffolding ───────────────────────────────────────────────────────


def test_tool_result_failure_is_a_value_not_an_exception() -> None:
    """A failed tool is information the model can act on."""
    result = ToolResult.failure("division by zero")

    assert result.ok is False
    assert result.error == "division by zero"


def test_duplicate_tool_names_are_rejected() -> None:
    """Otherwise the active implementation depends on declaration order."""

    @tool("same", "First.", {"type": "object"})
    async def first(_arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
        return ToolResult.success(1)

    @tool("same", "Second.", {"type": "object"})
    async def second(_arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
        return ToolResult.success(2)

    with pytest.raises(ValueError, match="duplicate tool name"):
        build_table(first, second)


def test_missing_arguments_produce_a_readable_error() -> None:
    """Models omit arguments routinely; a KeyError traceback helps nobody."""
    with pytest.raises(ValueError, match="missing required argument 'query'"):
        require({}, "query")


def test_wrong_argument_type_names_the_expectation() -> None:
    """The message has to be actionable by the model that made the mistake."""
    with pytest.raises(ValueError, match="must be str"):
        require({"query": 42}, "query")


def test_metrics_render_in_prometheus_format() -> None:
    """The scrape config in infra/prometheus depends on this shape."""
    metrics = Metrics()
    metrics.record("calculate", ok=True, duration_ms=5.0)
    metrics.record("calculate", ok=False, duration_ms=3.0)

    rendered = metrics.render("calculator")

    assert 'mcp_tool_calls_total{server="calculator",tool="calculate"} 2' in rendered
    assert 'mcp_tool_failures_total{server="calculator",tool="calculate"} 1' in rendered


# ── HTTP surface ─────────────────────────────────────────────────────────────


def test_every_server_exposes_the_same_surface(calculator: ModuleType) -> None:
    """One client implementation serves every server, so the surface is fixed."""
    client = TestClient(calculator.app)

    health = client.get("/healthz")
    tools = client.get("/tools")

    assert health.status_code == 200
    assert health.json()["server"] == "calculator"
    assert tools.status_code == 200
    assert len(tools.json()) == 5


def test_a_failing_tool_returns_200_with_ok_false(calculator: ModuleType) -> None:
    """A 5xx would make the client retry something guaranteed to fail again."""
    client = TestClient(calculator.app)

    response = client.post(
        "/call",
        json={"name": "calculate", "arguments": {"expression": "1/0"}, "tenant_id": "t"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is False


def test_an_unknown_tool_lists_what_is_available(calculator: ModuleType) -> None:
    """The model can correct itself from the error without another round trip."""
    client = TestClient(calculator.app)

    response = client.post("/call", json={"name": "nope", "arguments": {}, "tenant_id": "t"})

    body = response.json()
    assert body["ok"] is False
    assert "calculate" in body["error"]


def test_a_raising_handler_becomes_a_failed_result(calculator: ModuleType) -> None:
    """An unhandled exception must not reach the client as a 500."""
    client = TestClient(calculator.app)

    response = client.post("/call", json={"name": "calculate", "arguments": {}, "tenant_id": "t"})

    assert response.status_code == 200
    assert "missing required argument" in response.json()["error"]


def test_a_call_without_a_tenant_is_rejected(calculator: ModuleType) -> None:
    """The servers scope their own data rather than trusting the caller."""
    client = TestClient(calculator.app)

    response = client.post("/call", json={"name": "calculate", "arguments": {}})

    assert response.status_code == 422


def test_metrics_endpoint_reflects_real_calls(calculator: ModuleType) -> None:
    """Instrumentation that nothing updates is worse than none."""
    client = TestClient(calculator.app)
    client.post(
        "/call",
        json={"name": "calculate", "arguments": {"expression": "1+1"}, "tenant_id": "t"},
    )

    assert "mcp_tool_calls_total" in client.get("/metrics").text


# ── calculator ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 2 * 3", "8"), ("(1200 * 1.08) / 12", "108.000000000000"), ("1/3", "1/3")],
)
async def test_arithmetic_is_exact(calculator: ModuleType, expression: str, expected: str) -> None:
    """The point of the tool: numbers that cannot be plausibly wrong."""
    result = await calculator.evaluate_expression(expression)

    assert result.ok is True
    assert result.content["result"] == expected


@pytest.mark.parametrize(
    "hostile",
    [
        '__import__("os").system("whoami")',
        "eval('1+1')",
        "open('/etc/passwd')",
        "system(1)",
        "().__class__",
    ],
)
async def test_expressions_that_are_not_arithmetic_are_refused(
    calculator: ModuleType, hostile: str
) -> None:
    """The expression arrives from a model repeating untrusted document text."""
    assert (await calculator.evaluate_expression(hostile)).ok is False


async def test_an_enormous_expression_is_refused(calculator: ModuleType) -> None:
    """`9**9**9` is trivial to write and will not finish."""
    assert (await calculator.evaluate_expression("x" * 600)).ok is False


async def test_single_letters_remain_valid_symbols(calculator: ModuleType) -> None:
    """Algebra must still work; only multi-character unknown names are refused."""
    assert (await calculator.evaluate_expression("2 * x + sqrt(4)")).ok is True


async def test_unit_conversion(calculator: ModuleType) -> None:
    """Unit handling belongs in a library, not in a model's head."""
    result = await calculator.convert(100, "mile", "km")

    assert result.ok is True
    assert result.content["value"] == pytest.approx(160.934, rel=1e-4)


async def test_incompatible_units_fail_cleanly(calculator: ModuleType) -> None:
    """The model should learn what it did wrong, not receive a stack trace."""
    assert (await calculator.convert(1, "kilogram", "metre")).ok is False


async def test_percentage_change_from_zero_is_refused(calculator: ModuleType) -> None:
    """It is undefined, and returning infinity would be quietly wrong."""
    assert (await calculator.percentage("change", 0, 5)).ok is False


async def test_date_arithmetic(calculator: ModuleType) -> None:
    """Date maths is another thing models get confidently wrong."""
    result = await calculator.date_difference("2024-01-01", "2024-03-15", "days")

    assert result.content["value"] == 74


# ── code-exec ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "data", "expected"),
    [
        ("result = sum([1, 2, 3])", None, 6),
        ("result = max(data)", [3, 1, 2], 3),
        ("result = statistics.mean(data)", [2, 4, 6], 4),
    ],
)
async def test_legitimate_snippets_run(
    code_exec: ModuleType, code: str, data: Any, expected: Any
) -> None:
    """The tool has to be useful, not merely safe."""
    result = await code_exec.execute(code, data)

    assert result.ok is True
    assert result.content["result"] == expected


@pytest.mark.parametrize(
    "escape",
    [
        '__import__("os").system("whoami")',
        "import os",
        'open("/etc/passwd").read()',
        "result = (1).__class__.__bases__[0].__subclasses__()",
        "result = globals()",
        'exec("result = 1")',
        "result = [].__class__",
    ],
)
async def test_sandbox_escapes_are_blocked(code_exec: ModuleType, escape: str) -> None:
    """Each of these is a documented way out of a naive Python sandbox."""
    result = await code_exec.execute(escape)

    assert result.ok is False


@pytest.mark.slow
async def test_an_infinite_loop_is_actually_terminated(code_exec: ModuleType) -> None:
    """The reason this runs in a subprocess: a thread could not be stopped, and
    the timeout would silently leak a spinning core for the life of the server."""
    result = await code_exec.execute("while True: pass")

    assert result.ok is False
    assert "terminated" in (result.error or "")


async def test_printed_output_is_captured(code_exec: ModuleType) -> None:
    """print is how a model narrates its working, and it is useful to see."""
    result = await code_exec.execute("print('hello')\nresult = 1")

    assert "hello" in result.content["stdout"]


async def test_oversized_code_is_refused(code_exec: ModuleType) -> None:
    """A ceiling before the sandbox, so the subprocess is not even started."""
    assert (await code_exec.execute("x = 1\n" * 20_000)).ok is False


# ── sql-analytics ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "SELECT count(*) FROM v_documents",
        "SELECT region, sum(amount_usd) FROM v_sales GROUP BY region",
        "WITH recent AS (SELECT * FROM v_documents) SELECT count(*) FROM recent",
        "SELECT d.title FROM v_documents d JOIN v_chunks c ON c.document_id = d.id",
    ],
)
def test_legitimate_queries_are_allowed(sql_analytics: ModuleType, query: str) -> None:
    """Validation has to permit the queries the tool exists to run."""
    assert sql_analytics.validate(query).ok is True


@pytest.mark.parametrize(
    "attack",
    [
        "DROP TABLE v_documents",
        "SELECT 1; DROP TABLE v_documents",
        "DELETE FROM v_documents",
        "UPDATE v_documents SET title = 'x'",
        "INSERT INTO v_documents VALUES (1)",
        "GRANT ALL ON v_documents TO public",
        "SELECT * FROM pg_user",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT 1 /*; DROP TABLE x */ FROM v_documents; TRUNCATE v_chunks",
        "select * from V_DOCUMENTS; delete from v_chunks",
    ],
)
def test_write_and_escape_attempts_are_rejected(sql_analytics: ModuleType, attack: str) -> None:
    """Structural validation, so casing, comments and stacking do not help."""
    assert sql_analytics.validate(attack).ok is False


def test_a_string_literal_containing_delete_is_fine(sql_analytics: ModuleType) -> None:
    """The exact case a grep-based validator gets wrong."""
    assert sql_analytics.validate("SELECT 'delete me' AS note FROM v_documents").ok is True


def test_common_table_expressions_are_not_mistaken_for_tables(
    sql_analytics: ModuleType,
) -> None:
    """Otherwise every legitimate CTE is rejected as an unknown table."""
    tables = sql_analytics.extract_tables(
        "WITH recent AS (SELECT 1 FROM v_documents) SELECT * FROM recent"
    )

    assert tables == {"v_documents"}


def test_the_row_limit_wraps_rather_than_appends(sql_analytics: ModuleType) -> None:
    """Appending breaks on an existing LIMIT, a trailing semicolon and UNION."""
    wrapped = sql_analytics.apply_row_limit("SELECT 1 LIMIT 5000", limit=10)

    assert wrapped.endswith("LIMIT 10")
    assert "agent_query" in wrapped


def test_the_documented_schema_matches_the_allowlist(sql_analytics: ModuleType) -> None:
    """A view the agent is told about but may not query is a dead end."""
    assert set(sql_analytics.SCHEMA_DESCRIPTION) == set(sql_analytics.ALLOWED_TABLES)


# ── web-fetch ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1"]
)
def test_private_and_metadata_addresses_are_recognised(web_fetch: ModuleType, address: str) -> None:
    """169.254.169.254 is the cloud metadata endpoint and the usual SSRF target."""
    assert web_fetch.is_private_address(address) is True


def test_public_addresses_are_allowed(web_fetch: ModuleType) -> None:
    """The check must not reject the entire internet."""
    assert web_fetch.is_private_address("93.184.216.34") is False


def test_an_unparseable_address_is_treated_as_private(web_fetch: ModuleType) -> None:
    """Failing open here would be an SSRF; failing closed costs one fetch."""
    assert web_fetch.is_private_address("not-an-ip") is True


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "ftp://arxiv.org/paper",
        "http://internal.corp/secrets",
    ],
)
def test_urls_off_the_allowlist_are_refused(web_fetch: ModuleType, url: str) -> None:
    """An allowlist, because the set of interesting internal hosts is unbounded."""
    assert web_fetch.validate_url(url, resolve=False).ok is False


def test_an_allowlisted_domain_passes(web_fetch: ModuleType) -> None:
    """The tool has to be able to fetch something."""
    assert web_fetch.validate_url("https://arxiv.org/abs/2401.00001", resolve=False).ok is True


def test_subdomains_of_allowlisted_domains_pass(web_fetch: ModuleType) -> None:
    """Blocking them would make most allowlists useless."""
    assert web_fetch.validate_url("https://export.arxiv.org/abs/1", resolve=False).ok is True


def test_a_lookalike_domain_does_not_pass(web_fetch: ModuleType) -> None:
    """`arxiv.org.evil.com` ends with a suffix that must not match."""
    assert web_fetch.validate_url("https://arxiv.org.evil.com/x", resolve=False).ok is False


def test_page_titles_are_extracted_and_unescaped(web_fetch: ModuleType) -> None:
    """Entities would otherwise show up literally in the citation panel."""
    assert web_fetch.extract_title("<html><title>A &amp; B</title></html>") == "A & B"


# ── docs-search ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2024-03-15", "2024-03-15"), ("15 March 2024", "2024-03-15"), ("nonsense", None)],
)
def test_dates_are_normalised_from_whatever_the_model_produced(
    docs_search: ModuleType, value: str, expected: str | None
) -> None:
    """Models emit dates in half a dozen formats."""
    assert docs_search.normalise_date(value) == expected


def test_long_snippets_are_truncated(docs_search: ModuleType) -> None:
    """Several full chunks per tool call would eat the context the answer needs."""
    formatted = docs_search.format_hit({"content": "x" * 5000, "chunk_id": "c1"})

    assert len(formatted["snippet"]) <= docs_search.MAX_SNIPPET_CHARS + 3


# ── kg-query ─────────────────────────────────────────────────────────────────


def test_every_query_shape_binds_the_tenant(kg_query: ModuleType) -> None:
    """A traversal must not walk from one tenant's subgraph into another's."""
    for name in kg_query.QUERY_SHAPES:
        assert "$tenant_id" in kg_query.build_cypher(name)


def test_the_model_cannot_supply_raw_cypher(kg_query: ModuleType) -> None:
    """It picks a named shape; `MATCH (n) RETURN n` is not expressible."""
    assert kg_query.validate_parameters("MATCH (n) RETURN n", {}) is not None


def test_missing_parameters_are_reported(kg_query: ModuleType) -> None:
    """The model can correct itself without a round trip to the database."""
    assert "missing parameter" in (kg_query.validate_parameters("connections", {}) or "")


def test_parameters_containing_cypher_syntax_are_refused(kg_query: ModuleType) -> None:
    """Parameters are bound, so this is belt and braces — and a signal worth refusing."""
    verdict = kg_query.validate_parameters("connections", {"name": "x'}) DETACH DELETE (n"})

    assert verdict is not None


def test_traversal_depth_is_bounded(kg_query: ModuleType) -> None:
    """Past a few hops the neighbourhood is most of the graph."""
    assert "1..3" in kg_query.build_cypher("path_between", hops=99)
    assert "1..1" in kg_query.build_cypher("path_between", hops=0)


def test_valid_parameters_are_accepted(kg_query: ModuleType) -> None:
    """Ordinary entity names must pass."""
    assert kg_query.validate_parameters("connections", {"name": "ACME Corp"}) is None
