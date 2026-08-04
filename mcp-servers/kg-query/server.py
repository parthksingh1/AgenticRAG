"""kg-query-mcp — read-only Cypher over a workspace's knowledge graph.

Answers the questions vector search structurally cannot: "which of our vendors
are also customers", "who reports to the person who signed this contract". Those
need traversal, and similarity has no notion of a path.

**Cypher is never assembled from model text.** The model chooses a *named query
shape* and supplies parameters; the shapes are written here and the parameters
are bound. Letting a model emit raw Cypher would be handing it read access to
every relationship in the graph, with ``MATCH (n) RETURN n`` one plausible
generation away — and Cypher has no equivalent of a read-only role to fall back
on. The named-shape design also makes the tool easier for a model to use
correctly, because the shapes describe what the graph can actually answer.

Every shape carries ``tenant_id`` as a bound parameter and filters on it in the
MATCH itself, so a traversal cannot walk from one tenant's subgraph into
another's through a shared node.

Example:
    >>> sorted(QUERY_SHAPES)[:2]
    ['connections', 'entity_documents']
"""

from __future__ import annotations

import os
import re
from typing import Any

from mcp_common.server import ToolResult, build_app, build_table, require, tool

VERSION = "1.0.0"

MAX_RESULTS = 50
MAX_HOPS = 3
QUERY_TIMEOUT_SECONDS = 10

#: The queries this tool can run. Each is a fixed, parameterised Cypher shape:
#: the model picks a name and supplies parameters, and never writes Cypher.
QUERY_SHAPES: dict[str, dict[str, Any]] = {
    "find_entity": {
        "description": "Find entities whose name matches a search term.",
        "parameters": ["term"],
        "cypher": """
            MATCH (e:Entity)
            WHERE e.tenant_id = $tenant_id AND toLower(e.name) CONTAINS toLower($term)
            RETURN e.name AS name, e.type AS type, e.description AS description
            ORDER BY size(e.name) ASC
            LIMIT $limit
        """,
    },
    "connections": {
        "description": "The entities directly related to a named entity, and how.",
        "parameters": ["name"],
        "cypher": """
            MATCH (e:Entity)-[r:RELATED]-(other:Entity)
            WHERE e.tenant_id = $tenant_id
              AND other.tenant_id = $tenant_id
              AND toLower(e.name) = toLower($name)
            RETURN other.name AS name, other.type AS type, r.type AS relation,
                   r.description AS detail
            LIMIT $limit
        """,
    },
    "path_between": {
        "description": "How two entities are connected, if they are.",
        "parameters": ["from_name", "to_name"],
        "cypher": """
            MATCH (a:Entity), (b:Entity)
            WHERE a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id
              AND toLower(a.name) = toLower($from_name)
              AND toLower(b.name) = toLower($to_name)
            MATCH path = shortestPath((a)-[:RELATED*1..%(hops)d]-(b))
            WHERE ALL(r IN relationships(path) WHERE r.tenant_id = $tenant_id)
            RETURN [n IN nodes(path) | n.name] AS entities,
                   [r IN relationships(path) | r.type] AS relations,
                   length(path) AS hops
            LIMIT $limit
        """,
    },
    "entity_documents": {
        "description": "The documents that mention a named entity.",
        "parameters": ["name"],
        "cypher": """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.tenant_id = $tenant_id
              AND c.tenant_id = $tenant_id
              AND toLower(e.name) = toLower($name)
            RETURN DISTINCT c.document_id AS document_id,
                   c.document_title AS document_title,
                   count(c) AS mentions
            ORDER BY mentions DESC
            LIMIT $limit
        """,
    },
    "entities_by_type": {
        "description": "All entities of a given type, e.g. Organisation or Person.",
        "parameters": ["type"],
        "cypher": """
            MATCH (e:Entity)
            WHERE e.tenant_id = $tenant_id AND toLower(e.type) = toLower($type)
            RETURN e.name AS name, e.description AS description
            ORDER BY e.name
            LIMIT $limit
        """,
    },
    "shared_connections": {
        "description": "Entities related to both of two named entities.",
        "parameters": ["first", "second"],
        "cypher": """
            MATCH (a:Entity)-[:RELATED]-(shared:Entity)-[:RELATED]-(b:Entity)
            WHERE a.tenant_id = $tenant_id
              AND b.tenant_id = $tenant_id
              AND shared.tenant_id = $tenant_id
              AND toLower(a.name) = toLower($first)
              AND toLower(b.name) = toLower($second)
            RETURN DISTINCT shared.name AS name, shared.type AS type
            LIMIT $limit
        """,
    },
}


def build_cypher(shape: str, *, hops: int = 2) -> str:
    """Render a query shape.

    ``hops`` is the only value interpolated, and it is bounded to an integer
    range first — Cypher does not allow a parameter in a variable-length pattern,
    so this is the one place a value reaches the query text, and it is coerced
    rather than trusted.

    Example:
        >>> "1..2" in build_cypher("path_between", hops=2)
        True
        >>> "1..3" in build_cypher("path_between", hops=99)
        True
    """
    template = QUERY_SHAPES[shape]["cypher"]
    bounded = min(max(int(hops), 1), MAX_HOPS)
    return template % {"hops": bounded} if "%(hops)d" in template else template


def validate_parameters(shape: str, parameters: dict[str, Any]) -> str | None:
    """Check that a shape's required parameters are present and reasonable.

    Returns an error message, or None when the parameters are acceptable.

    Example:
        >>> validate_parameters("connections", {"name": "ACME"}) is None
        True
        >>> validate_parameters("connections", {})
        "missing parameter 'name' for query 'connections'"
    """
    definition = QUERY_SHAPES.get(shape)
    if definition is None:
        available = ", ".join(sorted(QUERY_SHAPES))
        return f"unknown query {shape!r}; available: {available}"

    for name in definition["parameters"]:
        value = parameters.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return f"missing parameter {name!r} for query {shape!r}"
        if isinstance(value, str) and len(value) > 200:
            return f"parameter {name!r} is too long"
        # Parameters are bound, not interpolated, so this is belt and braces —
        # but a name containing Cypher syntax is a sign of an attempted
        # injection and worth refusing loudly.
        if isinstance(value, str) and re.search(r"[{}()\[\]$;]|--", value):
            return f"parameter {name!r} contains characters that are not allowed"
    return None


class GraphBackend:
    """Runs parameterised Cypher against Neo4j."""

    def __init__(
        self, uri: str | None = None, user: str | None = None, password: str | None = None
    ) -> None:
        """Configure the connection from the environment."""
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver: Any | None = None

    async def driver(self) -> Any:
        """Return the shared driver, opening it on first use."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    async def run(self, cypher: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a query in a read transaction.

        The read transaction is a second line of defence behind the fixed query
        shapes: even a shape edited to contain a write would be refused by the
        server.
        """
        driver = await self.driver()
        async with driver.session(default_access_mode="READ") as session:
            result = await session.run(cypher, parameters, timeout=QUERY_TIMEOUT_SECONDS)
            return [record.data() async for record in result]

    async def close(self) -> None:
        """Close the driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None


BACKEND = GraphBackend()


@tool(
    "graph_query",
    "Query the knowledge graph extracted from this workspace's documents. Choose "
    "a query by name and supply its parameters. Available queries: "
    + "; ".join(f"{name} ({d['description']})" for name, d in sorted(QUERY_SHAPES.items())),
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "enum": sorted(QUERY_SHAPES)},
            "parameters": {"type": "object", "description": "Parameters for the chosen query."},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
            "hops": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_HOPS,
                "description": "Traversal depth, for path_between only.",
            },
        },
        "required": ["query"],
    },
    deterministic=False,
    read_only=True,
)
async def graph_query_handler(arguments: dict[str, Any], tenant_id: str) -> ToolResult:
    """Run a named graph query."""
    shape = require(arguments, "query")
    parameters = dict(arguments.get("parameters") or {})

    error = validate_parameters(shape, parameters)
    if error:
        return ToolResult.failure(error)

    bound = {
        **parameters,
        "tenant_id": tenant_id,
        "limit": min(int(arguments.get("limit") or 20), MAX_RESULTS),
    }
    cypher = build_cypher(shape, hops=int(arguments.get("hops") or 2))

    try:
        rows = await BACKEND.run(cypher, bound)
    except Exception as exc:  # noqa: BLE001 - a failed query is a usable result
        return ToolResult.failure(f"graph query failed: {exc}")

    return ToolResult.success(
        {"query": shape, "results": rows, "count": len(rows)},
        summary=f"{len(rows)} result{'s' if len(rows) != 1 else ''} from {shape}",
    )


@tool(
    "describe_graph",
    "List the graph queries available and the parameters each one takes.",
    {"type": "object", "properties": {}},
    deterministic=True,
    read_only=True,
)
async def describe_graph_handler(_arguments: dict[str, Any], _tenant_id: str) -> ToolResult:
    """Describe the available query shapes."""
    described = {
        name: {"description": d["description"], "parameters": d["parameters"]}
        for name, d in sorted(QUERY_SHAPES.items())
    }
    return ToolResult.success({"queries": described}, summary=f"{len(described)} query shapes")


TOOLS = build_table(graph_query_handler, describe_graph_handler)


async def check_graph() -> list[str]:
    """Report the graph as degraded when Neo4j is unreachable."""
    try:
        driver = await BACKEND.driver()
        await driver.verify_connectivity()
    except Exception:  # noqa: BLE001 - a degraded dependency, not a crash
        return ["neo4j"]
    return []


app = build_app(
    name="kg-query",
    version=VERSION,
    tools=TOOLS,
    description=__doc__ or "",
    health_check=check_graph,
)
