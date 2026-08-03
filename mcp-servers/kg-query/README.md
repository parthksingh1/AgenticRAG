# kg-query-mcp

Read-only Cypher over a workspace's knowledge graph.

## Tools

- **`graph_query`** — Query the knowledge graph extracted from this workspace's documents.
- **`describe_graph`** — List the graph queries available and the parameters each one takes.  Deterministic; results are cached.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-kg-query
curl localhost:8105/healthz
curl localhost:8105/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:kg-query uvicorn server:app --app-dir kg-query --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8105/call   -H 'content-type: application/json'   -d '{"name": "graph_query", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest kg-query
```

## Design notes

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
