# docs-search-mcp

Tenant-scoped search over a workspace's documents.

## Tools

- **`search_documents`** — Search this workspace's documents.
- **`list_documents`** — List the documents available in this workspace, most recent first.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-docs-search
curl localhost:8101/healthz
curl localhost:8101/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:docs-search uvicorn server:app --app-dir docs-search --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8101/call   -H 'content-type: application/json'   -d '{"name": "search_documents", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest docs-search
```

## Design notes

The tool the agent reaches for most. It exposes the same hybrid retrieval the
main graph uses, but as a *tool* the model can call deliberately — which matters
for multi-hop questions, where the model needs to search several times with
different terms rather than once with whatever the user happened to type.

Tenant scoping is enforced here rather than trusted from the caller. This server
runs as a separate process and could in principle be reached by anything on the
network, so it takes the tenant from the request and applies it itself. The
alternative — trusting a caller-supplied filter — makes isolation depend on
every future caller being correct.

Filters are typed and validated. A model asked to search "documents from last
quarter tagged finance" will produce dates in half a dozen formats, so the tool
normalises them and reports what it actually applied, rather than silently
ignoring a filter it could not parse.
