# web-fetch-mcp

Allowlisted web fetching, converted to markdown.

## Tools

- **`fetch_url`** — Fetch a web page from an allowlisted domain and return its main content as markdown.
- **`list_allowed_domains`** — List the domains this deployment is permitted to fetch from.  Deterministic; results are cached.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-web-fetch
curl localhost:8103/healthz
curl localhost:8103/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:web-fetch uvicorn server:app --app-dir web-fetch --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8103/call   -H 'content-type: application/json'   -d '{"name": "fetch_url", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest web-fetch
```

## Design notes

The corrective-RAG fallback needs the open web when a tenant's documents cannot
answer a question. Fetching arbitrary URLs on behalf of a model is a
server-side request forgery primitive, so the constraint is not "be careful"
but "the model cannot express a request that reaches anywhere interesting".

Four layers:

* **A domain allowlist**, configured per deployment. Not a denylist: the set of
  interesting internal hostnames is unbounded and grows with the infrastructure.
* **DNS resolution before connecting**, with every resolved address checked
  against the private, loopback, link-local and reserved ranges. Checking the
  hostname alone is defeated by a public name with an ``A`` record pointing at
  ``169.254.169.254`` — the cloud metadata endpoint — which is the standard SSRF
  escalation.
* **Redirects followed manually**, revalidating each hop. Following redirects
  automatically lets an allowlisted domain bounce the request anywhere.
* **robots.txt respected**, and a size and time ceiling on the response.
