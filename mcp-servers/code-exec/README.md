# code-exec-mcp

Sandboxed Python for analysis of retrieved tabular data.

## Tools

- **`run_python`** — Execute a short Python snippet in a sandbox to analyse data.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-code-exec
curl localhost:8106/healthz
curl localhost:8106/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:code-exec uvicorn server:app --app-dir code-exec --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8106/call   -H 'content-type: application/json'   -d '{"name": "run_python", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest code-exec
```

## Design notes

This is the highest-risk server in the system, so its design is defence in depth
rather than one mechanism. The code it runs is written by a model, which may in
turn be repeating text from a document an attacker uploaded, so it is treated as
hostile input at every layer:

1. **RestrictedPython compiles it**, not :func:`compile`. Dunder attribute
   access, imports and the statements that reach interpreter internals are
   rejected at compile time.
2. **The namespace is an allowlist.** ``__builtins__`` is replaced entirely.
3. **It runs in a subprocess that can be killed**, which is the only mechanism
   here that actually stops a runaway loop — a thread cannot be cancelled, so a
   thread-based timeout leaks a spinning core forever.
4. **An address-space limit** where the platform provides one, so an allocation
   bomb kills one worker rather than the server.
5. **No network, no filesystem, no environment.**

The honest position is that RestrictedPython is a hardening layer, not a security
boundary. In production this server runs in its own container with no network
egress and a read-only root filesystem; see the Dockerfile and docs/SECURITY.md.
The layers here shrink the blast radius, and the container is what contains it.

Results are never cached: the tool is not deterministic and may be given
different data on each call.
