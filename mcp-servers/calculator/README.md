# calculator-mcp

Deterministic arithmetic, units and dates.

## Tools

- **`calculate`** — Evaluate a mathematical expression exactly.  Deterministic; results are cached.
- **`convert_units`** — Convert a value between physical units, e.  Deterministic; results are cached.
- **`date_difference`** — The interval between two dates, in days, weeks, months or years.  Deterministic; results are cached.
- **`date_shift`** — Add or subtract a duration from a date.  Deterministic; results are cached.
- **`percentage`** — Percentage change between two values, a percentage of a value, or a value increased or decreased by a percentage.  Deterministic; results are cached.

## Running it

```bash
# From the repository root
docker compose --profile mcp up -d mcp-calculator
curl localhost:8104/healthz
curl localhost:8104/tools | jq

# Directly, for development
cd mcp-servers
PYTHONPATH=common:calculator uvicorn server:app --app-dir calculator --port 8080
```

## Calling a tool

```bash
curl -X POST localhost:8104/call   -H 'content-type: application/json'   -d '{"name": "calculate", "arguments": {}, "tenant_id": "ten_demo"}'
```

A tool that fails returns HTTP 200 with `ok: false`. A non-200 means the server
itself failed, which is a different problem: the client retries the second and
shows the first to the model.

## Tests

```bash
cd mcp-servers && pytest calculator
```

## Design notes

Models are unreliable at arithmetic in a way that is hard to notice: the answer
looks right, has the right number of digits, and is wrong. This server exists so
the numbers in an answer come from something that cannot be plausibly wrong.

Everything here is deterministic and side-effect free, which is why the client
caches its results for a day.

Expressions are evaluated with SymPy, never with :func:`eval`. That matters more
than it might appear: an expression arrives from a model that may itself be
repeating text from a retrieved document, so it is untrusted input, and
``eval("__import__('os').system(...)")`` is a working remote code execution
against the tool server.

Tools:
    calculate       — evaluate a mathematical expression exactly.
    convert_units   — convert between physical units.
    date_difference — the interval between two dates.
    date_shift      — add or subtract a duration from a date.
    percentage      — percentage change, of, and increase/decrease.
