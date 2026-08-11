# agrag — Python client for AgenticRAG

```bash
pip install agrag
```

```python
from agrag import AgRag

client = AgRag(api_key="agr_...")
answer = client.ask("What is the carry-over limit for annual leave?")

print(answer.content)
for citation in answer.citations:
    print(f"  [{citation.index}] {citation.document_title} p.{citation.page_number}")
```

## Check whether it refused

A refusal is a successful 200 response. Treating it as an answer is how
"I don't have that information" ends up quoted in a report.

```python
if answer.refused:
    print("The corpus does not cover this.")
```

## Async

The async client is the real implementation; the sync one wraps it.

```python
from agrag import AsyncAgRag

async with AsyncAgRag(api_key="agr_...") as client:
    async for token in client.stream("Summarise the leave policy"):
        print(token, end="", flush=True)
```

## Errors

| Exception | When |
|---|---|
| `AuthenticationError` | 401 or 403 — bad key or missing scope |
| `RateLimitError` | 429 — carries `retry_after` when the server sends it |
| `APIError` | anything else; carries `status` and `code` |

Transient failures (408, 429, 5xx) retry automatically with exponential backoff
and **full jitter**. Without jitter, every client that failed at the same moment
retries at the same moment, and a service recovering from a blip is knocked over
by its own clients.

## OpenAI compatibility

If you would rather not add a dependency, the API speaks the OpenAI wire format:

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.agrag.dev/v1", api_key="agr_...")
client.chat.completions.create(model="claude-sonnet-5", messages=[...])
```

Citations arrive in a non-standard `citations` field, which OpenAI SDKs ignore.
