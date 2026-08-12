# @agrag/sdk

Zero-dependency TypeScript client for AgenticRAG. Uses `fetch`, so the same
build runs in Node 18+, Deno, Bun, browsers and edge runtimes.

```bash
npm install @agrag/sdk
```

```ts
import { AgRag, isRefusal } from "@agrag/sdk";

const client = new AgRag({ apiKey: process.env.AGRAG_API_KEY! });

const answer = await client.ask("What is the carry-over limit for annual leave?");
if (isRefusal(answer)) {
  console.log("The corpus does not cover this.");
} else {
  console.log(answer.content);
  for (const c of answer.citations) console.log(`  [${c.index}] ${c.document_title}`);
}
```

## Streaming

```ts
for await (const token of client.stream("Summarise the leave policy")) {
  process.stdout.write(token);
}
```

Pass an `AbortSignal` to stop early:

```ts
const controller = new AbortController();
setTimeout(() => controller.abort(), 5000);
for await (const token of client.stream("...", { signal: controller.signal })) { }
```

## Errors

`AuthenticationError` (401/403), `RateLimitError` (429, with `retryAfterSeconds`)
and `APIError` (everything else, with `status` and `code`).

Transient failures retry with exponential backoff and full jitter, honouring
`Retry-After` when the server sends it.
