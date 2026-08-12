/**
 * The AgenticRAG TypeScript client.
 *
 * ```ts
 * import { AgRag } from "@agrag/sdk";
 *
 * const client = new AgRag({ apiKey: "agr_..." });
 * const answer = await client.ask("What is the carry-over limit for annual leave?");
 * console.log(answer.content, answer.citations);
 * ```
 *
 * No dependencies. It uses `fetch`, which is standard in Node 18+, Deno, Bun,
 * every browser and every edge runtime — so the same file runs in all of them
 * rather than needing a build per target.
 */

/** Status codes worth retrying. 4xx other than 408/429 is the caller's problem. */
const RETRYABLE = new Set([408, 429, 500, 502, 503, 504]);

const DEFAULT_BASE_URL = "https://api.agrag.dev";
const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 8_000;

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  snippet: string;
  page_number?: number | null;
  section_path?: string[];
  score?: number;
}

export interface Answer {
  content: string;
  citations: Citation[];
  conversation_id?: string;
  message_id?: string;
  model?: string;
  stop_reason?: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  latency_ms: number;
}

export interface Doc {
  id: string;
  title: string;
  status: string;
  chunk_count: number;
  error_message?: string | null;
  tags: string[];
}

export interface SearchHit {
  chunk_id: string;
  document_id: string;
  document_title: string;
  content: string;
  score: number;
  rerank_score?: number | null;
  page_number?: number | null;
}

export interface AgRagOptions {
  apiKey: string;
  baseUrl?: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** Injectable for tests and for runtimes with a non-global fetch. */
  fetch?: typeof globalThis.fetch;
}

/** Base class for every error this client throws. */
export class AgRagError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The API key is missing, wrong, or lacks the required scope. */
export class AuthenticationError extends AgRagError {}

/** The rate limit or token budget is exhausted. */
export class RateLimitError extends AgRagError {
  constructor(
    message: string,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
  }
}

/** Any other API error. Carries the status so callers can branch on it. */
export class APIError extends AgRagError {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
  }
}

/**
 * Whether an answer is a refusal rather than an answer.
 *
 * A refusal is a successful 200 response. Treating it as an answer is how
 * "I don't have that information" ends up quoted in a report.
 */
export function isRefusal(answer: Answer): boolean {
  return answer.stop_reason === "guardrail_blocked" || answer.stop_reason === "refused";
}

export class AgRag {
  readonly #apiKey: string;
  readonly #baseUrl: string;
  readonly #timeoutMs: number;
  readonly #maxRetries: number;
  readonly #fetch: typeof globalThis.fetch;

  constructor(options: AgRagOptions) {
    if (!options.apiKey) {
      // Thrown at construction rather than on the first request, so the mistake
      // is reported where it was made.
      throw new AgRagError("an API key is required");
    }
    this.#apiKey = options.apiKey;
    this.#baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.#timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.#maxRetries = options.maxRetries ?? MAX_RETRIES;
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  /** Ask a question and wait for the complete answer. */
  async ask(
    message: string,
    options: { conversationId?: string; model?: string } = {},
  ): Promise<Answer> {
    return this.#request<Answer>("POST", "/api/chat", {
      message,
      conversation_id: options.conversationId,
      model: options.model,
    });
  }

  /** Retrieve without generating an answer. */
  async search(query: string, options: { topK?: number } = {}): Promise<SearchHit[]> {
    const body = await this.#request<{ results: SearchHit[] }>("POST", "/api/search", {
      query,
      top_k: options.topK ?? 5,
    });
    return body.results ?? [];
  }

  /** List the workspace's documents. */
  async documents(): Promise<Doc[]> {
    return this.#request<Doc[]>("GET", "/api/documents");
  }

  /**
   * Stream an answer token by token.
   *
   * Yields content tokens only. Citations arrive on a separate event; a
   * generator of strings cannot carry them without making the common case
   * awkward, so callers that need them should `ask` instead.
   */
  async *stream(
    message: string,
    options: { conversationId?: string; signal?: AbortSignal } = {},
  ): AsyncGenerator<string> {
    const response = await this.#fetch(`${this.#baseUrl}/api/chat/stream`, {
      method: "POST",
      headers: this.#headers(),
      body: JSON.stringify({
        message,
        conversation_id: options.conversationId,
        include_thinking: false,
      }),
      signal: options.signal,
    });

    if (!response.ok || !response.body) {
      await this.#throwFor(response);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line. What follows the last separator
      // is a partial frame and stays buffered until the rest arrives.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        let event = "message";
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data = line.slice(5).trim();
        }
        if (event !== "token" || !data) continue;
        try {
          const text = (JSON.parse(data) as { text?: string }).text;
          if (text) yield text;
        } catch {
          // One malformed frame must not discard an answer already half
          // delivered.
          continue;
        }
      }
    }
  }

  async #request<T>(method: string, path: string, body?: unknown): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.#maxRetries; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.#timeoutMs);

      let response: Response;
      try {
        response = await this.#fetch(`${this.#baseUrl}${path}`, {
          method,
          headers: this.#headers(),
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (error) {
        const aborted = error instanceof Error && error.name === "AbortError";
        lastError = new APIError(aborted ? "the request timed out" : "could not reach the API", 0);
        if (attempt < this.#maxRetries) {
          await sleep(backoffMs(attempt));
          continue;
        }
        throw lastError;
      } finally {
        clearTimeout(timer);
      }

      if (RETRYABLE.has(response.status) && attempt < this.#maxRetries) {
        // Honour Retry-After when present: backing off less than asked turns a
        // rate limit into a rate-limit loop.
        await sleep(retryAfterMs(response) ?? backoffMs(attempt));
        continue;
      }

      if (!response.ok) await this.#throwFor(response);

      const text = await response.text();
      return (text ? JSON.parse(text) : {}) as T;
    }

    throw lastError ?? new APIError("the request failed", 0);
  }

  #headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.#apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": "agrag-ts/0.1.0",
    };
  }

  async #throwFor(response: Response): Promise<never> {
    let body: Record<string, unknown> = {};
    try {
      body = (await response.json()) as Record<string, unknown>;
    } catch {
      // A non-JSON error body is still an error; the status carries the meaning.
    }

    const message =
      (body.detail as string) ?? (body.message as string) ?? `the API returned ${response.status}`;

    if (response.status === 401 || response.status === 403) {
      throw new AuthenticationError(message);
    }
    if (response.status === 429) {
      throw new RateLimitError(message, (retryAfterMs(response) ?? 0) / 1000 || undefined);
    }
    throw new APIError(message, response.status, body.code as string | undefined);
  }
}

/**
 * Exponential backoff with full jitter.
 *
 * Full jitter, not a fixed multiplier: without it every client that failed at
 * the same moment retries at the same moment, and a service recovering from a
 * blip is knocked over by its own clients.
 */
function backoffMs(attempt: number): number {
  return Math.random() * Math.min(BASE_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS);
}

function retryAfterMs(response: Response): number | undefined {
  const raw = response.headers.get("Retry-After");
  if (!raw) return undefined;
  const seconds = Number(raw);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
