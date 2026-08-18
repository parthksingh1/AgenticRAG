/**
 * The typed client for the AgenticRAG API.
 *
 * Every response is validated with Zod before it reaches a component. That is
 * not ceremony: the backend and the frontend deploy independently, so a shape
 * that changed on one side must fail loudly at the boundary rather than as
 * `Cannot read properties of undefined` three components deep, in production,
 * with no indication of which field went missing.
 *
 * Requests go through Next's rewrite at `/api/backend/*` rather than straight to
 * the API host, which keeps the session cookie same-origin and removes CORS from
 * the deployment's list of things that can be misconfigured.
 */

import { z } from "zod";

const BASE = "/api/backend";

/** How long a non-streaming request may take before it is abandoned. */
const TIMEOUT_MS = 60_000;

export const citationSchema = z.object({
  index: z.number(),
  chunk_id: z.string(),
  document_id: z.string(),
  document_title: z.string(),
  snippet: z.string(),
  page_number: z.number().nullable().optional(),
  section_path: z.array(z.string()).default([]),
  score: z.number().optional(),
});
export type Citation = z.infer<typeof citationSchema>;

export const answerSchema = z.object({
  message_id: z.string(),
  conversation_id: z.string(),
  content: z.string(),
  model: z.string(),
  citations: z.array(citationSchema).default([]),
  thinking: z.array(z.string()).default([]),
  stop_reason: z.string().nullable().optional(),
  prompt_tokens: z.number().default(0),
  completion_tokens: z.number().default(0),
  cost_usd: z.number().default(0),
  latency_ms: z.number().default(0),
  cache_hit: z.string().nullable().optional(),
});
export type Answer = z.infer<typeof answerSchema>;

export const documentSchema = z.object({
  id: z.string(),
  title: z.string(),
  filename: z.string().nullable().optional(),
  mime_type: z.string().nullable().optional(),
  status: z.string(),
  error_message: z.string().nullable().optional(),
  created_at: z.string(),
  chunk_count: z.number().default(0),
  page_count: z.number().nullable().optional(),
  byte_size: z.number().nullable().optional(),
  tags: z.array(z.string()).default([]),
});
export type Doc = z.infer<typeof documentSchema>;

export const searchHitSchema = z.object({
  chunk_id: z.string(),
  document_id: z.string(),
  document_title: z.string(),
  content: z.string(),
  score: z.number(),
  rerank_score: z.number().nullable().optional(),
  page_number: z.number().nullable().optional(),
  section_path: z.array(z.string()).default([]),
});

export const searchResponseSchema = z.object({
  results: z.array(searchHitSchema).default([]),
  strategy: z.string(),
  latency_ms: z.number().default(0),
  source_latencies_ms: z.record(z.number()).default({}),
  expanded: z.boolean().default(false),
  crag_verdict: z.string().nullable().optional(),
  web_fallback_used: z.boolean().default(false),
});
export type SearchResponse = z.infer<typeof searchResponseSchema>;

export const costSchema = z.object({
  total_cost_usd: z.number().default(0),
  total_tokens: z.number().default(0),
  requests: z.number().default(0),
  by_model: z.record(z.number()).default({}),
  by_day: z
    .array(
      z.object({
        usage_date: z.string(),
        model: z.string(),
        cost_usd: z.number(),
        prompt_tokens: z.number(),
        completion_tokens: z.number(),
        requests: z.number(),
      }),
    )
    .default([]),
  budget_remaining_tokens: z.number().default(0),
  budget_fraction_used: z.number().default(0),
  anomalies: z.array(z.string()).default([]),
});
export type CostSummary = z.infer<typeof costSchema>;

/** An API error carrying the status, so a component can distinguish 401 from 500. */
export class ApiError extends Error {
  constructor(
    override readonly message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  init?: RequestInit,
): Promise<z.output<S>> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (error) {
    // An aborted request and a dead network look the same to fetch; the user
    // needs to be told which, because one is worth retrying immediately.
    const aborted = error instanceof Error && error.name === "AbortError";
    throw new ApiError(aborted ? "The request timed out." : "Could not reach the API.", 0);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      body.detail ?? body.message ?? `Request failed with ${response.status}`,
      response.status,
      body.code,
    );
  }

  const parsed = schema.safeParse(await response.json()) as z.SafeParseReturnType<
    unknown,
    z.output<S>
  >;
  if (!parsed.success) {
    // Deliberately loud. A silently-dropped field becomes a blank panel that
    // nobody can explain three sprints later.
    console.error(`Response from ${path} did not match its schema`, parsed.error.format());
    throw new ApiError("The API returned an unexpected shape.", 502);
  }
  return parsed.data;
}

export const api = {
  ask: (body: { message: string; conversation_id?: string; model?: string }) =>
    request("/chat", answerSchema, { method: "POST", body: JSON.stringify(body) }),

  documents: () => request("/documents", z.array(documentSchema)),

  document: (id: string) => request(`/documents/${id}`, documentSchema),

  deleteDocument: (id: string) =>
    request(`/documents/${id}`, z.unknown(), { method: "DELETE" }),

  search: (body: { query: string; top_k?: number; strategies?: string[] }) =>
    request("/search", searchResponseSchema, { method: "POST", body: JSON.stringify(body) }),

  cost: (days = 30) => request(`/admin/cost?days=${days}`, costSchema),
};

/**
 * Stream an answer token by token.
 *
 * Written against the SSE wire format directly rather than with EventSource,
 * because EventSource cannot send a POST body — and the question, the
 * conversation id and the model all belong in a body rather than a query string
 * where they would end up in every access log.
 *
 * Events arrive as `event: <name>` / `data: <json>` pairs. Anything unparseable
 * is skipped rather than throwing: one malformed frame must not discard an
 * answer that is halfway rendered.
 */
export async function* streamAnswer(
  body: { message: string; conversation_id?: string; include_thinking?: boolean },
  signal?: AbortSignal,
): AsyncGenerator<{ type: string; data: Record<string, unknown> }> {
  const response = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError("The stream could not be opened.", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. Anything after the last separator
    // is a partial frame and stays in the buffer until the rest arrives.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      const dataLines: string[] = [];

      for (const line of frame.split("\n")) {
        if (line.startsWith(":")) continue; // keepalive comment
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }

      if (dataLines.length === 0) continue;
      const payload = dataLines.join("\n");
      if (payload === "[DONE]") return;

      try {
        yield { type: event, data: JSON.parse(payload) };
      } catch {
        continue;
      }
    }
  }
}
