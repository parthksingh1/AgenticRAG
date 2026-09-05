"use client";

import { useState } from "react";
import { Play } from "lucide-react";
import { api, type SearchResponse } from "@/lib/api";
import { cn } from "@/lib/cn";

const STRATEGIES = [
  { id: "dense", label: "Dense (pgvector)" },
  { id: "sparse", label: "BM25" },
  { id: "hybrid_rrf", label: "RRF fusion" },
  { id: "rerank", label: "Cross-encoder rerank" },
  { id: "hyde", label: "HyDE" },
  { id: "multi_query", label: "Multi-query" },
  { id: "corrective", label: "Corrective RAG" },
  { id: "graph", label: "GraphRAG" },
] as const;

/**
 * Side-by-side retrieval comparison.
 *
 * The point of running two configurations against the same query at the same
 * time is that "does HyDE help our corpus" gets answered with evidence instead
 * of an opinion. Sequential runs against a changing index are not comparable.
 */
export default function PlaygroundPage() {
  const [query, setQuery] = useState("What is the carry-over limit for annual leave?");
  const [left, setLeft] = useState<string[]>(["dense"]);
  const [right, setRight] = useState<string[]>(["hybrid_rrf", "rerank"]);
  const [results, setResults] = useState<{ a?: SearchResponse; b?: SearchResponse }>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const [a, b] = await Promise.all([
        api.search({ query, strategies: left, top_k: 8 }),
        api.search({ query, strategies: right, top_k: 8 }),
      ]);
      setResults({ a, b });
    } catch (e) {
      setError(e instanceof Error ? e.message : "The comparison failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="sticky top-0 z-10 border-b border-line bg-bg/85 px-5 py-3 backdrop-blur-md">
        <h1 className="text-[13.5px] font-semibold tracking-tight">Retrieval playground</h1>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-5">
        <div className="mx-auto w-full max-w-5xl">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void run()}
              aria-label="Query to compare"
              className="flex-1 surface px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              type="button"
              onClick={() => void run()}
              disabled={busy}
              className="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              <Play size={14} aria-hidden /> {busy ? "Running…" : "Compare"}
            </button>
          </div>

          {error && (
            <p className="mt-3 rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-sm text-red-500">
              {error}
            </p>
          )}

          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <Column
              title="A"
              selected={left}
              onToggle={setLeft}
              result={results.a}
              other={results.b}
            />
            <Column
              title="B"
              selected={right}
              onToggle={setRight}
              result={results.b}
              other={results.a}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function Column({
  title,
  selected,
  onToggle,
  result,
  other,
}: {
  title: string;
  selected: string[];
  onToggle: (next: string[]) => void;
  result?: SearchResponse;
  other?: SearchResponse;
}) {
  // Chunks the other side did not return. This is the whole comparison: the
  // difference between two strategies is what each one found that the other
  // missed, not their overlap.
  const otherIds = new Set(other?.results.map((r) => r.chunk_id) ?? []);

  return (
    <section className="rounded-xl border border-line">
      <header className="border-b border-line px-4 py-3">
        <h2 className="text-sm font-medium">Configuration {title}</h2>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {STRATEGIES.map((s) => {
            const on = selected.includes(s.id);
            return (
              <button
                key={s.id}
                type="button"
                aria-pressed={on}
                onClick={() =>
                  onToggle(on ? selected.filter((x) => x !== s.id) : [...selected, s.id])
                }
                className={cn(
                  "rounded-full border px-2.5 py-1 text-xs transition-colors",
                  on
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-line text-muted hover:text-fg",
                )}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      </header>

      {result ? (
        <>
          <dl className="flex flex-wrap gap-x-5 gap-y-1 border-b border-line px-4 py-2 text-xs text-muted">
            <Stat label="strategy" value={result.strategy} />
            <Stat label="latency" value={`${result.latency_ms}ms`} />
            <Stat label="hits" value={String(result.results.length)} />
            {result.expanded && <Stat label="expanded" value="yes" />}
            {result.crag_verdict && <Stat label="CRAG" value={result.crag_verdict} />}
            {result.web_fallback_used && <Stat label="web fallback" value="used" />}
          </dl>

          <ol className="divide-y divide-line">
            {result.results.map((hit, i) => {
              const unique = other != null && !otherIds.has(hit.chunk_id);
              return (
                <li
                  key={hit.chunk_id}
                  className={cn("px-4 py-3", unique && "border-l-2 border-l-accent bg-accent/5")}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <p className="truncate text-xs font-medium">
                      <span className="mr-1.5 tabular-nums text-muted">{i + 1}.</span>
                      {hit.document_title}
                    </p>
                    <span className="shrink-0 tabular-nums text-xs text-muted">
                      {hit.score.toFixed(4)}
                      {hit.rerank_score != null ? ` → ${hit.rerank_score.toFixed(4)}` : ""}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-3 text-xs text-muted">{hit.content}</p>
                  {unique && (
                    <p className="mt-1 text-[11px] font-medium text-accent">
                      only found by this configuration
                    </p>
                  )}
                </li>
              );
            })}
          </ol>
        </>
      ) : (
        <p className="px-4 py-8 text-center text-sm text-muted">Run a comparison</p>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-1">
      <dt>{label}</dt>
      <dd className="font-medium text-fg">{value}</dd>
    </div>
  );
}
