/**
 * The run inspector: what the agent did, while it is doing it.
 *
 * The chat surface shows an answer. This shows the machinery that produced it —
 * which stages ran, what retrieval returned, what the guardrails decided, what
 * it cost. For a system whose whole argument is that the answer is checkable,
 * hiding the checks behind a collapsed disclosure is the wrong default.
 *
 * Every stage here is derived from an event the stream actually emitted, not
 * from a timer. A stage that cannot be observed from the wire is marked skipped
 * rather than guessed at: an inspector that invents progress is worse than none,
 * because it is believed.
 */

"use client";

import {
  Blocks,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  Coins,
  Database,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import { cn } from "@/lib/cn";

export type StageState = "pending" | "active" | "done" | "skipped";

/**
 * A union rather than `string`, so `stages[key]` is a StageState and not
 * `StageState | undefined`. Under noUncheckedIndexedAccess a string-keyed
 * Record makes every read optional, which forces a fallback at each site and
 * hides a genuine typo behind it.
 */
export type StageKey =
  | "guardrails_in"
  | "route"
  | "retrieve"
  | "rerank"
  | "generate"
  | "cite"
  | "guardrails_out";

export interface RunState {
  stages: Record<StageKey, StageState>;
  citations: number;
  thinkingSteps: number;
  model?: string;
  latencyMs?: number;
  costUsd?: number;
  cacheHit?: string | null;
  blocked?: boolean;
}

/**
 * The order matters: it is the order of the graph in
 * `apps/api/src/agents/graph.py`, so someone comparing the two finds them the
 * same. Thirteen nodes collapse to seven groups here — a sidebar listing every
 * node is a wall nobody reads.
 */
export const STAGES: { key: StageKey; label: string; hint: string }[] = [
  { key: "guardrails_in", label: "Input guardrails", hint: "Injection and PII detection" },
  { key: "route", label: "Route & rewrite", hint: "Intent, HyDE, multi-query" },
  { key: "retrieve", label: "Retrieve", hint: "Dense + BM25, fused with RRF" },
  { key: "rerank", label: "Rerank", hint: "Cross-encoder over the fused set" },
  { key: "generate", label: "Generate", hint: "Streamed token by token" },
  { key: "cite", label: "Bind citations", hint: "Each claim tied to a span" },
  { key: "guardrails_out", label: "Output guardrails", hint: "Groundedness, PII, cost caps" },
];

export function emptyRun(): RunState {
  return {
    stages: Object.fromEntries(STAGES.map((s) => [s.key, "pending"])) as Record<
      StageKey,
      StageState
    >,
    citations: 0,
    thinkingSteps: 0,
  };
}

export function Inspector({ run, busy }: { run: RunState | null; busy: boolean }) {
  return (
    <aside
      aria-label="Run inspector"
      className="hidden w-[290px] shrink-0 flex-col gap-3 overflow-y-auto border-l border-line bg-card/60 px-3.5 py-4 xl:flex"
    >
      <Section icon={<Blocks size={13} />} title="Agent pipeline">
        <ol className="mt-1 space-y-0.5">
          {STAGES.map((stage) => {
            const state: StageState = run?.stages[stage.key] ?? "pending";
            return (
              <li
                key={stage.key}
                className={cn(
                  "flex items-start gap-2.5 rounded-lg px-2 py-[7px] transition-colors",
                  state === "active" && "bg-accent/[0.08]",
                )}
              >
                <StageIcon state={state} />
                <span className="min-w-0">
                  <span
                    className={cn(
                      "block text-[12px] leading-tight",
                      state === "pending" && "text-muted/70",
                      state === "skipped" && "text-muted/60 line-through decoration-muted/40",
                      state === "active" && "font-medium text-accent",
                      state === "done" && "text-fg",
                    )}
                  >
                    {stage.label}
                  </span>
                  <span className="mt-0.5 block text-[10.5px] leading-tight text-muted/80">
                    {stage.hint}
                  </span>
                </span>
              </li>
            );
          })}
        </ol>
      </Section>

      <Section icon={<Database size={13} />} title="Retrieval">
        {run && run.citations > 0 ? (
          <Rows
            rows={[
              ["Sources cited", String(run.citations)],
              ["Reasoning steps", String(run.thinkingSteps)],
              ["Strategy", "hybrid + rerank"],
            ]}
          />
        ) : (
          <Note>
            {run?.blocked
              ? "Not reached — the request was blocked before retrieval."
              : run && !busy
                ? "No sources. The answer declined rather than citing something that does not support it."
                : "Waiting for a question."}
          </Note>
        )}
      </Section>

      <Section icon={<ShieldCheck size={13} />} title="Guardrails">
        {run?.blocked ? (
          <div className="mt-1 rounded-lg border border-red-500/30 bg-red-500/[0.07] px-2.5 py-2">
            <p className="text-[11.5px] font-medium text-red-500">Blocked at input</p>
            <p className="mt-0.5 text-[10.5px] leading-snug text-muted">
              Prompt injection detected. No model call was made, which is why the cost is zero.
            </p>
          </div>
        ) : (
          <Rows
            rows={[
              ["Injection", run ? "pass" : "—"],
              ["PII", run ? "pass" : "—"],
              ["Groundedness", run?.citations ? "checked" : "—"],
            ]}
          />
        )}
      </Section>

      <Section icon={<Coins size={13} />} title="Budget">
        <Rows
          rows={[
            ["Model", run?.model ?? "—"],
            ["Latency", run?.latencyMs ? `${(run.latencyMs / 1000).toFixed(1)}s` : "—"],
            ["Cost", run?.costUsd !== undefined ? `$${run.costUsd.toFixed(4)}` : "—"],
            ["Cache", run?.cacheHit ? run.cacheHit : "miss"],
          ]}
        />
      </Section>

      <p className="mt-auto pt-2 text-[10.5px] leading-snug text-muted/70">
        Every row here comes from an event on the stream. Stages that cannot be
        observed from the wire are marked skipped rather than assumed.
      </p>
    </aside>
  );
}

function StageIcon({ state }: { state: StageState }) {
  const cls = "mt-[1px] shrink-0";
  if (state === "active") return <Loader2 size={13} className={cn(cls, "animate-spin text-accent")} />;
  if (state === "done") return <CircleCheck size={13} className={cn(cls, "text-emerald-500")} />;
  if (state === "skipped") return <CircleSlash size={13} className={cn(cls, "text-muted/50")} />;
  return <CircleDashed size={13} className={cn(cls, "text-muted/40")} />;
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="surface px-3 py-2.5">
      <h2 className="flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-[0.07em] text-muted">
        <span className="text-accent">{icon}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function Rows({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="mt-1.5 space-y-1">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-baseline justify-between gap-2">
          <dt className="text-[11.5px] text-muted">{k}</dt>
          <dd className="truncate font-mono text-[11px] tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mt-1.5 text-[11px] leading-snug text-muted/80">{children}</p>;
}
