"use client";

import { useEffect, useState } from "react";
import { api, type CostSummary } from "@/lib/api";
import { cn } from "@/lib/cn";
import { BarChart, LineChart, ReliabilityDiagram } from "@/components/charts";

interface Tab {
  id: string;
  label: string;
}

const TABS: Tab[] = [
  { id: "cost", label: "Cost" },
  { id: "evals", label: "Evals" },
  { id: "calibration", label: "Judge calibration" },
  { id: "drift", label: "Drift" },
  { id: "failures", label: "Failures" },
  { id: "prompts", label: "Prompts" },
  { id: "keys", label: "API keys" },
  { id: "audit", label: "Audit log" },
];

export default function AdminPage() {
  const [tab, setTab] = useState("cost");

  return (
    <div className="flex h-dvh flex-col">
      <header className="sticky top-0 z-10 border-b border-line bg-bg/85 px-5 pt-4 backdrop-blur-md">
        <h1 className="text-[13.5px] font-semibold tracking-tight">Admin</h1>
        <div role="tablist" className="mt-3 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "whitespace-nowrap border-b-2 px-3 py-2.5 text-[13px] transition-colors",
                tab === t.id
                  ? "border-accent font-medium text-fg"
                  : "border-transparent text-muted hover:text-fg",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-5">
        <div className="mx-auto w-full max-w-5xl">
          {tab === "cost" ? <CostPanel /> : <ApiPanel tab={tab} />}
        </div>
      </div>
    </div>
  );
}

function CostPanel() {
  const [data, setData] = useState<CostSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .cost(30)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <Message text={error} tone="error" />;
  if (!data) return <Message text="Loading…" />;

  return (
    <>
      <div className="grid gap-3 sm:grid-cols-4">
        <Card label="Spend (30d)" value={`$${data.total_cost_usd.toFixed(2)}`} />
        <Card label="Tokens" value={data.total_tokens.toLocaleString()} />
        <Card label="Requests" value={data.requests.toLocaleString()} />
        <Card
          label="Daily budget used"
          value={`${(data.budget_fraction_used * 100).toFixed(0)}%`}
          tone={data.budget_fraction_used > 0.9 ? "warn" : undefined}
        />
      </div>

      {data.anomalies.length > 0 && (
        <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/[0.07] px-3.5 py-3 text-sm">
          <p className="font-medium text-amber-500">Cost anomalies</p>
          <p className="mt-0.5 text-xs text-muted">
            Days more than 3σ above the mean. Computed the same way as the
            Prometheus alert, so the dashboard and the pager cannot disagree.
          </p>
          <p className="mt-1 font-mono text-xs">{data.anomalies.join(", ")}</p>
        </div>
      )}

      {data.by_day.length > 1 && (
        <>
          <h2 className="mb-2 mt-7 text-[13.5px] font-semibold tracking-tight">Daily spend</h2>
          <BarChart
            points={data.by_day.map((d) => ({
              label: d.usage_date.slice(5),
              value: d.cost_usd,
            }))}
            format={(v) => `$${v.toFixed(3)}`}
          />
        </>
      )}

      <h2 className="mb-2 mt-7 text-[13.5px] font-semibold tracking-tight">By model</h2>
      {Object.keys(data.by_model).length === 0 ? (
        <Message text="No usage recorded yet." />
      ) : (
        <table className="w-full text-[13px]">
          <thead className="text-left text-[11px] uppercase tracking-[0.05em] text-muted">
            <tr className="border-b border-line">
              <th className="py-2 font-medium">Model</th>
              <th className="py-2 text-right font-medium">Spend</th>
              <th className="py-2 text-right font-medium">Share</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.by_model)
              .sort(([, a], [, b]) => b - a)
              .map(([model, cost]) => (
                <tr key={model} className="border-b border-line">
                  <td className="py-2 font-mono text-xs">{model}</td>
                  <td className="py-2 text-right tabular-nums">${cost.toFixed(4)}</td>
                  <td className="py-2 text-right tabular-nums text-muted">
                    {data.total_cost_usd > 0
                      ? `${((cost / data.total_cost_usd) * 100).toFixed(0)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </>
  );
}

/**
 * Module scope, not component scope, and that placement is load-bearing.
 *
 * Built inside the component this is a fresh object on every render, so the
 * `config` derived from it is a new reference every time. It is in the effect's
 * dependency list, so the effect re-ran on every render, set rows back to null,
 * and re-fetched — a loop that never escaped "Loading…". Hoisting it makes the
 * reference stable.
 */
const ENDPOINTS: Record<string, { path: string; empty: string }> = {
  evals: { path: "/admin/evals/runs", empty: "No eval runs recorded. Run `python -m evals.run`." },
  calibration: {
    path: "/admin/evals/calibration",
    empty: "No calibration yet. It is computed weekly from human labels.",
  },
  drift: { path: "/admin/drift", empty: "No drift snapshots yet. They accrue nightly." },
  failures: { path: "/admin/failures", empty: "No thumbs-down feedback to triage." },
  prompts: { path: "/admin/prompts", empty: "No prompts loaded." },
  keys: { path: "/admin/api-keys", empty: "No API keys." },
  audit: { path: "/admin/audit", empty: "No audit entries." },
};

/**
 * The remaining admin tabs, each backed by its own endpoint.
 *
 * Rendered generically because the panels differ only in which endpoint they
 * read and which columns they show; eight near-identical components would drift
 * apart within a month.
 */
function ApiPanel({ tab }: { tab: string }) {
  const config = ENDPOINTS[tab];
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!config) return;
    setRows(null);
    setError(null);
    api
      .adminPanel(config.path)
      .then(setRows)
      .catch((e: Error) => setError(e.message));
  }, [tab, config]);

  if (!config) return <Message text="Unknown panel." tone="error" />;
  if (error) return <Message text={error} tone="error" />;
  if (rows === null) return <Message text="Loading…" />;
  if (rows.length === 0) return <Message text={config.empty} />;

  const columns = Object.keys(rows[0] ?? {}).filter(
    (key) => !["before", "after", "reliability_bins", "histogram"].includes(key),
  );

  return (
    <>
      <PanelChart tab={tab} rows={rows} />
      <div className="surface overflow-x-auto px-4 py-1">
      <table className="w-full text-[13px]">
        <thead className="text-left text-[11px] uppercase tracking-[0.05em] text-muted">
          <tr className="border-b border-line">
            {columns.map((c) => (
              <th key={c} className="whitespace-nowrap py-2 pr-4 font-medium">
                {c.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-line align-top">
              {columns.map((c) => (
                <td key={c} className="max-w-xs truncate py-2 pr-4 tabular-nums">
                  {format(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </>
  );
}

/**
 * The chart above a panel, where one is meaningful.
 *
 * A table of eval runs answers "what was the number"; the chart answers "which
 * way is it moving, and how close is it to the floor" — which is the question
 * anyone actually opens this page to ask. Rendered from the same rows as the
 * table, so the two cannot disagree.
 */
function PanelChart({ tab, rows }: { tab: string; rows: Record<string, unknown>[] }) {
  if (tab === "evals") {
    // Oldest first: a trend read right-to-left is a trend read wrong.
    const points = [...rows]
      .reverse()
      .filter((r) => typeof r.groundedness === "number")
      .map((r) => ({
        label: String(r.run_id ?? "").replace(/^run-/, ""),
        value: r.groundedness as number,
      }));
    if (points.length < 2) return null;

    return (
      <section className="surface mb-6 p-5">
        <h3 className="text-[13.5px] font-semibold tracking-tight">Groundedness by run</h3>
        <p className="mt-0.5 text-xs text-muted">
          The dashed line is the absolute floor in the gate. A delta-only gate would let
          this ratchet downward one acceptable drop at a time.
        </p>
        <div className="mt-3">
          <LineChart points={points} floor={0.88} />
        </div>
      </section>
    );
  }

  if (tab === "calibration") {
    const judges = rows.filter((r) => Array.isArray(r.reliability_bins));
    if (judges.length === 0) return null;

    return (
      <section className="surface mb-6 p-5">
        <h3 className="text-[13.5px] font-semibold tracking-tight">Reliability</h3>
        <p className="mt-0.5 text-xs text-muted">
          Stated confidence against observed accuracy. The dashed diagonal is perfect
          calibration; a curve below it is a judge claiming more certainty than it earns,
          which is why its vote is weighted <code>1 / (1 + ECE)</code>.
        </p>
        <div className="mt-3 flex flex-wrap gap-8">
          {judges.map((j) => (
            <figure key={String(j.judge)} className="m-0">
              <ReliabilityDiagram
                bins={j.reliability_bins as { confidence: number; accuracy: number; n: number }[]}
                label={String(j.judge)}
              />
              <figcaption className="mt-1 text-center text-xs text-muted">
                {String(j.judge)} · ECE {Number(j.ece).toFixed(3)}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    );
  }

  return null;
}

function format(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function Card({ label, value, tone }: { label: string; value: string; tone?: "warn" }) {
  return (
    <div className="surface p-4 transition-shadow hover:shadow-md">
      <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted">{label}</p>
      <p
        className={cn(
          "mt-1.5 text-[26px] font-semibold leading-none tracking-[-0.02em] tabular-nums",
          tone === "warn" && "text-amber-500",
        )}
      >
        {value}
      </p>
    </div>
  );
}

function Message({ text, tone }: { text: string; tone?: "error" }) {
  return (
    <p className={cn("py-6 text-sm text-muted", tone === "error" && "text-red-500")}>{text}</p>
  );
}
