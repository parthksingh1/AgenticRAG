"use client";

import { useEffect, useState } from "react";
import { api, type CostSummary } from "@/lib/api";
import { cn } from "@/lib/cn";

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
      <header className="border-b border-line px-5 pt-3">
        <h1 className="text-sm font-semibold">Admin</h1>
        <div role="tablist" className="mt-3 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              type="button"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "whitespace-nowrap border-b-2 px-3 py-2 text-sm transition-colors",
                tab === t.id
                  ? "border-accent text-fg"
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
        <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-sm">
          <p className="font-medium text-amber-500">Cost anomalies</p>
          <p className="mt-0.5 text-xs text-muted">
            Days more than 3σ above the mean. Computed the same way as the
            Prometheus alert, so the dashboard and the pager cannot disagree.
          </p>
          <p className="mt-1 font-mono text-xs">{data.anomalies.join(", ")}</p>
        </div>
      )}

      <h2 className="mt-6 text-sm font-medium">By model</h2>
      {Object.keys(data.by_model).length === 0 ? (
        <Message text="No usage recorded yet." />
      ) : (
        <table className="mt-2 w-full text-sm">
          <thead className="text-left text-xs text-muted">
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
 * The remaining admin tabs, each backed by its own endpoint.
 *
 * Rendered generically because the panels differ only in which endpoint they
 * read and which columns they show; eight near-identical components would drift
 * apart within a month.
 */
function ApiPanel({ tab }: { tab: string }) {
  const endpoints: Record<string, { path: string; empty: string }> = {
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

  const config = endpoints[tab];
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!config) return;
    setRows(null);
    setError(null);
    fetch(`/api/backend${config.path}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`The API returned ${r.status}.`);
        return r.json();
      })
      .then((body) => setRows(Array.isArray(body) ? body : [body]))
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
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted">
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
  );
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
    <div className="rounded-xl border border-line bg-card p-4">
      <p className={cn("text-2xl font-semibold tabular-nums", tone === "warn" && "text-amber-500")}>
        {value}
      </p>
      <p className="mt-0.5 text-xs text-muted">{label}</p>
    </div>
  );
}

function Message({ text, tone }: { text: string; tone?: "error" }) {
  return (
    <p className={cn("py-6 text-sm text-muted", tone === "error" && "text-red-500")}>{text}</p>
  );
}
