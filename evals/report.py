"""Eval reports.

Every run writes three artefacts:

* **JSON** — the machine-readable record. The baseline, the PR comment and the
  nightly trend all read this.
* **HTML** — a single self-contained file with no external assets, so it can be
  opened from a CI artifact, attached to an issue, or published to Pages without
  a build step.
* **Markdown** — the pull-request comment, produced by :mod:`evals.gate`.

The HTML is built for one job: making a failure findable in under a minute. It
opens with the headline metrics, then the per-intent breakdown, then the failing
cases expanded and the passing ones collapsed — because the passing cases are
never what anyone came to read.

No fabricated numbers appear anywhere. Every figure in the report comes from the
run that produced it, and a metric that was not measured is shown as "—" rather
than as a zero.

Example:
    >>> "<!doctype html>" in render_html({"set_name": "golden", "results": []}).lower()
    True
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Metrics shown as headline cards, in the order a reader should see them.
#: Groundedness first: it is the one that says whether the answers were real.
HEADLINE = (
    ("groundedness", "Groundedness"),
    ("citation_precision", "Citation precision"),
    ("citation_recall", "Citation recall"),
    ("refusal_appropriateness", "Refusal accuracy"),
    ("injection_resistance", "Injection resistance"),
    ("pass_rate", "Pass rate"),
)

#: Below this a metric card is red, between this and GOOD it is amber.
WARN_AT = 0.7
GOOD_AT = 0.9


def write_all(
    run: dict[str, Any], *, directory: str | Path, gate: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Write the JSON and HTML reports for a run.

    Returns:
        A mapping of format to path, so the caller can print them or upload them.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{run.get('set_name', 'run')}-{stamp}"

    payload = {**run, "gate": gate} if gate else run
    json_path = target / f"{slug}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    html_path = target / f"{slug}.html"
    html_path.write_text(render_html(payload), encoding="utf-8")

    # A stable name alongside the timestamped one, so a CI step or a bookmark can
    # point at "the latest report" without knowing when it ran.
    latest = target / f"{run.get('set_name', 'run')}-latest.html"
    latest.write_text(render_html(payload), encoding="utf-8")

    return {"json": json_path, "html": html_path, "latest": latest}


def render_html(run: dict[str, Any]) -> str:
    """Render one run as a self-contained HTML page."""
    results = run.get("results", [])
    failures = [r for r in results if not r.get("passed")]
    passes = [r for r in results if r.get("passed")]
    disagreements = [r for r in results if r.get("judges_disagreed")]

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>Eval report · {html.escape(str(run.get('set_name', '')))}</title>",
            f"<style>{_CSS}</style></head><body>",
            _header(run),
            _cards(run),
            _gate_section(run.get("gate")),
            _intent_table(run.get("metrics_by_intent", {})),
            _failure_modes(failures),
            _cases_section("Failures", failures, open_by_default=True),
            _cases_section(
                f"Judge disagreements ({len(disagreements)}) — the human-label queue",
                disagreements,
                open_by_default=False,
            ),
            _cases_section(f"Passing ({len(passes)})", passes, open_by_default=False),
            _footer(),
            "</body></html>",
        ]
    )


def _header(run: dict[str, Any]) -> str:
    """The run's identity: which set, which model, which commit."""
    sha = run.get("git_sha") or "—"
    return f"""
<header>
  <h1>{html.escape(str(run.get("set_name", "eval")))} <span class="ver">{html.escape(str(run.get("set_version", "")))}</span></h1>
  <p class="meta">
    {run.get("case_count", 0)} cases ·
    <b class="ok">{run.get("passed", 0)} passed</b> ·
    <b class="bad">{run.get("failed", 0)} failed</b> ·
    {html.escape(str(run.get("model", "")))} ·
    <code>{html.escape(str(sha))[:12]}</code> ·
    ${float(run.get("total_cost_usd", 0) or 0):.2f} ·
    {float(run.get("duration_seconds", 0) or 0):.0f}s
  </p>
</header>"""


def _cards(run: dict[str, Any]) -> str:
    """Headline metric cards."""
    metrics = run.get("metrics", {})
    cards = []
    for key, label in HEADLINE:
        value = metrics.get(key)
        if value is None:
            # Shown as unmeasured rather than as zero. A metric that did not
            # apply to this set must not read as a failure.
            cards.append(
                f'<div class="card"><span class="n dim">—</span><span>{label}</span></div>'
            )
            continue
        klass = "good" if value >= GOOD_AT else ("warn" if value >= WARN_AT else "bad")
        cards.append(
            f'<div class="card"><span class="n {klass}">{value:.3f}</span><span>{label}</span></div>'
        )
    return f'<section class="cards">{"".join(cards)}</section>'


def _gate_section(gate: dict[str, Any] | None) -> str:
    """The CI gate's verdict, when the run was gated."""
    if not gate:
        return ""
    passed = gate.get("passed")
    banner = "Gate passed" if passed else "Gate failed"
    klass = "ok" if passed else "bad"
    parts = [f'<section class="gate {klass}"><h2>{banner}</h2>']
    for failure in gate.get("failures", []):
        parts.append(f'<p class="bad">✗ {html.escape(str(failure))}</p>')
    for warning in gate.get("warnings", []):
        parts.append(f'<p class="warn">! {html.escape(str(warning))}</p>')
    if regressed := gate.get("regressed_cases"):
        parts.append(
            f"<p>Regressed: {', '.join(f'<code>{html.escape(c)}</code>' for c in regressed)}</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _intent_table(by_intent: dict[str, Any]) -> str:
    """Per-intent breakdown.

    The most valuable table in the report: it is where a suite that looks healthy
    on average turns out to be failing every multi-hop question.
    """
    if not by_intent:
        return ""

    columns = ["pass_rate", "groundedness", "citation_precision", "citation_recall"]
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in columns)
    rows = []
    for intent, values in by_intent.items():
        cells = []
        for column in columns:
            value = values.get(column)
            cells.append("<td>—</td>" if value is None else f"<td>{value:.3f}</td>")
        rows.append(
            f"<tr><th>{html.escape(intent)}</th><td>{int(values.get('count', 0))}</td>"
            + "".join(cells)
            + "</tr>"
        )

    return f"""
<section>
  <h2>By intent</h2>
  <table><thead><tr><th>intent</th><th>n</th>{head}</tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
</section>"""


def _failure_modes(failures: list[dict[str, Any]]) -> str:
    """Failure counts by named mode, so triage starts with the biggest bucket."""
    if not failures:
        return ""
    counts: dict[str, int] = {}
    for failure in failures:
        mode = failure.get("failure_mode") or "unknown"
        counts[mode] = counts.get(mode, 0) + 1

    items = "".join(
        f"<li><b>{count}</b> {html.escape(mode)}</li>"
        for mode, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return f'<section><h2>Failure modes</h2><ul class="modes">{items}</ul></section>'


def _cases_section(title: str, cases: list[dict[str, Any]], *, open_by_default: bool) -> str:
    """A collapsible list of cases."""
    if not cases:
        return ""
    body = "".join(_case(case) for case in cases)
    attr = " open" if open_by_default else ""
    return f"<section><details{attr}><summary><h2>{html.escape(title)}</h2></summary>{body}</details></section>"


def _case(case: dict[str, Any]) -> str:
    """One case, with everything needed to understand why it scored as it did."""
    mode = case.get("failure_mode")
    badge = (
        f'<span class="badge bad">{html.escape(str(mode))}</span>'
        if mode
        else '<span class="badge ok">passed</span>'
    )
    metrics = " · ".join(
        f"{k}={v:.2f}"
        for k, v in sorted((case.get("metrics") or {}).items())
        if isinstance(v, int | float)
    )
    expected = case.get("expected_sources") or []
    cited = case.get("citations") or []

    reasoning = []
    for judge, verdict in (case.get("judge_scores") or {}).items():
        if judge.startswith("_") or not isinstance(verdict, dict):
            continue
        reasoning.append(
            f"<p class='judge'><b>{html.escape(judge)}</b> "
            f"{float(verdict.get('score', 0)):.2f} — {html.escape(str(verdict.get('reasoning', '')))}</p>"
        )

    return f"""
<article class="case">
  <h3><code>{html.escape(str(case.get("case_id", "")))}</code> {badge}
    <span class="tag">{html.escape(str(case.get("intent", "")))}</span></h3>
  <p class="q">{html.escape(str(case.get("query", "")))}</p>
  <pre class="a">{html.escape(str(case.get("actual_answer") or case.get("error") or "(no answer)"))}</pre>
  <p class="src">expected: {", ".join(f"<code>{html.escape(s)}</code>" for s in expected) or "—"}
     · cited: {", ".join(f"<code>{html.escape(s)}</code>" for s in cited) or "—"}</p>
  <p class="metrics">{html.escape(metrics)}</p>
  {"".join(reasoning)}
</article>"""


def _footer() -> str:
    """Provenance, so a reader knows how to reproduce the numbers above."""
    return (
        "<footer><p>Generated by <code>python -m evals.run</code>. "
        "Every number on this page comes from that run; nothing here is illustrative.</p></footer>"
    )


_CSS = """
:root { --bg:#fff; --fg:#16181d; --muted:#6b7280; --line:#e5e7eb;
        --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --card:#f9fafb; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e6e8eb; --muted:#9ca3af; --line:#262b33;
          --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; --card:#161a20; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:var(--bg); color:var(--fg);
       font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; max-width:1100px; }
h1 { margin:0 0 .25rem; font-size:1.6rem; }
h2 { font-size:1.1rem; margin:0; display:inline; }
h3 { font-size:.95rem; margin:0 0 .4rem; font-weight:600; }
.ver { color:var(--muted); font-weight:400; font-size:1rem; }
.meta { color:var(--muted); margin:0 0 1.5rem; }
.ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
.good { color:var(--ok); } .dim { color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin-bottom:2rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:1rem;
        display:flex; flex-direction:column; gap:.25rem; }
.card .n { font-size:1.7rem; font-weight:650; font-variant-numeric:tabular-nums; }
.card span:last-child { color:var(--muted); font-size:.8rem; }
section { margin-bottom:2rem; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:.5rem .7rem; border-bottom:1px solid var(--line); }
thead th, tbody th { text-align:left; font-weight:600; }
summary { cursor:pointer; padding:.5rem 0; }
.modes { list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:.75rem; }
.modes li { background:var(--card); border:1px solid var(--line); border-radius:999px; padding:.3rem .8rem; }
.case { border:1px solid var(--line); border-radius:10px; padding:1rem; margin:.75rem 0; }
.case .q { font-weight:600; margin:.2rem 0 .6rem; }
.case .a { background:var(--card); border-radius:8px; padding:.75rem; white-space:pre-wrap;
           word-break:break-word; max-height:22rem; overflow:auto; margin:0 0 .6rem; font-size:.88rem; }
.case .src, .case .metrics, .judge { color:var(--muted); font-size:.82rem; margin:.25rem 0; }
.badge { font-size:.72rem; border-radius:999px; padding:.1rem .55rem; border:1px solid currentColor; }
.tag { color:var(--muted); font-size:.75rem; font-weight:400; }
.gate { border:1px solid var(--line); border-left-width:4px; border-radius:8px; padding:1rem; }
.gate.ok { border-left-color:var(--ok); } .gate.bad { border-left-color:var(--bad); }
code { background:var(--card); border-radius:4px; padding:.05rem .3rem; font-size:.85em; }
footer { color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; }
"""
