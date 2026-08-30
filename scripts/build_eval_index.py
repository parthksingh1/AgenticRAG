"""Build the GitHub Pages index for the nightly eval reports.

    python scripts/build_eval_index.py reports/ site/

The nightly run produces one HTML report per set. This assembles them into a
site with a landing page showing the trend, because a directory listing of
timestamped files is not something anyone reads twice.

The trend is the point. A single night's groundedness number says little; the
same number falling for nine consecutive nights is the signal that no individual
pull request was responsible for.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

#: Metrics charted on the landing page, in the order they matter.
TRACKED = (
    ("groundedness", "Groundedness"),
    ("citation_precision", "Citation precision"),
    ("citation_recall", "Citation recall"),
    ("refusal_appropriateness", "Refusal accuracy"),
    ("injection_resistance", "Injection resistance"),
    ("pass_rate", "Pass rate"),
)

#: Runs kept on the landing page. Beyond this the page becomes a scroll rather
#: than a summary; older reports stay on disk and stay linked.
MAX_RUNS = 30


def main(source: str, target: str) -> int:
    """Copy the reports and write an index over them."""
    source_dir = Path(source)
    target_dir = Path(target)
    target_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for json_path in sorted(source_dir.rglob("*.json")):
        try:
            run = json.loads(json_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "set_name" not in run:
            continue

        html_path = json_path.with_suffix(".html")
        if html_path.exists():
            shutil.copy2(html_path, target_dir / html_path.name)
            run["_href"] = html_path.name
        runs.append(run)

    runs.sort(key=lambda r: str(r.get("finished_at", "")), reverse=True)
    (target_dir / "index.html").write_text(render(runs[:MAX_RUNS]), encoding="utf-8")

    print(f"wrote {target_dir / 'index.html'} from {len(runs)} runs")
    return 0


def render(runs: list[dict[str, Any]]) -> str:
    """Render the landing page."""
    by_set: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_set.setdefault(str(run.get("set_name", "unknown")), []).append(run)

    sections = "".join(_section(name, entries) for name, entries in sorted(by_set.items()))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgenticRAG — nightly evals</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>Nightly evals</h1>
  <p>Produced by <code>python -m evals.run</code> against a real stack, every
     night at 03:00 UTC. Every number here comes from that run; nothing on this
     page is illustrative.</p>
</header>
{sections or "<p>No runs recorded yet.</p>"}
<footer><p>AgenticRAG · <a href="https://github.com">source</a></p></footer>
</body></html>"""


def _section(name: str, runs: list[dict[str, Any]]) -> str:
    """One set's trend table."""
    latest = runs[0]
    metrics = latest.get("metrics", {})

    cards = "".join(
        f'<div class="card"><span class="n">{metrics[key]:.3f}</span><span>{label}</span></div>'
        if isinstance(metrics.get(key), int | float)
        else f'<div class="card"><span class="n dim">—</span><span>{label}</span></div>'
        for key, label in TRACKED
    )

    header = "".join(f"<th>{label}</th>" for _, label in TRACKED)
    rows = "".join(_row(run) for run in runs)

    return f"""
<section>
  <h2>{html.escape(name)}</h2>
  <div class="cards">{cards}</div>
  <table>
    <thead><tr><th>run</th><th>cases</th><th>passed</th>{header}<th>cost</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>"""


def _row(run: dict[str, Any]) -> str:
    """One night's row, linked to its full report."""
    when = str(run.get("finished_at", ""))[:16].replace("T", " ")
    href = run.get("_href")
    label = (
        f'<a href="{html.escape(str(href))}">{html.escape(when)}</a>' if href else html.escape(when)
    )

    cells = ""
    for key, _ in TRACKED:
        value = run.get("metrics", {}).get(key)
        cells += f"<td>{value:.3f}</td>" if isinstance(value, int | float) else "<td>—</td>"

    return (
        f"<tr><th>{label}</th>"
        f"<td>{run.get('case_count', 0)}</td>"
        f"<td>{run.get('passed', 0)}</td>"
        f"{cells}"
        f"<td>${float(run.get('total_cost_usd', 0) or 0):.2f}</td></tr>"
    )


_CSS = """
:root { --bg:#fff; --fg:#16181d; --muted:#6b7280; --line:#e5e7eb; --card:#f9fafb; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e6e8eb; --muted:#9ca3af; --line:#262b33; --card:#161a20; }
}
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:var(--bg); color:var(--fg);
       font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; max-width:1200px; }
h1 { font-size:1.6rem; margin:0 0 .3rem; }
h2 { font-size:1.1rem; margin:2rem 0 .8rem; }
header p { color:var(--muted); max-width:60ch; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.7rem; margin-bottom:1rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.9rem;
        display:flex; flex-direction:column; }
.card .n { font-size:1.5rem; font-weight:650; font-variant-numeric:tabular-nums; }
.card .dim { color:var(--muted); }
.card span:last-child { color:var(--muted); font-size:.78rem; }
table { border-collapse:collapse; width:100%; font-size:.86rem; font-variant-numeric:tabular-nums; }
th,td { padding:.45rem .6rem; border-bottom:1px solid var(--line); text-align:right; }
thead th, tbody th { text-align:left; font-weight:600; white-space:nowrap; }
a { color:inherit; }
footer { margin-top:3rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; }
code { background:var(--card); border-radius:4px; padding:.05rem .3rem; }
"""


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
