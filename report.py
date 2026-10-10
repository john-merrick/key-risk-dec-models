#!/usr/bin/env python3
"""Turn a risk_pipeline JSON run into a single-file HTML report anyone can read.

Usage:
  python report.py output/POOL_2017-02-24_2026-07-29_risk_timeseries.json
  python report.py run.json --out report.html

The report is annual: each column is a 10-K, the one filing a year that restates every
risk factor. Quarterly rows stay in the CSV/JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

SEVERE = 1.5  # materiality score at or above this counts as "severe"
MODERATE = 0.5  # at or above this (and below SEVERE) counts as "moderate"
CONFIDENT = 0.6  # trajectory confidence needed to call out a move in the heatmap
TRAJECTORIES = ("increased", "unchanged", "decreased", "resolved")
TRAJECTORY_COLOURS = {
    "increased": "#b3261e",
    "unchanged": "#c9c4b8",
    "decreased": "#2e7d5b",
    "resolved": "#3b6ea5",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class Run:
    """The pipeline output reshaped around annual (10-K) observations."""

    def __init__(self, document: dict):
        self.ticker = document["ticker"]
        self.company = document["company"]
        self.models = document["models"]
        self.filings = document["filings"]
        self.risks = {r["risk_id"]: r for r in document["risks"]}
        self.rows = document["timeseries"]
        self.cost = sum(row.get("cost_usd") or 0 for row in self.rows)

        self.annual_dates = [
            f["filing_date"]
            for f in self.filings
            if f["form"].startswith("10-K") and f["section_is_full"]
        ]
        if len(self.annual_dates) < 2:
            raise ValueError("a report needs at least two full 10-K filings in the run")
        self.cell: dict[tuple[str, str], dict] = {
            (row["risk_id"], row["filing_date"]): row
            for row in self.rows
            if row["filing_date"] in set(self.annual_dates)
        }
        self.first_date = self.annual_dates[0]
        self.last_date = self.annual_dates[-1]

    def first_seen(self, risk_id: str) -> str:
        return self.risks[risk_id].get("first_seen", self.first_date)

    def series(self, risk_id: str) -> list[tuple[str, dict]]:
        return [
            (d, self.cell[(risk_id, d)]) for d in self.annual_dates if (risk_id, d) in self.cell
        ]

    def active(self, date: str) -> list[str]:
        return [r for r in self.risks if (r, date) in self.cell]

    def score(self, risk_id: str, date: str) -> float | None:
        row = self.cell.get((risk_id, date))
        return None if row is None else row["materiality_score"]


def band(score: float) -> str:
    return "severe" if score >= SEVERE else "moderate" if score >= MODERATE else "low"


def year(date: str) -> str:
    return date[:4]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def yearly_stats(run: Run) -> list[dict]:
    """One record per annual filing: risk counts, mean score, trajectory mix."""
    stats = []
    for date in run.annual_dates:
        ids = run.active(date)
        scores = [run.score(r, date) for r in ids]
        trajectory = defaultdict(int)
        for r in ids:
            t = run.cell[(r, date)]["trajectory"]
            if t != "baseline":
                trajectory[t] += 1
        stats.append(
            {
                "date": date,
                "tracked": len(ids),
                "mean": mean(scores),
                "severe": sum(s >= SEVERE for s in scores),
                "moderate": sum(MODERATE <= s < SEVERE for s in scores),
                "low": sum(s < MODERATE for s in scores),
                "new": sum(1 for r in ids if run.first_seen(r) == date and date != run.first_date),
                "trajectory": dict(trajectory),
            }
        )
    return stats


def risk_changes(run: Run) -> list[dict]:
    """First-to-latest annual move for each risk seen in at least two 10-Ks."""
    changes = []
    for risk_id, risk in run.risks.items():
        series = run.series(risk_id)
        if len(series) < 2:
            continue
        first, last = series[0][1]["materiality_score"], series[-1][1]["materiality_score"]
        scores = [row["materiality_score"] for _, row in series]
        changes.append(
            {
                "risk": risk,
                "first_date": series[0][0],
                "first": first,
                "last": last,
                "delta": last - first,
                "peak": max(scores),
                "peak_date": series[scores.index(max(scores))][0],
                "severe_years": sum(s >= SEVERE for s in scores),
                "years": len(scores),
                "latest_trajectory": series[-1][1]["trajectory"],
            }
        )
    return changes


def category_stats(run: Run) -> list[dict]:
    out = []
    for category in sorted({r["category"] for r in run.risks.values()}):
        ids = [r for r, v in run.risks.items() if v["category"] == category]
        latest = [run.score(r, run.last_date) for r in ids if run.score(r, run.last_date) is not None]
        earliest = [
            s for r in ids if (s := run.score(r, run.first_date)) is not None
        ]
        out.append(
            {
                "category": category,
                "count": len(ids),
                "first_mean": mean(earliest),
                "last_mean": mean(latest),
                "severe_latest": sum(s >= SEVERE for s in latest),
            }
        )
    return out


def lowest_confidence_share(run: Run) -> float:
    calls = [
        row["trajectory_confidence"]
        for row in run.cell.values()
        if row["trajectory"] != "baseline" and row["trajectory_confidence"] is not None
    ]
    return sum(c < 0.5 for c in calls) / len(calls) if calls else float("nan")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def heat(score: float) -> tuple[str, str]:
    """Background and text colour for a 0-2 materiality score."""
    t = max(0.0, min(1.0, score / 2))
    low, high = (244, 241, 234), (179, 38, 30)
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
    return f"rgb{rgb}", "#fff" if t > 0.55 else "#1c1b18"


def line_chart(stats: list[dict]) -> str:
    """Inline SVG: mean materiality (left) and count of severe risks (bars)."""
    width, height, left, right, top, bottom = 720, 260, 44, 44, 16, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(stats)
    x = lambda i: left + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)  # noqa: E731
    max_count = max(max(s["severe"] for s in stats), 1)
    y_mean = lambda v: top + plot_h * (1 - v / 2)  # noqa: E731
    y_count = lambda v: top + plot_h * (1 - v / max_count)  # noqa: E731

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Mean materiality and severe risk count by year">']
    for tick in (0, 0.5, 1.0, 1.5, 2.0):
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y_mean(tick):.1f}" y2="{y_mean(tick):.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y_mean(tick) + 4:.1f}" class="axis" text-anchor="end">{tick:g}</text>'
        )
    bar_w = min(34, plot_w / n * 0.5)
    for i, s in enumerate(stats):
        h = plot_h - (y_count(s["severe"]) - top)
        parts.append(
            f'<rect x="{x(i) - bar_w / 2:.1f}" y="{y_count(s["severe"]):.1f}" width="{bar_w:.1f}" '
            f'height="{h:.1f}" class="bar"><title>{year(s["date"])}: {s["severe"]} severe risks</title></rect>'
            f'<text x="{x(i):.1f}" y="{height - 14}" class="axis" text-anchor="middle">{year(s["date"])}</text>'
        )
    points = " ".join(f"{x(i):.1f},{y_mean(s['mean']):.1f}" for i, s in enumerate(stats))
    parts.append(f'<polyline points="{points}" class="line"/>')
    for i, s in enumerate(stats):
        parts.append(
            f'<circle cx="{x(i):.1f}" cy="{y_mean(s["mean"]):.1f}" r="4" class="dot">'
            f'<title>{year(s["date"])}: mean {s["mean"]:.2f}</title></circle>'
            f'<text x="{x(i):.1f}" y="{y_mean(s["mean"]) - 9:.1f}" class="val" text-anchor="middle">{s["mean"]:.2f}</text>'
        )
    parts.append(
        f'<text x="{width - right + 8}" y="{top + 10}" class="axis">{max_count}</text>'
        f'<text x="{width - right + 8}" y="{top + plot_h}" class="axis">0</text>'
        "</svg>"
    )
    return "".join(parts)


def takeaways(run: Run, stats: list[dict], changes: list[dict]) -> list[str]:
    first, last = stats[0], stats[-1]
    out = []
    peak = max(stats, key=lambda s: s["mean"])
    out.append(
        f"Across {len(stats)} annual reports ({year(run.first_date)}–{year(run.last_date)}), "
        f"{run.company} disclosed {len(run.risks)} distinct risks. The average materiality "
        f"moved from {first['mean']:.2f} to {last['mean']:.2f} on a 0–2 scale"
        + (f", peaking in {year(peak['date'])} at {peak['mean']:.2f}." if peak is not last else ".")
    )
    top = max(stats, key=lambda s: s["severe"])
    if top["severe"] == 0:
        out.append(
            f"No risk reached the severe band (score ≥ {SEVERE:g}) in any annual report; "
            f"{last['moderate']} were moderate in {year(last['date'])}."
        )
    else:
        out.append(
            f"Severe risks (score ≥ {SEVERE:g}) peaked at {top['severe']} in {year(top['date'])} "
            f"and numbered {last['severe']} in {year(last['date'])}."
        )
    if last["new"]:
        out.append(f"{last['new']} risk(s) first appeared in the latest 10-K.")
    risers = sorted((c for c in changes if c["delta"] > 0.15), key=lambda c: -c["delta"])[:3]
    if risers:
        names = "; ".join(f"“{c['risk']['title']}” (+{c['delta']:.2f})" for c in risers)
        out.append(f"Largest increases in materiality: {names}.")
    fallers = sorted((c for c in changes if c["delta"] < -0.15), key=lambda c: c["delta"])[:3]
    if fallers:
        names = "; ".join(f"“{c['risk']['title']}” ({c['delta']:.2f})" for c in fallers)
        out.append(f"Largest decreases: {names}.")
    persistent = [c for c in changes if c["years"] >= 3 and c["severe_years"] / c["years"] >= 0.5]
    if persistent:
        names = "; ".join(f"“{c['risk']['title']}”" for c in persistent[:3])
        out.append(
            f"Chronic exposure (severe in at least half of the years tracked): {names}."
        )
    resolved = [
        r
        for r in run.risks
        if any(row["trajectory"] == "resolved" for _, row in run.series(r))
    ]
    out.append(
        f"{len(resolved)} risk(s) were flagged as resolved (dropped from the filing or "
        "reported as concluded)."
        if resolved
        else "No risk was flagged as resolved: everything first disclosed is still disclosed."
    )
    return out


CSS = """
:root{--bg:#fbfaf7;--fg:#1c1b18;--muted:#6b675d;--line:#e2ded3;--card:#fff;--accent:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161512;--fg:#ece9e0;--muted:#9a968a;--line:#2e2c26;--card:#1d1c18}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 6px}
.sub{color:var(--muted);margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:24px;line-height:1.2}.card span{color:var(--muted);font-size:13px}
ul.take{padding-left:18px}ul.take li{margin:6px 0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{font-weight:600;background:color-mix(in srgb,var(--line) 40%,transparent)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
.hm td.c{text-align:center;font-variant-numeric:tabular-nums;min-width:52px;padding:4px 2px}
.hm td.t{min-width:260px}
.up{color:#b3261e;font-weight:700}.down{color:#2e7d5b;font-weight:700}
.note{color:var(--muted);font-size:13px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:8px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-2px}
svg{width:100%;height:auto;max-width:720px}
.grid{stroke:var(--line)}.axis{fill:var(--muted);font-size:11px}.val{fill:var(--fg);font-size:11px}
.line{fill:none;stroke:var(--accent);stroke-width:2.5}.dot{fill:var(--accent)}.bar{fill:var(--line)}
.stack{display:flex;height:16px;border-radius:3px;overflow:hidden;min-width:160px}
.stack span{display:block}
.tag{font-size:11px;padding:1px 6px;border-radius:9px;background:var(--line);white-space:nowrap}
@media print{body{background:#fff}h2{break-after:avoid}.scroll{overflow:visible}}
"""


def render(run: Run) -> str:
    stats = yearly_stats(run)
    changes = risk_changes(run)
    categories = category_stats(run)
    last = stats[-1]
    cost = run.cost
    low_conf = lowest_confidence_share(run)
    o = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{escape(run.ticker)} risk factor report</title><style>{CSS}</style></head><body><main>"]

    o.append(
        f"<h1>{escape(run.company)} ({escape(run.ticker)}): how disclosed risks changed</h1>"
        f"<p class='sub'>From the “Risk Factors” section (Item 1A) of {len(stats)} annual reports, "
        f"{year(run.first_date)} to {year(run.last_date)}. Scores are AI assessments of the filing text.</p>"
    )
    o.append(
        "<div class='cards'>"
        f"<div class='card'><b>{len(run.risks)}</b><span>distinct risks tracked</span></div>"
        f"<div class='card'><b>{max(x['severe'] for x in stats)}</b><span>most risks rated severe "
        f"in one year ({year(max(stats, key=lambda x: x['severe'])['date'])}); "
        f"{last['severe']} in {year(run.last_date)}</span></div>"
        f"<div class='card'><b>{last['mean']:.2f}</b><span>average materiality, 0–2 "
        f"(was {stats[0]['mean']:.2f})</span></div>"
        f"<div class='card'><b>{sum(s['new'] for s in stats)}</b><span>risks added after the first report</span></div>"
        "</div>"
    )

    o.append("<h2>Key takeaways</h2><ul class='take'>")
    o.extend(f"<li>{escape(t)}</li>" for t in takeaways(run, stats, changes))
    o.append("</ul>")

    o.append(
        "<h2>How to read this report</h2>"
        "<p><b>Materiality score (0–2)</b>: how prominent and consequential the filing makes the "
        "risk. <b>0</b> = barely mentioned or boilerplate; <b>1</b> = a specific, active concern; "
        "<b>2</b> = severe, materialised or critical focus. It is a probability-weighted position, "
        "so 1.4 means “between active and severe”.</p>"
        "<p><b>Trajectory</b>: whether the filing’s disclosure of a risk grew, shrank, stayed "
        "the same or ended, compared with the previous annual report. It reflects how the "
        "company <i>describes</i> the risk, not whether the risk event occurred.</p>"
    )

    o.append("<h2>Overall risk load by year</h2>")
    o.append("<div class='legend'><span><i style='background:var(--accent)'></i>Average materiality (line, left axis)</span>"
             "<span><i style='background:var(--line)'></i>Number of severe risks (bars, right axis)</span></div>")
    o.append(line_chart(stats))
    o.append(
        "<div class='scroll'><table><tr><th>10-K filed</th><th class='n'>Risks tracked</th>"
        "<th class='n'>Severe</th><th class='n'>Moderate</th><th class='n'>Low</th>"
        "<th class='n'>Average score</th><th class='n'>New this year</th></tr>"
    )
    for s in stats:
        o.append(
            f"<tr><td>{s['date']}</td><td class='n'>{s['tracked']}</td><td class='n'>{s['severe']}</td>"
            f"<td class='n'>{s['moderate']}</td><td class='n'>{s['low']}</td>"
            f"<td class='n'>{s['mean']:.2f}</td><td class='n'>{s['new'] or '–'}</td></tr>"
        )
    o.append("</table></div>")

    o.append("<h2>Year-over-year direction</h2>")
    o.append("<p class='note'>Each year, how many risks were disclosed more strongly, the same, "
             "more weakly, or ended, versus the prior annual report.</p>")
    o.append("<div class='legend'>" + "".join(
        f"<span><i style='background:{TRAJECTORY_COLOURS[t]}'></i>{t}</span>" for t in TRAJECTORIES) + "</div>")
    o.append("<div class='scroll'><table><tr><th>10-K filed</th><th>Mix</th>"
             + "".join(f"<th class='n'>{t.capitalize()}</th>" for t in TRAJECTORIES) + "</tr>")
    for s in stats[1:]:
        total = sum(s["trajectory"].values()) or 1
        mix = "".join(
            f"<span style='width:{100 * s['trajectory'].get(t, 0) / total:.1f}%;background:{TRAJECTORY_COLOURS[t]}'></span>"
            for t in TRAJECTORIES
        )
        o.append(
            f"<tr><td>{s['date']}</td><td><div class='stack'>{mix}</div></td>"
            + "".join(f"<td class='n'>{s['trajectory'].get(t, 0)}</td>" for t in TRAJECTORIES)
            + "</tr>"
        )
    o.append("</table></div>")

    o.append("<h2>By category</h2><div class='scroll'><table><tr><th>Category</th><th class='n'>Risks</th>"
             f"<th class='n'>Avg score {year(run.first_date)}</th><th class='n'>Avg score {year(run.last_date)}</th>"
             f"<th class='n'>Severe in {year(run.last_date)}</th></tr>")
    for c in categories:
        o.append(
            f"<tr><td>{escape(c['category'].capitalize())}</td><td class='n'>{c['count']}</td>"
            f"<td class='n'>{c['first_mean']:.2f}</td><td class='n'>{c['last_mean']:.2f}</td>"
            f"<td class='n'>{c['severe_latest']}</td></tr>"
        )
    o.append("</table></div>")

    def movers(title: str, items: list[dict]) -> None:
        if not items:
            return
        o.append(f"<h3>{title}</h3><div class='scroll'><table><tr><th>Risk</th><th>First scored</th>"
                 f"<th class='n'>Then</th><th class='n'>Now</th><th class='n'>Change</th></tr>")
        for c in items:
            cls = "up" if c["delta"] > 0 else "down"
            o.append(
                f"<tr><td>{escape(c['risk']['title'])}</td><td>{year(c['first_date'])}</td>"
                f"<td class='n'>{c['first']:.2f}</td><td class='n'>{c['last']:.2f}</td>"
                f"<td class='n {cls}'>{c['delta']:+.2f}</td></tr>"
            )
        o.append("</table></div>")

    o.append(f"<h2>What moved most, {year(run.first_date)} → {year(run.last_date)}</h2>")
    movers("Rising", sorted((c for c in changes if c["delta"] > 0), key=lambda c: -c["delta"])[:8])
    movers("Falling", sorted((c for c in changes if c["delta"] < 0), key=lambda c: c["delta"])[:8])

    o.append("<h2>Today’s most material risks</h2><div class='scroll'><table><tr><th class='n'>#</th>"
             f"<th>Risk</th><th>Category</th><th class='n'>Score {year(run.last_date)}</th>"
             "<th class='n'>Peak</th><th>First seen</th><th class='n'>Years rated severe</th></tr>")
    current = sorted(changes, key=lambda c: -c["last"])[:10]
    for rank, c in enumerate(current, start=1):
        o.append(
            f"<tr><td class='n'>{rank}</td><td>{escape(c['risk']['title'])}</td>"
            f"<td>{escape(c['risk']['category'])}</td><td class='n'>{c['last']:.2f}</td>"
            f"<td class='n'>{c['peak']:.2f} ({year(c['peak_date'])})</td>"
            f"<td>{year(run.first_seen(c['risk']['risk_id']))}</td>"
            f"<td class='n'>{c['severe_years']} of {c['years']}</td></tr>"
        )
    o.append("</table></div>")

    o.append("<h2>Full heatmap</h2><p class='note'>Materiality score by annual report. "
             "▲ / ▼ mark a confident (≥ 0.6) increase or decrease versus the prior report; "
             "a blank cell means the risk had not yet appeared.</p>")
    o.append("<div class='legend'>" + "".join(
        f"<span><i style='background:{heat(v)[0]}'></i>{v:g}</span>" for v in (0, 0.5, 1, 1.5, 2)) + "</div>")
    o.append("<div class='scroll'><table class='hm'><tr><th>Risk</th><th>Category</th>"
             + "".join(f"<th class='n'>{year(d)}</th>" for d in run.annual_dates) + "</tr>")
    order = sorted(run.risks.values(), key=lambda r: (r["category"], run.first_seen(r["risk_id"]), r["risk_id"]))
    for risk in order:
        o.append(f"<tr><td class='t'>{escape(risk['title'])}</td><td>{escape(risk['category'])}</td>")
        for date in run.annual_dates:
            row = run.cell.get((risk["risk_id"], date))
            if row is None:
                o.append("<td class='c'></td>")
                continue
            bg, fg = heat(row["materiality_score"])
            arrow = ""
            if row["trajectory"] in ("increased", "decreased") and (row["trajectory_confidence"] or 0) >= CONFIDENT:
                arrow = " ▲" if row["trajectory"] == "increased" else " ▼"
            if row["trajectory"] == "resolved" and (row["trajectory_confidence"] or 0) >= CONFIDENT:
                arrow = " ✕"
            o.append(f"<td class='c' style='background:{bg};color:{fg}'>{row['materiality_score']:.1f}{arrow}</td>")
        o.append("</tr>")
    o.append("</table></div>")

    o.append("<h2>Risk register</h2><div class='scroll'><table><tr><th>ID</th><th>Risk</th><th>Category</th>"
             "<th>First seen</th><th>Description</th></tr>")
    for risk in sorted(run.risks.values(), key=lambda r: r["risk_id"]):
        o.append(
            f"<tr><td>{escape(risk['risk_id'])}</td><td>{escape(risk['title'])}</td>"
            f"<td>{escape(risk['category'])}</td><td>{year(run.first_seen(risk['risk_id']))}</td>"
            f"<td>{escape(risk['description'])}</td></tr>"
        )
    o.append("</table></div>")

    o.append(
        "<h2>Method and limits</h2><ul class='take'>"
        f"<li><b>Source</b>: Item 1A of each 10-K from SEC EDGAR. Quarterly 10-Qs mostly say "
        "“no material changes”, so this report uses annual filings; quarterly rows are in the CSV.</li>"
        f"<li><b>Register</b>: the first 10-K was read by {escape(run.models['baseline'])} to list the "
        "risks; each later 10-K was screened for risks not already listed, so newer risks are tracked "
        "from the year they first appear.</li>"
        f"<li><b>Scoring</b>: {escape(run.models['decision'])} rated each risk against each filing "
        f"(cost of scoring about ${cost:.2f}). Materiality and trajectory are model judgments about "
        "the text, not measured outcomes.</li>"
        "<li><b>Scale</b>: most risks score between 0.5 and 1.5, so rank and change matter more "
        "than the absolute number. A score of 0.9 is a normal, specific concern, not a mild one.</li>"
        f"<li><b>Confidence</b>: {low_conf:.0%} of trajectory calls had confidence below 0.5. Treat "
        "small score changes (under about 0.2) as noise; the stronger signal is a sustained move "
        "across several years.</li>"
        "<li><b>Not advice</b>: a change in disclosure is a prompt to read the filing, not a "
        "conclusion about the company’s prospects.</li></ul>"
    )
    o.append("</main></body></html>")
    return "".join(o)


def build_report(json_path: Path, out: Path | None = None) -> Path:
    """Render the report for a pipeline JSON run and return the HTML path.

    Raises ValueError when the run has fewer than two full 10-K filings.
    """
    run = Run(json.loads(json_path.read_text(encoding="utf-8")))
    out = out or json_path.parent / f"{run.ticker}_risk_report.html"
    out.write_text(render(run), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an HTML report from a pipeline JSON run.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, help="Output path (default: <ticker>_risk_report.html)")
    args = parser.parse_args(argv)
    try:
        print(f"Wrote {build_report(args.json_path, args.out)}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
