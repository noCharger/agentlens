"""HTML report generator for evolution history."""

from __future__ import annotations

from datetime import datetime
from html import escape

from agentlens.core.models import EvolutionRecord

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_evolution_report(
    records: list[EvolutionRecord],
    *,
    project_name: str = "",
) -> str:
    """Return a self-contained HTML string for the evolution history.

    Parameters
    ----------
    records:
        List of EvolutionRecord instances (any order; sorted newest-first internally).
    project_name:
        Optional display name shown in the page title and header.
    """
    title = f"Evolution History — {project_name}" if project_name else "Evolution History"
    generated_at = datetime.now().isoformat(timespec="seconds")

    if not records:
        return _empty_page(title, generated_at)

    # Sort newest-first for rendering; keep originals for chart (cycle order).
    sorted_records = sorted(records, key=lambda r: r.created_at, reverse=True)
    chart_records = sorted(records, key=lambda r: r.cycle)

    # Summary stats
    total_cycles = len(records)
    accepted = sum(1 for r in records if r.accepted)
    rejected = total_cycles - accepted
    net_gain = sum(r.delta_pass_rate for r in records if r.accepted)

    summary_cards = _summary_cards(total_cycles, accepted, rejected, net_gain)
    bar_chart = _bar_chart(chart_records)
    cycle_cards = "\n".join(_cycle_card(r) for r in sorted_records)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0;
    background: #f0f2f7;
    color: #222;
  }}
  header {{
    background: #1a1a2e;
    color: #fff;
    padding: 1.25rem 2rem;
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
  }}
  header h1 {{ margin: 0; font-size: 1.4rem; font-weight: 700; }}
  header .subtitle {{
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
  }}
  main {{ padding: 1.5rem 2rem; max-width: 1100px; margin: 0 auto; }}
  /* Summary cards */
  .summary {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.75rem;
  }}
  .card {{
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    min-width: 130px;
    flex: 1 1 130px;
  }}
  .card h3 {{
    margin: 0 0 0.35rem;
    color: #666;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .card .value {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
  }}
  .value.green {{ color: #16a34a; }}
  .value.red {{ color: #dc2626; }}
  .value.blue {{ color: #2563eb; }}
  /* Bar chart */
  .chart-section {{
    background: #fff;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    margin-bottom: 1.75rem;
  }}
  .chart-section h2 {{
    margin: 0 0 1rem;
    font-size: 1rem;
    color: #1a1a2e;
  }}
  .chart-legend {{
    display: flex;
    gap: 1.25rem;
    font-size: 0.8rem;
    color: #555;
    margin-bottom: 0.75rem;
  }}
  .legend-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
  }}
  /* Per-cycle cards */
  .cycles-heading {{
    font-size: 1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 0.75rem;
  }}
  .cycle-card {{
    background: #fff;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
    margin-bottom: 1rem;
  }}
  .cycle-header {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
  }}
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 5px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }}
  .badge-cycle {{ background: #e0e7ff; color: #3730a3; }}
  .badge-accepted {{ background: #dcfce7; color: #15803d; }}
  .badge-rejected {{ background: #fee2e2; color: #b91c1c; }}
  .rate-arrow {{
    font-size: 0.9rem;
    color: #555;
    white-space: nowrap;
  }}
  .rate-arrow .delta-pos {{ color: #16a34a; font-weight: 700; }}
  .rate-arrow .delta-neg {{ color: #dc2626; font-weight: 700; }}
  /* Chips */
  .chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.65rem; }}
  .chip {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    font-size: 0.76rem;
    color: #475569;
  }}
  .chip-none {{ color: #94a3b8; font-style: italic; }}
  /* Rationale */
  .rationale {{
    font-size: 0.88rem;
    color: #444;
    line-height: 1.55;
    margin-bottom: 0.75rem;
  }}
  /* Details blocks */
  details {{
    margin-bottom: 0.5rem;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    overflow: hidden;
  }}
  details summary {{
    cursor: pointer;
    padding: 0.45rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #374151;
    background: #f9fafb;
    user-select: none;
    list-style: none;
  }}
  details summary::-webkit-details-marker {{ display: none; }}
  details summary::before {{
    content: '▶ ';
    font-size: 0.65rem;
    color: #6b7280;
  }}
  details[open] summary::before {{ content: '▼ '; }}
  .details-body {{
    padding: 0.65rem 0.9rem;
    font-size: 0.83rem;
    color: #374151;
    background: #fff;
  }}
  .details-body p {{ margin: 0.25rem 0; }}
  .details-body strong {{ color: #1a1a2e; }}
  pre.prompt-box {{
    margin: 0;
    padding: 0.6rem 0.75rem;
    background: #f8fafc;
    border-radius: 4px;
    font-family: 'SFMono-Regular', Consolas, 'Courier New', monospace;
    font-size: 0.78rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
  }}
  pre.prompt-box.accepted {{ border-left: 3px solid #16a34a; }}
  pre.prompt-box.rejected {{ border-left: 3px solid #f97316; }}
  svg text {{ font-family: inherit; }}
</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <span class="subtitle">Generated {escape(generated_at)}</span>
</header>
<main>
  {summary_cards}
  {bar_chart}
  <p class="cycles-heading">Cycle Details (newest first)</p>
  {cycle_cards}
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_pct(rate: float) -> str:
    """Format a 0–1 pass rate as a percentage string."""
    return f"{rate * 100:.1f}%"


def _empty_page(title: str, generated_at: str) -> str:
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(title)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0; background: #f0f2f7;
  }}
  header {{
    background: #1a1a2e; color: #fff;
    padding: 1.25rem 2rem;
  }}
  header h1 {{ margin: 0; font-size: 1.4rem; }}
  .empty {{
    text-align: center;
    padding: 5rem 2rem;
    color: #6b7280;
    font-size: 1rem;
  }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1></header>
<div class="empty">
  No evolution records found. Run <code>EvolutionCycle.run()</code> to generate them.
</div>
</body>
</html>
"""


def _summary_cards(
    total_cycles: int,
    accepted: int,
    rejected: int,
    net_gain: float,
) -> str:
    gain_cls = "green" if net_gain >= 0 else "red"
    gain_sign = "+" if net_gain >= 0 else ""
    return f"""\
<div class="summary">
  <div class="card">
    <h3>Total Cycles</h3>
    <div class="value blue">{total_cycles}</div>
  </div>
  <div class="card">
    <h3>Accepted</h3>
    <div class="value green">{accepted}</div>
  </div>
  <div class="card">
    <h3>Rejected</h3>
    <div class="value red">{rejected}</div>
  </div>
  <div class="card">
    <h3>Net Pass Rate Gain</h3>
    <div class="value {gain_cls}">{gain_sign}{net_gain * 100:.1f}%</div>
  </div>
</div>"""


def _bar_chart(chart_records: list[EvolutionRecord]) -> str:
    """Render an inline SVG bar chart: two bars per cycle."""
    if not chart_records:
        return ""

    BAR_W = 18
    GAP = 8          # gap between baseline and candidate bar within same cycle
    GROUP_GAP = 28   # gap between cycle groups
    LEFT_PAD = 40    # space for Y-axis labels
    TOP_PAD = 10
    BOTTOM_PAD = 34  # space for cycle labels
    CHART_H = 120    # height of bar area

    n = len(chart_records)
    total_w = LEFT_PAD + n * (2 * BAR_W + GAP + GROUP_GAP) - GROUP_GAP + 20
    svg_h = TOP_PAD + CHART_H + BOTTOM_PAD

    bars: list[str] = []

    # Y-axis guide lines at 0%, 25%, 50%, 75%, 100%
    for pct in (0, 25, 50, 75, 100):
        y = TOP_PAD + CHART_H - int(CHART_H * pct / 100)
        bars.append(
            f'<line x1="{LEFT_PAD}" y1="{y}" x2="{total_w}" y2="{y}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        bars.append(
            f'<text x="{LEFT_PAD - 4}" y="{y + 4}" text-anchor="end" '
            f'font-size="9" fill="#9ca3af">{pct}%</text>'
        )

    for idx, rec in enumerate(chart_records):
        x_base = LEFT_PAD + idx * (2 * BAR_W + GAP + GROUP_GAP)
        x_cand = x_base + BAR_W + GAP

        # Baseline pass rate from signal_summary if available, else derive from delta
        ss = rec.signal_summary
        baseline_rate: float = float(ss.get("pass_rate", 0.0)) if ss else 0.0
        candidate_rate = max(0.0, min(1.0, baseline_rate + rec.delta_pass_rate))

        bh = max(2, int(CHART_H * baseline_rate))
        ch = max(2, int(CHART_H * candidate_rate))
        by = TOP_PAD + CHART_H - bh
        cy = TOP_PAD + CHART_H - ch

        # Baseline bar (gray)
        bars.append(
            f'<rect x="{x_base}" y="{by}" width="{BAR_W}" height="{bh}" '
            f'fill="#9ca3af" rx="2"/>'
        )
        # Candidate bar (green if accepted, blue otherwise)
        cand_color = "#16a34a" if rec.accepted else "#3b82f6"
        bars.append(
            f'<rect x="{x_cand}" y="{cy}" width="{BAR_W}" height="{ch}" '
            f'fill="{cand_color}" rx="2"/>'
        )

        # Cycle label on X axis
        label_x = x_base + BAR_W + GAP // 2
        bars.append(
            f'<text x="{label_x}" y="{TOP_PAD + CHART_H + 14}" '
            f'text-anchor="middle" font-size="10" fill="#6b7280">C{rec.cycle}</text>'
        )
        # Tooltip-style title elements (native SVG)
        bars.append(
            f'<title>Cycle {rec.cycle} — baseline: {_fmt_pct(baseline_rate)}, '
            f'candidate: {_fmt_pct(candidate_rate)}</title>'
        )

    bars_svg = "\n  ".join(bars)
    svg = (
        f'<svg width="{total_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="overflow:visible">\n  {bars_svg}\n</svg>'
    )

    legend = (
        '<div class="chart-legend">'
        '<span><span class="legend-dot" style="background:#9ca3af"></span>Baseline</span>'
        '<span><span class="legend-dot" style="background:#16a34a"></span>Candidate (accepted)</span>'
        '<span><span class="legend-dot" style="background:#3b82f6"></span>Candidate (rejected)</span>'
        "</div>"
    )

    return f"""\
<div class="chart-section">
  <h2>Pass Rate Trend per Cycle</h2>
  {legend}
  <div style="overflow-x:auto">{svg}</div>
</div>"""


def _cycle_card(rec: EvolutionRecord) -> str:
    status_badge = (
        '<span class="badge badge-accepted">ACCEPTED</span>'
        if rec.accepted
        else '<span class="badge badge-rejected">REJECTED</span>'
    )

    ss = rec.signal_summary
    baseline_rate: float = float(ss.get("pass_rate", 0.0)) if ss else 0.0
    candidate_rate = baseline_rate + rec.delta_pass_rate

    delta_sign = "+" if rec.delta_pass_rate >= 0 else ""
    delta_cls = "delta-pos" if rec.delta_pass_rate >= 0 else "delta-neg"
    rate_display = (
        f'<span class="rate-arrow">'
        f'{_fmt_pct(baseline_rate)} &rarr; {_fmt_pct(candidate_rate)} '
        f'(<span class="{delta_cls}">{delta_sign}{rec.delta_pass_rate * 100:.1f}%</span>)'
        f"</span>"
    )

    # Targeted patterns as chips
    if rec.targeted_patterns:
        chips_html = "".join(
            f'<span class="chip">{escape(p)}</span>' for p in rec.targeted_patterns
        )
    else:
        chips_html = '<span class="chip chip-none">none</span>'

    rationale_html = (
        f'<div class="rationale">{escape(rec.rationale)}</div>'
        if rec.rationale
        else ""
    )

    signal_details = _signal_details_block(rec)
    original_block = _prompt_details_block(
        "Original Prompt", rec.original_prompt, style_class="rejected"
    )
    prompt_cls = "accepted" if rec.accepted else "rejected"
    evolved_block = _prompt_details_block(
        "Evolved Prompt", rec.evolved_prompt, style_class=prompt_cls
    )

    created = rec.created_at.strftime("%Y-%m-%d %H:%M UTC") if rec.created_at else ""

    return f"""\
<div class="cycle-card">
  <div class="cycle-header">
    <span class="badge badge-cycle">Cycle {rec.cycle}</span>
    {status_badge}
    {rate_display}
    <span style="flex:1"></span>
    <span style="font-size:0.75rem;color:#9ca3af">{escape(created)}</span>
  </div>
  <div class="chips">{chips_html}</div>
  {rationale_html}
  {signal_details}
  {original_block}
  {evolved_block}
</div>"""


def _signal_details_block(rec: EvolutionRecord) -> str:
    ss = rec.signal_summary
    if not ss:
        return ""

    lines: list[str] = []

    pass_rate = ss.get("pass_rate")
    if pass_rate is not None:
        lines.append(f"<p><strong>Pass rate:</strong> {float(pass_rate) * 100:.1f}%</p>")

    mem = ss.get("memory_retention_score")
    if mem is not None:
        lines.append(f"<p><strong>Memory retention:</strong> {float(mem):.2f}</p>")

    failure_patterns = ss.get("dominant_failure_patterns") or []
    if failure_patterns:
        top3 = list(failure_patterns)[:3]
        items = "".join(
            f"<li>{escape(str(pat))} &times; {escape(str(cnt))}</li>"
            for pat, cnt in top3
        )
        lines.append(f"<p><strong>Top failure patterns:</strong></p><ul>{items}</ul>")

    weak_dims = ss.get("weak_dimensions") or []
    if weak_dims:
        top3 = list(weak_dims)[:3]
        items = "".join(
            f"<li>{escape(str(dim))}: {escape(str(score))}</li>"
            for dim, score in top3
        )
        lines.append(f"<p><strong>Weak dimensions:</strong></p><ul>{items}</ul>")

    if not lines:
        return ""

    body = "\n".join(lines)
    return f"""\
<details>
  <summary>Signal Summary</summary>
  <div class="details-body">{body}</div>
</details>"""


def _prompt_details_block(label: str, prompt_text: str, *, style_class: str) -> str:
    escaped = escape(prompt_text or "")
    return f"""\
<details>
  <summary>{escape(label)}</summary>
  <div class="details-body">
    <pre class="prompt-box {style_class}">{escaped}</pre>
  </div>
</details>"""
