"""HTML report generator for multi-model sweep results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

from agentlens.eval.benchmarks import UNASSIGNED_BENCHMARK, summarize_results_by_benchmark

if TYPE_CHECKING:
    from agentlens.eval.sweep import SweepResult


@dataclass
class ScenarioGridRow:
    scenario_id: str
    scenario_name: str
    benchmark: str
    cells: dict[str, str]  # agent_model -> "PASS" | "FAIL" | "ERROR" | "—"


def build_scenario_grid(sweep: SweepResult) -> list[ScenarioGridRow]:
    results_by_model: dict[str, dict[str, object]] = {
        run.agent_model: {r.scenario.id: r for r in run.results}
        for run in sweep.model_runs
    }

    seen: set[str] = set()
    ordered_ids: list[str] = []
    for run in sweep.model_runs:
        for r in run.results:
            if r.scenario.id not in seen:
                seen.add(r.scenario.id)
                ordered_ids.append(r.scenario.id)

    id_to_meta: dict[str, tuple[str, str]] = {}
    for run in sweep.model_runs:
        for r in run.results:
            if r.scenario.id not in id_to_meta:
                id_to_meta[r.scenario.id] = (r.scenario.name, r.scenario.benchmark or "")

    rows: list[ScenarioGridRow] = []
    for sid in ordered_ids:
        name, benchmark = id_to_meta.get(sid, (sid, ""))
        cells: dict[str, str] = {}
        for model in [run.agent_model for run in sweep.model_runs]:
            result = results_by_model[model].get(sid)
            if result is None:
                cells[model] = "—"
            elif result.error:
                cells[model] = "ERROR"
            elif result.passed:
                cells[model] = "PASS"
            else:
                cells[model] = "FAIL"
        rows.append(ScenarioGridRow(
            scenario_id=sid,
            scenario_name=name,
            benchmark=benchmark,
            cells=cells,
        ))

    rows.sort(key=lambda r: (r.benchmark == "", r.benchmark, r.scenario_id))
    return rows


SWEEP_REPORT_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AgentLens Sweep Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f8f9fa; }
  h1, h2, h3 { color: #1a1a2e; }
  .models { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0 2rem; }
  .model-card { background: white; border-radius: 8px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 180px; }
  .model-card h3 { margin: 0 0 0.75rem; font-size: 0.9rem; color: #555; word-break: break-all; }
  .model-card .pass-rate { font-size: 2rem; font-weight: bold; color: #1a1a2e; }
  .model-card .meta { font-size: 0.8rem; color: #888; margin-top: 0.4rem; }
  .winner-badge { display: inline-block; background: #fef08a; color: #713f12; font-size: 0.7rem; font-weight: bold; padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.4rem; vertical-align: middle; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 1rem; }
  th { background: #1a1a2e; color: white; padding: 0.75rem 1rem; text-align: left; }
  td { padding: 0.6rem 1rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }
  tr:hover { background: #f0f0f5; }
  .badge { padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.78rem; font-weight: bold; display: inline-block; }
  .badge-pass { background: #dcfce7; color: #166534; }
  .badge-fail { background: #fee2e2; color: #991b1b; }
  .badge-error { background: #fef9c3; color: #854d0e; }
  .badge-na { background: #f1f5f9; color: #64748b; }
  .regression { color: #991b1b; }
  .improvement { color: #166534; }
  .section { margin: 2rem 0; }
</style>
</head>
<body>
<h1>AgentLens Sweep Report</h1>
<p>Sweep ID: <code>{{ sweep_id }}</code> &nbsp;·&nbsp; Generated: {{ generated_at }}</p>

<h2>Model Summary</h2>
<div class="models">
{% for run in ranked_runs %}
  <div class="model-card">
    <h3>{{ run.agent_model }}{% if loop.first %}<span class="winner-badge">★ best</span>{% endif %}</h3>
    <div class="pass-rate">{{ run.pass_rate }}%</div>
    <div class="meta">{{ run.results | length }} scenarios &nbsp;·&nbsp; {{ run.avg_steps }} avg steps &nbsp;·&nbsp; {{ run.avg_tokens | int }} avg tokens</div>
  </div>
{% endfor %}
</div>

{% if all_benchmarks %}
<div class="section">
<h2>Per-Benchmark Breakdown</h2>
<table>
<thead>
  <tr>
    <th>Benchmark</th>
    {% for model in model_labels %}<th>{{ model }}</th>{% endfor %}
  </tr>
</thead>
<tbody>
{% for benchmark in all_benchmarks %}
  <tr>
    <td>{{ benchmark }}</td>
    {% for model in model_labels %}
      {% set summary = benchmark_summaries_by_model.get(model, {}).get(benchmark) %}
      <td>{% if summary %}{{ summary.pass_rate }}% ({{ summary.passed }}/{{ summary.total }}){% else %}—{% endif %}</td>
    {% endfor %}
  </tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}

<div class="section">
<h2>Scenario Grid</h2>
<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Name</th>
    <th>Benchmark</th>
    {% for model in model_labels %}<th>{{ model }}</th>{% endfor %}
  </tr>
</thead>
<tbody>
{% for row in grid %}
  <tr>
    <td>{{ row.scenario_id }}</td>
    <td>{{ row.scenario_name }}</td>
    <td>{{ row.benchmark or '—' }}</td>
    {% for model in model_labels %}
      {% set cell = row.cells.get(model, '—') %}
      <td><span class="badge {% if cell == 'PASS' %}badge-pass{% elif cell == 'FAIL' %}badge-fail{% elif cell == 'ERROR' %}badge-error{% else %}badge-na{% endif %}">{{ cell }}</span></td>
    {% endfor %}
  </tr>
{% endfor %}
</tbody>
</table>
</div>

{% if comparison %}
<div class="section">
<h2>Pairwise Comparison: {{ comparison.baseline_config.agent_model }} → {{ comparison.candidate_config.agent_model }}</h2>
<p>Pass rate: <strong>{{ comparison.baseline_pass_rate }}%</strong> → <strong>{{ comparison.candidate_pass_rate }}%</strong>
  (<span class="{% if comparison.delta_pass_rate >= 0 %}improvement{% else %}regression{% endif %}">{{ '%+.1f' | format(comparison.delta_pass_rate) }}%</span>)
</p>

{% if comparison.regressions %}
<h3 class="regression">Regressions ({{ comparison.regressions | length }})</h3>
<table>
<thead><tr><th>Scenario</th><th>Benchmark</th><th>Baseline</th><th>Candidate</th></tr></thead>
<tbody>
{% for r in comparison.regressions %}
  <tr>
    <td>{{ r.scenario_name }} <span style="color:#888;font-size:0.8rem">({{ r.scenario_id }})</span></td>
    <td>{{ r.benchmark or '—' }}</td>
    <td>{{ r.baseline_status | upper }}</td>
    <td>{{ r.candidate_status | upper }}</td>
  </tr>
{% endfor %}
</tbody>
</table>
{% endif %}

{% if comparison.improvements %}
<h3 class="improvement">Improvements ({{ comparison.improvements | length }})</h3>
<table>
<thead><tr><th>Scenario</th><th>Benchmark</th><th>Baseline</th><th>Candidate</th></tr></thead>
<tbody>
{% for r in comparison.improvements %}
  <tr>
    <td>{{ r.scenario_name }} <span style="color:#888;font-size:0.8rem">({{ r.scenario_id }})</span></td>
    <td>{{ r.benchmark or '—' }}</td>
    <td>{{ r.baseline_status | upper }}</td>
    <td>{{ r.candidate_status | upper }}</td>
  </tr>
{% endfor %}
</tbody>
</table>
{% endif %}
</div>
{% endif %}

</body>
</html>
""")


def generate_sweep_report(sweep: SweepResult, output_path: Path | None = None) -> str:
    model_labels = [run.agent_model for run in sweep.model_runs]
    ranked_runs = sweep.ranked_models

    benchmark_summaries_by_model: dict[str, dict[str, object]] = {}
    all_benchmark_set: set[str] = set()
    for run in sweep.model_runs:
        summaries = {
            s.name: s
            for s in summarize_results_by_benchmark(run.results)
            if s.slug != UNASSIGNED_BENCHMARK
        }
        benchmark_summaries_by_model[run.agent_model] = summaries
        all_benchmark_set.update(summaries.keys())

    all_benchmarks = sorted(all_benchmark_set)
    grid = build_scenario_grid(sweep)

    html = SWEEP_REPORT_TEMPLATE.render(
        sweep_id=sweep.sweep_id,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        ranked_runs=ranked_runs,
        model_labels=model_labels,
        benchmark_summaries_by_model=benchmark_summaries_by_model,
        all_benchmarks=all_benchmarks,
        grid=grid,
        comparison=sweep.pairwise_comparison,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html
