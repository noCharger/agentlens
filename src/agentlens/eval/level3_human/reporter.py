"""HTML report generator for eval results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Template

from agentlens.eval.benchmarks import UNASSIGNED_BENCHMARK, summarize_results_by_benchmark
from agentlens.eval.runner import EvalResult

REPORT_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AgentLens Eval Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f8f9fa; }
  h1 { color: #1a1a2e; }
  .summary { display: flex; gap: 1rem; margin: 1rem 0; }
  .card { background: white; border-radius: 8px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card h3 { margin: 0 0 0.5rem; color: #555; font-size: 0.85rem; text-transform: uppercase; }
  .card .value { font-size: 2rem; font-weight: bold; }
  .pass { color: #22c55e; }
  .fail { color: #ef4444; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 1rem; }
  th { background: #1a1a2e; color: white; padding: 0.75rem 1rem; text-align: left; }
  td { padding: 0.75rem 1rem; border-bottom: 1px solid #eee; }
  tr:hover { background: #f0f0f5; }
  .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
  .badge-pass { background: #dcfce7; color: #166534; }
  .badge-partial { background: #fef9c3; color: #854d0e; }
  .badge-risky { background: #ffedd5; color: #9a3412; }
  .badge-fail { background: #fee2e2; color: #991b1b; }
  details { margin: 0.5rem 0; }
  summary { cursor: pointer; font-weight: bold; }
  .details-content { padding: 0.5rem; background: #f8f8f8; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; }
  .reason { color: #991b1b; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>AgentLens Eval Report</h1>
<p>Generated: {{ generated_at }}</p>

<div class="summary">
  <div class="card">
    <h3>Total Scenarios</h3>
    <div class="value">{{ total }}</div>
  </div>
  <div class="card">
    <h3>Passed</h3>
    <div class="value pass">{{ passed }}</div>
  </div>
  <div class="card">
    <h3>Partial</h3>
    <div class="value" style="color: #854d0e;">{{ partial_success }}</div>
  </div>
  <div class="card">
    <h3>Risky</h3>
    <div class="value" style="color: #9a3412;">{{ risky_success }}</div>
  </div>
  <div class="card">
    <h3>Failed</h3>
    <div class="value fail">{{ failed }}</div>
  </div>
  <div class="card">
    <h3>Pass Rate</h3>
    <div class="value">{{ pass_rate }}%</div>
  </div>
  <div class="card">
    <h3>Benchmarks</h3>
    <div class="value">{{ benchmarks_covered }}</div>
  </div>
</div>

{% if benchmark_summaries %}
<h2>Benchmark Summary</h2>
<table>
<thead>
  <tr>
    <th>Benchmark</th>
    <th>Scenarios</th>
    <th>Passed</th>
    <th>Failed</th>
    <th>Pass Rate</th>
  </tr>
</thead>
<tbody>
{% for summary in benchmark_summaries %}
  <tr>
    <td>{{ summary.name }}</td>
    <td>{{ summary.total }}</td>
    <td>{{ summary.passed }}</td>
    <td>{{ summary.failed }}</td>
    <td>{{ summary.pass_rate }}%</td>
  </tr>
{% endfor %}
</tbody>
</table>
{% endif %}

<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Name</th>
    <th>Benchmark</th>
    <th>Category</th>
    <th>Tool Usage</th>
    <th>Output</th>
    <th>Trajectory</th>
    <th>Overall</th>
    <th>Details</th>
  </tr>
</thead>
<tbody>
{% for r in results %}
  <tr>
    <td>{{ r.scenario.id }}</td>
    <td>{{ r.scenario.name }}</td>
    <td>{{ r.scenario.benchmark_name or 'Unassigned' }}</td>
    <td>{{ r.scenario.category }}</td>
    <td><span class="badge {{ 'badge-pass' if r.level1.tool_usage.passed else 'badge-fail' }}">{{ 'PASS' if r.level1.tool_usage.passed else 'FAIL' }}</span></td>
    <td><span class="badge {{ 'badge-pass' if r.level1.output_format.passed else 'badge-fail' }}">{{ 'PASS' if r.level1.output_format.passed else 'FAIL' }}</span></td>
    <td><span class="badge {{ 'badge-pass' if r.level1.trajectory.passed else 'badge-fail' }}">{{ 'PASS' if r.level1.trajectory.passed else 'FAIL' }}</span></td>
    <td><span class="badge {% if r.status.value == 'passed' %}badge-pass{% elif r.status.value == 'partial_success' %}badge-partial{% elif r.status.value == 'risky_success' %}badge-risky{% else %}badge-fail{% endif %}">{{ r.status.value | upper }}</span></td>
    <td>
      <details>
        <summary>Details</summary>
        <div class="details-content">
          <p><strong>Benchmark:</strong> {{ r.scenario.benchmark_name or 'Unassigned' }}</p>
          <p><strong>Query:</strong> {{ r.scenario.input_query }}</p>
          <p><strong>Expected tools:</strong> {{ r.level1.tool_usage.expected_tools | join(', ') }}</p>
          <p><strong>Actual tools:</strong> {{ r.level1.tool_usage.actual_tools | join(', ') or 'none' }}</p>
          {% if r.level1.tool_usage.missing_tools %}
          <p class="reason">Missing tools: {{ r.level1.tool_usage.missing_tools | join(', ') }}</p>
          {% endif %}
          {% if r.level1.output_format.missing_substrings %}
          <p class="reason">Missing output: {{ r.level1.output_format.missing_substrings | join(', ') }}</p>
          {% endif %}
          <p><strong>Steps:</strong> {{ r.level1.trajectory.total_steps }} / {{ r.level1.trajectory.max_steps }}</p>
          <p><strong>Tokens:</strong> {{ r.level1.trajectory.total_prompt_tokens }} prompt + {{ r.level1.trajectory.total_completion_tokens }} completion</p>
          {% if r.level1.trajectory.has_loop %}
          <p class="reason">Loop detected!</p>
          {% endif %}
          {% for reason in r.level1.trajectory.reasons %}
          <p class="reason">{{ reason }}</p>
          {% endfor %}
          {% if r.level2_scores %}
          <p><strong>L2 Scores:</strong></p>
          {% for dim, score in r.level2_scores.items() %}
          <p>&nbsp;&nbsp;{{ dim }}: {{ score }}/5</p>
          {% endfor %}
          {% endif %}
          {% if r.risk_signals %}
          <p><strong>Risk Signals:</strong></p>
          {% for signal in r.risk_signals %}
          <p class="reason" style="color: #9a3412;">&nbsp;&nbsp;⚠ {{ signal }}</p>
          {% endfor %}
          {% endif %}
          {% if r.error %}
          <p class="reason">Error: {{ r.error }}</p>
          {% endif %}
        </div>
      </details>
    </td>
  </tr>
{% endfor %}
</tbody>
</table>
</body>
</html>
""")


def generate_report(results: list[EvalResult], output_path: Path | None = None) -> str:
    from agentlens.core.models import TraceStatus

    total = len(results)
    passed = sum(1 for r in results if r.status == TraceStatus.PASSED)
    partial_success = sum(1 for r in results if r.status == TraceStatus.PARTIAL_SUCCESS)
    risky_success = sum(1 for r in results if r.status == TraceStatus.RISKY_SUCCESS)
    failed = total - passed - partial_success - risky_success
    pass_rate = round(passed / total * 100, 1) if total > 0 else 0
    benchmark_summaries = [
        summary
        for summary in summarize_results_by_benchmark(results)
        if summary.slug != UNASSIGNED_BENCHMARK
    ]

    html = REPORT_TEMPLATE.render(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        total=total,
        passed=passed,
        partial_success=partial_success,
        risky_success=risky_success,
        failed=failed,
        pass_rate=pass_rate,
        benchmarks_covered=len(benchmark_summaries),
        benchmark_summaries=benchmark_summaries,
        results=results,
    )

    if output_path:
        output_path.write_text(html)

    return html
