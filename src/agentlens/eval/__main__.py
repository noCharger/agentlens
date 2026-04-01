"""CLI entry point for the eval runner.

Usage:
    python -m agentlens.eval --dry-run
    python -m agentlens.eval --scenario-id tc-001
    python -m agentlens.eval --benchmark swe-bench-pro
    python -m agentlens.eval --list-benchmarks
    python -m agentlens.eval --level2 --output report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentlens.deepseek import DeepSeekPreflightError, validate_deepseek_preflight
from agentlens.eval.benchmarks import (
    UNASSIGNED_BENCHMARK,
    collect_benchmark_inventory,
    filter_scenarios_by_benchmark,
    summarize_results_by_benchmark,
)
from agentlens.eval.scenarios import load_runtime_scenarios
from agentlens.eval.runner import EvalResult, execute_and_eval, QuotaExhaustedError
from agentlens.eval.level3_human.reporter import generate_report

console = Console()


def _badge(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


def _print_results(results: list[EvalResult]) -> None:
    table = Table(title="AgentLens Eval Results")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Benchmark")
    table.add_column("Category")
    table.add_column("Tools", justify="center")
    table.add_column("Output", justify="center")
    table.add_column("Trajectory", justify="center")
    table.add_column("L2 Score", justify="center")
    table.add_column("Overall", justify="center")

    for r in results:
        l2 = ""
        if r.level2_scores:
            scores = [v for v in r.level2_scores.values() if v > 0]
            if scores:
                l2 = f"{sum(scores) / len(scores):.1f}/5"

        table.add_row(
            r.scenario.id, r.scenario.name, r.scenario.benchmark_name or "\u2014", r.scenario.category,
            _badge(r.level1.tool_usage.passed),
            _badge(r.level1.output_format.passed),
            _badge(r.level1.trajectory.passed),
            l2 or "\u2014",
            _badge(r.passed),
        )

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = round(passed / total * 100, 1) if total else 0
    console.print(f"\n[bold]{passed}/{total} passed ({rate}%)[/bold]")

    benchmark_summaries = summarize_results_by_benchmark(results)
    if any(summary.slug != UNASSIGNED_BENCHMARK for summary in benchmark_summaries):
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Benchmark")
        summary_table.add_column("Scenarios", justify="right")
        summary_table.add_column("Passed", justify="right")
        summary_table.add_column("Pass Rate", justify="right")
        for summary in benchmark_summaries:
            if summary.slug == UNASSIGNED_BENCHMARK:
                continue
            summary_table.add_row(
                summary.name,
                str(summary.total),
                str(summary.passed),
                f"{summary.pass_rate:.1f}%",
            )
        console.print()
        console.print(summary_table)

    errors = [r for r in results if r.error]
    if errors:
        console.print(f"\n[red]{len(errors)} scenario(s) had errors:[/red]")
        for r in errors:
            console.print(f"  [cyan]{r.scenario.id}[/cyan]: {r.error}")


def _print_benchmark_inventory(scenarios) -> None:
    table = Table(title="AgentLens Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Slug")
    table.add_column("Scenarios", justify="right")
    table.add_column("Status")

    for row in collect_benchmark_inventory(scenarios):
        table.add_row(
            row.name,
            row.slug if row.slug != UNASSIGNED_BENCHMARK else "\u2014",
            str(row.scenario_count),
            "built-in" if row.built_in else "custom",
        )

    console.print(table)


def _init_metrics(settings):
    try:
        from agentlens.observability.setup import create_meter_provider
        from opentelemetry import metrics as otel_metrics

        provider = create_meter_provider(settings)
        otel_metrics.set_meter_provider(provider)
        console.print("[dim]Metrics enabled[/dim]")
        return provider
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="AgentLens Eval Runner")
    parser.add_argument("--scenarios", type=Path, default=Path("src/agentlens/scenarios"))
    parser.add_argument(
        "--benchmark-data-root",
        type=Path,
        default=Path("data/benchmarks"),
        help="Directory containing downloaded raw benchmark files for dynamic loading",
    )
    parser.add_argument("--output", type=Path, help="HTML report output path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--level2", action="store_true", help="Enable LLM-as-Judge")
    parser.add_argument(
        "--agent-model",
        type=str,
        help="Override agent model, for example gemini:gemini-2.5-flash or deepseek:deepseek-chat",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Override judge model, for example gemini:gemini-2.5-flash-lite or deepseek:deepseek-chat",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Run only scenarios for the given benchmark slug or display name",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List supported benchmarks and loaded scenario counts",
    )
    parser.add_argument("--scenario-id", type=str, help="Run a single scenario")
    parser.add_argument("--preset", type=str, default="full")
    args = parser.parse_args()

    scenarios = load_runtime_scenarios(
        args.scenarios,
        benchmark_data_root=args.benchmark_data_root,
        benchmarks=None if args.list_benchmarks else args.benchmark,
    )
    console.print(f"Loaded [bold]{len(scenarios)}[/bold] scenarios from {args.scenarios}")

    if args.list_benchmarks:
        console.print()
        _print_benchmark_inventory(scenarios)
        return

    if args.scenario_id:
        scenarios = [s for s in scenarios if s.id == args.scenario_id]
        if not scenarios:
            console.print(f"[red]Scenario '{args.scenario_id}' not found[/red]")
            sys.exit(1)

    if args.benchmark:
        scenarios = filter_scenarios_by_benchmark(scenarios, args.benchmark)
        if not scenarios:
            requested = ", ".join(args.benchmark)
            console.print(f"[red]No scenarios matched benchmark filter: {requested}[/red]")
            sys.exit(1)

    if args.dry_run:
        console.print("[yellow]Dry run mode[/yellow]")
        for s in scenarios:
            benchmark = f" [{s.benchmark_name}]" if s.benchmark_name else ""
            console.print(f"  [{s.category}]{benchmark} {s.id}: {s.name}")
        return

    try:
        from agentlens.config import get_settings
        settings_overrides = {}
        if args.agent_model:
            settings_overrides["agent_model"] = args.agent_model
        if args.judge_model:
            settings_overrides["judge_model"] = args.judge_model
        settings = get_settings(**settings_overrides)
    except Exception as e:
        console.print(f"[red]Failed to load settings: {e}[/red]")
        console.print("[yellow]Set GOOGLE_API_KEY and/or DEEPSEEK_API_KEY in .env or environment[/yellow]")
        sys.exit(1)

    meter_provider = _init_metrics(settings)

    try:
        deepseek_balance = validate_deepseek_preflight(
            settings,
            require_judge=args.level2,
        )
    except DeepSeekPreflightError as e:
        console.print(f"[red]DeepSeek preflight failed:[/red] {e}")
        sys.exit(1)

    if deepseek_balance is not None:
        console.print(
            f"[dim]DeepSeek balance check passed ({deepseek_balance.formatted_totals})[/dim]"
        )

    console.print(f"\nRunning [bold]{len(scenarios)}[/bold] scenarios with [cyan]{settings.agent_model}[/cyan]")
    if args.level2:
        console.print(f"L2 judge: [cyan]{settings.judge_model}[/cyan]")
    console.print()

    results: list[EvalResult] = []
    for i, scenario in enumerate(scenarios, 1):
        console.print(f"[{i}/{len(scenarios)}] [cyan]{scenario.id}[/cyan]: {scenario.name}...", end=" ")

        try:
            result = execute_and_eval(
                scenario=scenario, settings=settings,
                preset=args.preset, with_level2=args.level2,
            )
        except QuotaExhaustedError:
            console.print("[red]QUOTA EXHAUSTED[/red]")
            remaining = [s.id for s in scenarios[i:]]
            if remaining:
                console.print(f"  [yellow]Skipping {len(remaining)}: {', '.join(remaining)}[/yellow]")
            break

        results.append(result)
        if result.passed:
            console.print("[green]PASS[/green]")
        else:
            console.print("[red]FAIL[/red]")
            if result.error:
                console.print(f"  [dim]{result.error}[/dim]")

    console.print()
    _print_results(results)

    if meter_provider:
        try:
            meter_provider.force_flush()
            meter_provider.shutdown()
        except Exception:
            pass

    if args.output:
        generate_report(results, output_path=args.output)
        console.print(f"\n[green]Report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
