"""CLI entry point for the eval runner.

Usage:
    python -m agentlens.eval --dry-run
    python -m agentlens.eval --scenario-id tc-001
    python -m agentlens.eval --level2 --output report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentlens.eval.scenarios import load_scenarios_from_dir
from agentlens.eval.runner import EvalResult, execute_and_eval, QuotaExhaustedError
from agentlens.eval.level3_human.reporter import generate_report

console = Console()


def _badge(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


def _print_results(results: list[EvalResult]) -> None:
    table = Table(title="AgentLens Eval Results")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
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
            r.scenario.id, r.scenario.name, r.scenario.category,
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

    errors = [r for r in results if r.error]
    if errors:
        console.print(f"\n[red]{len(errors)} scenario(s) had errors:[/red]")
        for r in errors:
            console.print(f"  [cyan]{r.scenario.id}[/cyan]: {r.error}")


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
    parser.add_argument("--output", type=Path, help="HTML report output path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--level2", action="store_true", help="Enable LLM-as-Judge")
    parser.add_argument("--scenario-id", type=str, help="Run a single scenario")
    parser.add_argument("--preset", type=str, default="full")
    args = parser.parse_args()

    scenarios = load_scenarios_from_dir(args.scenarios)
    console.print(f"Loaded [bold]{len(scenarios)}[/bold] scenarios from {args.scenarios}")

    if args.scenario_id:
        scenarios = [s for s in scenarios if s.id == args.scenario_id]
        if not scenarios:
            console.print(f"[red]Scenario '{args.scenario_id}' not found[/red]")
            sys.exit(1)

    if args.dry_run:
        console.print("[yellow]Dry run mode[/yellow]")
        for s in scenarios:
            console.print(f"  [{s.category}] {s.id}: {s.name}")
        return

    try:
        from agentlens.config import get_settings
        settings = get_settings()
    except Exception as e:
        console.print(f"[red]Failed to load settings: {e}[/red]")
        console.print("[yellow]Set GOOGLE_API_KEY in .env or environment[/yellow]")
        sys.exit(1)

    meter_provider = _init_metrics(settings)

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
        except QuotaExhaustedError as e:
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
