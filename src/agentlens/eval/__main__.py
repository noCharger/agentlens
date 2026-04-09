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

from agentlens.dataset.builder import (
    build_dataset_version_from_scenarios,
    compute_dataset_fingerprint,
    dataset_version_to_scenarios,
    load_dataset_version_from_path,
    make_deterministic_id_factory,
)
from agentlens.deepseek import DeepSeekPreflightError, validate_deepseek_preflight
from agentlens.openrouter import OpenRouterPreflightError, validate_openrouter_preflight
from agentlens.zhipu import ZhipuPreflightError, validate_zhipu_preflight
from agentlens.eval.benchmarks import (
    UNASSIGNED_BENCHMARK,
    collect_benchmark_inventory,
    filter_scenarios_by_benchmark,
    summarize_results_by_benchmark,
)
from agentlens.eval.scenarios import load_runtime_scenarios
from agentlens.eval.runner import EvalResult, execute_and_eval, QuotaExhaustedError
from agentlens.eval.level3_human.reporter import generate_report
from agentlens.core.models import DatasetSource

console = Console()


def _badge(passed: bool) -> str:
    return "[green]PASS[/green]" if passed else "[red]FAIL[/red]"


def _truncate(text: str, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _format_l1_extra(level1) -> str:
    checks = level1.supplemental_checks
    if not checks:
        return "\u2014"
    labels = []
    for name, passed in checks.items():
        labels.append(f"{name}={'PASS' if passed else 'FAIL'}")
    return ", ".join(labels)


def _reason_preview(result: EvalResult) -> str:
    if result.error:
        return result.error
    if result.level1.failure_reasons:
        return result.level1.failure_reasons[0]
    if result.level2_reason_lines:
        return result.level2_reason_lines[0]
    if result.risk_signals:
        return result.risk_signals[0]
    return ""


def _print_results(results: list[EvalResult]) -> None:
    table = Table(title="AgentLens Eval Results")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Benchmark")
    table.add_column("Category")
    table.add_column("Tools", justify="center")
    table.add_column("Output", justify="center")
    table.add_column("Trajectory", justify="center")
    table.add_column("L1 Extra")
    table.add_column("L2 Score", justify="center")
    table.add_column("Overall", justify="center")
    table.add_column("Why")

    for r in results:
        l2 = ""
        if r.judge_overall_score is not None:
            l2 = f"{r.judge_overall_score:.1f}/5"

        table.add_row(
            r.scenario.id, r.scenario.name, r.scenario.benchmark_name or "\u2014", r.scenario.category,
            _badge(r.level1.tool_usage.passed),
            _badge(r.level1.output_format.passed),
            _badge(r.level1.trajectory.passed),
            _format_l1_extra(r.level1),
            l2 or "\u2014",
            _badge(r.passed),
            _truncate(_reason_preview(r)) or "\u2014",
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

    detailed_results = [
        r for r in results
        if r.error or r.level1.failure_reasons or r.level2_reason_lines or r.risk_signals
    ]
    if detailed_results:
        console.print("\n[bold]Scenario Details[/bold]")
        for r in detailed_results:
            console.print(f"[cyan]{r.scenario.id}[/cyan]: {r.scenario.name}")
            if r.error:
                console.print(f"  [red]Error:[/red] {r.error}")
            if r.level1.failure_reasons:
                console.print("  [bold]L1 Reasons:[/bold]")
                for reason in r.level1.failure_reasons:
                    console.print(f"    - {reason}")
            if r.level1.supplemental_checks:
                console.print(
                    "  [bold]L1 Extra Checks:[/bold] "
                    + _format_l1_extra(r.level1)
                )
            if r.level2_reason_lines:
                console.print("  [bold]Judge Explanations:[/bold]")
                for reason in r.level2_reason_lines:
                    console.print(f"    - {reason}")
            if r.risk_signals:
                console.print("  [bold]Risk Signals:[/bold]")
                for signal in r.risk_signals:
                    console.print(f"    - {signal}")


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


def _resolve_platform_dataset_name(args) -> str:
    if args.platform_dataset_name:
        return args.platform_dataset_name
    if len(args.benchmark) == 1:
        return f"{args.benchmark[0]}-dataset"
    return "agentlens-dataset"


def _resolve_platform_run_name(args) -> str:
    if args.platform_run_name:
        return args.platform_run_name
    return "agentlens-eval-run"


def _resolve_platform_project_name(args) -> str:
    if args.platform_project:
        return args.platform_project
    if len(args.benchmark) == 1:
        return f"{args.benchmark[0]} project"
    return "AgentLens Local Project"


def _resolve_platform_project_slug(args) -> str:
    if args.platform_project_slug:
        return args.platform_project_slug

    from agentlens.core.repository import slugify_project_name

    return slugify_project_name(_resolve_platform_project_name(args))


def _build_dataset_repositories(args):
    repositories: list[tuple[str, object]] = []
    if args.platform_store:
        from agentlens.core.repository import FileCoreRepository

        repositories.append(("store", FileCoreRepository(args.platform_store)))
    if args.platform_sqlite:
        from agentlens.core.sqlite_repository import SQLiteCoreRepository

        repositories.append(("sqlite", SQLiteCoreRepository(args.platform_sqlite)))
    return repositories


def _resolve_runtime_dataset_source(args) -> DatasetSource:
    if args.benchmark:
        return DatasetSource.BENCHMARK_IMPORT
    return DatasetSource.MANUAL_CURATION


def _filter_eval_scenarios(args, scenarios):
    filtered = scenarios

    if args.scenario_id:
        filtered = [scenario for scenario in filtered if scenario.id == args.scenario_id]
        if not filtered:
            console.print(f"[red]Scenario '{args.scenario_id}' not found[/red]")
            sys.exit(1)

    if args.benchmark:
        filtered = filter_scenarios_by_benchmark(filtered, args.benchmark)
        if not filtered:
            requested = ", ".join(args.benchmark)
            console.print(f"[red]No scenarios matched benchmark filter: {requested}[/red]")
            sys.exit(1)

    return filtered


def _load_dataset_version_from_platform(args):
    repositories = _build_dataset_repositories(args)
    if not repositories:
        raise ValueError(
            "--dataset-version-id requires storage access: "
            "pass --platform-store or --platform-sqlite"
        )
    project_slug = _resolve_platform_project_slug(args)
    for _, repository in repositories:
        dataset_version = repository.load_dataset_version(project_slug, args.dataset_version_id)
        if dataset_version is not None:
            return dataset_version

    raise FileNotFoundError(
        f"Dataset version '{args.dataset_version_id}' not found in project '{project_slug}'"
    )


def _persist_runtime_dataset_version(args, dataset_version) -> None:
    if args.dataset_version_file or args.dataset_version_id:
        return

    repositories = _build_dataset_repositories(args)
    if not repositories:
        return

    project_name = _resolve_platform_project_name(args)
    project_slug = _resolve_platform_project_slug(args)
    for label, repository in repositories:
        existing = repository.load_dataset_version(project_slug, dataset_version.id)
        if existing is not None:
            continue
        repository.save_dataset_version(
            project_name=project_name,
            project_slug=project_slug,
            dataset_version=dataset_version,
        )
        console.print(
            f"[dim]Persisted runtime dataset version {dataset_version.id} "
            f"to {label} backend[/dim]"
        )


def _resolve_eval_dataset_and_scenarios(args):
    if args.dataset_version_file and args.dataset_version_id:
        raise ValueError("Use either --dataset-version-file or --dataset-version-id, not both")

    if args.dataset_version_file:
        dataset_version = load_dataset_version_from_path(args.dataset_version_file)
        scenarios = dataset_version_to_scenarios(dataset_version)
        scenarios = _filter_eval_scenarios(args, scenarios)
        console.print(
            f"Loaded dataset version [bold]{dataset_version.id}[/bold] "
            f"({len(dataset_version.items)} items) from {args.dataset_version_file}"
        )
        return dataset_version, scenarios

    if args.dataset_version_id:
        dataset_version = _load_dataset_version_from_platform(args)
        scenarios = dataset_version_to_scenarios(dataset_version)
        scenarios = _filter_eval_scenarios(args, scenarios)
        console.print(
            f"Loaded dataset version [bold]{dataset_version.id}[/bold] "
            f"({len(dataset_version.items)} items) from project storage"
        )
        return dataset_version, scenarios

    scenarios = load_runtime_scenarios(
        args.scenarios,
        benchmark_data_root=args.benchmark_data_root,
        benchmarks=args.benchmark,
    )
    console.print(f"Loaded [bold]{len(scenarios)}[/bold] scenarios from {args.scenarios}")
    scenarios = _filter_eval_scenarios(args, scenarios)
    fingerprint = compute_dataset_fingerprint(scenarios)
    dataset_version = build_dataset_version_from_scenarios(
        scenarios,
        name=_resolve_platform_dataset_name(args),
        source=_resolve_runtime_dataset_source(args),
        dataset_fingerprint=fingerprint,
        id_factory=make_deterministic_id_factory(fingerprint),
        metadata={
            "builder": "agentlens.eval",
            "benchmarks": sorted({scenario.benchmark for scenario in scenarios if scenario.benchmark}),
            "scenario_count": len(scenarios),
        },
    )

    project_slug = _resolve_platform_project_slug(args)
    for label, repository in _build_dataset_repositories(args):
        existing = repository.load_dataset_version(project_slug, dataset_version.id)
        if existing is not None:
            console.print(
                f"[dim]Reusing dataset version {existing.id} from {label} backend[/dim]"
            )
            dataset_version = existing
            break
    return dataset_version, scenarios


def _handle_platform_outputs(args, results, settings, *, dataset_version=None) -> None:
    if not args.platform_export and not args.platform_store and not args.platform_sqlite:
        return

    from agentlens.core.exporters import (
        build_closed_loop_snapshot,
        write_closed_loop_snapshot,
    )

    snapshot = build_closed_loop_snapshot(
        results,
        dataset_name=_resolve_platform_dataset_name(args),
        run_name=_resolve_platform_run_name(args),
        source="cli",
        agent_framework=getattr(settings, "agent_framework", "langgraph"),
        agent_model=settings.agent_model,
        judge_model=settings.judge_model if args.level2 else "",
        dataset_record=dataset_version,
    )

    if args.platform_export:
        write_closed_loop_snapshot(snapshot, args.platform_export)
        console.print(f"[green]Platform snapshot saved to {args.platform_export}[/green]")

    if args.platform_store:
        from agentlens.core.repository import FileCoreRepository

        repository = FileCoreRepository(args.platform_store)
        stored = repository.save_snapshot(
            project_name=_resolve_platform_project_name(args),
            project_slug=args.platform_project_slug,
            snapshot=snapshot,
            idempotency_key=args.platform_idempotency_key,
        )
        console.print(f"[green]Platform records stored in {stored.project_dir}[/green]")

    if args.platform_sqlite:
        from agentlens.core.sqlite_repository import SQLiteCoreRepository

        repository = SQLiteCoreRepository(args.platform_sqlite)
        database_path = repository.save_snapshot(
            project_name=_resolve_platform_project_name(args),
            project_slug=args.platform_project_slug,
            snapshot=snapshot,
            idempotency_key=args.platform_idempotency_key,
        )
        console.print(f"[green]Platform records stored in SQLite {database_path}[/green]")


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
        help=(
            "Override agent model, for example gemini:gemini-2.5-flash, "
            "deepseek:deepseek-chat, openrouter:openai/gpt-4o-mini, or zhipu:glm-4-plus"
        ),
    )
    parser.add_argument(
        "--agent-framework",
        choices=["langgraph", "ag2"],
        help="Override the agent runtime framework.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help=(
            "Override judge model, for example gemini:gemini-2.5-flash-lite, "
            "deepseek:deepseek-chat, openrouter:openai/gpt-4o-mini, or zhipu:glm-4-plus"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Run only scenarios for the given benchmark slug or display name",
    )
    parser.add_argument(
        "--dataset-version-id",
        type=str,
        help="Run eval against a persisted dataset version id from platform store/sqlite",
    )
    parser.add_argument(
        "--dataset-version-file",
        type=Path,
        help="Run eval against a dataset version JSON file",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List supported benchmarks and loaded scenario counts",
    )
    parser.add_argument("--scenario-id", type=str, help="Run a single scenario")
    parser.add_argument("--preset", type=str, default="full")
    parser.add_argument("--geval", action="store_true", help="Enable G-Eval CoT meta-evaluation for L2 judge")
    parser.add_argument("--task-completion", action="store_true", help="Enable task completion metric")
    parser.add_argument("--answer-relevancy", action="store_true", help="Enable answer relevancy metric")
    parser.add_argument("--hallucination", action="store_true", help="Enable hallucination detection metric")
    parser.add_argument("--faithfulness", action="store_true", help="Enable context-support faithfulness metric")
    parser.add_argument("--all-metrics", action="store_true", help="Enable all L2 extension metrics")
    parser.add_argument(
        "--platform-export",
        type=Path,
        help="Write a closed-loop platform snapshot JSON for the completed run",
    )
    parser.add_argument(
        "--platform-dataset-name",
        type=str,
        help="Dataset name to use when exporting a platform snapshot",
    )
    parser.add_argument(
        "--platform-run-name",
        type=str,
        help="Run name to use when exporting a platform snapshot",
    )
    parser.add_argument(
        "--platform-store",
        type=Path,
        help="Persist platform records to a local store (and load dataset version ids from it)",
    )
    parser.add_argument(
        "--platform-sqlite",
        type=Path,
        help="Persist platform records to SQLite (and load dataset version ids from it)",
    )
    parser.add_argument(
        "--platform-project",
        type=str,
        help="Project name to use when persisting a platform store snapshot",
    )
    parser.add_argument(
        "--platform-project-slug",
        type=str,
        help="Project slug to use when persisting a platform store snapshot",
    )
    parser.add_argument(
        "--platform-idempotency-key",
        type=str,
        help="Optional idempotency key for resilient platform snapshot persistence",
    )
    args = parser.parse_args()

    if args.list_benchmarks:
        if args.dataset_version_id or args.dataset_version_file:
            parser.error("--list-benchmarks cannot be used with dataset-version inputs")
        scenarios = load_runtime_scenarios(
            args.scenarios,
            benchmark_data_root=args.benchmark_data_root,
            benchmarks=None,
        )
        console.print(f"Loaded [bold]{len(scenarios)}[/bold] scenarios from {args.scenarios}")
        console.print()
        _print_benchmark_inventory(scenarios)
        return

    try:
        dataset_version, scenarios = _resolve_eval_dataset_and_scenarios(args)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if args.dry_run:
        console.print("[yellow]Dry run mode[/yellow]")
        for s in scenarios:
            benchmark = f" [{s.benchmark_name}]" if s.benchmark_name else ""
            console.print(f"  [{s.category}]{benchmark} {s.id}: {s.name}")
        return

    _persist_runtime_dataset_version(args, dataset_version)

    try:
        from agentlens.config import get_settings
        settings_overrides = {}
        if args.agent_model:
            settings_overrides["agent_model"] = args.agent_model
        if args.agent_framework:
            settings_overrides["agent_framework"] = args.agent_framework
        if args.judge_model:
            settings_overrides["judge_model"] = args.judge_model
        settings = get_settings(**settings_overrides)
    except Exception as e:
        console.print(f"[red]Failed to load settings: {e}[/red]")
        console.print(
            "[yellow]Set GOOGLE_API_KEY and/or DEEPSEEK_API_KEY and/or OPENROUTER_API_KEY "
            "and/or ZHIPU_API_KEY "
            "in .env or environment[/yellow]"
        )
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

    try:
        openrouter_summary = validate_openrouter_preflight(
            settings,
            require_judge=args.level2,
        )
    except OpenRouterPreflightError as e:
        console.print(f"[red]OpenRouter preflight failed:[/red] {e}")
        sys.exit(1)

    if openrouter_summary is not None:
        console.print(
            f"[dim]OpenRouter key check passed ({openrouter_summary.formatted_status})[/dim]"
        )

    try:
        zhipu_summary = validate_zhipu_preflight(
            settings,
            require_judge=args.level2,
        )
    except ZhipuPreflightError as e:
        console.print(f"[red]Zhipu preflight failed:[/red] {e}")
        sys.exit(1)

    if zhipu_summary is not None:
        console.print(
            f"[dim]Zhipu key check passed ({zhipu_summary.formatted_status})[/dim]"
        )

    agent_framework = getattr(settings, "agent_framework", "langgraph")
    console.print(
        f"\nRunning [bold]{len(scenarios)}[/bold] scenarios with "
        f"[cyan]{settings.agent_model}[/cyan] via [magenta]{agent_framework}[/magenta]"
    )
    if args.level2:
        console.print(f"L2 judge: [cyan]{settings.judge_model}[/cyan]")
    console.print()

    results: list[EvalResult] = []
    interrupted_by_user = False
    for i, scenario in enumerate(scenarios, 1):
        console.print(f"[{i}/{len(scenarios)}] [cyan]{scenario.id}[/cyan]: {scenario.name}...", end=" ")

        use_geval = args.geval or args.all_metrics or settings.judge_use_geval
        use_task_completion = args.task_completion or args.all_metrics or settings.judge_task_completion
        use_answer_relevancy = args.answer_relevancy or args.all_metrics or settings.judge_answer_relevancy
        use_hallucination = args.hallucination or args.all_metrics or settings.judge_hallucination
        use_faithfulness = args.faithfulness or args.all_metrics or settings.judge_faithfulness

        try:
            result = execute_and_eval(
                scenario=scenario, settings=settings,
                preset=args.preset, with_level2=args.level2,
                use_geval=use_geval,
                task_completion=use_task_completion,
                answer_relevancy=use_answer_relevancy,
                hallucination=use_hallucination,
                faithfulness=use_faithfulness,
            )
        except KeyboardInterrupt:
            interrupted_by_user = True
            console.print("[yellow]INTERRUPTED[/yellow]")
            console.print(
                f"  [yellow]Interrupted by user (Ctrl+C). "
                f"Keeping {len(results)} completed result(s) for reporting.[/yellow]"
            )
            break
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
            elif args.level2 and result.level2_scores and result.scenario.evaluation_mode == "llm_judge":
                scored = {k: v for k, v in result.level2_scores.items() if v >= 0}
                if scored:
                    details = ", ".join(f"{dim}={score:.1f}" for dim, score in scored.items())
                    overall = result.judge_overall_score
                    overall_text = f"{overall:.1f}" if overall is not None else "n/a"
                    console.print(
                        "  [dim]L2 below threshold: "
                        f"overall={overall_text}, threshold={result.scenario.judge_threshold:.1f}, "
                        f"details=({details})[/dim]"
                    )
            if result.level1.failure_reasons:
                for reason in result.level1.failure_reasons[:3]:
                    console.print(f"  [dim]L1: {reason}[/dim]")
            if result.level2_reason_lines:
                for reason in result.level2_reason_lines[:2]:
                    console.print(f"  [dim]Judge: {reason}[/dim]")

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
        status = "partial report" if interrupted_by_user else "report"
        console.print(f"\n[green]{status.capitalize()} saved to {args.output}[/green]")

    _handle_platform_outputs(args, results, settings, dataset_version=dataset_version)


if __name__ == "__main__":
    main()
