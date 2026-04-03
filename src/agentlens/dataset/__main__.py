"""CLI entry point for dataset-version generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from agentlens.dataset.builder import (
    build_dataset_version_from_scenarios,
    write_dataset_version,
)
from agentlens.eval.benchmarks import filter_scenarios_by_benchmark
from agentlens.eval.scenarios import load_runtime_scenarios
from agentlens.core.models import DatasetSource
from agentlens.core.repository import FileCoreRepository
from agentlens.core.sqlite_repository import SQLiteCoreRepository

console = Console()


def _resolve_dataset_name(args) -> str:
    if args.name:
        return args.name
    if len(args.benchmark) == 1:
        return f"{args.benchmark[0]}-dataset"
    return "agentlens-dataset"


def _resolve_dataset_source(args) -> DatasetSource:
    if args.source:
        return DatasetSource(args.source)
    if args.benchmark:
        return DatasetSource.BENCHMARK_IMPORT
    return DatasetSource.MANUAL_CURATION


def _build_repository(args):
    if args.sqlite:
        return SQLiteCoreRepository(args.sqlite)
    if args.store:
        return FileCoreRepository(args.store)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable dataset versions for eval runs")
    parser.add_argument("--scenarios", type=Path, default=Path("src/agentlens/scenarios"))
    parser.add_argument(
        "--benchmark-data-root",
        type=Path,
        default=Path("data/benchmarks"),
        help="Directory containing raw benchmark files for dynamic loading",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Include only scenarios for the given benchmark slug or display name",
    )
    parser.add_argument("--scenario-id", type=str, help="Include only one scenario id")
    parser.add_argument("--name", type=str, help="Dataset name")
    parser.add_argument("--version", type=str, default="v1", help="Dataset version label")
    parser.add_argument(
        "--source",
        choices=[source.value for source in DatasetSource],
        help="Dataset source type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write dataset version JSON to this path",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Persist dataset version into a local file repository",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        help="Persist dataset version into a local SQLite repository",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="AgentLens Local Project",
        help="Project name used when persisting to store/sqlite",
    )
    parser.add_argument(
        "--project-slug",
        type=str,
        help="Project slug used when persisting to store/sqlite",
    )
    args = parser.parse_args()

    if args.store and args.sqlite:
        parser.error("Choose only one of --store or --sqlite")

    scenarios = load_runtime_scenarios(
        args.scenarios,
        benchmark_data_root=args.benchmark_data_root,
        benchmarks=args.benchmark or None,
    )

    if args.scenario_id:
        scenarios = [scenario for scenario in scenarios if scenario.id == args.scenario_id]
        if not scenarios:
            console.print(f"[red]Scenario '{args.scenario_id}' not found[/red]")
            sys.exit(1)

    if args.benchmark:
        scenarios = filter_scenarios_by_benchmark(scenarios, args.benchmark)
        if not scenarios:
            requested = ", ".join(args.benchmark)
            console.print(f"[red]No scenarios matched benchmark filter: {requested}[/red]")
            sys.exit(1)

    if not scenarios:
        console.print("[red]No scenarios found for dataset build[/red]")
        sys.exit(1)

    dataset_version = build_dataset_version_from_scenarios(
        scenarios,
        name=_resolve_dataset_name(args),
        version=args.version,
        source=_resolve_dataset_source(args),
        metadata={
            "builder": "agentlens.dataset",
            "scenario_count": len(scenarios),
            "benchmarks": sorted({scenario.benchmark for scenario in scenarios if scenario.benchmark}),
        },
    )

    console.print(
        f"[green]Built dataset version[/green] {dataset_version.id} "
        f"({dataset_version.item_count} items)"
    )

    if args.output:
        write_dataset_version(dataset_version, args.output)
        console.print(f"[green]Dataset version JSON saved to {args.output}[/green]")

    repository = _build_repository(args)
    if repository is not None:
        repository.save_dataset_version(
            project_name=args.project,
            project_slug=args.project_slug,
            dataset_version=dataset_version,
        )
        target = args.sqlite if args.sqlite else args.store
        console.print(
            f"[green]Dataset version stored in {target} for project "
            f"{args.project_slug or args.project}[/green]"
        )


if __name__ == "__main__":
    main()
