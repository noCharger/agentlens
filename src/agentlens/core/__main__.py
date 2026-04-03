"""CLI helpers for inspecting locally persisted platform records."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentlens.core.repository import FileCoreRepository
from agentlens.core.sqlite_repository import SQLiteCoreRepository

console = Console()


def _build_repository(args):
    if args.sqlite:
        return SQLiteCoreRepository(args.sqlite)
    return FileCoreRepository(args.store)


def _print_projects(repository, *, limit: int | None = None, offset: int = 0) -> None:
    projects = repository.list_projects(limit=limit, offset=offset)
    table = Table(title="AgentLens Platform Projects")
    table.add_column("Name")
    table.add_column("Slug", style="cyan")
    table.add_column("Tags")

    for project in projects:
        table.add_row(
            project.name,
            project.slug,
            ", ".join(project.tags) or "-",
        )

    console.print(table)


def _print_eval_runs(
    repository,
    project_slug: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    runs = repository.list_eval_runs(project_slug, limit=limit, offset=offset)
    table = Table(title=f"Eval Runs for {project_slug}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name")
    table.add_column("Source")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Cases", justify="right")

    for run in runs:
        table.add_row(
            run.id,
            run.name,
            run.source,
            f"{run.summary.pass_rate:.1f}%",
            str(run.summary.total),
        )

    console.print(table)


def _print_dataset_versions(
    repository,
    project_slug: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    datasets = repository.list_dataset_versions(project_slug, limit=limit, offset=offset)
    table = Table(title=f"Dataset Versions for {project_slug}")
    table.add_column("Dataset Version ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Items", justify="right")

    for dataset in datasets:
        table.add_row(
            dataset.id,
            dataset.name,
            dataset.version,
            dataset.source.value,
            str(dataset.item_count),
        )

    console.print(table)


def _print_snapshot(repository: FileCoreRepository, project_slug: str, run_id: str) -> None:
    snapshot = repository.load_snapshot(project_slug, run_id)
    console.print(
        f"[bold]Project:[/bold] {project_slug}\n"
        f"[bold]Run:[/bold] {snapshot.eval_run.name} ({snapshot.eval_run.id})\n"
        f"[bold]Dataset:[/bold] {snapshot.dataset_version.name} ({snapshot.dataset_version.version})\n"
        f"[bold]Traces:[/bold] {len(snapshot.traces)}\n"
        f"[bold]Annotation Tasks:[/bold] {len(snapshot.annotation_tasks)}"
    )


def _print_alert_rules(
    repository,
    project_slug: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    rules = repository.list_alert_rules(project_slug, limit=limit, offset=offset)
    table = Table(title=f"Alert Rules for {project_slug}")
    table.add_column("Rule ID", style="cyan")
    table.add_column("Name")
    table.add_column("Metric")
    table.add_column("Operator")
    table.add_column("Threshold", justify="right")
    table.add_column("Severity")
    table.add_column("Enabled")

    for rule in rules:
        table.add_row(
            rule.id,
            rule.name,
            rule.metric_key,
            rule.operator,
            str(rule.threshold),
            rule.severity.value,
            "yes" if rule.enabled else "no",
        )

    console.print(table)


def _print_alert_events(
    repository,
    project_slug: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    events = repository.list_alert_events(project_slug, limit=limit, offset=offset)
    table = Table(title=f"Alert Events for {project_slug}")
    table.add_column("Event ID", style="cyan")
    table.add_column("Rule ID")
    table.add_column("Run ID")
    table.add_column("Severity")
    table.add_column("Message")
    table.add_column("Triggered At")

    for event in events:
        table.add_row(
            event.id,
            event.rule_id,
            event.eval_run_id,
            event.severity.value,
            event.message,
            event.triggered_at.isoformat(),
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AgentLens platform store data")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(".agentlens-platform"),
        help="Path to the local platform store",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        help="Path to the local platform SQLite database",
    )
    parser.add_argument("--list-projects", action="store_true", help="List stored projects")
    parser.add_argument("--project", type=str, help="Project slug to inspect")
    parser.add_argument("--list-eval-runs", action="store_true", help="List eval runs for a project")
    parser.add_argument(
        "--list-dataset-versions",
        action="store_true",
        help="List dataset versions for a project",
    )
    parser.add_argument(
        "--list-alert-rules",
        action="store_true",
        help="List alert rules for a project",
    )
    parser.add_argument(
        "--list-alert-events",
        action="store_true",
        help="List alert events for a project",
    )
    parser.add_argument("--run-id", type=str, help="Eval run id to show")
    parser.add_argument("--limit", type=int, help="Max records to return for list commands")
    parser.add_argument("--offset", type=int, default=0, help="Offset for list commands")
    args = parser.parse_args()

    repository = _build_repository(args)

    if args.list_projects:
        _print_projects(repository, limit=args.limit, offset=args.offset)
        return

    if args.project and args.list_eval_runs:
        _print_eval_runs(repository, args.project, limit=args.limit, offset=args.offset)
        return

    if args.project and args.list_dataset_versions:
        _print_dataset_versions(repository, args.project, limit=args.limit, offset=args.offset)
        return

    if args.project and args.list_alert_rules:
        _print_alert_rules(repository, args.project, limit=args.limit, offset=args.offset)
        return

    if args.project and args.list_alert_events:
        _print_alert_events(repository, args.project, limit=args.limit, offset=args.offset)
        return

    if args.project and args.run_id:
        _print_snapshot(repository, args.project, args.run_id)
        return

    parser.error(
        "Choose one of: --list-projects, --project <slug> --list-eval-runs, "
        "--project <slug> --list-dataset-versions, "
        "--project <slug> --list-alert-rules, "
        "--project <slug> --list-alert-events, "
        "or --project <slug> --run-id <id>"
    )


if __name__ == "__main__":
    main()
