"""CLI entry point for viewing evolution history.

Usage
-----
    # Rich table of all cycles across all projects
    python -m agentlens.evolution

    # Filter by project slug
    python -m agentlens.evolution --project my-project

    # Generate HTML report (same data)
    python -m agentlens.evolution --html evolution.html

    # Override store path
    python -m agentlens.evolution --store .agentlens-platform
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _delta_text(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}%"


def _status_markup(accepted: bool) -> str:
    if accepted:
        return "[bold green]ACCEPTED[/bold green]"
    return "[bold red]REJECTED[/bold red]"


def _created_str(record) -> str:
    try:
        return record.created_at.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(record.created_at)


def _short_patterns(patterns: list[str], max_len: int = 50) -> str:
    if not patterns:
        return "—"
    joined = ", ".join(patterns)
    if len(joined) <= max_len:
        return joined
    return joined[:max_len - 3].rstrip() + "..."


def _load_records(store_path: Path, project_slug: str | None):
    """Return a list of (project_slug, EvolutionRecord) pairs.

    If *project_slug* is given, load only that project's records.
    Otherwise load records from every project found in the store.
    """
    from agentlens.core.repository import FileCoreRepository

    repo = FileCoreRepository(store_path)

    if project_slug:
        project = repo.load_project(project_slug)
        if project is None:
            console.print(
                f"[red]Project '{project_slug}' not found in {store_path}[/red]"
            )
            sys.exit(1)
        records = repo.load_evolution_records(project_slug)
        return [(project_slug, r) for r in records], {project_slug: project.name}

    projects = repo.list_projects()
    if not projects:
        return [], {}

    slug_to_name: dict[str, str] = {p.slug: p.name for p in projects}
    pairs: list[tuple[str, object]] = []
    for project in projects:
        for rec in repo.load_evolution_records(project.slug):
            pairs.append((project.slug, rec))

    return pairs, slug_to_name


def _print_table(
    pairs: list[tuple[str, object]],
    slug_to_name: dict[str, str],
    *,
    single_project: bool,
) -> None:
    table = Table(title="AgentLens Evolution History")

    if not single_project:
        table.add_column("Project", style="magenta")
    table.add_column("Cycle", justify="right", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ", justify="right")           # Greek delta
    table.add_column("Targeted Patterns")
    table.add_column("Created At", style="dim")

    # Sort by project slug then cycle
    sorted_pairs = sorted(pairs, key=lambda t: (t[0], t[1].cycle))

    for slug, rec in sorted_pairs:
        ss = rec.signal_summary or {}
        baseline_rate: float = float(ss.get("pass_rate", 0.0))
        candidate_rate = baseline_rate + rec.delta_pass_rate

        delta_color = "green" if rec.delta_pass_rate >= 0 else "red"
        delta_markup = f"[{delta_color}]{_delta_text(rec.delta_pass_rate)}[/{delta_color}]"

        row_values: list[str] = []
        if not single_project:
            row_values.append(slug_to_name.get(slug, slug))
        row_values += [
            str(rec.cycle),
            _status_markup(rec.accepted),
            _fmt_pct(baseline_rate),
            _fmt_pct(candidate_rate),
            delta_markup,
            _short_patterns(rec.targeted_patterns),
            _created_str(rec),
        ]
        table.add_row(*row_values)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m agentlens.evolution",
        description="View evolution history stored in a FileCoreRepository.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(".agentlens-platform"),
        metavar="PATH",
        help="Path to the FileCoreRepository root (default: .agentlens-platform)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="SLUG",
        help="Filter by project slug. If omitted, all projects are shown.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write an HTML report to this file path.",
    )
    args = parser.parse_args()

    store_path: Path = args.store
    if not store_path.exists():
        console.print(
            f"[yellow]Store path '{store_path}' does not exist. "
            "No records to display.[/yellow]"
        )
        sys.exit(0)

    pairs, slug_to_name = _load_records(store_path, args.project)

    if not pairs:
        target = f"project '{args.project}'" if args.project else "any project"
        console.print(
            f"[dim]No evolution records found for {target} in {store_path}.[/dim]"
        )
        console.print(
            "[dim]Run EvolutionCycle.run() to generate records.[/dim]"
        )
        sys.exit(0)

    single_project = args.project is not None
    _print_table(pairs, slug_to_name, single_project=single_project)

    if args.html is not None:
        from agentlens.evolution.reporter import generate_evolution_report
        from agentlens.core.models import EvolutionRecord

        records: list[EvolutionRecord] = [rec for _, rec in pairs]

        # Determine a display project name
        if args.project and args.project in slug_to_name:
            project_name = slug_to_name[args.project]
        elif len(slug_to_name) == 1:
            project_name = next(iter(slug_to_name.values()))
        else:
            project_name = ""

        html = generate_evolution_report(records, project_name=project_name)

        output_path: Path = args.html
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        console.print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
