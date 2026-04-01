"""Built-in benchmark registry and helpers for benchmark-aware reporting."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from agentlens.eval.runner import EvalResult
    from agentlens.eval.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    slug: str
    name: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    slug: str
    name: str
    total: int
    passed: int
    failed: int
    pass_rate: float


@dataclass(frozen=True, slots=True)
class BenchmarkInventoryRow:
    slug: str
    name: str
    scenario_count: int
    built_in: bool


UNASSIGNED_BENCHMARK = "__unassigned__"


_BUILTIN_BENCHMARKS: tuple[BenchmarkDefinition, ...] = (
    BenchmarkDefinition(
        slug="swe-bench-pro",
        name="SWE Bench Pro",
        description="Repository-level software engineering benchmark focused on code changes.",
        aliases=("swebench-pro", "swe bench pro"),
    ),
    BenchmarkDefinition(
        slug="multi-swe-bench",
        name="Multi-SWE Bench",
        description="Multi-repo or multi-file software engineering benchmark with broader task context.",
        aliases=("multi swe bench", "multi-swebench"),
    ),
    BenchmarkDefinition(
        slug="vibe-pro",
        name="VIBE-Pro",
        description="Agent reliability benchmark centered on broader tool-using workflows.",
        aliases=("vibe pro", "vibepro"),
    ),
    BenchmarkDefinition(
        slug="mle-bench-lite",
        name="MLE-Bench lite",
        description="Machine-learning engineering benchmark for lightweight experimentation workflows.",
        aliases=("mle bench lite", "mlebench lite"),
    ),
    BenchmarkDefinition(
        slug="gdpval-aa",
        name="GDPval-AA",
        description="Analytical reasoning benchmark for grounded problem decomposition and validation.",
        aliases=("gdpval aa", "gdpval"),
    ),
    BenchmarkDefinition(
        slug="toolathlon",
        name="Toolathlon",
        description="Tool-use benchmark for multi-step tool planning and execution quality.",
        aliases=("tool athlon",),
    ),
    BenchmarkDefinition(
        slug="mm-clawbench",
        name="MM-ClawBench",
        description="Multimodal reasoning and action benchmark with complex evidence gathering.",
        aliases=("mm clawbench", "mmclawbench"),
    ),
    BenchmarkDefinition(
        slug="artificial-analysis",
        name="Artificial Analysis",
        description="General-purpose agent benchmark family used for model comparison and analysis tasks.",
        aliases=("artificial analysis",),
    ),
)

_BENCHMARKS_BY_SLUG = {benchmark.slug: benchmark for benchmark in _BUILTIN_BENCHMARKS}

_ALIASES_TO_SLUG: dict[str, str] = {}
for _benchmark in _BUILTIN_BENCHMARKS:
    for candidate in (_benchmark.slug, _benchmark.name, *_benchmark.aliases):
        _ALIASES_TO_SLUG[re.sub(r"[^a-z0-9]+", "-", candidate.casefold()).strip("-")] = (
            _benchmark.slug
        )


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def normalize_benchmark_name(value: str | None) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    slug = _slugify(raw)
    return _ALIASES_TO_SLUG.get(slug, slug)


def get_benchmark_definition(value: str | None) -> BenchmarkDefinition | None:
    slug = normalize_benchmark_name(value)
    if not slug:
        return None
    return _BENCHMARKS_BY_SLUG.get(slug)


def display_benchmark_name(value: str | None) -> str:
    definition = get_benchmark_definition(value)
    if definition:
        return definition.name
    raw = (value or "").strip()
    return raw


def list_supported_benchmarks() -> list[BenchmarkDefinition]:
    return list(_BUILTIN_BENCHMARKS)


def infer_benchmark_from_path(path) -> str:
    if path is None:
        return ""

    # Prefer explicit benchmark directories such as scenarios/benchmarks/<slug>/...
    if "benchmarks" in path.parts:
        index = path.parts.index("benchmarks")
        if index + 1 < len(path.parts):
            return normalize_benchmark_name(path.parts[index + 1])

    # Fall back to matching any parent folder that maps to a known benchmark slug.
    for parent in path.parents:
        slug = normalize_benchmark_name(parent.name)
        if slug in _BENCHMARKS_BY_SLUG:
            return slug
    return ""


def normalize_benchmark_filters(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    return {slug for value in values if (slug := normalize_benchmark_name(value))}


def filter_scenarios_by_benchmark(
    scenarios: Iterable["Scenario"],
    benchmark_filters: Iterable[str] | None = None,
) -> list["Scenario"]:
    selected = normalize_benchmark_filters(benchmark_filters)
    if not selected:
        return list(scenarios)
    return [scenario for scenario in scenarios if scenario.benchmark in selected]


def collect_benchmark_inventory(scenarios: Iterable["Scenario"]) -> list[BenchmarkInventoryRow]:
    scenario_list = list(scenarios)
    counts = Counter(
        scenario.benchmark for scenario in scenario_list if scenario.benchmark
    )
    rows: list[BenchmarkInventoryRow] = []
    seen: set[str] = set()

    for benchmark in _BUILTIN_BENCHMARKS:
        rows.append(
            BenchmarkInventoryRow(
                slug=benchmark.slug,
                name=benchmark.name,
                scenario_count=counts.get(benchmark.slug, 0),
                built_in=True,
            )
        )
        seen.add(benchmark.slug)

    for slug, count in sorted(counts.items()):
        if slug not in seen:
            rows.append(
                BenchmarkInventoryRow(
                    slug=slug,
                    name=display_benchmark_name(slug) or slug,
                    scenario_count=count,
                    built_in=False,
                )
            )

    unassigned = sum(1 for scenario in scenario_list if not scenario.benchmark)
    if unassigned:
        rows.append(
            BenchmarkInventoryRow(
                slug=UNASSIGNED_BENCHMARK,
                name="Unassigned",
                scenario_count=unassigned,
                built_in=False,
            )
        )

    return rows


def summarize_results_by_benchmark(results: Iterable["EvalResult"]) -> list[BenchmarkSummary]:
    grouped: dict[str, dict[str, int | str]] = {}

    for result in results:
        benchmark = result.scenario.benchmark or UNASSIGNED_BENCHMARK
        bucket = grouped.setdefault(
            benchmark,
            {
                "name": (
                    display_benchmark_name(result.scenario.benchmark)
                    if result.scenario.benchmark
                    else "Unassigned"
                ),
                "total": 0,
                "passed": 0,
            },
        )
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1

    summaries: list[BenchmarkSummary] = []
    for slug, bucket in grouped.items():
        total = int(bucket["total"])
        passed = int(bucket["passed"])
        failed = total - passed
        pass_rate = round(passed / total * 100, 1) if total else 0.0
        summaries.append(
            BenchmarkSummary(
                slug=slug,
                name=str(bucket["name"]),
                total=total,
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
            )
        )

    return sorted(
        summaries,
        key=lambda summary: (summary.slug == UNASSIGNED_BENCHMARK, summary.name.casefold()),
    )
