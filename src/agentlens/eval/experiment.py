"""Multi-dimensional experiment comparison.

Extends the basic ExperimentRecord with:
- Prompt version tracking
- Tool configuration versioning
- Per-dimension score comparison (not just pass_rate)
- Performance metrics comparison (latency, token usage, memory retention)
- Detailed regression analysis with L1 sub-check granularity and tool diffs
- Soft regression detection (PASSED → RISKY_SUCCESS transitions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.runner import EvalResult

_FEATURE_FLAG_NAMES = (
    "geval",
    "task_completion",
    "answer_relevancy",
    "hallucination",
    "faithfulness",
)


@dataclass
class VersionedConfig:
    """Captures a specific configuration version for comparison."""
    agent_framework: str = "langgraph"
    agent_model: str = ""
    judge_model: str = ""
    prompt_version: str = ""
    prompt_hash: str = ""
    tool_preset: str = ""
    tool_versions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DimensionComparison:
    """Comparison of a single evaluation dimension across two runs."""
    dimension: str
    baseline_value: float
    candidate_value: float
    delta: float
    improved: bool
    regressed: bool


@dataclass
class PerformanceComparison:
    """Performance metric comparison between two runs."""
    baseline_avg_steps: float
    candidate_avg_steps: float
    baseline_avg_tokens: float
    candidate_avg_tokens: float
    baseline_avg_prompt_tokens: float
    candidate_avg_prompt_tokens: float
    step_delta: float
    token_delta: float
    # Memory retention (0.0 when no memory scenarios ran)
    baseline_avg_retention: float
    candidate_avg_retention: float
    retention_delta: float


@dataclass
class ScenarioRegression:
    """Detailed regression info for a single scenario."""
    scenario_id: str
    scenario_name: str
    benchmark: str
    baseline_status: str
    candidate_status: str
    baseline_score: float | None
    candidate_score: float | None
    # L1 sub-check granularity
    baseline_l1_checks: dict[str, bool] = field(default_factory=dict)
    candidate_l1_checks: dict[str, bool] = field(default_factory=dict)
    l1_flipped: list[str] = field(default_factory=list)  # e.g. ["tool_usage: pass→fail"]
    # Tool call diff
    baseline_tools: list[str] = field(default_factory=list)
    candidate_tools: list[str] = field(default_factory=list)
    tools_only_baseline: list[str] = field(default_factory=list)  # dropped by candidate
    tools_only_candidate: list[str] = field(default_factory=list)  # added by candidate


@dataclass
class ExperimentComparison:
    """Complete multi-dimensional experiment comparison."""
    baseline_config: VersionedConfig
    candidate_config: VersionedConfig
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta_pass_rate: float
    baseline_status_counts: dict[str, int]
    candidate_status_counts: dict[str, int]
    dimension_comparisons: list[DimensionComparison]
    performance: PerformanceComparison
    regressions: list[ScenarioRegression]        # pass → fail
    improvements: list[ScenarioRegression]       # fail → pass
    soft_regressions: list[ScenarioRegression]   # PASSED → RISKY_SUCCESS
    soft_improvements: list[ScenarioRegression]  # RISKY_SUCCESS → PASSED
    unchanged: int
    total_scenarios: int


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _status_counts(results: list["EvalResult"]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        key = r.status.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _collect_dimension_scores(results: list["EvalResult"]) -> dict[str, list[float]]:
    """Collect L2 judge scores per dimension across all results."""
    scores: dict[str, list[float]] = {}
    for r in results:
        for dim, score in r.level2_scores.items():
            if score >= 0:
                scores.setdefault(dim, []).append(score)
    return scores


def _collect_feature_flags(results: list["EvalResult"]) -> dict[str, bool]:
    flags = {name: False for name in _FEATURE_FLAG_NAMES}
    for result in results:
        for name in _FEATURE_FLAG_NAMES:
            if result.feature_flags.get(name, False):
                flags[name] = True
    return flags


def _l1_checks(result: "EvalResult") -> dict[str, bool]:
    """Extract all L1 check results as a flat dict."""
    checks = {
        "tool_usage": result.level1.tool_usage.passed,
        "output_format": result.level1.output_format.passed,
        "trajectory": result.level1.trajectory.passed,
    }
    checks.update(result.level1.supplemental_checks)
    return checks


def _l1_flipped(base: dict[str, bool], cand: dict[str, bool]) -> list[str]:
    """Return check names that changed between baseline and candidate."""
    all_keys = sorted(set(base) | set(cand))
    flipped = []
    for k in all_keys:
        bv = base.get(k, True)
        cv = cand.get(k, True)
        if bv != cv:
            arrow = "pass→fail" if bv else "fail→pass"
            flipped.append(f"{k}: {arrow}")
    return flipped


def _tool_diff(base: list[str], cand: list[str]) -> tuple[list[str], list[str]]:
    """Return (tools only in baseline, tools only in candidate) as sorted unique lists."""
    base_set = set(base)
    cand_set = set(cand)
    return sorted(base_set - cand_set), sorted(cand_set - base_set)


def _avg_retention(results: list["EvalResult"]) -> float:
    scores = [
        r.level1.memory_retention.retention_score
        for r in results
        if r.level1.memory_retention is not None
    ]
    return _avg(scores)


def _make_scenario_regression(base: "EvalResult", cand: "EvalResult") -> ScenarioRegression:
    base_checks = _l1_checks(base)
    cand_checks = _l1_checks(cand)
    base_tools = base.level1.tool_usage.actual_tools
    cand_tools = cand.level1.tool_usage.actual_tools
    only_base, only_cand = _tool_diff(base_tools, cand_tools)
    return ScenarioRegression(
        scenario_id=base.scenario.id,
        scenario_name=base.scenario.name,
        benchmark=base.scenario.benchmark,
        baseline_status=base.status.value,
        candidate_status=cand.status.value,
        baseline_score=base.judge_overall_score,
        candidate_score=cand.judge_overall_score,
        baseline_l1_checks=base_checks,
        candidate_l1_checks=cand_checks,
        l1_flipped=_l1_flipped(base_checks, cand_checks),
        baseline_tools=base_tools,
        candidate_tools=cand_tools,
        tools_only_baseline=only_base,
        tools_only_candidate=only_cand,
    )


def compare_experiments(
    baseline_results: list["EvalResult"],
    candidate_results: list["EvalResult"],
    *,
    baseline_config: VersionedConfig | None = None,
    candidate_config: VersionedConfig | None = None,
) -> ExperimentComparison:
    """Run multi-dimensional comparison between two experiment runs."""
    baseline_config = baseline_config or VersionedConfig()
    candidate_config = candidate_config or VersionedConfig()
    baseline_config.metadata = dict(baseline_config.metadata)
    candidate_config.metadata = dict(candidate_config.metadata)
    baseline_config.metadata.setdefault("feature_flags", _collect_feature_flags(baseline_results))
    candidate_config.metadata.setdefault("feature_flags", _collect_feature_flags(candidate_results))

    base_by_id = {r.scenario.id: r for r in baseline_results}
    cand_by_id = {r.scenario.id: r for r in candidate_results}
    all_ids = sorted(set(base_by_id) | set(cand_by_id))

    base_passed = sum(1 for r in baseline_results if r.passed)
    cand_passed = sum(1 for r in candidate_results if r.passed)
    base_rate = round(base_passed / len(baseline_results) * 100, 1) if baseline_results else 0
    cand_rate = round(cand_passed / len(candidate_results) * 100, 1) if candidate_results else 0

    regressions: list[ScenarioRegression] = []
    improvements: list[ScenarioRegression] = []
    soft_regressions: list[ScenarioRegression] = []
    soft_improvements: list[ScenarioRegression] = []
    unchanged = 0

    for sid in all_ids:
        base = base_by_id.get(sid)
        cand = cand_by_id.get(sid)
        if base is None or cand is None:
            continue

        regression_info = _make_scenario_regression(base, cand)

        if base.passed and not cand.passed:
            regressions.append(regression_info)
        elif not base.passed and cand.passed:
            improvements.append(regression_info)
        else:
            # Both passed or both failed — check for soft risk transitions
            base_status = base.status.value
            cand_status = cand.status.value
            if base_status == "passed" and cand_status == "risky_success":
                soft_regressions.append(regression_info)
            elif base_status == "risky_success" and cand_status == "passed":
                soft_improvements.append(regression_info)
            else:
                unchanged += 1

    base_dims = _collect_dimension_scores(baseline_results)
    cand_dims = _collect_dimension_scores(candidate_results)
    all_dims = sorted(set(base_dims) | set(cand_dims))

    dim_comparisons = []
    for dim in all_dims:
        base_avg = _avg(base_dims.get(dim, []))
        cand_avg = _avg(cand_dims.get(dim, []))
        delta = round(cand_avg - base_avg, 2)
        dim_comparisons.append(DimensionComparison(
            dimension=dim,
            baseline_value=base_avg,
            candidate_value=cand_avg,
            delta=delta,
            improved=delta > 0.1,
            regressed=delta < -0.1,
        ))

    base_steps = [r.level1.trajectory.total_steps for r in baseline_results]
    cand_steps = [r.level1.trajectory.total_steps for r in candidate_results]
    base_tokens = [
        r.level1.trajectory.total_prompt_tokens + r.level1.trajectory.total_completion_tokens
        for r in baseline_results
    ]
    cand_tokens = [
        r.level1.trajectory.total_prompt_tokens + r.level1.trajectory.total_completion_tokens
        for r in candidate_results
    ]
    base_prompt = [r.level1.trajectory.total_prompt_tokens for r in baseline_results]
    cand_prompt = [r.level1.trajectory.total_prompt_tokens for r in candidate_results]

    base_retention = _avg_retention(baseline_results)
    cand_retention = _avg_retention(candidate_results)

    performance = PerformanceComparison(
        baseline_avg_steps=_avg(base_steps),
        candidate_avg_steps=_avg(cand_steps),
        baseline_avg_tokens=_avg(base_tokens),
        candidate_avg_tokens=_avg(cand_tokens),
        baseline_avg_prompt_tokens=_avg(base_prompt),
        candidate_avg_prompt_tokens=_avg(cand_prompt),
        step_delta=round(_avg(cand_steps) - _avg(base_steps), 2),
        token_delta=round(_avg(cand_tokens) - _avg(base_tokens), 2),
        baseline_avg_retention=base_retention,
        candidate_avg_retention=cand_retention,
        retention_delta=round(cand_retention - base_retention, 2),
    )

    return ExperimentComparison(
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        baseline_pass_rate=base_rate,
        candidate_pass_rate=cand_rate,
        delta_pass_rate=round(cand_rate - base_rate, 1),
        baseline_status_counts=_status_counts(baseline_results),
        candidate_status_counts=_status_counts(candidate_results),
        dimension_comparisons=dim_comparisons,
        performance=performance,
        regressions=regressions,
        improvements=improvements,
        soft_regressions=soft_regressions,
        soft_improvements=soft_improvements,
        unchanged=unchanged,
        total_scenarios=len(all_ids),
    )
