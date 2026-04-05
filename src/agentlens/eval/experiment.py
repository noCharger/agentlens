"""Multi-dimensional experiment comparison.

Extends the basic ExperimentRecord with:
- Prompt version tracking
- Tool configuration versioning
- Per-dimension score comparison (not just pass_rate)
- Performance metrics comparison (latency, token usage)
- Detailed regression analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.runner import EvalResult


@dataclass
class VersionedConfig:
    """Captures a specific configuration version for comparison."""
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
    regressions: list[ScenarioRegression]
    improvements: list[ScenarioRegression]
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

    base_by_id = {r.scenario.id: r for r in baseline_results}
    cand_by_id = {r.scenario.id: r for r in candidate_results}
    all_ids = sorted(set(base_by_id) | set(cand_by_id))

    base_passed = sum(1 for r in baseline_results if r.passed)
    cand_passed = sum(1 for r in candidate_results if r.passed)
    base_rate = round(base_passed / len(baseline_results) * 100, 1) if baseline_results else 0
    cand_rate = round(cand_passed / len(candidate_results) * 100, 1) if candidate_results else 0

    regressions: list[ScenarioRegression] = []
    improvements: list[ScenarioRegression] = []
    unchanged = 0

    for sid in all_ids:
        base = base_by_id.get(sid)
        cand = cand_by_id.get(sid)
        if base is None or cand is None:
            continue

        base_passed_flag = base.passed
        cand_passed_flag = cand.passed

        regression_info = ScenarioRegression(
            scenario_id=sid,
            scenario_name=base.scenario.name,
            benchmark=base.scenario.benchmark,
            baseline_status=base.status.value,
            candidate_status=cand.status.value,
            baseline_score=base.judge_overall_score,
            candidate_score=cand.judge_overall_score,
        )

        if base_passed_flag and not cand_passed_flag:
            regressions.append(regression_info)
        elif not base_passed_flag and cand_passed_flag:
            improvements.append(regression_info)
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

    performance = PerformanceComparison(
        baseline_avg_steps=_avg(base_steps),
        candidate_avg_steps=_avg(cand_steps),
        baseline_avg_tokens=_avg(base_tokens),
        candidate_avg_tokens=_avg(cand_tokens),
        baseline_avg_prompt_tokens=_avg(base_prompt),
        candidate_avg_prompt_tokens=_avg(cand_prompt),
        step_delta=round(_avg(cand_steps) - _avg(base_steps), 2),
        token_delta=round(_avg(cand_tokens) - _avg(base_tokens), 2),
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
        unchanged=unchanged,
        total_scenarios=len(all_ids),
    )
