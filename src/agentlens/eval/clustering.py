"""Failure sample clustering for eval results.

Groups failed evaluation results by multiple dimensions:
- Error type (tool failure, timeout, loop, output mismatch, etc.)
- Failure pattern (from trajectory failure map)
- Benchmark / category
- Status (failed, partial_success, error)

Produces a structured clustering result for analysis and visualization.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.runner import EvalResult


@dataclass
class ClusterMember:
    """A single failure within a cluster."""
    scenario_id: str
    scenario_name: str
    benchmark: str
    category: str
    error_summary: str


@dataclass
class FailureCluster:
    """A group of failures sharing a common characteristic."""
    cluster_key: str
    cluster_type: str  # error_type, failure_pattern, benchmark, category, status
    label: str
    members: list[ClusterMember] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass
class ClusteringResult:
    """Complete clustering analysis of failed samples."""
    total_failures: int
    clusters_by_error_type: list[FailureCluster]
    clusters_by_failure_pattern: list[FailureCluster]
    clusters_by_benchmark: list[FailureCluster]
    clusters_by_category: list[FailureCluster]
    clusters_by_status: list[FailureCluster]

    @property
    def all_clusters(self) -> list[FailureCluster]:
        return (
            self.clusters_by_error_type
            + self.clusters_by_failure_pattern
            + self.clusters_by_benchmark
            + self.clusters_by_category
            + self.clusters_by_status
        )

    def top_clusters(self, n: int = 5) -> list[FailureCluster]:
        """Return the N largest clusters across all dimensions."""
        all_c = self.all_clusters
        all_c.sort(key=lambda c: c.count, reverse=True)
        return all_c[:n]


def _classify_error_type(result: "EvalResult") -> str:
    """Classify a failed result into an error type."""
    if result.error:
        error_lower = result.error.lower()
        if "setup failed" in error_lower:
            return "setup_failure"
        if "quota" in error_lower:
            return "quota_exhausted"
        if "ssl" in error_lower or "network" in error_lower:
            return "network_error"
        if "timeout" in error_lower:
            return "timeout"
        if "external benchmark" in error_lower:
            return "external_harness"
        return "agent_error"

    l1 = result.level1
    issues = []
    if not l1.tool_usage.passed:
        issues.append("tool_mismatch")
    if not l1.output_format.passed:
        issues.append("output_mismatch")
    if l1.trajectory.has_loop:
        issues.append("loop")
    if not l1.trajectory.passed and l1.trajectory.total_steps > l1.trajectory.max_steps:
        issues.append("step_overflow")
    if l1.safety and not l1.safety.passed:
        issues.append("safety_violation")
    if l1.termination and not l1.termination.passed:
        issues.append("termination_issue")

    if not issues:
        score = result.judge_overall_score
        if score is not None and score < result.scenario.judge_threshold:
            return "low_judge_score"
        return "unknown"

    return "+".join(sorted(issues))


def _extract_failure_patterns(result: "EvalResult") -> list[str]:
    """Extract failure pattern names from trajectory analysis."""
    patterns = []
    if result.level1.trajectory.has_loop:
        patterns.append("loop_trap")
    if result.risk_signals:
        for signal in result.risk_signals:
            if signal.startswith("safety_"):
                patterns.append(signal.split(":")[0])
            elif signal.startswith("excessive_retries"):
                patterns.append("excessive_retries")
            elif signal.startswith("unexpected_privileged"):
                patterns.append("unexpected_privilege")
    return patterns or ["unclassified"]


def _make_member(result: "EvalResult", error_summary: str = "") -> ClusterMember:
    if not error_summary:
        error_summary = result.error or _classify_error_type(result)
    return ClusterMember(
        scenario_id=result.scenario.id,
        scenario_name=result.scenario.name,
        benchmark=result.scenario.benchmark,
        category=result.scenario.category,
        error_summary=error_summary,
    )


def cluster_failures(results: list["EvalResult"]) -> ClusteringResult:
    """Cluster failed eval results by multiple dimensions."""
    from agentlens.core.models import TraceStatus

    failures = [r for r in results if r.status not in (TraceStatus.PASSED,)]

    if not failures:
        return ClusteringResult(
            total_failures=0,
            clusters_by_error_type=[],
            clusters_by_failure_pattern=[],
            clusters_by_benchmark=[],
            clusters_by_category=[],
            clusters_by_status=[],
        )

    by_error: dict[str, list[ClusterMember]] = defaultdict(list)
    for r in failures:
        error_type = _classify_error_type(r)
        by_error[error_type].append(_make_member(r))

    by_pattern: dict[str, list[ClusterMember]] = defaultdict(list)
    for r in failures:
        for pattern in _extract_failure_patterns(r):
            by_pattern[pattern].append(_make_member(r))

    by_benchmark: dict[str, list[ClusterMember]] = defaultdict(list)
    for r in failures:
        key = r.scenario.benchmark or "unassigned"
        by_benchmark[key].append(_make_member(r))

    by_category: dict[str, list[ClusterMember]] = defaultdict(list)
    for r in failures:
        by_category[r.scenario.category].append(_make_member(r))

    by_status: dict[str, list[ClusterMember]] = defaultdict(list)
    for r in failures:
        by_status[r.status.value].append(_make_member(r))

    def _build_clusters(groups: dict, cluster_type: str) -> list[FailureCluster]:
        clusters = [
            FailureCluster(
                cluster_key=key,
                cluster_type=cluster_type,
                label=key.replace("_", " ").title(),
                members=members,
            )
            for key, members in groups.items()
        ]
        clusters.sort(key=lambda c: c.count, reverse=True)
        return clusters

    return ClusteringResult(
        total_failures=len(failures),
        clusters_by_error_type=_build_clusters(by_error, "error_type"),
        clusters_by_failure_pattern=_build_clusters(by_pattern, "failure_pattern"),
        clusters_by_benchmark=_build_clusters(by_benchmark, "benchmark"),
        clusters_by_category=_build_clusters(by_category, "category"),
        clusters_by_status=_build_clusters(by_status, "status"),
    )
