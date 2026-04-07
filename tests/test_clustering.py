"""Tests for failure sample clustering."""

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.clustering import cluster_failures


def _make_scenario(id: str, **kw) -> Scenario:
    defaults = dict(
        name=f"Test {id}",
        category="test",
        input="test query",
        expected=ExpectedResult(tools_called=["read_file"], output_contains=["ok"]),
    )
    defaults.update(kw)
    return Scenario(id=id, **defaults)


def _make_result(scenario, *, passed=True, error=None, tool_passed=True, output_passed=True, traj_passed=True, risk_signals=None):
    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(tool_passed, ["read_file"], ["read_file"] if tool_passed else [], [] if tool_passed else ["read_file"], []),
            output_format=OutputFormatResult(output_passed, "ok" if output_passed else "wrong", ["ok"], [] if output_passed else ["ok"]),
            trajectory=TrajectoryResult(traj_passed, 2, 10, not traj_passed, 0, 0, None, []),
        ),
        error=error,
        risk_signals=risk_signals or [],
    )


class TestClustering:
    def test_no_failures(self):
        results = [_make_result(_make_scenario("s1"))]
        clustering = cluster_failures(results)
        assert clustering.total_failures == 0
        assert clustering.clusters_by_error_type == []

    def test_cluster_by_error_type(self):
        results = [
            _make_result(_make_scenario("s1"), tool_passed=False, output_passed=True, traj_passed=True),
            _make_result(_make_scenario("s2"), tool_passed=False, output_passed=True, traj_passed=True),
            _make_result(_make_scenario("s3"), error="Setup failed: missing file"),
        ]
        clustering = cluster_failures(results)
        assert clustering.total_failures == 3
        error_types = {c.cluster_key for c in clustering.clusters_by_error_type}
        assert "tool_mismatch" in error_types
        assert "setup_failure" in error_types

    def test_cluster_by_benchmark(self):
        results = [
            _make_result(_make_scenario("s1", benchmark="gdpval-aa"), tool_passed=False),
            _make_result(_make_scenario("s2", benchmark="gdpval-aa"), tool_passed=False),
            _make_result(_make_scenario("s3", benchmark="swe-bench-pro"), tool_passed=False),
        ]
        clustering = cluster_failures(results)
        bm_keys = {c.cluster_key for c in clustering.clusters_by_benchmark}
        assert "gdpval-aa" in bm_keys
        assert "swe-bench-pro" in bm_keys

    def test_cluster_by_status(self):
        s1 = _make_scenario("s1")
        s2 = _make_scenario("s2")
        results = [
            _make_result(s1, tool_passed=False),
            _make_result(s2, error="some error"),
        ]
        clustering = cluster_failures(results)
        # partial_success (output ok, tools wrong) and error
        assert len(clustering.clusters_by_status) >= 1

    def test_top_clusters(self):
        results = [
            _make_result(_make_scenario(f"s{i}"), tool_passed=False)
            for i in range(5)
        ]
        clustering = cluster_failures(results)
        top = clustering.top_clusters(n=3)
        assert len(top) <= 3
        assert all(c.count > 0 for c in top)
