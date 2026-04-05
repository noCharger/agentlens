"""Tests for multi-dimensional experiment comparison."""

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.experiment import (
    ExperimentComparison,
    VersionedConfig,
    compare_experiments,
)


def _make_result(id: str, *, passed=True, score=None, steps=2, tokens=100):
    scenario = Scenario(
        id=id,
        name=f"Test {id}",
        category="test",
        input="query",
        expected=ExpectedResult(tools_called=[], output_contains=[]),
    )
    l2 = {}
    if score is not None:
        l2 = {"accuracy": score}
    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(passed, [], [], [], []),
            output_format=OutputFormatResult(passed, "ok" if passed else "", [], []),
            trajectory=TrajectoryResult(True, steps, 10, False, tokens, tokens // 2, None, []),
        ),
        level2_scores=l2,
    )


class TestExperimentComparison:
    def test_basic_comparison(self):
        baseline = [_make_result("s1"), _make_result("s2", passed=False)]
        candidate = [_make_result("s1"), _make_result("s2")]

        result = compare_experiments(baseline, candidate)
        assert result.baseline_pass_rate == 50.0
        assert result.candidate_pass_rate == 100.0
        assert result.delta_pass_rate == 50.0
        assert len(result.improvements) == 1
        assert len(result.regressions) == 0

    def test_regression_detected(self):
        baseline = [_make_result("s1"), _make_result("s2")]
        candidate = [_make_result("s1"), _make_result("s2", passed=False)]

        result = compare_experiments(baseline, candidate)
        assert len(result.regressions) == 1
        assert result.regressions[0].scenario_id == "s2"

    def test_dimension_comparison(self):
        baseline = [_make_result("s1", score=4.0), _make_result("s2", score=3.0)]
        candidate = [_make_result("s1", score=4.5), _make_result("s2", score=4.0)]

        result = compare_experiments(baseline, candidate)
        assert len(result.dimension_comparisons) == 1
        dim = result.dimension_comparisons[0]
        assert dim.dimension == "accuracy"
        assert dim.improved is True
        assert dim.delta > 0

    def test_performance_comparison(self):
        baseline = [_make_result("s1", steps=3, tokens=200)]
        candidate = [_make_result("s1", steps=5, tokens=400)]

        result = compare_experiments(baseline, candidate)
        assert result.performance.step_delta > 0  # candidate uses more steps
        assert result.performance.token_delta > 0

    def test_versioned_config(self):
        baseline_config = VersionedConfig(
            agent_model="gemini:gemini-2.5-flash",
            prompt_version="v1.0",
        )
        candidate_config = VersionedConfig(
            agent_model="gemini:gemini-2.5-flash",
            prompt_version="v2.0",
        )

        result = compare_experiments(
            [_make_result("s1")],
            [_make_result("s1")],
            baseline_config=baseline_config,
            candidate_config=candidate_config,
        )
        assert result.baseline_config.prompt_version == "v1.0"
        assert result.candidate_config.prompt_version == "v2.0"

    def test_status_distribution(self):
        baseline = [_make_result("s1"), _make_result("s2", passed=False)]
        candidate = [_make_result("s1"), _make_result("s2")]

        result = compare_experiments(baseline, candidate)
        assert "passed" in result.candidate_status_counts
