"""Tests for multi-dimensional experiment comparison."""

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.level1_deterministic.memory_retention import MemoryRetentionResult
from agentlens.eval.level1_deterministic.safety import SafetyResult
from agentlens.eval.experiment import (
    VersionedConfig,
    compare_experiments,
)


def _make_result(
    id: str,
    *,
    passed=True,
    score=None,
    steps=2,
    tokens=100,
    feature_flags=None,
    actual_tools: list[str] | None = None,
    tool_usage_passed: bool | None = None,
    output_format_passed: bool | None = None,
    trajectory_passed: bool = True,
    risk_signals: list[str] | None = None,
    retention_score: float | None = None,
    safety_passed: bool | None = None,
):
    if tool_usage_passed is None:
        tool_usage_passed = passed
    if output_format_passed is None:
        output_format_passed = passed
    tools = actual_tools or []
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

    memory_retention = None
    if retention_score is not None:
        memory_retention = MemoryRetentionResult(
            passed=retention_score == 1.0,
            anchors_recalled=int(retention_score * 3),
            anchors_actively_used=int(retention_score * 2),
            anchors_total=3,
            poison_hallucinated=0,
            lost_anchors=[],
            retention_score=retention_score,
        )

    safety = None
    if safety_passed is not None:
        safety = SafetyResult(passed=safety_passed, violations=[])

    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(tool_usage_passed, [], tools, [], []),
            output_format=OutputFormatResult(output_format_passed, "ok" if output_format_passed else "", [], []),
            trajectory=TrajectoryResult(trajectory_passed, steps, 10, False, tokens, tokens // 2, None, []),
            memory_retention=memory_retention,
            safety=safety,
        ),
        level2_scores=l2,
        feature_flags=feature_flags or {},
        risk_signals=risk_signals or [],
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
            agent_framework="langgraph",
            agent_model="gemini:gemini-2.5-flash",
            prompt_version="v1.0",
        )
        candidate_config = VersionedConfig(
            agent_framework="ag2",
            agent_model="gemini:gemini-2.5-flash",
            prompt_version="v2.0",
        )

        result = compare_experiments(
            [_make_result("s1")],
            [_make_result("s1")],
            baseline_config=baseline_config,
            candidate_config=candidate_config,
        )
        assert result.baseline_config.agent_framework == "langgraph"
        assert result.candidate_config.agent_framework == "ag2"
        assert result.baseline_config.prompt_version == "v1.0"
        assert result.candidate_config.prompt_version == "v2.0"

    def test_status_distribution(self):
        baseline = [_make_result("s1"), _make_result("s2", passed=False)]
        candidate = [_make_result("s1"), _make_result("s2")]

        result = compare_experiments(baseline, candidate)
        assert "passed" in result.candidate_status_counts

    def test_feature_flags_recorded_in_versioned_config_metadata(self):
        baseline = [_make_result("s1", feature_flags={"geval": False, "task_completion": False})]
        candidate = [_make_result("s1", feature_flags={"geval": True, "faithfulness": True})]

        result = compare_experiments(baseline, candidate)

        assert result.baseline_config.metadata["feature_flags"]["geval"] is False
        assert result.candidate_config.metadata["feature_flags"]["geval"] is True
        assert result.candidate_config.metadata["feature_flags"]["faithfulness"] is True


class TestL1SubCheckGranularity:
    def test_l1_flipped_shown_in_regression(self):
        # baseline: all pass. candidate: tool_usage fails.
        base = _make_result("s1", passed=True, actual_tools=["read_file"])
        cand = _make_result("s1", tool_usage_passed=False, output_format_passed=True, actual_tools=[])
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert any("tool_usage" in f for f in reg.l1_flipped)
        assert any("pass→fail" in f for f in reg.l1_flipped)

    def test_l1_checks_populated_for_all_scenarios(self):
        base = _make_result("s1")
        cand = _make_result("s1", passed=False)
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert "tool_usage" in reg.baseline_l1_checks
        assert "output_format" in reg.baseline_l1_checks
        assert "trajectory" in reg.baseline_l1_checks

    def test_supplemental_check_safety_included(self):
        base = _make_result("s1", safety_passed=True)
        cand = _make_result("s1", safety_passed=False, passed=False)
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert "safety" in reg.baseline_l1_checks
        assert "safety" in reg.candidate_l1_checks
        assert any("safety" in f for f in reg.l1_flipped)

    def test_memory_retention_check_included(self):
        base = _make_result("s1", retention_score=1.0)
        cand = _make_result("s1", retention_score=0.5, passed=False)
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert "memory_retention" in reg.baseline_l1_checks

    def test_no_l1_flipped_when_same_checks(self):
        base = _make_result("s1", passed=True)
        cand = _make_result("s1", passed=True)
        result = compare_experiments([base], [cand])
        assert result.unchanged == 1
        assert result.regressions == []

    def test_l1_flip_fail_to_pass_shown_in_improvement(self):
        base = _make_result("s1", tool_usage_passed=False, output_format_passed=True, passed=False)
        cand = _make_result("s1", passed=True)
        result = compare_experiments([base], [cand])
        imp = result.improvements[0]
        assert any("fail→pass" in f for f in imp.l1_flipped)


class TestSoftRegressions:
    def test_passed_to_risky_success_is_soft_regression(self):
        # baseline: PASSED (no risk signals), candidate: RISKY_SUCCESS (has risk signals)
        base = _make_result("s1", passed=True, risk_signals=[])
        cand = _make_result("s1", passed=True, risk_signals=["safety_boundary_crossed"])
        result = compare_experiments([base], [cand])
        assert len(result.soft_regressions) == 1
        assert result.soft_regressions[0].scenario_id == "s1"
        assert len(result.regressions) == 0

    def test_risky_success_to_passed_is_soft_improvement(self):
        base = _make_result("s1", passed=True, risk_signals=["safety_boundary_crossed"])
        cand = _make_result("s1", passed=True, risk_signals=[])
        result = compare_experiments([base], [cand])
        assert len(result.soft_improvements) == 1
        assert len(result.soft_regressions) == 0

    def test_soft_regressions_not_counted_in_unchanged(self):
        base = _make_result("s1", risk_signals=[])
        cand = _make_result("s1", risk_signals=["safety_boundary_crossed"])
        result = compare_experiments([base], [cand])
        assert result.unchanged == 0

    def test_two_same_status_counted_as_unchanged(self):
        base = _make_result("s1", risk_signals=[])
        cand = _make_result("s1", risk_signals=[])
        result = compare_experiments([base], [cand])
        assert result.unchanged == 1
        assert result.soft_regressions == []
        assert result.soft_improvements == []

    def test_soft_regression_has_l1_and_tool_fields(self):
        base = _make_result("s1", risk_signals=[], actual_tools=["read_file"])
        cand = _make_result("s1", risk_signals=["safety_boundary_crossed"], actual_tools=["read_file", "shell"])
        result = compare_experiments([base], [cand])
        sr = result.soft_regressions[0]
        assert "tool_usage" in sr.baseline_l1_checks
        assert "shell" in sr.tools_only_candidate


class TestToolDiff:
    def test_tool_added_by_candidate(self):
        base = _make_result("s1", passed=False, actual_tools=["read_file"])
        cand = _make_result("s1", passed=True, actual_tools=["read_file", "write_file"])
        result = compare_experiments([base], [cand])
        imp = result.improvements[0]
        assert "write_file" in imp.tools_only_candidate
        assert imp.tools_only_baseline == []

    def test_tool_dropped_by_candidate(self):
        base = _make_result("s1", passed=True, actual_tools=["read_file", "list_dir"])
        cand = _make_result("s1", passed=False, actual_tools=["read_file"])
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert "list_dir" in reg.tools_only_baseline
        assert reg.tools_only_candidate == []

    def test_no_tool_diff_when_identical(self):
        base = _make_result("s1", actual_tools=["read_file", "write_file"])
        cand = _make_result("s1", passed=False, actual_tools=["read_file", "write_file"])
        result = compare_experiments([base], [cand])
        reg = result.regressions[0]
        assert reg.tools_only_baseline == []
        assert reg.tools_only_candidate == []

    def test_tools_stored_on_all_scenario_types(self):
        # improvements also carry tool diff
        base = _make_result("s1", passed=False, actual_tools=["shell"])
        cand = _make_result("s1", passed=True, actual_tools=["read_file"])
        result = compare_experiments([base], [cand])
        imp = result.improvements[0]
        assert "shell" in imp.tools_only_baseline
        assert "read_file" in imp.tools_only_candidate


class TestMemoryRetentionDelta:
    def test_retention_delta_computed(self):
        base = [_make_result("s1", retention_score=0.5), _make_result("s2", retention_score=0.7)]
        cand = [_make_result("s1", retention_score=0.9), _make_result("s2", retention_score=1.0)]
        result = compare_experiments(base, cand)
        assert result.performance.baseline_avg_retention == 0.6
        assert result.performance.candidate_avg_retention == 0.95
        assert result.performance.retention_delta > 0

    def test_retention_zero_when_no_memory_scenarios(self):
        base = [_make_result("s1")]
        cand = [_make_result("s1")]
        result = compare_experiments(base, cand)
        assert result.performance.baseline_avg_retention == 0.0
        assert result.performance.candidate_avg_retention == 0.0
        assert result.performance.retention_delta == 0.0

    def test_retention_delta_negative_when_candidate_worse(self):
        base = [_make_result("s1", retention_score=1.0)]
        cand = [_make_result("s1", retention_score=0.33)]
        result = compare_experiments(base, cand)
        assert result.performance.retention_delta < 0
