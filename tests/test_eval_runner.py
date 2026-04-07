"""Tests for the eval runner orchestration logic."""

from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.runner import (
    EvalResult,
    Level1Result,
    _annotate_eval_span,
    _has_level2_rubric,
    _run_level2,
    evaluate_scenario,
    execute_and_eval,
    run_level1_eval,
)
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.tool_params import ToolParamsResult, ToolParamViolation
from agentlens.eval.level1_deterministic.termination import TerminationResult
from agentlens.eval.level1_deterministic.safety import SafetyResult, SafetyViolation
from agentlens.eval.level1_deterministic.trajectory import (
    FailureMapResult,
    StructuralAnalysis,
    TrajectoryAnalysis,
    TrajectoryResult,
)
from agentlens.eval.level2_llm_judge.rubrics import JudgeResult, JudgeScore


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="test-001",
        name="Test",
        category="test",
        input="test query",
        setup=[],
        expected=ExpectedResult(
            tools_called=["read_file"],
            max_steps=5,
            output_contains=["hello"],
        ),
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def _make_spans(*span_configs):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    for config in span_configs:
        with tracer.start_as_current_span(config.get("name", "test")) as span:
            for k, v in config.get("attributes", {}).items():
                span.set_attribute(k, v)

    provider.force_flush()
    spans = list(exporter.get_finished_spans())
    provider.shutdown()
    return spans


def test_run_level1_eval_passes():
    scenario = _make_scenario()
    spans = _make_spans(
        {"name": "t", "attributes": {"tool.name": "read_file"}},
        {"name": "agent.run", "attributes": {"agent.output": "hello world"}},
        {"name": "agent.step", "attributes": {"step.index": 0, "step.action": "read"}},
    )
    result = run_level1_eval(scenario, spans)
    assert result.passed is True
    assert result.tool_usage.passed is True
    assert result.output_format.passed is True
    assert result.trajectory.passed is True


def test_run_level1_eval_fails_missing_tool():
    scenario = _make_scenario()
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "hello world"}},
    )
    result = run_level1_eval(scenario, spans)
    assert result.passed is False
    assert result.tool_usage.passed is False


def test_run_level1_eval_fails_missing_output():
    scenario = _make_scenario()
    spans = _make_spans(
        {"name": "t", "attributes": {"tool.name": "read_file"}},
        {"name": "agent.run", "attributes": {"agent.output": "wrong output"}},
    )
    result = run_level1_eval(scenario, spans)
    assert result.passed is False
    assert result.output_format.passed is False


def test_evaluate_scenario_returns_eval_result():
    scenario = _make_scenario()
    spans = _make_spans(
        {"name": "t", "attributes": {"tool.name": "read_file"}},
        {"name": "agent.run", "attributes": {"agent.output": "hello there"}},
    )
    result = evaluate_scenario(scenario, spans)
    assert result.scenario.id == "test-001"
    assert result.passed is True
    assert result.error is None


def test_evaluate_scenario_no_expected_tools():
    scenario = _make_scenario(
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[])
    )
    spans = _make_spans({"name": "agent.run", "attributes": {"agent.output": "anything"}})
    result = evaluate_scenario(scenario, spans)
    assert result.passed is True


def test_eval_result_llm_judge_passes_with_threshold():
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_threshold=4.0,
    )
    result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
        level2_scores={"custom": 4.5},
    )
    assert result.passed is True


def test_eval_result_llm_judge_fails_below_threshold():
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_threshold=4.0,
    )
    result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
        level2_scores={"custom": 3.0},
    )
    assert result.passed is False


def test_execute_and_eval_external_mode_short_circuits():
    scenario = _make_scenario(evaluation_mode="external")
    result = execute_and_eval(scenario, settings=SimpleNamespace())
    assert result.passed is False
    assert "external benchmark harness" in (result.error or "")


def test_execute_and_eval_llm_judge_requires_level2():
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
    )
    result = execute_and_eval(scenario, settings=SimpleNamespace(), with_level2=False)
    assert result.passed is False
    assert "requires LLM-as-Judge" in (result.error or "")


def test_has_level2_rubric_with_judge_rubric_text_only():
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        judge_rubric="",
        judge_rubric_text="Score quality from 1-5.",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
    )
    assert _has_level2_rubric(scenario) is True


def test_run_level2_sets_error_when_judge_returns_no_scores(monkeypatch):
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_rubric_text="Score quality from 1-5.",
    )
    eval_result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
    )

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.create_judge_llm",
        lambda settings: object(),
    )

    class DummyJudgeResult:
        scores = []

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.judge_scenario",
        lambda **kwargs: DummyJudgeResult(),
    )

    _run_level2(eval_result, spans=[], scenario=scenario, settings=SimpleNamespace())

    assert eval_result.level2_scores == {"error": -1}
    assert "returned no scores" in (eval_result.error or "")


def test_run_level2_preserves_score_explanations(monkeypatch):
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_rubric_text="Score quality from 1-5.",
    )
    eval_result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
    )

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.create_judge_llm",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.judge_scenario",
        lambda **kwargs: JudgeResult(
            scores=[JudgeScore(dimension="custom", score=4, explanation="Grounded and complete")]
        ),
    )

    _run_level2(eval_result, spans=[], scenario=scenario, settings=SimpleNamespace())

    assert eval_result.level2_scores == {"custom": 4}
    assert eval_result.level2_explanations == {"custom": "Grounded and complete"}


def test_run_level2_clears_geval_step_cache(monkeypatch):
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_rubric="accuracy",
    )
    eval_result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
    )

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.create_judge_llm",
        lambda settings: object(),
    )

    calls = {"cleared": 0}

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.geval.clear_step_cache",
        lambda: calls.__setitem__("cleared", calls["cleared"] + 1),
    )
    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.geval.geval_judge_scenario",
        lambda **kwargs: JudgeResult(
            scores=[JudgeScore(dimension="accuracy", score=4, explanation="Guided by fresh steps")]
        ),
    )

    _run_level2(
        eval_result,
        spans=[],
        scenario=scenario,
        settings=SimpleNamespace(),
        use_geval=True,
    )

    assert calls["cleared"] == 1
    assert eval_result.level2_scores == {"accuracy": 4}


def test_run_level2_adds_faithfulness_metric(monkeypatch):
    scenario = _make_scenario(
        evaluation_mode="llm_judge",
        expected=ExpectedResult(tools_called=[], max_steps=5, output_contains=[]),
        judge_rubric="accuracy",
        context=["hello world"],
    )
    eval_result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, [], [], [], []),
            output_format=OutputFormatResult(True, "ok", [], []),
            trajectory=TrajectoryResult(True, 1, 5, False, 0, 0, None, []),
        ),
    )

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.create_judge_llm",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.judge_scenario",
        lambda **kwargs: JudgeResult(
            scores=[JudgeScore(dimension="accuracy", score=4, explanation="Accurate")]
        ),
    )
    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.faithfulness.evaluate_faithfulness",
        lambda llm, spans, query, context=None: JudgeScore(
            dimension="faithfulness",
            score=5,
            explanation="Fully supported",
        ),
    )

    _run_level2(
        eval_result,
        spans=[],
        scenario=scenario,
        settings=SimpleNamespace(),
        faithfulness=True,
    )

    assert eval_result.level2_scores == {"accuracy": 4, "faithfulness": 5}
    assert eval_result.level2_explanations["faithfulness"] == "Fully supported"


def test_level1_failure_reasons_include_hidden_checks():
    level1 = Level1Result(
        tool_usage=ToolUsageResult(True, ["read_file"], ["read_file"], [], []),
        output_format=OutputFormatResult(True, "ok", ["hello"], []),
        trajectory=TrajectoryResult(True, 2, 5, False, 0, 0, None, []),
        tool_params=ToolParamsResult(
            passed=False,
            violations=[
                ToolParamViolation(
                    tool_name="read_file",
                    param_name="path",
                    reason="value_mismatch",
                    expected="config.yaml",
                    actual="other.txt",
                )
            ],
            checked_count=1,
        ),
        termination=TerminationResult(
            passed=False,
            termination_type="premature",
            reasons=["Agent used only 0 tools, expected at least 1"],
            has_escalation=False,
        ),
        safety=SafetyResult(
            passed=False,
            violations=[
                SafetyViolation(
                    violation_type="leakage",
                    description="Potential api_key leakage detected",
                    severity="high",
                )
            ],
            checked_spans=1,
        ),
    )

    assert level1.supplemental_checks == {
        "params": False,
        "termination": False,
        "safety": False,
    }
    assert "Tool params read_file.path: value_mismatch" in level1.failure_reasons
    assert "Agent used only 0 tools, expected at least 1" in level1.failure_reasons
    assert "Safety leakage: Potential api_key leakage detected" in level1.failure_reasons


def test_annotate_eval_span_emits_semantic_attributes_and_events():
    scenario = _make_scenario(evaluation_mode="llm_judge", judge_threshold=4.0)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    result = EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(True, ["read_file"], ["read_file"], [], []),
            output_format=OutputFormatResult(True, "ok", ["hello"], []),
            trajectory=TrajectoryResult(True, 2, 5, False, 10, 5, None, []),
            trajectory_analysis=TrajectoryAnalysis(
                basic=TrajectoryResult(True, 2, 5, False, 10, 5, None, []),
                structural=StructuralAnalysis([], 0, 0, 0.0, 0, 0.0),
                failure_map=FailureMapResult(patterns=[], dominant_pattern=None, risk_score=0.0),
            ),
        ),
        level2_scores={"accuracy": 4.5},
        feature_flags={"geval": True, "task_completion": False, "answer_relevancy": False, "hallucination": True, "faithfulness": False},
        risk_signals=["unexpected_privileged_tool:shell"],
    )

    with tracer.start_as_current_span("agent.run") as span:
        _annotate_eval_span(span, scenario, result)

    spans = exporter.get_finished_spans()
    span = spans[0]
    assert span.attributes["eval.status"] == "risky_success"
    assert span.attributes["eval.level1.tool_usage_passed"] is True
    assert span.attributes["eval.risk_signal_count"] == 1
    assert span.attributes["eval.flags.geval"] is True
    assert span.attributes["eval.flags.hallucination"] is True
    assert span.attributes["eval.flags.task_completion"] is False
    event_names = [event.name for event in span.events]
    assert "eval.risk_signal" in event_names
    assert "eval.judge_score" in event_names
