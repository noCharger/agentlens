"""Tests for the eval runner orchestration logic."""

from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.runner import (
    EvalResult,
    Level1Result,
    evaluate_scenario,
    execute_and_eval,
    run_level1_eval,
)
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult


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
