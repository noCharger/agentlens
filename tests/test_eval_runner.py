"""Tests for the eval runner orchestration logic."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.scenarios import Scenario, ExpectedResult
from agentlens.eval.runner import run_level1_eval, evaluate_scenario


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
