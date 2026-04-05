"""Tests for unified trajectory analysis and fault diagnosis module.

Covers: structural analysis (Tier 2) and failure map detection (Tier 3).
Basic metrics (Tier 1) are covered in test_eval_deterministic.py.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level1_deterministic.trajectory import (
    analyze_structure,
    analyze_trajectory,
    detect_failure_patterns,
    evaluate_trajectory,
)


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


# --- Tier 2: Structural Analysis ---


class TestStructuralAnalysis:
    def test_empty_spans(self):
        spans = _make_spans()
        result = analyze_structure(spans)
        assert result.tool_sequence == []
        assert result.unique_tools_used == 0
        assert result.strategy_drift_score == 0.0

    def test_consistent_strategy(self):
        """Same tool used repeatedly = low drift."""
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "read_file"}},
        )
        result = analyze_structure(spans)
        assert result.unique_tools_used == 1
        assert result.tool_switch_count == 0
        assert result.strategy_drift_score == 0.5  # same transition repeated

    def test_high_drift(self):
        """Many different tools = high drift."""
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "shell"}},
            {"name": "t", "attributes": {"tool.name": "write_file"}},
            {"name": "t", "attributes": {"tool.name": "duckduckgo_search"}},
            {"name": "t", "attributes": {"tool.name": "read_file"}},
        )
        result = analyze_structure(spans)
        assert result.unique_tools_used == 4
        assert result.tool_switch_count == 4
        assert result.strategy_drift_score == 1.0

    def test_tool_switch_count(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "shell"}},
            {"name": "t", "attributes": {"tool.name": "shell"}},
            {"name": "t", "attributes": {"tool.name": "write_file"}},
        )
        result = analyze_structure(spans)
        assert result.tool_switch_count == 2


# --- Tier 3: Failure Map Detection ---


class TestFailureMap:
    def test_no_failures_on_clean_trace(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file", "input.value": "config.yaml"}},
            {"name": "t", "attributes": {"tool.name": "write_file", "input.value": "output.txt"}},
            {"name": "agent.run", "attributes": {"agent.output": "Done."}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert not result.has_failures
        assert result.risk_score == 0.0

    def test_detect_loop_trap(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"openinference.span.kind": "TOOL", "tool.name": "shell", "input.value": "ls /missing"}},
            {"name": "t", "attributes": {"openinference.span.kind": "TOOL", "tool.name": "shell", "input.value": "ls /missing"}},
            {"name": "t", "attributes": {"openinference.span.kind": "TOOL", "tool.name": "shell", "input.value": "ls /missing"}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert result.has_failures
        assert any(p.pattern_type == "loop_trap" for p in result.patterns)
        assert result.risk_score >= 0.8

    def test_detect_confabulation(self):
        long_output = " ".join(["word"] * 60)
        spans = _make_spans(
            {"name": "agent.run", "attributes": {"agent.output": long_output}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert any(p.pattern_type == "confabulation" for p in result.patterns)

    def test_detect_missing_confirmation_first_privileged(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "shell"}},
            {"name": "t", "attributes": {"tool.name": "read_file"}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert any(p.pattern_type == "missing_confirmation" for p in result.patterns)

    def test_no_confabulation_with_tool_calls(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "shell"}},
            {"name": "t", "attributes": {"tool.name": "write_file"}},
            {"name": "agent.run", "attributes": {"agent.output": "The file has been updated."}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert not any(p.pattern_type == "confabulation" for p in result.patterns)

    def test_detect_context_forgetting(self):
        """Agent redoes same work far apart in the sequence."""
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file", "input.value": "config.yaml"}},
            {"name": "t", "attributes": {"tool.name": "shell", "input.value": "ls"}},
            {"name": "t", "attributes": {"tool.name": "write_file", "input.value": "out.txt"}},
            {"name": "t", "attributes": {"tool.name": "shell", "input.value": "echo test"}},
            {"name": "t", "attributes": {"tool.name": "read_file", "input.value": "config.yaml"}},
            {"name": "t", "attributes": {"tool.name": "shell", "input.value": "ls"}},
        )
        basic = evaluate_trajectory(spans, max_steps=10)
        structure = analyze_structure(spans)
        result = detect_failure_patterns(spans, basic, structure)
        assert any(p.pattern_type == "context_forgetting" for p in result.patterns)


# --- Full trajectory analysis ---


class TestFullAnalysis:
    def test_analyze_trajectory_clean(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "agent.run", "attributes": {"agent.output": "Done"}},
        )
        analysis = analyze_trajectory(spans, max_steps=10)
        assert analysis.basic.passed is True
        assert analysis.structural.unique_tools_used == 1
        assert analysis.failure_map.risk_score == 0.0

    def test_analyze_trajectory_with_issues(self):
        long_output = " ".join(["analysis"] * 80)
        spans = _make_spans(
            {"name": "agent.run", "attributes": {"agent.output": long_output}},
        )
        analysis = analyze_trajectory(spans, max_steps=10)
        assert analysis.failure_map.has_failures
        assert analysis.failure_map.dominant_pattern == "confabulation"
