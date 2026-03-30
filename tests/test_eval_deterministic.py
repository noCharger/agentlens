"""Tests for Level 1 deterministic evaluators using mock spans."""

import pytest

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level1_deterministic.tool_usage import evaluate_tool_usage, extract_tool_names
from agentlens.eval.level1_deterministic.output_format import evaluate_output_format, extract_output
from agentlens.eval.level1_deterministic.trajectory import (
    evaluate_trajectory,
    count_steps,
    detect_loop,
    sum_tokens,
)


def _make_spans(*span_configs) -> list[ReadableSpan]:
    """Create mock spans from config dicts.

    Each config: {"name": "...", "attributes": {...}}
    """
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


# --- Tool Usage Tests ---


class TestToolUsage:
    def test_extract_tool_names_from_tool_name_attr(self):
        spans = _make_spans({"name": "tool_call", "attributes": {"tool.name": "read_file"}})
        assert extract_tool_names(spans) == ["read_file"]

    def test_extract_tool_names_from_function_name(self):
        spans = _make_spans(
            {"name": "call", "attributes": {"tool_call.function.name": "shell"}}
        )
        assert extract_tool_names(spans) == ["shell"]

    def test_extract_tool_names_from_span_prefix(self):
        spans = _make_spans({"name": "Tool: write_file", "attributes": {}})
        assert extract_tool_names(spans) == ["write_file"]

    def test_evaluate_all_expected_tools_called(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "write_file"}},
        )
        result = evaluate_tool_usage(spans, ["read_file", "write_file"])
        assert result.passed is True
        assert result.missing_tools == []

    def test_evaluate_missing_tool(self):
        spans = _make_spans({"name": "t", "attributes": {"tool.name": "read_file"}})
        result = evaluate_tool_usage(spans, ["read_file", "write_file"])
        assert result.passed is False
        assert "write_file" in result.missing_tools

    def test_evaluate_unexpected_tool_still_passes(self):
        spans = _make_spans(
            {"name": "t", "attributes": {"tool.name": "read_file"}},
            {"name": "t", "attributes": {"tool.name": "shell"}},
        )
        result = evaluate_tool_usage(spans, ["read_file"])
        assert result.passed is True
        assert "shell" in result.unexpected_tools

    def test_evaluate_empty_expected(self):
        spans = _make_spans({"name": "t", "attributes": {"tool.name": "shell"}})
        result = evaluate_tool_usage(spans, [])
        assert result.passed is True


# --- Output Format Tests ---


class TestOutputFormat:
    def test_extract_output_from_agent_output(self):
        spans = _make_spans({"name": "agent.run", "attributes": {"agent.output": "hello world"}})
        assert extract_output(spans) == "hello world"

    def test_extract_output_from_openinference(self):
        spans = _make_spans({"name": "Chain", "attributes": {"output.value": "result text"}})
        assert extract_output(spans) == "result text"

    def test_extract_output_empty(self):
        spans = _make_spans({"name": "other", "attributes": {}})
        assert extract_output(spans) == ""

    def test_evaluate_all_substrings_present(self):
        spans = _make_spans(
            {"name": "agent.run", "attributes": {"agent.output": "The answer is 1081"}}
        )
        result = evaluate_output_format(spans, ["1081"])
        assert result.passed is True
        assert result.missing_substrings == []

    def test_evaluate_case_insensitive(self):
        spans = _make_spans(
            {"name": "agent.run", "attributes": {"agent.output": "Hello World"}}
        )
        result = evaluate_output_format(spans, ["hello", "world"])
        assert result.passed is True

    def test_evaluate_missing_substring(self):
        spans = _make_spans(
            {"name": "agent.run", "attributes": {"agent.output": "Some output"}}
        )
        result = evaluate_output_format(spans, ["missing_text"])
        assert result.passed is False
        assert "missing_text" in result.missing_substrings


# --- Trajectory Tests ---


class TestTrajectory:
    def test_count_steps(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.index": 0}},
            {"name": "agent.step", "attributes": {"step.index": 1}},
            {"name": "LLM", "attributes": {}},
        )
        assert count_steps(spans) == 2

    def test_detect_loop_no_loop(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.action": "read"}},
            {"name": "agent.step", "attributes": {"step.action": "write"}},
            {"name": "agent.step", "attributes": {"step.action": "shell"}},
        )
        assert detect_loop(spans) is False

    def test_detect_loop_found(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.action": "read"}},
            {"name": "agent.step", "attributes": {"step.action": "read"}},
            {"name": "agent.step", "attributes": {"step.action": "read"}},
        )
        assert detect_loop(spans) is True

    def test_detect_loop_too_few_steps(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.action": "read"}},
        )
        assert detect_loop(spans) is False

    def test_sum_tokens(self):
        spans = _make_spans(
            {"name": "LLM", "attributes": {"llm.token_count.prompt": 100, "llm.token_count.completion": 50}},
            {"name": "LLM", "attributes": {"llm.token_count.prompt": 200, "llm.token_count.completion": 80}},
        )
        prompt, completion = sum_tokens(spans)
        assert prompt == 300
        assert completion == 130

    def test_evaluate_within_budget(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.index": 0}},
            {"name": "agent.step", "attributes": {"step.index": 1}},
        )
        result = evaluate_trajectory(spans, max_steps=5)
        assert result.passed is True
        assert result.total_steps == 2

    def test_evaluate_exceeds_steps(self):
        spans = _make_spans(
            *[{"name": "agent.step", "attributes": {"step.index": i}} for i in range(6)]
        )
        result = evaluate_trajectory(spans, max_steps=3)
        assert result.passed is False
        assert "exceeds" in result.reasons[0]

    def test_evaluate_with_loop(self):
        spans = _make_spans(
            {"name": "agent.step", "attributes": {"step.action": "x", "step.index": 0}},
            {"name": "agent.step", "attributes": {"step.action": "x", "step.index": 1}},
            {"name": "agent.step", "attributes": {"step.action": "x", "step.index": 2}},
        )
        result = evaluate_trajectory(spans, max_steps=10)
        assert result.passed is False
        assert result.has_loop is True

    def test_evaluate_token_budget_exceeded(self):
        spans = _make_spans(
            {"name": "LLM", "attributes": {"llm.token_count.prompt": 500, "llm.token_count.completion": 300}},
        )
        result = evaluate_trajectory(spans, max_tokens=500)
        assert result.passed is False
        assert "Token usage" in result.reasons[0]

    def test_evaluate_token_budget_ok(self):
        spans = _make_spans(
            {"name": "LLM", "attributes": {"llm.token_count.prompt": 100, "llm.token_count.completion": 50}},
        )
        result = evaluate_trajectory(spans, max_tokens=500)
        assert result.passed is True
