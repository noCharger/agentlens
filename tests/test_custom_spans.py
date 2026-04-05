from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentlens.observability.custom_spans import (
    agent_run_span,
    agent_step_span,
    record_recovery_event,
    finalize_run_span,
    set_custom_tracer_provider,
)

import pytest


@pytest.fixture(autouse=True)
def trace_capture():
    """Create a fresh tracer provider + in-memory exporter for each test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_custom_tracer_provider(provider)
    yield exporter
    provider.shutdown()
    set_custom_tracer_provider(None)


def test_agent_run_span_basic(trace_capture):
    exporter = trace_capture
    with agent_run_span(
        scenario_id="tc-001",
        query="test query",
        benchmark="swe-bench-pro",
        category="qa",
        evaluation_mode="deterministic",
    ) as span:
        finalize_run_span(span, total_steps=3, success=True, output="result")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "agent.run"
    assert s.attributes["agent.scenario_id"] == "tc-001"
    assert s.attributes["agent.benchmark"] == "swe-bench-pro"
    assert s.attributes["agent.category"] == "qa"
    assert s.attributes["eval.evaluation_mode"] == "deterministic"
    assert s.attributes["agent.input_query"] == "test query"
    assert s.attributes["agent.total_steps"] == 3
    assert s.attributes["agent.success"] is True
    assert s.attributes["agent.output"] == "result"
    assert s.status.status_code == StatusCode.OK


def test_agent_run_span_failure(trace_capture):
    exporter = trace_capture
    with agent_run_span(query="bad query") as span:
        finalize_run_span(span, total_steps=1, success=False, error="tool crashed")

    spans = exporter.get_finished_spans()
    s = spans[0]
    assert s.attributes["agent.success"] is False
    assert s.status.status_code == StatusCode.ERROR


def test_agent_step_span(trace_capture):
    exporter = trace_capture
    with agent_step_span(step_index=0, thought="I need to read a file", action="read_file"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "agent.step"
    assert s.attributes["step.index"] == 0
    assert s.attributes["step.thought"] == "I need to read a file"
    assert s.attributes["step.action"] == "read_file"


def test_recovery_event(trace_capture):
    exporter = trace_capture
    with agent_step_span(step_index=1) as span:
        record_recovery_event(
            span,
            error_type="FileNotFoundError",
            retry_count=1,
            fallback_action="create_file",
        )

    spans = exporter.get_finished_spans()
    s = spans[0]
    events = s.events
    assert len(events) == 1
    e = events[0]
    assert e.name == "recovery"
    assert e.attributes["recovery.error_type"] == "FileNotFoundError"
    assert e.attributes["recovery.retry_count"] == 1
    assert e.attributes["recovery.fallback_action"] == "create_file"


def test_nested_run_and_step_spans(trace_capture):
    exporter = trace_capture
    with agent_run_span(scenario_id="tc-002", query="nested test") as run_span:
        with agent_step_span(step_index=0, action="shell"):
            pass
        with agent_step_span(step_index=1, action="write_file"):
            pass
        finalize_run_span(run_span, total_steps=2, success=True)

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    names = [s.name for s in spans]
    assert names.count("agent.step") == 2
    assert names.count("agent.run") == 1
