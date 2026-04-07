"""Tests for NLI-based hallucination detection metric."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level2_llm_judge.hallucination import (
    _parse_contradiction_results,
    _ratio_to_score,
    detect_contradictions,
    evaluate_hallucination,
    extract_agent_context,
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


# --- Contradiction Result Parsing ---


def test_parse_contradiction_results_valid():
    text = '{"results": [{"context": "Earth is round", "contradicted": false, "explanation": "consistent"}, {"context": "Water is wet", "contradicted": true, "explanation": "output says water is dry"}]}'
    results = _parse_contradiction_results(text)
    assert len(results) == 2
    assert results[0] == ("Earth is round", False, "consistent")
    assert results[1] == ("Water is wet", True, "output says water is dry")


def test_parse_contradiction_results_markdown():
    text = '```json\n{"results": [{"context": "A", "contradicted": false, "explanation": "ok"}]}\n```'
    results = _parse_contradiction_results(text)
    assert len(results) == 1


def test_parse_contradiction_results_invalid():
    results = _parse_contradiction_results("not json")
    assert results == []


# --- Ratio to Score ---


def test_ratio_to_score_no_contradictions():
    assert _ratio_to_score(1.0) == 5


def test_ratio_to_score_all_contradictions():
    assert _ratio_to_score(0.0) == 1


def test_ratio_to_score_half():
    assert _ratio_to_score(0.5) == 3


# --- Context Extraction from Spans ---


def test_extract_agent_context_from_tool_outputs():
    spans = _make_spans(
        {"name": "tool", "attributes": {"tool.name": "read_file", "tool.output": "The file contains important data"}},
        {"name": "tool", "attributes": {"tool.name": "shell", "tool.output": "command output here"}},
    )
    context = extract_agent_context(spans)
    assert len(context) == 2
    assert "important data" in context[0]


def test_extract_agent_context_skips_short_outputs():
    spans = _make_spans(
        {"name": "tool", "attributes": {"tool.name": "shell", "tool.output": "ok"}},  # too short (<=5)
    )
    context = extract_agent_context(spans)
    assert len(context) == 0


def test_extract_agent_context_empty_spans():
    spans = _make_spans({"name": "other", "attributes": {}})
    context = extract_agent_context(spans)
    assert context == []


# --- Detect Contradictions ---


def test_detect_contradictions_no_contradictions():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "Earth is round", "contradicted": false, "explanation": "output agrees"}]}'
    )
    results = detect_contradictions(mock_llm, "The Earth is round.", ["Earth is round"])
    assert len(results) == 1
    assert results[0][1] is False


def test_detect_contradictions_with_contradiction():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "Earth is round", "contradicted": true, "explanation": "output claims flat"}]}'
    )
    results = detect_contradictions(mock_llm, "The Earth is flat.", ["Earth is round"])
    assert len(results) == 1
    assert results[0][1] is True


def test_detect_contradictions_empty_context():
    mock_llm = MagicMock()
    results = detect_contradictions(mock_llm, "Some output", [])
    assert results == []
    mock_llm.invoke.assert_not_called()


def test_detect_contradictions_empty_output():
    mock_llm = MagicMock()
    results = detect_contradictions(mock_llm, "", ["Some context"])
    assert results == []
    mock_llm.invoke.assert_not_called()


# --- End-to-End ---


def test_evaluate_hallucination_no_contradictions():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "Earth is round", "contradicted": false, "explanation": "consistent"}]}'
    )
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The Earth is round."}},
    )
    score = evaluate_hallucination(mock_llm, spans, "What shape is Earth?", context=["Earth is round"])
    assert score.dimension == "hallucination"
    assert score.score == 5
    assert "0/1" in score.explanation


def test_evaluate_hallucination_with_contradiction():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "Earth is round", "contradicted": true, "explanation": "says flat"}]}'
    )
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The Earth is flat."}},
    )
    score = evaluate_hallucination(mock_llm, spans, "What shape is Earth?", context=["Earth is round"])
    assert score.dimension == "hallucination"
    assert score.score == 1  # 100% contradiction -> score 1


def test_evaluate_hallucination_partial_contradiction():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "A is true", "contradicted": false, "explanation": "ok"}, {"context": "B is false", "contradicted": true, "explanation": "says B is true"}]}'
    )
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "A is true. B is true."}},
    )
    score = evaluate_hallucination(mock_llm, spans, "query", context=["A is true", "B is false"])
    assert score.dimension == "hallucination"
    assert score.score == 3  # 50% contradiction -> score 3


def test_evaluate_hallucination_no_context_uses_trace():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"context": "file says hello world", "contradicted": false, "explanation": "consistent"}]}'
    )
    spans = _make_spans(
        {"name": "tool", "attributes": {"tool.name": "read_file", "tool.output": "file says hello world"}},
        {"name": "agent.run", "attributes": {"agent.output": "The file says hello world."}},
    )
    score = evaluate_hallucination(mock_llm, spans, "What does the file say?")
    assert score.dimension == "hallucination"
    assert score.score >= 3  # Should detect from trace context
    mock_llm.invoke.assert_called_once()


def test_evaluate_hallucination_no_context_no_trace():
    mock_llm = MagicMock()
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "Some answer"}},
    )
    score = evaluate_hallucination(mock_llm, spans, "query")
    assert score.dimension == "hallucination"
    assert score.score == 3  # Neutral score when no context available
    mock_llm.invoke.assert_not_called()


def test_evaluate_hallucination_no_answer():
    mock_llm = MagicMock()
    spans = _make_spans({"name": "other", "attributes": {}})
    score = evaluate_hallucination(mock_llm, spans, "query", context=["some context"])
    assert score.dimension == "hallucination"
    assert score.score == 1


# --- Schema backward compatibility ---


def test_scenario_context_field_default():
    """Verify Scenario loads without context field (backward compat)."""
    from agentlens.eval.scenarios import Scenario, ExpectedResult

    scenario = Scenario(
        id="test-001",
        name="Test",
        category="test",
        input="Do something",
        expected=ExpectedResult(),
    )
    assert scenario.context == []


def test_scenario_context_field_explicit():
    """Verify Scenario loads with explicit context field."""
    from agentlens.eval.scenarios import Scenario, ExpectedResult

    scenario = Scenario(
        id="test-002",
        name="Test with context",
        category="test",
        input="Do something",
        expected=ExpectedResult(),
        context=["fact 1", "fact 2"],
    )
    assert scenario.context == ["fact 1", "fact 2"]


def test_scenario_from_dict_with_context():
    """Verify Scenario.from_dict handles context field."""
    from agentlens.eval.scenarios import Scenario

    data = {
        "id": "test-003",
        "name": "Test from dict",
        "category": "test",
        "input": {"query": "Do something"},
        "expected": {},
        "context": ["ctx 1", "ctx 2"],
    }
    scenario = Scenario.from_dict(data)
    assert scenario.context == ["ctx 1", "ctx 2"]


def test_scenario_from_dict_without_context():
    """Verify Scenario.from_dict works without context field."""
    from agentlens.eval.scenarios import Scenario

    data = {
        "id": "test-004",
        "name": "Test no context",
        "category": "test",
        "input": {"query": "Do something"},
        "expected": {},
    }
    scenario = Scenario.from_dict(data)
    assert scenario.context == []
