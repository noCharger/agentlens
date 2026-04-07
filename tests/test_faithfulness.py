"""Tests for faithfulness metric."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level2_llm_judge.faithfulness import (
    _parse_support_results,
    _ratio_to_score,
    detect_support,
    evaluate_faithfulness,
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


def test_parse_support_results_valid():
    text = '{"results": [{"statement": "Earth is round", "supported": true, "explanation": "present in context"}]}'
    results = _parse_support_results(text)
    assert results == [("Earth is round", True, "present in context")]


def test_parse_support_results_invalid():
    assert _parse_support_results("not json") == []


def test_ratio_to_score_all_supported():
    assert _ratio_to_score(1.0) == 5


def test_ratio_to_score_none_supported():
    assert _ratio_to_score(0.0) == 1


def test_detect_support_empty_inputs():
    mock_llm = MagicMock()
    assert detect_support(mock_llm, [], ["ctx"]) == []
    assert detect_support(mock_llm, ["stmt"], []) == []
    mock_llm.invoke.assert_not_called()


def test_detect_support_calls_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"statement": "Earth is round", "supported": true, "explanation": "supported"}]}'
    )
    results = detect_support(mock_llm, ["Earth is round"], ["Earth is round"])
    assert results[0][1] is True


def test_evaluate_faithfulness_all_supported():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Earth is round"]'),
        AIMessage(content='{"results": [{"statement": "Earth is round", "supported": true, "explanation": "supported"}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "Earth is round."}},
    )
    score = evaluate_faithfulness(mock_llm, spans, "What shape is Earth?", context=["Earth is round"])
    assert score.dimension == "faithfulness"
    assert score.score == 5
    assert "1/1 statements supported" in score.explanation


def test_evaluate_faithfulness_unsupported_claims():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Earth is flat"]'),
        AIMessage(content='{"results": [{"statement": "Earth is flat", "supported": false, "explanation": "context says round"}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "Earth is flat."}},
    )
    score = evaluate_faithfulness(mock_llm, spans, "What shape is Earth?", context=["Earth is round"])
    assert score.dimension == "faithfulness"
    assert score.score == 1


def test_evaluate_faithfulness_no_context_uses_trace():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["The file says hello world"]'),
        AIMessage(content='{"results": [{"statement": "The file says hello world", "supported": true, "explanation": "supported"}]}'),
    ]
    spans = _make_spans(
        {"name": "tool", "attributes": {"tool.name": "read_file", "tool.output": "The file says hello world"}},
        {"name": "agent.run", "attributes": {"agent.output": "The file says hello world."}},
    )
    score = evaluate_faithfulness(mock_llm, spans, "What does the file say?")
    assert score.dimension == "faithfulness"
    assert score.score == 5


def test_evaluate_faithfulness_no_context_returns_neutral():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='["Some answer"]')
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "Some answer"}},
    )
    score = evaluate_faithfulness(mock_llm, spans, "query")
    assert score.dimension == "faithfulness"
    assert score.score == 3


def test_evaluate_faithfulness_no_answer():
    mock_llm = MagicMock()
    spans = _make_spans({"name": "other", "attributes": {}})
    score = evaluate_faithfulness(mock_llm, spans, "query", context=["some context"])
    assert score.dimension == "faithfulness"
    assert score.score == 1
