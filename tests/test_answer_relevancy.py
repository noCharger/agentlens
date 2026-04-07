"""Tests for atomic statement-level answer relevancy metric."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level2_llm_judge.answer_relevancy import (
    _parse_relevance_results,
    _parse_string_list,
    _ratio_to_score,
    evaluate_answer_relevancy,
    extract_statements,
    judge_statement_relevance,
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


# --- String List Parsing ---


def test_parse_string_list_json():
    result = _parse_string_list('["Statement 1", "Statement 2"]')
    assert len(result) == 2


def test_parse_string_list_markdown():
    result = _parse_string_list('```json\n["A", "B"]\n```')
    assert len(result) == 2


def test_parse_string_list_embedded():
    result = _parse_string_list('Here:\n["X", "Y"]\nDone.')
    assert len(result) == 2


def test_parse_string_list_empty():
    result = _parse_string_list("")
    assert result == []


# --- Relevance Results Parsing ---


def test_parse_relevance_results_valid():
    text = '{"results": [{"statement": "The sky is blue", "relevant": true}, {"statement": "Pizza is good", "relevant": false}]}'
    results = _parse_relevance_results(text)
    assert len(results) == 2
    assert results[0] == ("The sky is blue", True)
    assert results[1] == ("Pizza is good", False)


def test_parse_relevance_results_markdown():
    text = '```json\n{"results": [{"statement": "A", "relevant": true}]}\n```'
    results = _parse_relevance_results(text)
    assert len(results) == 1


def test_parse_relevance_results_invalid():
    results = _parse_relevance_results("not json")
    assert results == []


# --- Ratio to Score ---


def test_ratio_to_score_full():
    assert _ratio_to_score(1.0) == 5


def test_ratio_to_score_zero():
    assert _ratio_to_score(0.0) == 1


def test_ratio_to_score_half():
    assert _ratio_to_score(0.5) == 3


# --- Extract Statements ---


def test_extract_statements_calls_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["The file contains hello", "The file is in /tmp"]'
    )
    stmts = extract_statements(mock_llm, "The file contains hello and is in /tmp")
    assert len(stmts) == 2
    mock_llm.invoke.assert_called_once()


def test_extract_statements_empty_input():
    mock_llm = MagicMock()
    stmts = extract_statements(mock_llm, "")
    assert stmts == []
    mock_llm.invoke.assert_not_called()


# --- Judge Statement Relevance ---


def test_judge_statement_relevance_all_relevant():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"statement": "Today is sunny", "relevant": true}, {"statement": "Temperature is 25C", "relevant": true}]}'
    )
    results = judge_statement_relevance(mock_llm, ["Today is sunny", "Temperature is 25C"], "What is the weather?")
    assert len(results) == 2
    assert all(r for _, r in results)


def test_judge_statement_relevance_mixed():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"results": [{"statement": "Today is sunny", "relevant": true}, {"statement": "Pizza is great", "relevant": false}]}'
    )
    results = judge_statement_relevance(mock_llm, ["Today is sunny", "Pizza is great"], "What is the weather?")
    assert results[0][1] is True
    assert results[1][1] is False


def test_judge_statement_relevance_empty():
    mock_llm = MagicMock()
    results = judge_statement_relevance(mock_llm, [], "query")
    assert results == []
    mock_llm.invoke.assert_not_called()


# --- End-to-End ---


def test_evaluate_answer_relevancy_all_relevant():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["The weather is sunny", "Temperature is 25C"]'),
        AIMessage(content='{"results": [{"statement": "The weather is sunny", "relevant": true}, {"statement": "Temperature is 25C", "relevant": true}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The weather is sunny. Temperature is 25C."}},
    )
    score = evaluate_answer_relevancy(mock_llm, spans, "What is the weather?")
    assert score.dimension == "answer_relevancy"
    assert score.score == 5
    assert "2/2" in score.explanation


def test_evaluate_answer_relevancy_mixed():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["The weather is sunny", "Pizza is great", "History of Beijing", "Temperature is 25C"]'),
        AIMessage(content='{"results": [{"statement": "The weather is sunny", "relevant": true}, {"statement": "Pizza is great", "relevant": false}, {"statement": "History of Beijing", "relevant": false}, {"statement": "Temperature is 25C", "relevant": true}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The weather is sunny. Pizza is great. History of Beijing. Temperature is 25C."}},
    )
    score = evaluate_answer_relevancy(mock_llm, spans, "What is the weather?")
    assert score.dimension == "answer_relevancy"
    assert score.score == 3  # 50% relevant -> score 3


def test_evaluate_answer_relevancy_no_answer():
    mock_llm = MagicMock()
    spans = _make_spans({"name": "other", "attributes": {}})
    score = evaluate_answer_relevancy(mock_llm, spans, "query")
    assert score.dimension == "answer_relevancy"
    assert score.score == 1


def test_evaluate_answer_relevancy_no_statements_extracted():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="[]")
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "OK"}},
    )
    score = evaluate_answer_relevancy(mock_llm, spans, "query")
    assert score.dimension == "answer_relevancy"
    assert score.score == 1
