"""Tests for trace-based task completion metric."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level2_llm_judge.task_completion import (
    ExtractedTask,
    TaskCompletionResult,
    _parse_completion_result,
    _parse_task_list,
    _ratio_to_score,
    evaluate_task_completion,
    extract_tasks,
    judge_task_completion,
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


# --- Task List Parsing ---


def test_parse_task_list_json_array():
    text = '["Read the file", "Extract content", "Summarize"]'
    tasks = _parse_task_list(text)
    assert len(tasks) == 3
    assert tasks[0] == "Read the file"


def test_parse_task_list_markdown_fenced():
    text = '```json\n["Task A", "Task B"]\n```'
    tasks = _parse_task_list(text)
    assert len(tasks) == 2


def test_parse_task_list_embedded_array():
    text = 'Here are the tasks:\n["Task 1", "Task 2"]\nDone.'
    tasks = _parse_task_list(text)
    assert len(tasks) == 2


def test_parse_task_list_plain_text_fallback():
    text = "Read the file and summarize it"
    tasks = _parse_task_list(text)
    assert len(tasks) == 1
    assert "Read the file" in tasks[0]


def test_parse_task_list_empty():
    tasks = _parse_task_list("")
    assert tasks == []


# --- Completion Result Parsing ---


def test_parse_completion_result_valid():
    text = '{"tasks": [{"task": "Read file", "completed": true, "evidence": "File was read"}, {"task": "Summarize", "completed": false, "evidence": "No summary produced"}]}'
    results = _parse_completion_result(text)
    assert len(results) == 2
    assert results[0].completed is True
    assert results[1].completed is False


def test_parse_completion_result_markdown():
    text = '```json\n{"tasks": [{"task": "Do X", "completed": true, "evidence": "done"}]}\n```'
    results = _parse_completion_result(text)
    assert len(results) == 1
    assert results[0].completed is True


def test_parse_completion_result_invalid():
    results = _parse_completion_result("not json at all")
    assert results == []


# --- Ratio to Score ---


def test_ratio_to_score_full():
    assert _ratio_to_score(1.0) == 5


def test_ratio_to_score_zero():
    assert _ratio_to_score(0.0) == 1


def test_ratio_to_score_half():
    assert _ratio_to_score(0.5) == 3


# --- TaskCompletionResult ---


def test_task_completion_result_ratio():
    result = TaskCompletionResult(tasks=[
        ExtractedTask(task="A", completed=True),
        ExtractedTask(task="B", completed=False),
        ExtractedTask(task="C", completed=True),
    ])
    assert abs(result.completion_ratio - 2 / 3) < 0.01


def test_task_completion_result_empty():
    result = TaskCompletionResult(tasks=[])
    assert result.completion_ratio == 0.0


# --- Extract Tasks ---


def test_extract_tasks_calls_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["Read the file", "Check its content"]'
    )
    tasks = extract_tasks(mock_llm, "Read the file and check its content")
    assert len(tasks) == 2
    mock_llm.invoke.assert_called_once()


# --- Judge Task Completion ---


def test_judge_task_completion_all_done():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"tasks": [{"task": "Read file", "completed": true, "evidence": "File read successfully"}]}'
    )
    result = judge_task_completion(
        mock_llm,
        tasks=["Read file"],
        trajectory="Tool: read_file(path=/tmp/test) -> hello",
        final_answer="The file contains: hello",
    )
    assert result.completion_ratio == 1.0


def test_judge_task_completion_none_done():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"tasks": [{"task": "Read file", "completed": false, "evidence": "Agent did not read any file"}]}'
    )
    result = judge_task_completion(
        mock_llm,
        tasks=["Read file"],
        trajectory="(no trajectory captured)",
        final_answer="I cannot help with that.",
    )
    assert result.completion_ratio == 0.0


# --- End-to-End ---


def test_evaluate_task_completion_full_pipeline():
    mock_llm = MagicMock()
    # First call: task extraction. Second call: completion judgment.
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Read the file", "Report content"]'),
        AIMessage(content='{"tasks": [{"task": "Read the file", "completed": true, "evidence": "done"}, {"task": "Report content", "completed": true, "evidence": "done"}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The file says hello"}},
    )
    score = evaluate_task_completion(mock_llm, spans, "Read the file and report its content")
    assert score.dimension == "task_completion"
    assert score.score == 5
    assert "2/2" in score.explanation


def test_evaluate_task_completion_partial():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Read file", "Write summary"]'),
        AIMessage(content='{"tasks": [{"task": "Read file", "completed": true, "evidence": "done"}, {"task": "Write summary", "completed": false, "evidence": "not done"}]}'),
    ]
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "File content: hello"}},
    )
    score = evaluate_task_completion(mock_llm, spans, "Read file and write summary")
    assert score.dimension == "task_completion"
    assert score.score == 3  # 50% completion -> score 3


def test_evaluate_task_completion_no_tasks_extracted():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="")
    spans = _make_spans()
    score = evaluate_task_completion(mock_llm, spans, "")
    assert score.dimension == "task_completion"
    assert score.score == 1
