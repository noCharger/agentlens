"""Tests for Level 2 LLM-as-Judge (using mocked LLM responses)."""

from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from langchain_core.messages import AIMessage

from agentlens.eval.level2_llm_judge.rubrics import (
    JudgeScore,
    JudgeResult,
    RUBRIC_DEFINITIONS,
)
from agentlens.eval.level2_llm_judge.judge import (
    _format_trajectory,
    _extract_final_answer,
    _parse_judge_response,
    create_judge_llm,
    judge_scenario,
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


# --- Rubrics Tests ---


def test_judge_score_validation():
    s = JudgeScore(dimension="accuracy", score=4, explanation="good")
    assert s.score == 4


def test_judge_result_overall_score():
    r = JudgeResult(scores=[
        JudgeScore(dimension="accuracy", score=4, explanation=""),
        JudgeScore(dimension="reasoning", score=2, explanation=""),
    ])
    assert r.overall_score == 3.0


def test_judge_result_empty():
    r = JudgeResult(scores=[])
    assert r.overall_score == 0.0


def test_rubric_definitions_exist():
    assert "accuracy" in RUBRIC_DEFINITIONS
    assert "reasoning_quality" in RUBRIC_DEFINITIONS
    assert "tool_appropriateness" in RUBRIC_DEFINITIONS
    assert "recovery_behavior" in RUBRIC_DEFINITIONS


# --- Trajectory Formatting ---


def test_format_trajectory_with_steps():
    spans = _make_spans(
        {"name": "agent.step", "attributes": {"step.index": 0, "step.thought": "think", "step.action": "read"}},
        {"name": "agent.step", "attributes": {"step.index": 1, "step.thought": "done", "step.action": "respond"}},
    )
    text = _format_trajectory(spans)
    assert "Step 0" in text
    assert "Step 1" in text
    assert "think" in text
    assert "read" in text


def test_format_trajectory_with_tools():
    spans = _make_spans(
        {"name": "tool", "attributes": {"tool.name": "shell", "tool.params": "ls", "tool.output": "a.py"}},
    )
    text = _format_trajectory(spans)
    assert "shell" in text


def test_format_trajectory_empty():
    spans = _make_spans({"name": "other", "attributes": {}})
    text = _format_trajectory(spans)
    assert "no trajectory" in text


# --- Final Answer Extraction ---


def test_extract_final_answer():
    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "The answer is 42"}},
    )
    assert _extract_final_answer(spans) == "The answer is 42"


def test_extract_final_answer_fallback():
    spans = _make_spans(
        {"name": "Chain", "attributes": {"output.value": "fallback answer"}},
    )
    assert _extract_final_answer(spans) == "fallback answer"


def test_extract_final_answer_none():
    spans = _make_spans({"name": "other", "attributes": {}})
    assert "no answer" in _extract_final_answer(spans)


# --- JSON Parsing ---


def test_parse_judge_response_plain_json():
    text = '{"dimension": "accuracy", "score": 4, "explanation": "good answer"}'
    result = _parse_judge_response(text, "accuracy")
    assert result.score == 4
    assert result.dimension == "accuracy"
    assert result.explanation == "good answer"


def test_parse_judge_response_markdown_json():
    text = '```json\n{"dimension": "accuracy", "score": 3, "explanation": "ok"}\n```'
    result = _parse_judge_response(text, "accuracy")
    assert result.score == 3


def test_parse_judge_response_with_wrapped_text():
    text = (
        "Here is the evaluation result:\n"
        '{"dimension":"accuracy","score":4,"explanation":"mostly correct"}\n'
        "done."
    )
    result = _parse_judge_response(text, "accuracy")
    assert result.score == 4


def test_parse_judge_response_list_content_blocks():
    content = [
        {"type": "text", "text": '{"dimension":"accuracy","score":5,"explanation":"great"}'},
    ]
    result = _parse_judge_response(content, "accuracy")
    assert result.score == 5


def test_parse_judge_response_clamps_zero_score_to_one():
    text = '{"dimension":"accuracy","score":0,"explanation":"bad"}'
    result = _parse_judge_response(text, "accuracy")
    assert result.score == 1


def test_parse_judge_response_clamps_large_score_to_five():
    text = '{"dimension":"accuracy","score":9,"explanation":"great"}'
    result = _parse_judge_response(text, "accuracy")
    assert result.score == 5


# --- Judge Scenario (mocked LLM) ---


def test_judge_scenario_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"dimension": "accuracy", "score": 5, "explanation": "perfect"}'
    )

    spans = _make_spans(
        {"name": "agent.step", "attributes": {"step.index": 0, "step.action": "read"}},
        {"name": "agent.run", "attributes": {"agent.output": "hello world"}},
    )

    result = judge_scenario(
        llm=mock_llm,
        spans=spans,
        query="Read the file",
        reference_answer="hello world",
        rubric_name="accuracy",
    )

    assert len(result.scores) == 1
    assert result.scores[0].score == 5
    assert result.overall_score == 5.0
    mock_llm.invoke.assert_called_once()


def test_judge_scenario_unknown_rubric():
    mock_llm = MagicMock()
    result = judge_scenario(
        llm=mock_llm,
        spans=[],
        query="test",
        reference_answer="test",
        rubric_name="nonexistent_rubric",
    )
    assert len(result.scores) == 0
    mock_llm.invoke.assert_not_called()


def test_judge_scenario_with_custom_rubric_text():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"dimension": "custom", "score": 4, "explanation": "solid"}'
    )

    result = judge_scenario(
        llm=mock_llm,
        spans=[],
        query="Do the thing",
        reference_answer="",
        rubric_name="",
        rubric_text="Score whether the task was completed cleanly.",
    )

    assert len(result.scores) == 1
    assert result.scores[0].score == 4
    assert result.scores[0].dimension == "custom"
    mock_llm.invoke.assert_called_once()


def test_create_judge_llm_uses_configured_max_tokens(monkeypatch):
    captured = {}

    def fake_create_chat_llm(settings, model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "agentlens.eval.level2_llm_judge.judge.create_chat_llm",
        fake_create_chat_llm,
    )

    settings = MagicMock()
    settings.judge_model = "openrouter:openai/gpt-4.1"
    settings.judge_max_tokens = 333

    _ = create_judge_llm(settings)

    assert captured["model"] == "openrouter:openai/gpt-4.1"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 333
