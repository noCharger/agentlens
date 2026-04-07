"""Tests for G-Eval CoT meta-evaluation framework."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.eval.level2_llm_judge.geval import (
    _parse_steps,
    clear_step_cache,
    generate_evaluation_steps,
    geval_judge_scenario,
    geval_score,
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


# --- Step Parsing ---


def test_parse_steps_json_array():
    text = '["Check accuracy", "Verify completeness", "Rate confidence"]'
    steps = _parse_steps(text)
    assert len(steps) == 3
    assert "Check accuracy" in steps[0]


def test_parse_steps_markdown_fenced():
    text = '```json\n["Step 1", "Step 2"]\n```'
    steps = _parse_steps(text)
    assert len(steps) == 2


def test_parse_steps_numbered_list():
    text = "1. Check if the answer is correct\n2. Verify the reasoning chain\n3. Score accordingly"
    steps = _parse_steps(text)
    assert len(steps) == 3
    assert "Check if the answer is correct" in steps[0]


def test_parse_steps_bullet_list():
    text = "- Check accuracy\n- Verify completeness"
    steps = _parse_steps(text)
    assert len(steps) == 2


def test_parse_steps_embedded_json():
    text = 'Here are the steps:\n["Step A", "Step B"]\nDone.'
    steps = _parse_steps(text)
    assert len(steps) == 2


def test_parse_steps_empty():
    steps = _parse_steps("")
    assert steps == []


# --- Step Generation ---


def test_generate_evaluation_steps_calls_llm():
    clear_step_cache()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["Check answer accuracy", "Verify reference alignment", "Score 1-5"]'
    )

    steps = generate_evaluation_steps(mock_llm, "accuracy", "Rate accuracy 1-5.")
    assert len(steps) == 3
    mock_llm.invoke.assert_called_once()
    clear_step_cache()


def test_generate_evaluation_steps_caches():
    clear_step_cache()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["Step 1", "Step 2"]'
    )

    steps1 = generate_evaluation_steps(mock_llm, "accuracy", "Rate accuracy 1-5.")
    steps2 = generate_evaluation_steps(mock_llm, "accuracy", "Rate accuracy 1-5.")

    assert steps1 == steps2
    assert mock_llm.invoke.call_count == 1  # Only called once due to caching
    clear_step_cache()


def test_generate_evaluation_steps_different_rubric_not_cached():
    clear_step_cache()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='["Step 1", "Step 2"]'
    )

    generate_evaluation_steps(mock_llm, "accuracy", "Rate accuracy 1-5.")
    generate_evaluation_steps(mock_llm, "reasoning", "Rate reasoning 1-5.")

    assert mock_llm.invoke.call_count == 2  # Called twice for different rubrics
    clear_step_cache()


def test_generate_evaluation_steps_different_model_not_cached():
    clear_step_cache()

    model_a = MagicMock()
    model_a.model_name = "judge-a"
    model_a.invoke.return_value = AIMessage(content='["Step A1", "Step A2"]')

    model_b = MagicMock()
    model_b.model_name = "judge-b"
    model_b.invoke.return_value = AIMessage(content='["Step B1", "Step B2"]')

    steps_a = generate_evaluation_steps(model_a, "accuracy", "Rate accuracy 1-5.")
    steps_b = generate_evaluation_steps(model_b, "accuracy", "Rate accuracy 1-5.")

    assert steps_a != steps_b
    assert model_a.invoke.call_count == 1
    assert model_b.invoke.call_count == 1
    clear_step_cache()


def test_generate_evaluation_steps_fallback_on_empty():
    clear_step_cache()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="")

    rubric_text = "Rate accuracy 1-5."
    steps = generate_evaluation_steps(mock_llm, "accuracy", rubric_text)
    assert steps == [rubric_text]  # Falls back to rubric text
    clear_step_cache()


# --- G-Eval Scoring ---


def test_geval_score_returns_judge_score():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content='{"dimension": "accuracy", "score": 4, "explanation": "Good, followed steps"}'
    )

    score = geval_score(
        llm=mock_llm,
        query="What is 2+2?",
        trajectory="Step 0: thought='calculate', action='compute'",
        final_answer="4",
        reference_answer="4",
        dimension="accuracy",
        rubric_text="Rate accuracy 1-5.",
        evaluation_steps=["Check if answer matches", "Verify reasoning"],
    )

    assert score.score == 4
    assert score.dimension == "accuracy"

    # Verify the prompt includes evaluation steps.
    call_args = mock_llm.invoke.call_args[0][0]
    user_msg = call_args[1].content
    assert "1. Check if answer matches" in user_msg
    assert "2. Verify reasoning" in user_msg


# --- Full G-Eval Scenario ---


def test_geval_judge_scenario_two_phase():
    clear_step_cache()
    mock_llm = MagicMock()
    # First call: step generation. Second call: scoring.
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Check correctness", "Compare with reference", "Assign score"]'),
        AIMessage(content='{"dimension": "accuracy", "score": 5, "explanation": "perfect"}'),
    ]

    spans = _make_spans(
        {"name": "agent.run", "attributes": {"agent.output": "hello world"}},
    )

    result = geval_judge_scenario(
        llm=mock_llm,
        spans=spans,
        query="Read the file",
        reference_answer="hello world",
        rubric_name="accuracy",
    )

    assert len(result.scores) == 1
    assert result.scores[0].score == 5
    assert mock_llm.invoke.call_count == 2  # Phase 1 + Phase 2
    clear_step_cache()


def test_geval_judge_scenario_unknown_rubric():
    mock_llm = MagicMock()
    result = geval_judge_scenario(
        llm=mock_llm,
        spans=[],
        query="test",
        reference_answer="test",
        rubric_name="nonexistent_rubric",
    )
    assert len(result.scores) == 0
    mock_llm.invoke.assert_not_called()


def test_geval_judge_scenario_with_custom_rubric_text():
    clear_step_cache()
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content='["Evaluate completeness"]'),
        AIMessage(content='{"dimension": "custom", "score": 3, "explanation": "partial"}'),
    ]

    result = geval_judge_scenario(
        llm=mock_llm,
        spans=[],
        query="Do the thing",
        reference_answer="",
        rubric_name="",
        rubric_text="Score whether the task was completed.",
    )

    assert len(result.scores) == 1
    assert result.scores[0].score == 3
    clear_step_cache()
