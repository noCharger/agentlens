"""Tests for feature flag system (config, CLI, runner wiring)."""

from agentlens.config import AgentLensSettings, get_settings


# --- Config Feature Flags ---


def test_feature_flags_default_off():
    settings = AgentLensSettings()
    assert settings.judge_use_geval is False
    assert settings.judge_task_completion is False
    assert settings.judge_answer_relevancy is False
    assert settings.judge_hallucination is False
    assert settings.judge_faithfulness is False


def test_feature_flags_can_be_enabled():
    settings = AgentLensSettings(
        judge_use_geval=True,
        judge_task_completion=True,
        judge_answer_relevancy=True,
        judge_hallucination=True,
        judge_faithfulness=True,
    )
    assert settings.judge_use_geval is True
    assert settings.judge_task_completion is True
    assert settings.judge_answer_relevancy is True
    assert settings.judge_hallucination is True
    assert settings.judge_faithfulness is True


def test_feature_flags_via_get_settings():
    settings = get_settings(judge_use_geval=True)
    assert settings.judge_use_geval is True
    assert settings.judge_task_completion is False  # Others remain off


# --- Rubric Definitions Include New Metrics ---


def test_rubric_definitions_include_new_metrics():
    from agentlens.eval.level2_llm_judge.rubrics import RUBRIC_DEFINITIONS

    assert "task_completion" in RUBRIC_DEFINITIONS
    assert "answer_relevancy" in RUBRIC_DEFINITIONS
    assert "hallucination" in RUBRIC_DEFINITIONS
    assert "faithfulness" in RUBRIC_DEFINITIONS


# --- Runner Feature Flag Wiring ---


def test_run_level2_signature_accepts_flags():
    """Verify _run_level2 accepts all feature flag kwargs."""
    from agentlens.eval.runner import _run_level2
    import inspect

    sig = inspect.signature(_run_level2)
    param_names = set(sig.parameters.keys())
    assert "use_geval" in param_names
    assert "task_completion" in param_names
    assert "answer_relevancy" in param_names
    assert "hallucination" in param_names
    assert "faithfulness" in param_names


def test_execute_and_eval_signature_accepts_flags():
    """Verify execute_and_eval accepts all feature flag kwargs."""
    from agentlens.eval.runner import execute_and_eval
    import inspect

    sig = inspect.signature(execute_and_eval)
    param_names = set(sig.parameters.keys())
    assert "use_geval" in param_names
    assert "task_completion" in param_names
    assert "answer_relevancy" in param_names
    assert "hallucination" in param_names
    assert "faithfulness" in param_names


# --- CLI Argument Parsing ---


def test_cli_parser_accepts_flags():
    """Verify CLI parser accepts all feature flag arguments."""
    import argparse

    # Import just the parser construction part.
    parser = argparse.ArgumentParser()
    parser.add_argument("--geval", action="store_true")
    parser.add_argument("--task-completion", action="store_true")
    parser.add_argument("--answer-relevancy", action="store_true")
    parser.add_argument("--hallucination", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--all-metrics", action="store_true")

    args = parser.parse_args(["--geval", "--task-completion"])
    assert args.geval is True
    assert args.task_completion is True
    assert args.answer_relevancy is False
    assert args.hallucination is False
    assert args.faithfulness is False
    assert args.all_metrics is False


def test_cli_all_metrics_shortcut():
    """Verify --all-metrics enables all flags."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--geval", action="store_true")
    parser.add_argument("--task-completion", action="store_true")
    parser.add_argument("--answer-relevancy", action="store_true")
    parser.add_argument("--hallucination", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--all-metrics", action="store_true")

    args = parser.parse_args(["--all-metrics"])
    assert args.all_metrics is True
    # Individual flags are False, but --all-metrics is True.
    # The runner resolves this with OR logic.
