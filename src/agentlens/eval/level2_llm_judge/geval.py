"""G-Eval scoring helpers."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan

from agentlens.eval.level2_llm_judge.judge import (
    _extract_final_answer,
    _format_trajectory,
    _parse_judge_response,
)
from agentlens.eval.level2_llm_judge.prompts import (
    GEVAL_SCORING_SYSTEM_PROMPT,
    GEVAL_SCORING_USER_TEMPLATE,
    GEVAL_STEP_GENERATION_SYSTEM_PROMPT,
    GEVAL_STEP_GENERATION_USER_TEMPLATE,
)
from agentlens.eval.level2_llm_judge.rubrics import (
    JudgeResult,
    JudgeScore,
    RUBRIC_DEFINITIONS,
)

log = logging.getLogger("agentlens.eval.geval")

_step_cache: dict[tuple[str, str, str, str], list[str]] = {}


def _cache_identity(llm, dimension: str, rubric_text: str) -> tuple[str, str, str, str]:
    model_name = str(
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or llm.__class__.__name__
    )
    return (
        model_name,
        GEVAL_STEP_GENERATION_SYSTEM_PROMPT,
        dimension,
        rubric_text,
    )


def _parse_steps(text: str) -> list[str]:
    """Parse LLM response into a list of evaluation steps."""
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed if str(s).strip()]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except json.JSONDecodeError:
            pass

    lines = text.strip().splitlines()
    steps = []
    for line in lines:
        cleaned = re.sub(r"^[\s]*[\d]+[.)]\s*", "", line).strip()
        cleaned = re.sub(r"^[\s]*[-*]\s*", "", cleaned).strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def generate_evaluation_steps(
    llm,
    dimension: str,
    rubric_text: str,
) -> list[str]:
    """Phase 1: Generate evaluation steps via chain-of-thought.

    Results are cached per (dimension, rubric_text) for the run.
    """
    cache_key = _cache_identity(llm, dimension, rubric_text)
    if cache_key in _step_cache:
        return _step_cache[cache_key]

    user_prompt = GEVAL_STEP_GENERATION_USER_TEMPLATE.format(
        dimension=dimension,
        rubric_text=rubric_text,
    )

    messages = [
        SystemMessage(content=GEVAL_STEP_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content]
        content = "\n".join(parts)

    steps = _parse_steps(content)
    if not steps:
        log.warning("G-Eval step generation returned no steps for %s; falling back to rubric text", dimension)
        steps = [rubric_text]

    _step_cache[cache_key] = steps
    return steps


def clear_step_cache() -> None:
    """Clear the in-memory step cache (useful between runs or in tests)."""
    _step_cache.clear()


def geval_score(
    llm,
    query: str,
    trajectory: str,
    final_answer: str,
    reference_answer: str,
    dimension: str,
    rubric_text: str,
    evaluation_steps: list[str],
) -> JudgeScore:
    """Phase 2: Score using the generated evaluation steps as guidance."""
    steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(evaluation_steps))

    user_prompt = GEVAL_SCORING_USER_TEMPLATE.format(
        query=query,
        trajectory=trajectory,
        final_answer=final_answer,
        reference_answer=reference_answer,
        dimension=dimension,
        rubric_text=rubric_text,
        evaluation_steps=steps_text,
    )

    messages = [
        SystemMessage(content=GEVAL_SCORING_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return _parse_judge_response(response.content, dimension)


def geval_judge_scenario(
    llm,
    spans: list[ReadableSpan],
    query: str,
    reference_answer: str,
    rubric_name: str,
    rubric_text: str = "",
) -> JudgeResult:
    """Run G-Eval two-phase evaluation on a single scenario."""
    resolved_rubric_text = rubric_text or RUBRIC_DEFINITIONS.get(rubric_name, "")
    if not resolved_rubric_text:
        return JudgeResult(scores=[])

    dimension = rubric_name or "custom"
    trajectory = _format_trajectory(spans)
    final_answer = _extract_final_answer(spans)

    steps = generate_evaluation_steps(llm, dimension, resolved_rubric_text)

    score = geval_score(
        llm=llm,
        query=query,
        trajectory=trajectory,
        final_answer=final_answer,
        reference_answer=reference_answer,
        dimension=dimension,
        rubric_text=resolved_rubric_text,
        evaluation_steps=steps,
    )

    return JudgeResult(scores=[score])
