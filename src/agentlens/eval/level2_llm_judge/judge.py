"""LLM-as-Judge evaluator using the configured chat model."""

from __future__ import annotations

import json
import math
import re

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan

from agentlens.config import AgentLensSettings
from agentlens.llms import create_chat_llm
from agentlens.eval.level2_llm_judge.rubrics import (
    JudgeScore,
    JudgeResult,
    RUBRIC_DEFINITIONS,
)
from agentlens.eval.level2_llm_judge.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


def _format_trajectory(spans: list[ReadableSpan]) -> str:
    """Format spans into a human-readable trajectory string."""
    lines = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if span.name == "agent.step":
            idx = attrs.get("step.index", "?")
            thought = attrs.get("step.thought", "")
            action = attrs.get("step.action", "")
            lines.append(f"Step {idx}: thought='{thought}', action='{action}'")
        elif attrs.get("tool.name"):
            tool = attrs["tool.name"]
            params = attrs.get("tool.params", "")
            output = attrs.get("tool.output", "")
            lines.append(f"Tool: {tool}({params}) -> {output}")
    return "\n".join(lines) if lines else "(no trajectory captured)"


def _extract_final_answer(spans: list[ReadableSpan]) -> str:
    for span in reversed(spans):
        attrs = dict(span.attributes or {})
        answer = attrs.get("agent.output") or attrs.get("output.value")
        if answer:
            return str(answer)
    return "(no answer captured)"


def _parse_judge_response(text: str, dimension: str) -> JudgeScore:
    """Parse JSON response from judge LLM.

    Handles common provider variations:
    - markdown fenced JSON
    - explanatory text surrounding a JSON object
    - list-structured content blocks (OpenAI-compatible responses)
    """
    normalized = _normalize_judge_response_text(text).strip()

    data = _try_load_json(normalized)
    if data is None:
        # Fallback: find first JSON object embedded in free-form text.
        match = re.search(r"\{[\s\S]*\}", normalized)
        if match:
            data = _try_load_json(match.group(0))
    if data is None:
        raise ValueError(f"Judge response is not valid JSON: {normalized[:240]}")

    return JudgeScore(
        dimension=data.get("dimension", dimension),
        score=_normalize_judge_score(data.get("score")),
        explanation=data.get("explanation", ""),
    )


def _normalize_judge_score(raw_score: object) -> int:
    if raw_score is None:
        raise ValueError("Judge response is missing required field 'score'.")

    numeric_source: object = raw_score
    if isinstance(raw_score, str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw_score)
        if match:
            numeric_source = match.group(0)

    try:
        numeric = float(numeric_source)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Judge response score is not numeric: {raw_score!r}"
        ) from exc

    if not math.isfinite(numeric):
        raise ValueError(f"Judge response score is not finite: {raw_score!r}")

    rounded = int(round(numeric))
    if rounded < 1:
        return 1
    if rounded > 5:
        return 5
    return rounded


def _normalize_judge_response_text(raw: object) -> str:
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(raw)

    text = text.strip()
    if text.startswith("```"):
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    return text


def _try_load_json(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def create_judge_llm(settings: AgentLensSettings):
    return create_chat_llm(
        settings,
        settings.judge_model,
        temperature=0.0,
        max_tokens=settings.judge_max_tokens,
    )


def judge_scenario(
    llm,
    spans: list[ReadableSpan],
    query: str,
    reference_answer: str,
    rubric_name: str,
    rubric_text: str = "",
) -> JudgeResult:
    """Run LLM-as-Judge on a single scenario for the specified rubric."""
    resolved_rubric_text = rubric_text or RUBRIC_DEFINITIONS.get(rubric_name, "")
    if not resolved_rubric_text:
        return JudgeResult(scores=[])
    dimension = rubric_name or "custom"

    trajectory = _format_trajectory(spans)
    final_answer = _extract_final_answer(spans)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        query=query,
        trajectory=trajectory,
        final_answer=final_answer,
        reference_answer=reference_answer,
        dimension=dimension,
        rubric_text=resolved_rubric_text,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    score = _parse_judge_response(response.content, dimension)
    return JudgeResult(scores=[score])
