"""Hallucination detection metric."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan

from agentlens.eval.level2_llm_judge.judge import (
    _extract_final_answer,
)
from agentlens.eval.level2_llm_judge.prompts import (
    CONTRADICTION_DETECTION_SYSTEM_PROMPT,
    CONTRADICTION_DETECTION_USER_TEMPLATE,
)
from agentlens.eval.level2_llm_judge.rubrics import JudgeScore

log = logging.getLogger("agentlens.eval.hallucination")


def _normalize_content(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw]
        return "\n".join(parts)
    return str(raw)


def extract_agent_context(spans: list[ReadableSpan]) -> list[str]:
    """Extract implicit context from tool output spans (zero-resource fallback).

    When no explicit context is provided, derives context from the tool
    outputs captured in the agent's execution trace.
    """
    context_items = []
    for span in spans:
        attrs = dict(span.attributes or {})
        tool_output = attrs.get("tool.output")
        if tool_output:
            output_str = str(tool_output).strip()
            if output_str and len(output_str) > 5:
                context_items.append(output_str)
    return context_items


def _parse_contradiction_results(text: str) -> list[tuple[str, bool, str]]:
    """Parse contradiction detection response."""
    text = _normalize_content(text).strip()
    if text.startswith("```"):
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        return []

    results = []
    for item in data.get("results", []):
        if isinstance(item, dict):
            results.append((
                item.get("context", ""),
                bool(item.get("contradicted", False)),
                item.get("explanation", ""),
            ))
    return results


def detect_contradictions(
    llm,
    output_text: str,
    context_items: list[str],
) -> list[tuple[str, bool, str]]:
    """Detect contradictions between output and context items.

    Returns a list of (context_item, is_contradicted, explanation).
    """
    if not context_items or not output_text.strip():
        return []

    context_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(context_items))
    user_prompt = CONTRADICTION_DETECTION_USER_TEMPLATE.format(
        output_text=output_text,
        context_items=context_text,
    )
    messages = [
        SystemMessage(content=CONTRADICTION_DETECTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    results = _parse_contradiction_results(response.content)

    if not results:
        return [(item, False, "parsing_failed") for item in context_items]
    return results


def _ratio_to_score(non_contradiction_ratio: float) -> int:
    """Map a 0-1 non-contradiction ratio to a 1-5 score.

    Higher ratio (fewer contradictions) = higher score.
    """
    score = round(non_contradiction_ratio * 4 + 1)
    return max(1, min(5, score))


def evaluate_hallucination(
    llm,
    spans: list[ReadableSpan],
    query: str,
    context: list[str] | None = None,
) -> JudgeScore:
    """End-to-end hallucination detection returning a JudgeScore.

    If explicit context is provided, uses it. Otherwise falls back to
    extracting context from tool outputs in the trace.
    """
    final_answer = _extract_final_answer(spans)

    if "no answer" in final_answer.lower():
        return JudgeScore(
            dimension="hallucination",
            score=1,
            explanation="No answer was captured from the agent.",
        )

    context_items = context if context else extract_agent_context(spans)

    if not context_items:
        return JudgeScore(
            dimension="hallucination",
            score=3,
            explanation="No context available for hallucination detection (neither explicit nor from tool outputs). Returning neutral score.",
        )

    results = detect_contradictions(llm, final_answer, context_items)
    contradicted = sum(1 for _, is_contradicted, _ in results if is_contradicted)
    total = len(results)
    non_contradiction_ratio = 1.0 - (contradicted / total) if total > 0 else 1.0

    parts = [f"{contradicted}/{total} context items contradicted."]
    contradicted_items = [(ctx, expl) for ctx, is_c, expl in results if is_c]
    if contradicted_items:
        preview = contradicted_items[:3]
        details = "; ".join(f"'{ctx[:50]}': {expl}" for ctx, expl in preview)
        parts.append(f"Contradictions: {details}")

    return JudgeScore(
        dimension="hallucination",
        score=_ratio_to_score(non_contradiction_ratio),
        explanation=" ".join(parts),
    )
