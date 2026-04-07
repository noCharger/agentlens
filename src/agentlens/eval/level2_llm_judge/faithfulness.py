"""Faithfulness metric."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan

from agentlens.eval.level2_llm_judge.answer_relevancy import extract_statements
from agentlens.eval.level2_llm_judge.hallucination import extract_agent_context
from agentlens.eval.level2_llm_judge.judge import _extract_final_answer
from agentlens.eval.level2_llm_judge.prompts import (
    SUPPORT_DETECTION_SYSTEM_PROMPT,
    SUPPORT_DETECTION_USER_TEMPLATE,
)
from agentlens.eval.level2_llm_judge.rubrics import JudgeScore

log = logging.getLogger("agentlens.eval.faithfulness")


def _normalize_content(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw]
        return "\n".join(parts)
    return str(raw)


def _parse_support_results(text: str) -> list[tuple[str, bool, str]]:
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
                item.get("statement", ""),
                bool(item.get("supported", False)),
                item.get("explanation", ""),
            ))
    return results


def detect_support(
    llm,
    statements: list[str],
    context_items: list[str],
) -> list[tuple[str, bool, str]]:
    """Detect whether each statement is supported by the available context."""
    if not statements or not context_items:
        return []

    statements_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(statements))
    context_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(context_items))
    user_prompt = SUPPORT_DETECTION_USER_TEMPLATE.format(
        statements=statements_text,
        context_items=context_text,
    )
    messages = [
        SystemMessage(content=SUPPORT_DETECTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    results = _parse_support_results(response.content)

    if not results:
        return [(statement, False, "parsing_failed") for statement in statements]
    return results


def _ratio_to_score(supported_ratio: float) -> int:
    score = round(supported_ratio * 4 + 1)
    return max(1, min(5, score))


def evaluate_faithfulness(
    llm,
    spans: list[ReadableSpan],
    query: str,
    context: list[str] | None = None,
) -> JudgeScore:
    """End-to-end faithfulness evaluation returning a JudgeScore."""
    del query

    final_answer = _extract_final_answer(spans)
    if "no answer" in final_answer.lower():
        return JudgeScore(
            dimension="faithfulness",
            score=1,
            explanation="No answer was captured from the agent.",
        )

    statements = extract_statements(llm, final_answer)
    if not statements:
        return JudgeScore(
            dimension="faithfulness",
            score=1,
            explanation="Could not extract any statements from the answer.",
        )

    context_items = context if context else extract_agent_context(spans)
    if not context_items:
        return JudgeScore(
            dimension="faithfulness",
            score=3,
            explanation="No context available for faithfulness evaluation (neither explicit nor from tool outputs). Returning neutral score.",
        )

    results = detect_support(llm, statements, context_items)
    supported = sum(1 for _, is_supported, _ in results if is_supported)
    total = len(results)
    supported_ratio = supported / total if total > 0 else 0.0

    parts = [f"{supported}/{total} statements supported by context."]
    unsupported_items = [(statement, explanation) for statement, ok, explanation in results if not ok]
    if unsupported_items:
        preview = unsupported_items[:3]
        details = "; ".join(f"'{statement[:50]}': {explanation}" for statement, explanation in preview)
        parts.append(f"Unsupported: {details}")

    return JudgeScore(
        dimension="faithfulness",
        score=_ratio_to_score(supported_ratio),
        explanation=" ".join(parts),
    )
