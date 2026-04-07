"""Answer relevancy metric."""

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
    STATEMENT_EXTRACTION_SYSTEM_PROMPT,
    STATEMENT_EXTRACTION_USER_TEMPLATE,
    STATEMENT_RELEVANCE_SYSTEM_PROMPT,
    STATEMENT_RELEVANCE_USER_TEMPLATE,
)
from agentlens.eval.level2_llm_judge.rubrics import JudgeScore

log = logging.getLogger("agentlens.eval.answer_relevancy")


def _normalize_content(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw]
        return "\n".join(parts)
    return str(raw)


def _parse_string_list(text: str) -> list[str]:
    """Parse LLM response into a list of strings."""
    text = _normalize_content(text).strip()
    if text.startswith("```"):
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    try:
        parsed = json.loads(text)
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

    return []


def _parse_relevance_results(text: str) -> list[tuple[str, bool]]:
    """Parse relevance judgment response."""
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
                bool(item.get("relevant", False)),
            ))
    return results


def extract_statements(llm, answer_text: str) -> list[str]:
    """Decompose answer text into atomic statements."""
    if not answer_text.strip():
        return []

    user_prompt = STATEMENT_EXTRACTION_USER_TEMPLATE.format(answer_text=answer_text)
    messages = [
        SystemMessage(content=STATEMENT_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return _parse_string_list(response.content)


def judge_statement_relevance(
    llm,
    statements: list[str],
    query: str,
) -> list[tuple[str, bool]]:
    """Judge relevance of each statement to the query."""
    if not statements:
        return []

    statements_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    user_prompt = STATEMENT_RELEVANCE_USER_TEMPLATE.format(
        query=query,
        statements=statements_text,
    )
    messages = [
        SystemMessage(content=STATEMENT_RELEVANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    results = _parse_relevance_results(response.content)

    if not results:
        return [(s, False) for s in statements]
    return results


def _ratio_to_score(ratio: float) -> int:
    """Map a 0-1 relevancy ratio to a 1-5 integer score."""
    score = round(ratio * 4 + 1)
    return max(1, min(5, score))


def evaluate_answer_relevancy(
    llm,
    spans: list[ReadableSpan],
    query: str,
) -> JudgeScore:
    """End-to-end answer relevancy evaluation returning a JudgeScore."""
    final_answer = _extract_final_answer(spans)

    if "no answer" in final_answer.lower():
        return JudgeScore(
            dimension="answer_relevancy",
            score=1,
            explanation="No answer was captured from the agent.",
        )

    statements = extract_statements(llm, final_answer)
    if not statements:
        return JudgeScore(
            dimension="answer_relevancy",
            score=1,
            explanation="Could not extract any statements from the answer.",
        )

    results = judge_statement_relevance(llm, statements, query)
    relevant_count = sum(1 for _, is_relevant in results if is_relevant)
    total = len(results)
    ratio = relevant_count / total if total > 0 else 0.0

    irrelevant_stmts = [s for s, r in results if not r]

    parts = [f"{relevant_count}/{total} statements relevant."]
    if irrelevant_stmts:
        preview = irrelevant_stmts[:3]
        parts.append(f"Irrelevant: {'; '.join(preview)}")

    return JudgeScore(
        dimension="answer_relevancy",
        score=_ratio_to_score(ratio),
        explanation=" ".join(parts),
    )
