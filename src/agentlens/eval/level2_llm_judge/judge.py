"""LLM-as-Judge evaluator using Gemini Flash-Lite."""

from __future__ import annotations

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan

from agentlens.config import AgentLensSettings
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
    """Parse JSON response from judge LLM."""
    # Try to extract JSON from the response
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    return JudgeScore(
        dimension=data.get("dimension", dimension),
        score=int(data["score"]),
        explanation=data.get("explanation", ""),
    )


def create_judge_llm(settings: AgentLensSettings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.judge_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
    )


def judge_scenario(
    llm: ChatGoogleGenerativeAI,
    spans: list[ReadableSpan],
    query: str,
    reference_answer: str,
    rubric_name: str,
) -> JudgeResult:
    """Run LLM-as-Judge on a single scenario for the specified rubric."""
    rubric_text = RUBRIC_DEFINITIONS.get(rubric_name, "")
    if not rubric_text:
        return JudgeResult(scores=[])

    trajectory = _format_trajectory(spans)
    final_answer = _extract_final_answer(spans)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        query=query,
        trajectory=trajectory,
        final_answer=final_answer,
        reference_answer=reference_answer,
        dimension=rubric_name,
        rubric_text=rubric_text,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    score = _parse_judge_response(response.content, rubric_name)
    return JudgeResult(scores=[score])
