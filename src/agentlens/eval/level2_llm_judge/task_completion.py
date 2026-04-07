"""Task completion metric."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from agentlens.eval.level2_llm_judge.judge import (
    _extract_final_answer,
    _format_trajectory,
)
from agentlens.eval.level2_llm_judge.prompts import (
    TASK_COMPLETION_SYSTEM_PROMPT,
    TASK_COMPLETION_USER_TEMPLATE,
    TASK_EXTRACTION_SYSTEM_PROMPT,
    TASK_EXTRACTION_USER_TEMPLATE,
)
from agentlens.eval.level2_llm_judge.rubrics import JudgeScore

log = logging.getLogger("agentlens.eval.task_completion")


class ExtractedTask(BaseModel):
    task: str
    completed: bool
    evidence: str = ""


class TaskCompletionResult(BaseModel):
    tasks: list[ExtractedTask]
    completion_ratio: float = 0.0

    def model_post_init(self, __context):
        if self.tasks:
            completed = sum(1 for t in self.tasks if t.completed)
            self.completion_ratio = completed / len(self.tasks)


def _normalize_content(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw]
        return "\n".join(parts)
    return str(raw)


def _parse_task_list(text: str) -> list[str]:
    """Parse LLM response into a list of task strings."""
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

    return [text.strip()] if text.strip() else []


def _parse_completion_result(text: str) -> list[ExtractedTask]:
    """Parse LLM completion judgment response."""
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

    tasks_raw = data.get("tasks", [])
    results = []
    for item in tasks_raw:
        if isinstance(item, dict):
            results.append(ExtractedTask(
                task=item.get("task", ""),
                completed=bool(item.get("completed", False)),
                evidence=item.get("evidence", ""),
            ))
    return results


def extract_tasks(llm, query: str) -> list[str]:
    """Decompose a user query into discrete sub-tasks."""
    user_prompt = TASK_EXTRACTION_USER_TEMPLATE.format(query=query)
    messages = [
        SystemMessage(content=TASK_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return _parse_task_list(response.content)


def judge_task_completion(
    llm,
    tasks: list[str],
    trajectory: str,
    final_answer: str,
) -> TaskCompletionResult:
    """Judge whether each sub-task was completed."""
    tasks_text = "\n".join(f"{i + 1}. {task}" for i, task in enumerate(tasks))
    user_prompt = TASK_COMPLETION_USER_TEMPLATE.format(
        tasks=tasks_text,
        trajectory=trajectory,
        final_answer=final_answer,
    )
    messages = [
        SystemMessage(content=TASK_COMPLETION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    extracted = _parse_completion_result(response.content)

    if not extracted:
        extracted = [ExtractedTask(task=t, completed=False) for t in tasks]

    return TaskCompletionResult(tasks=extracted)


def _ratio_to_score(ratio: float) -> int:
    """Map a 0-1 completion ratio to a 1-5 integer score."""
    score = round(ratio * 4 + 1)
    return max(1, min(5, score))


def evaluate_task_completion(
    llm,
    spans: list[ReadableSpan],
    query: str,
) -> JudgeScore:
    """End-to-end task completion evaluation returning a JudgeScore."""
    tasks = extract_tasks(llm, query)
    if not tasks:
        return JudgeScore(
            dimension="task_completion",
            score=1,
            explanation="Could not extract any sub-tasks from the query.",
        )

    trajectory = _format_trajectory(spans)
    final_answer = _extract_final_answer(spans)
    result = judge_task_completion(llm, tasks, trajectory, final_answer)

    completed = [t for t in result.tasks if t.completed]
    not_completed = [t for t in result.tasks if not t.completed]

    parts = []
    if completed:
        parts.append(f"Completed: {', '.join(t.task for t in completed)}")
    if not_completed:
        parts.append(f"Not completed: {', '.join(t.task for t in not_completed)}")
    explanation = f"{len(completed)}/{len(result.tasks)} sub-tasks completed. " + "; ".join(parts)

    return JudgeScore(
        dimension="task_completion",
        score=_ratio_to_score(result.completion_ratio),
        explanation=explanation,
    )
