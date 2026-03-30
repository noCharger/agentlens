"""Evaluation rubrics for LLM-as-Judge scoring."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    dimension: str
    score: int = Field(ge=1, le=5)
    explanation: str


class JudgeResult(BaseModel):
    scores: list[JudgeScore]
    overall_score: float = 0.0

    def model_post_init(self, __context):
        if self.scores:
            self.overall_score = sum(s.score for s in self.scores) / len(self.scores)


RUBRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": (
        "Rate the accuracy of the agent's final answer compared to the reference answer. "
        "1=completely wrong, 2=partially correct with major errors, "
        "3=mostly correct with minor errors, 4=correct with minor omissions, 5=fully correct."
    ),
    "reasoning_quality": (
        "Rate the quality of the agent's reasoning chain. "
        "1=no reasoning or incoherent, 2=poor reasoning with logical errors, "
        "3=adequate reasoning, 4=clear logical reasoning, 5=excellent step-by-step reasoning."
    ),
    "tool_appropriateness": (
        "Rate whether the agent used tools appropriately. "
        "1=completely wrong tools, 2=mostly wrong tool selection, "
        "3=acceptable but suboptimal, 4=good tool selection, 5=optimal tool usage."
    ),
    "recovery_behavior": (
        "Rate how well the agent recovered from errors or failures. "
        "1=no recovery attempt, 2=poor recovery (stuck or crashed), "
        "3=recovered but inefficiently, 4=good recovery with minor issues, "
        "5=excellent graceful recovery."
    ),
}
