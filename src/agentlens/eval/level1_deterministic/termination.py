"""Level 1 evaluator: termination behavior validation.

Evaluates whether the agent stopped at the right time:
- Did not stop prematurely (before completing the task)
- Did not continue unnecessarily after achieving the goal
- Correctly escalated to human when needed
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class TerminationResult:
    passed: bool
    termination_type: str  # normal, premature, over_execution, escalation
    reasons: list[str] = field(default_factory=list)
    steps_after_answer: int = 0
    has_escalation: bool = False


def _extract_tool_sequence(spans: list[ReadableSpan]) -> list[dict[str, object]]:
    """Extract ordered sequence of tool calls with timestamps."""
    tools = []
    for span in spans:
        attrs = dict(span.attributes or {})
        name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
        if name:
            tools.append({
                "name": str(name),
                "start_time": span.start_time or 0,
                "end_time": span.end_time or 0,
            })
    tools.sort(key=lambda t: t["start_time"])
    return tools


def _find_answer_span_time(spans: list[ReadableSpan]) -> int | None:
    """Find the timestamp when the agent first produced a final answer."""
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("agent.output") or attrs.get("output.value"):
            return span.start_time or 0
    return None


def evaluate_termination(
    spans: list[ReadableSpan],
    *,
    expected_min_steps: int = 1,
    expected_escalation: bool = False,
    max_steps_after_answer: int = 1,
) -> TerminationResult:
    """Evaluate whether the agent terminated correctly."""
    tools = _extract_tool_sequence(spans)
    reasons: list[str] = []

    # Check for escalation signals
    has_escalation = any(
        "escalat" in str(dict(s.attributes or {}).get("agent.output", "")).lower()
        or "human" in str(dict(s.attributes or {}).get("agent.output", "")).lower()
        or "cannot" in str(dict(s.attributes or {}).get("agent.output", "")).lower()
        for s in spans
    )

    # Premature termination: too few tool calls
    if len(tools) < expected_min_steps and not has_escalation:
        reasons.append(
            f"Agent used only {len(tools)} tools, expected at least {expected_min_steps}"
        )
        return TerminationResult(
            passed=False,
            termination_type="premature",
            reasons=reasons,
            has_escalation=has_escalation,
        )

    # Over-execution: continued working after producing answer
    answer_time = _find_answer_span_time(spans)
    steps_after = 0
    if answer_time is not None:
        steps_after = sum(
            1 for t in tools
            if (t["start_time"] or 0) > answer_time
        )

    if steps_after > max_steps_after_answer:
        reasons.append(
            f"Agent executed {steps_after} tool calls after producing answer "
            f"(max allowed: {max_steps_after_answer})"
        )
        return TerminationResult(
            passed=False,
            termination_type="over_execution",
            reasons=reasons,
            steps_after_answer=steps_after,
            has_escalation=has_escalation,
        )

    # Escalation check
    if expected_escalation and not has_escalation:
        reasons.append("Agent should have escalated to human but did not")
        return TerminationResult(
            passed=False,
            termination_type="normal",
            reasons=reasons,
            has_escalation=False,
        )

    if not expected_escalation and has_escalation:
        termination_type = "escalation"
    else:
        termination_type = "normal"

    return TerminationResult(
        passed=True,
        termination_type=termination_type,
        reasons=reasons,
        steps_after_answer=steps_after,
        has_escalation=has_escalation,
    )
