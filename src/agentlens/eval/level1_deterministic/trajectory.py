"""Level 1 evaluator: trajectory validation.

Counts ReAct loop iterations (AGENT spans from OpenInference),
detects loops via repeated tool calls, and checks token budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class TrajectoryResult:
    passed: bool
    total_steps: int
    max_steps: int
    has_loop: bool
    total_prompt_tokens: int
    total_completion_tokens: int
    max_tokens: int | None
    reasons: list[str]


def count_steps(spans: list[ReadableSpan]) -> int:
    """Count ReAct loop iterations.

    OpenInference emits one AGENT span per LangGraph node invocation.
    Each `agent` node call = one ReAct step (think → act → observe).
    Falls back to counting our custom `agent.step` spans if present.
    """
    agent_steps = sum(
        1 for s in spans
        if dict(s.attributes or {}).get("openinference.span.kind") == "AGENT"
    )
    if agent_steps > 0:
        return agent_steps

    return sum(1 for s in spans if s.name == "agent.step")


def detect_loop(spans: list[ReadableSpan], max_repeats: int = 3) -> bool:
    """Detect repeated identical tool calls (indicates stuck agent).

    Checks both OpenInference TOOL spans and custom agent.step spans.
    """
    actions = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("openinference.span.kind") == "TOOL":
            tool = attrs.get("tool.name", "")
            params = attrs.get("input.value", "")
            actions.append(f"{tool}:{params}")
        elif span.name == "agent.step":
            action = attrs.get("step.action", "")
            if action:
                actions.append(str(action))

    if len(actions) < max_repeats:
        return False

    for i in range(len(actions) - max_repeats + 1):
        window = actions[i: i + max_repeats]
        if len(set(window)) == 1 and window[0]:
            return True
    return False


def sum_tokens(spans: list[ReadableSpan]) -> tuple[int, int]:
    """Sum prompt and completion tokens from LLM spans."""
    prompt_total = 0
    completion_total = 0
    for span in spans:
        attrs = dict(span.attributes or {})
        pt = attrs.get("llm.token_count.prompt") or attrs.get("llm.usage.prompt_tokens") or 0
        ct = (
            attrs.get("llm.token_count.completion")
            or attrs.get("llm.usage.completion_tokens")
            or 0
        )
        prompt_total += int(pt)
        completion_total += int(ct)
    return prompt_total, completion_total


def evaluate_trajectory(
    spans: list[ReadableSpan],
    max_steps: int = 10,
    max_tokens: int | None = None,
) -> TrajectoryResult:
    steps = count_steps(spans)
    has_loop = detect_loop(spans)
    prompt_tokens, completion_tokens = sum_tokens(spans)
    total_tokens = prompt_tokens + completion_tokens

    reasons = []
    passed = True

    if steps > max_steps:
        passed = False
        reasons.append(f"Step count {steps} exceeds max {max_steps}")

    if has_loop:
        passed = False
        reasons.append("Detected repeated identical actions (loop)")

    if max_tokens is not None and total_tokens > max_tokens:
        passed = False
        reasons.append(f"Token usage {total_tokens} exceeds max {max_tokens}")

    return TrajectoryResult(
        passed=passed,
        total_steps=steps,
        max_steps=max_steps,
        has_loop=has_loop,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        max_tokens=max_tokens,
        reasons=reasons,
    )
