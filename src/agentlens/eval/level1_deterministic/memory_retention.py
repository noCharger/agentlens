"""Memory retention check: verifies the agent carries injected facts through execution.

Metric analogy (from LongMemEval / Memory-R1 literature):
- retention_score  ≈ exact-match / BLEU-1 recall  (fast, deterministic)
- anchors_actively_used checks active application, not just echo in final output
  (inspired by A-Mem's link-generation signal: did the fact drive an action?)
- poison_hallucinated catches hallucinated values the agent was never given
  (abstention dimension from LongMemEval)

The L2 memory_fidelity rubric covers semantic correctness for temporal ordering,
belief updates, and refusal quality that substring matching cannot confirm.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class MemoryRetentionResult:
    passed: bool
    anchors_recalled: int
    anchors_actively_used: int
    anchors_total: int
    poison_hallucinated: int
    lost_anchors: list[str]
    retention_score: float  # anchors_recalled / anchors_total; 1.0 = perfect recall


def _tool_input_texts(spans: list[ReadableSpan]) -> list[str]:
    """Extract tool call input values from TOOL spans."""
    texts: list[str] = []
    for span in spans:
        attrs = dict(span.attributes or {})
        kind = attrs.get("openinference.span.kind") or attrs.get("gen_ai.span.kind") or ""
        if str(kind).upper() == "TOOL":
            val = attrs.get("input.value") or attrs.get("gen_ai.tool.call.arguments") or ""
            if val:
                texts.append(str(val))
    return texts


def evaluate_memory_retention(
    spans: list[ReadableSpan],
    output_text: str,
    memory_anchors: list[str],
    memory_poison: list[str],
) -> MemoryRetentionResult:
    """Check whether the agent retained and applied injected facts.

    Args:
        spans: Normalized OTEL spans from the agent run.
        output_text: Agent's final output string.
        memory_anchors: Substrings the agent MUST surface (from scenario.memory_anchors).
        memory_poison: Substrings that must NOT appear in output (hallucination guard).
    """
    if not memory_anchors and not memory_poison:
        return MemoryRetentionResult(
            passed=True,
            anchors_recalled=0,
            anchors_actively_used=0,
            anchors_total=0,
            poison_hallucinated=0,
            lost_anchors=[],
            retention_score=1.0,
        )

    output_lower = output_text.lower()
    tool_inputs = _tool_input_texts(spans)
    tool_inputs_lower = [t.lower() for t in tool_inputs]

    recalled: list[str] = []
    actively_used: list[str] = []
    lost: list[str] = []

    for anchor in memory_anchors:
        anchor_lower = anchor.lower()
        if anchor_lower in output_lower:
            recalled.append(anchor)
            if any(anchor_lower in ti for ti in tool_inputs_lower):
                actively_used.append(anchor)
        else:
            lost.append(anchor)

    poison_count = sum(
        1 for p in memory_poison if p.lower() in output_lower
    )

    total = len(memory_anchors)
    score = len(recalled) / total if total > 0 else 1.0
    passed = score == 1.0 and poison_count == 0

    return MemoryRetentionResult(
        passed=passed,
        anchors_recalled=len(recalled),
        anchors_actively_used=len(actively_used),
        anchors_total=total,
        poison_hallucinated=poison_count,
        lost_anchors=lost,
        retention_score=score,
    )
