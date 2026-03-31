"""Level 1 evaluator: tool selection validation.

Reads TOOL spans from OpenInference to verify the agent called expected tools.
Supports checking both which tools and how many times (with ordered count).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class ToolUsageResult:
    passed: bool
    expected_tools: list[str]
    actual_tools: list[str]
    missing_tools: list[str]
    unexpected_tools: list[str]


def extract_tool_names(spans: list[ReadableSpan]) -> list[str]:
    """Extract ordered tool names from OpenInference TOOL spans."""
    tool_spans = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("openinference.span.kind") == "TOOL":
            name = attrs.get("tool.name")
            if name:
                tool_spans.append((span.start_time or 0, str(name)))
        elif attrs.get("tool.name") or attrs.get("tool_call.function.name"):
            name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
            tool_spans.append((span.start_time or 0, str(name)))
        elif span.name and span.name.startswith("Tool:"):
            tool_spans.append((span.start_time or 0, span.name.removeprefix("Tool:").strip()))

    tool_spans.sort(key=lambda t: t[0])
    return [name for _, name in tool_spans]


def evaluate_tool_usage(
    spans: list[ReadableSpan],
    expected_tools: list[str],
) -> ToolUsageResult:
    """Evaluate tool usage with count-aware comparison.

    expected_tools is treated as a multiset: ["write_file", "write_file"]
    means write_file must be called at least twice.
    """
    actual_tools = extract_tool_names(spans)

    expected_counts = Counter(expected_tools)
    actual_counts = Counter(actual_tools)

    missing = []
    for tool, count in expected_counts.items():
        actual_count = actual_counts.get(tool, 0)
        if actual_count < count:
            missing.extend([tool] * (count - actual_count))

    unexpected = sorted(set(actual_counts) - set(expected_counts))
    passed = len(missing) == 0

    return ToolUsageResult(
        passed=passed,
        expected_tools=expected_tools,
        actual_tools=actual_tools,
        missing_tools=missing,
        unexpected_tools=unexpected,
    )
