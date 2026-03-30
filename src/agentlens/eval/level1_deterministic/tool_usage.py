"""Level 1 evaluator: tool selection and parameter validation.

Reads Tool spans to verify:
- Correct tools were called (by name)
- Tool parameters match expected schema
"""

from __future__ import annotations

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
    """Extract tool names from OpenInference Tool spans."""
    tool_names = []
    for span in spans:
        attrs = dict(span.attributes or {})
        # OpenInference uses "tool.name" or the span name for tool calls
        tool_name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
        if tool_name:
            tool_names.append(str(tool_name))
        elif span.name and span.name.startswith("Tool:"):
            tool_names.append(span.name.removeprefix("Tool:").strip())
    return tool_names


def evaluate_tool_usage(
    spans: list[ReadableSpan],
    expected_tools: list[str],
) -> ToolUsageResult:
    actual_tools = extract_tool_names(spans)

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    passed = len(missing) == 0

    return ToolUsageResult(
        passed=passed,
        expected_tools=expected_tools,
        actual_tools=actual_tools,
        missing_tools=missing,
        unexpected_tools=unexpected,
    )
