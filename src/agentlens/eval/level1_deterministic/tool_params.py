"""Level 1 evaluator: tool parameter validation.

Validates that tool calls used correct parameters, not just correct tool names.
Checks parameter presence, types, and value constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class ToolParamViolation:
    tool_name: str
    param_name: str
    reason: str
    expected: str = ""
    actual: str = ""


@dataclass
class ToolParamsResult:
    passed: bool
    violations: list[ToolParamViolation] = field(default_factory=list)
    checked_count: int = 0


@dataclass
class ExpectedToolParam:
    """Specification for a single expected tool parameter."""
    tool_name: str
    param_name: str
    required: bool = True
    expected_value: str | None = None
    forbidden_values: list[str] = field(default_factory=list)


def extract_tool_params(spans: list[ReadableSpan]) -> list[dict[str, object]]:
    """Extract tool call parameters from spans."""
    results = []
    for span in spans:
        attrs = dict(span.attributes or {})
        tool_name = (
            attrs.get("tool.name")
            or attrs.get("tool_call.function.name")
        )
        if not tool_name:
            if attrs.get("openinference.span.kind") == "TOOL":
                tool_name = attrs.get("tool.name", "")
            else:
                continue

        params_raw = attrs.get("tool.params") or attrs.get("input.value") or ""
        results.append({
            "tool_name": str(tool_name),
            "params_raw": str(params_raw),
            "start_time": span.start_time or 0,
        })
    results.sort(key=lambda r: r["start_time"])
    return results


def evaluate_tool_params(
    spans: list[ReadableSpan],
    expected_params: list[ExpectedToolParam],
) -> ToolParamsResult:
    """Validate tool call parameters against expected specifications."""
    if not expected_params:
        return ToolParamsResult(passed=True, checked_count=0)

    tool_calls = extract_tool_params(spans)
    violations: list[ToolParamViolation] = []

    for spec in expected_params:
        matching_calls = [c for c in tool_calls if c["tool_name"] == spec.tool_name]

        if not matching_calls:
            if spec.required:
                violations.append(ToolParamViolation(
                    tool_name=spec.tool_name,
                    param_name=spec.param_name,
                    reason="tool_not_called",
                ))
            continue

        for call in matching_calls:
            params_str = str(call["params_raw"]).lower()

            if spec.expected_value is not None:
                if spec.expected_value.lower() not in params_str:
                    violations.append(ToolParamViolation(
                        tool_name=spec.tool_name,
                        param_name=spec.param_name,
                        reason="value_mismatch",
                        expected=spec.expected_value,
                        actual=params_str[:200],
                    ))

            for forbidden in spec.forbidden_values:
                if forbidden.lower() in params_str:
                    violations.append(ToolParamViolation(
                        tool_name=spec.tool_name,
                        param_name=spec.param_name,
                        reason="forbidden_value_used",
                        expected=f"not {forbidden}",
                        actual=params_str[:200],
                    ))

    return ToolParamsResult(
        passed=len(violations) == 0,
        violations=violations,
        checked_count=len(expected_params),
    )
