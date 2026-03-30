"""Level 1 evaluator: output format validation.

Reads Chain root span output to verify expected content is present.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class OutputFormatResult:
    passed: bool
    output_text: str
    expected_substrings: list[str]
    missing_substrings: list[str]


def extract_output(spans: list[ReadableSpan]) -> str:
    """Extract the final output from the root Chain span or agent.run span."""
    for span in reversed(spans):
        attrs = dict(span.attributes or {})
        # Check custom agent.run span first
        output = attrs.get("agent.output")
        if output:
            return str(output)
        # Check OpenInference output
        output = attrs.get("output.value")
        if output:
            return str(output)
    return ""


def evaluate_output_format(
    spans: list[ReadableSpan],
    expected_substrings: list[str],
) -> OutputFormatResult:
    output = extract_output(spans)
    output_lower = output.lower()

    missing = [s for s in expected_substrings if s.lower() not in output_lower]
    passed = len(missing) == 0

    return OutputFormatResult(
        passed=passed,
        output_text=output,
        expected_substrings=expected_substrings,
        missing_substrings=missing,
    )
