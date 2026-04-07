"""Tests for HTML report generation."""

import tempfile
from pathlib import Path

from agentlens.eval.scenarios import Scenario, ExpectedResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_params import ToolParamsResult, ToolParamViolation
from agentlens.eval.level1_deterministic.termination import TerminationResult
from agentlens.eval.level1_deterministic.safety import SafetyResult, SafetyViolation
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.level3_human.reporter import generate_report


def _make_eval_result(
    passed: bool,
    scenario_id: str = "test-001",
    benchmark: str = "",
) -> EvalResult:
    return EvalResult(
        scenario=Scenario(
            id=scenario_id,
            name=f"Test {scenario_id}",
            category="tool_calling",
            benchmark=benchmark,
            input="test query",
            setup=[],
            expected=ExpectedResult(tools_called=["read_file"], max_steps=5, output_contains=["hello"]),
        ),
        level1=Level1Result(
            tool_usage=ToolUsageResult(
                passed=passed,
                expected_tools=["read_file"],
                actual_tools=["read_file"] if passed else [],
                missing_tools=[] if passed else ["read_file"],
                unexpected_tools=[],
            ),
            output_format=OutputFormatResult(
                passed=passed,
                output_text="hello world" if passed else "wrong",
                expected_substrings=["hello"],
                missing_substrings=[] if passed else ["hello"],
            ),
            trajectory=TrajectoryResult(
                passed=True,
                total_steps=2,
                max_steps=5,
                has_loop=False,
                total_prompt_tokens=100,
                total_completion_tokens=50,
                max_tokens=None,
                reasons=[],
            ),
        ),
    )


def test_generate_report_returns_html():
    results = [_make_eval_result(True)]
    html = generate_report(results)
    assert "<html" in html
    assert "AgentLens Eval Report" in html
    assert "test-001" in html


def test_generate_report_pass_and_fail():
    results = [_make_eval_result(True, "pass-1"), _make_eval_result(False, "fail-1")]
    html = generate_report(results)
    assert "PASS" in html
    assert "FAIL" in html
    assert "50.0%" in html  # pass rate


def test_generate_report_empty():
    html = generate_report([])
    assert "<html" in html
    assert "0%" in html


def test_generate_report_writes_file():
    results = [_make_eval_result(True)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.html"
        generate_report(results, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert "AgentLens" in content


def test_generate_report_creates_parent_directory():
    results = [_make_eval_result(True)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "reports" / "nested" / "report.html"
        generate_report(results, output_path=path)
        assert path.exists()


def test_generate_report_with_l2_scores():
    result = _make_eval_result(True)
    result.level2_scores = {"accuracy": 4.5}
    html = generate_report([result])
    assert "4.5" in html


def test_generate_report_includes_judge_explanations_and_hidden_l1_reasons():
    result = _make_eval_result(True)
    result.level1.tool_params = ToolParamsResult(
        passed=False,
        violations=[
            ToolParamViolation(
                tool_name="read_file",
                param_name="path",
                reason="value_mismatch",
                expected="config.yaml",
                actual="other.txt",
            )
        ],
        checked_count=1,
    )
    result.level1.termination = TerminationResult(
        passed=False,
        termination_type="premature",
        reasons=["Agent used only 0 tools, expected at least 1"],
        has_escalation=False,
    )
    result.level1.safety = SafetyResult(
        passed=False,
        violations=[
            SafetyViolation(
                violation_type="leakage",
                description="Potential api_key leakage detected",
                severity="high",
            )
        ],
        checked_spans=1,
    )
    result.level2_scores = {"accuracy": 4.5}
    result.level2_explanations = {"accuracy": "Grounded and complete"}

    html = generate_report([result])

    assert "Grounded and complete" in html
    assert "Tool params read_file.path: value_mismatch" in html
    assert "Potential api_key leakage detected" in html


def test_generate_report_with_error():
    result = _make_eval_result(False)
    result.error = "Connection timeout"
    html = generate_report([result])
    assert "Connection timeout" in html


def test_generate_report_includes_benchmark_summary():
    results = [
        _make_eval_result(True, "pass-1", benchmark="SWE Bench Pro"),
        _make_eval_result(False, "fail-1", benchmark="Toolathlon"),
    ]

    html = generate_report(results)

    assert "Benchmark Summary" in html
    assert "SWE Bench Pro" in html
    assert "Toolathlon" in html
