import json

from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.core.exporters import (
    build_annotation_tasks,
    build_closed_loop_snapshot,
    build_dataset_version,
    build_eval_run_record,
    build_experiment_record,
    build_trace_records,
    snapshot_to_dict,
    write_closed_loop_snapshot,
)
from agentlens.core.models import TraceStatus


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="tc-001",
        name="Tool calling case",
        category="tool_calling",
        benchmark="gdpval-aa",
        evaluation_mode="deterministic",
        input="Inspect the file and summarize the result.",
        setup=[],
        expected=ExpectedResult(
            tools_called=["read_file"],
            max_steps=5,
            output_contains=["summary"],
        ),
        judge_rubric="",
        judge_rubric_text="",
        reference_answer="Expected summary",
        metadata={"source": "test"},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def _make_result(
    *,
    scenario: Scenario | None = None,
    passed: bool = True,
    error: str | None = None,
    judge_score: float | None = None,
) -> EvalResult:
    scenario = scenario or _make_scenario()
    level1 = Level1Result(
        tool_usage=ToolUsageResult(
            passed=passed,
            expected_tools=["read_file"],
            actual_tools=["read_file"] if passed else [],
            missing_tools=[] if passed else ["read_file"],
            unexpected_tools=[],
        ),
        output_format=OutputFormatResult(
            passed=passed,
            output_text="summary ready" if passed else "bad output",
            expected_substrings=["summary"],
            missing_substrings=[] if passed else ["summary"],
        ),
        trajectory=TrajectoryResult(
            passed=passed,
            total_steps=2,
            max_steps=5,
            has_loop=False,
            total_prompt_tokens=10,
            total_completion_tokens=4,
            max_tokens=None,
            reasons=[] if passed else ["failed"],
        ),
    )
    scores = {"quality": judge_score} if judge_score is not None else {}
    return EvalResult(
        scenario=scenario,
        level1=level1,
        level2_scores=scores,
        error=error,
    )


def test_build_trace_records_captures_eval_details():
    result = _make_result()
    traces = build_trace_records([result])

    assert len(traces) == 1
    trace = traces[0]
    assert trace.status == TraceStatus.PASSED
    assert trace.scenario_id == "tc-001"
    assert trace.actual_tools == ["read_file"]
    assert trace.prompt_tokens == 10
    assert trace.completion_tokens == 4


def test_build_dataset_version_links_items_to_trace_records():
    result = _make_result()
    traces = build_trace_records([result])
    dataset = build_dataset_version([result], name="Regression Set", trace_records=traces)

    assert dataset.name == "Regression Set"
    assert dataset.item_count == 1
    assert dataset.items[0].source_trace_id == traces[0].id
    assert dataset.items[0].reference_answer == "Expected summary"


def test_build_eval_run_record_summarizes_pass_rate():
    passed = _make_result()
    failed = _make_result(
        scenario=_make_scenario(id="tc-002", name="Broken case"),
        passed=False,
        error="agent crashed",
    )

    run = build_eval_run_record(
        [passed, failed],
        name="nightly",
        agent_model="gemini:gemini-2.5-flash",
    )

    assert run.summary.total == 2
    assert run.summary.passed == 1
    assert run.summary.failed == 1
    assert run.summary.pass_rate == 50.0
    assert run.agent_model == "gemini:gemini-2.5-flash"


def test_build_eval_run_record_preserves_semantic_statuses():
    passed = _make_result()
    risky = _make_result()
    risky.risk_signals = ["unexpected_privileged_tool:shell"]
    partial = _make_result(passed=False)
    partial.level1.tool_usage.passed = False
    partial.level1.output_format.passed = True
    partial.level1.trajectory.passed = True

    run = build_eval_run_record([passed, risky, partial], name="nightly")

    assert run.summary.passed == 1
    assert run.summary.risky_success == 1
    assert run.summary.partial_success == 1
    assert run.summary.failed == 0
    assert run.cases[1].status == TraceStatus.RISKY_SUCCESS
    assert run.cases[1].risk_signals == ["unexpected_privileged_tool:shell"]
    assert run.cases[2].status == TraceStatus.PARTIAL_SUCCESS


def test_build_annotation_tasks_defaults_to_failed_cases_only():
    passed = _make_result()
    failed = _make_result(
        scenario=_make_scenario(id="tc-002", name="Broken case"),
        passed=False,
        error="agent crashed",
    )

    tasks = build_annotation_tasks([passed, failed])

    assert len(tasks) == 1
    assert tasks[0].title == "Review tc-002"
    assert tasks[0].reason == "agent crashed"


def test_build_experiment_record_computes_delta():
    baseline = build_eval_run_record([_make_result(passed=True)], name="baseline")
    candidate = build_eval_run_record(
        [
            _make_result(passed=True),
            _make_result(
                scenario=_make_scenario(id="tc-003", name="Failing case"),
                passed=False,
                error="bad run",
            ),
        ],
        name="candidate",
    )

    experiment = build_experiment_record(
        name="Prompt A vs Prompt B",
        baseline_run=baseline,
        candidate_run=candidate,
    )

    assert experiment.baseline_pass_rate == 100.0
    assert experiment.candidate_pass_rate == 50.0
    assert experiment.delta_pass_rate == -50.0


def test_build_closed_loop_snapshot_links_dataset_and_run():
    results = [
        _make_result(),
        _make_result(
            scenario=_make_scenario(id="tc-010", name="Judge case", evaluation_mode="llm_judge"),
            passed=False,
            judge_score=2.0,
        ),
    ]

    snapshot = build_closed_loop_snapshot(
        results,
        dataset_name="triage",
        run_name="nightly-regression",
    )

    assert len(snapshot.traces) == 2
    assert snapshot.dataset_version.item_count == 2
    assert snapshot.eval_run.dataset_version_id == snapshot.dataset_version.id
    assert len(snapshot.annotation_tasks) == 1
    assert snapshot.annotation_tasks[0].dataset_item_id is not None


def test_snapshot_serialization_writes_json(tmp_path):
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly-regression",
    )

    payload = snapshot_to_dict(snapshot)
    assert payload["dataset_version"]["name"] == "triage"

    output_path = tmp_path / "snapshot.json"
    write_closed_loop_snapshot(snapshot, output_path)

    written = json.loads(output_path.read_text())
    assert written["eval_run"]["name"] == "nightly-regression"
    assert written["traces"][0]["scenario_id"] == "tc-001"
