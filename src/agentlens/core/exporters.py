"""Bridge the OSS eval runner into platform-friendly closed-loop records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from agentlens.eval.runner import EvalResult
from agentlens.core.models import (
    AnnotationStatus,
    AnnotationTaskRecord,
    DatasetItemRecord,
    DatasetSource,
    DatasetVersionRecord,
    EvalCaseRecord,
    EvalRunRecord,
    EvalRunSummary,
    ExperimentRecord,
    Role,
    TraceRecord,
    TraceStatus,
    utc_now,
)

IdFactory = Callable[[str], str]


def _default_id_factory(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _make_id(prefix: str, id_factory: IdFactory | None) -> str:
    factory = id_factory or _default_id_factory
    return factory(prefix)


def _timestamp(created_at: datetime | None) -> datetime:
    return created_at or utc_now()


def _trace_status(result: EvalResult) -> TraceStatus:
    if result.error:
        return TraceStatus.ERROR
    if result.passed:
        return TraceStatus.PASSED
    return TraceStatus.FAILED


def build_trace_records(
    results: list[EvalResult],
    *,
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> list[TraceRecord]:
    timestamp = _timestamp(created_at)
    traces: list[TraceRecord] = []

    for result in results:
        scenario = result.scenario
        traces.append(
            TraceRecord(
                id=_make_id("trace", id_factory),
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                benchmark=scenario.benchmark,
                category=scenario.category,
                evaluation_mode=scenario.evaluation_mode,
                status=_trace_status(result),
                created_at=timestamp,
                input_query=scenario.input_query,
                output_text=result.level1.output_format.output_text,
                expected_tools=list(result.level1.tool_usage.expected_tools),
                actual_tools=list(result.level1.tool_usage.actual_tools),
                missing_tools=list(result.level1.tool_usage.missing_tools),
                expected_output_contains=list(result.level1.output_format.expected_substrings),
                missing_output_contains=list(result.level1.output_format.missing_substrings),
                total_steps=result.level1.trajectory.total_steps,
                prompt_tokens=result.level1.trajectory.total_prompt_tokens,
                completion_tokens=result.level1.trajectory.total_completion_tokens,
                judge_scores=dict(result.level2_scores),
                judge_overall_score=result.judge_overall_score,
                error=result.error,
                metadata=dict(scenario.metadata),
            )
        )

    return traces


def build_dataset_version(
    results: list[EvalResult],
    *,
    name: str,
    version: str = "v1",
    source: DatasetSource = DatasetSource.TRACE_PROMOTION,
    dataset_id: str | None = None,
    trace_records: list[TraceRecord] | None = None,
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> DatasetVersionRecord:
    timestamp = _timestamp(created_at)
    traces = trace_records or build_trace_records(
        results,
        created_at=timestamp,
        id_factory=id_factory,
    )
    dataset_version_id = _make_id("dataset_version", id_factory)
    resolved_dataset_id = dataset_id or _make_id("dataset", id_factory)

    trace_by_scenario_id = {trace.scenario_id: trace for trace in traces}
    items: list[DatasetItemRecord] = []
    for result in results:
        scenario = result.scenario
        trace = trace_by_scenario_id.get(scenario.id)
        items.append(
            DatasetItemRecord(
                id=_make_id("dataset_item", id_factory),
                dataset_version_id=dataset_version_id,
                source_trace_id=trace.id if trace else None,
                scenario_id=scenario.id,
                name=scenario.name,
                category=scenario.category,
                benchmark=scenario.benchmark,
                evaluation_mode=scenario.evaluation_mode,
                input_query=scenario.input_query,
                setup_commands=list(scenario.setup_commands),
                max_steps=scenario.expected.max_steps,
                max_tokens=scenario.expected.max_tokens,
                reference_answer=scenario.reference_answer,
                expected_tools=list(scenario.expected.tools_called),
                expected_output_contains=list(scenario.expected.output_contains),
                judge_rubric=scenario.judge_rubric,
                judge_rubric_text=scenario.judge_rubric_text,
                judge_threshold=scenario.judge_threshold,
                metadata=dict(scenario.metadata),
            )
        )

    return DatasetVersionRecord(
        id=dataset_version_id,
        dataset_id=resolved_dataset_id,
        name=name,
        version=version,
        source=source,
        created_at=timestamp,
        items=items,
    )


def build_eval_run_record(
    results: list[EvalResult],
    *,
    name: str,
    source: str = "cli",
    dataset_version_id: str | None = None,
    trace_records: list[TraceRecord] | None = None,
    agent_model: str = "",
    judge_model: str = "",
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> EvalRunRecord:
    timestamp = _timestamp(created_at)
    traces = trace_records or build_trace_records(
        results,
        created_at=timestamp,
        id_factory=id_factory,
    )
    trace_by_scenario_id = {trace.scenario_id: trace for trace in traces}

    cases: list[EvalCaseRecord] = []
    for result in results:
        scenario = result.scenario
        trace = trace_by_scenario_id.get(scenario.id)
        cases.append(
            EvalCaseRecord(
                id=_make_id("eval_case", id_factory),
                trace_id=trace.id if trace else None,
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                benchmark=scenario.benchmark,
                category=scenario.category,
                evaluation_mode=scenario.evaluation_mode,
                passed=result.passed,
                level1_passed=result.level1.passed,
                judge_overall_score=result.judge_overall_score,
                error=result.error,
                metadata=dict(scenario.metadata),
            )
        )

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    failed = total - passed
    benchmarks = sorted({result.scenario.benchmark for result in results if result.scenario.benchmark})

    summary = EvalRunSummary(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=round((passed / total) * 100, 1) if total else 0.0,
        benchmarks=benchmarks,
    )

    return EvalRunRecord(
        id=_make_id("eval_run", id_factory),
        name=name,
        source=source,
        created_at=timestamp,
        dataset_version_id=dataset_version_id,
        agent_model=agent_model,
        judge_model=judge_model,
        cases=cases,
        summary=summary,
    )


def build_experiment_record(
    *,
    name: str,
    baseline_run: EvalRunRecord,
    candidate_run: EvalRunRecord,
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> ExperimentRecord:
    timestamp = _timestamp(created_at)
    baseline_pass_rate = baseline_run.summary.pass_rate
    candidate_pass_rate = candidate_run.summary.pass_rate
    return ExperimentRecord(
        id=_make_id("experiment", id_factory),
        name=name,
        created_at=timestamp,
        baseline_run_id=baseline_run.id,
        candidate_run_id=candidate_run.id,
        baseline_pass_rate=baseline_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        delta_pass_rate=round(candidate_pass_rate - baseline_pass_rate, 1),
    )


def build_annotation_tasks(
    results: list[EvalResult],
    *,
    trace_records: list[TraceRecord] | None = None,
    dataset_version: DatasetVersionRecord | None = None,
    only_failed: bool = True,
    low_score_threshold: float | None = None,
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> list[AnnotationTaskRecord]:
    del created_at
    traces = trace_records or build_trace_records(results, id_factory=id_factory)
    trace_by_scenario_id = {trace.scenario_id: trace for trace in traces}
    dataset_item_by_scenario_id = {
        item.scenario_id: item for item in (dataset_version.items if dataset_version else [])
    }

    tasks: list[AnnotationTaskRecord] = []
    for result in results:
        score = result.judge_overall_score
        should_enqueue = not only_failed or not result.passed
        if low_score_threshold is not None and score is not None and score < low_score_threshold:
            should_enqueue = True
        if not should_enqueue:
            continue

        trace = trace_by_scenario_id.get(result.scenario.id)
        dataset_item = dataset_item_by_scenario_id.get(result.scenario.id)
        reason = (
            result.error
            or f"Scenario failed in {result.scenario.evaluation_mode} evaluation"
            if not result.passed
            else f"Judge score {score:.1f} below threshold {low_score_threshold:.1f}"
        )
        tasks.append(
            AnnotationTaskRecord(
                id=_make_id("annotation", id_factory),
                trace_id=trace.id if trace else None,
                dataset_item_id=dataset_item.id if dataset_item else None,
                title=f"Review {result.scenario.id}",
                status=AnnotationStatus.PENDING,
                priority=1 if result.error else 2,
                requested_role=Role.QA,
                reason=reason,
                metadata={
                    "scenario_name": result.scenario.name,
                    "benchmark": result.scenario.benchmark,
                },
            )
        )

    return tasks


@dataclass(frozen=True, slots=True)
class ClosedLoopSnapshot:
    traces: list[TraceRecord]
    dataset_version: DatasetVersionRecord
    eval_run: EvalRunRecord
    annotation_tasks: list[AnnotationTaskRecord]


def build_closed_loop_snapshot(
    results: list[EvalResult],
    *,
    dataset_name: str,
    run_name: str,
    dataset_version: str = "v1",
    source: str = "cli",
    agent_model: str = "",
    judge_model: str = "",
    dataset_record: DatasetVersionRecord | None = None,
    created_at: datetime | None = None,
    id_factory: IdFactory | None = None,
) -> ClosedLoopSnapshot:
    timestamp = _timestamp(created_at)
    traces = build_trace_records(results, created_at=timestamp, id_factory=id_factory)
    if dataset_record is None:
        dataset = build_dataset_version(
            results,
            name=dataset_name,
            version=dataset_version,
            trace_records=traces,
            created_at=timestamp,
            id_factory=id_factory,
        )
    else:
        dataset = dataset_record
    eval_run = build_eval_run_record(
        results,
        name=run_name,
        source=source,
        dataset_version_id=dataset.id,
        trace_records=traces,
        agent_model=agent_model,
        judge_model=judge_model,
        created_at=timestamp,
        id_factory=id_factory,
    )
    annotation_tasks = build_annotation_tasks(
        results,
        trace_records=traces,
        dataset_version=dataset,
        id_factory=id_factory,
    )
    return ClosedLoopSnapshot(
        traces=traces,
        dataset_version=dataset,
        eval_run=eval_run,
        annotation_tasks=annotation_tasks,
    )


def snapshot_to_dict(snapshot: ClosedLoopSnapshot) -> dict[str, object]:
    return {
        "traces": [trace.model_dump(mode="json") for trace in snapshot.traces],
        "dataset_version": snapshot.dataset_version.model_dump(mode="json"),
        "eval_run": snapshot.eval_run.model_dump(mode="json"),
        "annotation_tasks": [task.model_dump(mode="json") for task in snapshot.annotation_tasks],
    }


def snapshot_from_dict(payload: dict[str, object]) -> ClosedLoopSnapshot:
    return ClosedLoopSnapshot(
        traces=[TraceRecord.model_validate(item) for item in payload["traces"]],
        dataset_version=DatasetVersionRecord.model_validate(payload["dataset_version"]),
        eval_run=EvalRunRecord.model_validate(payload["eval_run"]),
        annotation_tasks=[
            AnnotationTaskRecord.model_validate(item)
            for item in payload["annotation_tasks"]
        ],
    )


def write_closed_loop_snapshot(snapshot: ClosedLoopSnapshot, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(snapshot_to_dict(snapshot), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
