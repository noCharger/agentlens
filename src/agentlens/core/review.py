"""Human review workflow for annotation tasks.

Provides task lifecycle management:
- Create review tasks from eval results (auto-triage)
- Assign tasks to reviewers
- State transitions: PENDING -> IN_REVIEW -> RESOLVED
- Record review verdicts and feedback
- Batch operations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from agentlens.core.models import (
    AnnotationStatus,
    AnnotationTaskRecord,
    AuditEventRecord,
    Role,
    TraceStatus,
    utc_now,
)


@dataclass
class ReviewVerdict:
    """A reviewer's judgment on an annotation task."""
    task_id: str
    reviewer: str
    reviewer_role: Role
    verdict: str  # agree, disagree, needs_more_info, relabel
    corrected_status: TraceStatus | None = None
    notes: str = ""
    reviewed_at: datetime = field(default_factory=utc_now)


@dataclass
class ReviewStats:
    """Summary statistics for review progress."""
    total: int = 0
    pending: int = 0
    in_review: int = 0
    resolved: int = 0
    agreement_rate: float = 0.0


class ReviewWorkflow:
    """Manages the lifecycle of human review tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, AnnotationTaskRecord] = {}
        self._verdicts: dict[str, list[ReviewVerdict]] = {}
        self._audit_events: list[AuditEventRecord] = []

    def create_task(
        self,
        *,
        trace_id: str | None = None,
        dataset_item_id: str | None = None,
        title: str,
        reason: str,
        priority: int = 2,
        requested_role: Role = Role.QA,
        metadata: dict | None = None,
    ) -> AnnotationTaskRecord:
        task = AnnotationTaskRecord(
            id=f"review_{uuid4().hex[:10]}",
            trace_id=trace_id,
            dataset_item_id=dataset_item_id,
            title=title,
            status=AnnotationStatus.PENDING,
            priority=priority,
            requested_role=requested_role,
            reason=reason,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        self._verdicts[task.id] = []
        return task

    def create_tasks_from_eval_results(
        self,
        results: list,
        *,
        auto_triage: bool = True,
    ) -> list[AnnotationTaskRecord]:
        """Auto-create review tasks from eval results that need human review."""
        from agentlens.core.models import TraceStatus as TS

        tasks = []
        for result in results:
            status = result.status

            if status == TS.PASSED:
                continue

            if status == TS.ERROR:
                priority = 1
                reason = f"Execution error: {result.error or 'unknown'}"
            elif status == TS.RISKY_SUCCESS:
                priority = 1
                reason = f"Risk signals: {', '.join(result.risk_signals[:3])}"
            elif status == TS.PARTIAL_SUCCESS:
                priority = 2
                reason = "Partial success — verify if acceptable"
            else:
                priority = 3
                reason = "Evaluation failed"

            if auto_triage and status == TS.RISKY_SUCCESS:
                requested_role = Role.COMPLIANCE
            else:
                requested_role = Role.QA

            task = self.create_task(
                trace_id=result.scenario.id,
                title=f"Review: {result.scenario.name}",
                reason=reason,
                priority=priority,
                requested_role=requested_role,
                metadata={
                    "scenario_id": result.scenario.id,
                    "benchmark": result.scenario.benchmark,
                    "status": status.value,
                },
            )
            tasks.append(task)

        return tasks

    def start_review(self, task_id: str, reviewer: str) -> AnnotationTaskRecord:
        task = self._get_task(task_id)
        if task.status != AnnotationStatus.PENDING:
            raise ValueError(f"Cannot start review: task {task_id} is {task.status.value}")
        task.status = AnnotationStatus.IN_REVIEW
        task.metadata["reviewer"] = reviewer
        task.metadata["review_started_at"] = utc_now().isoformat()
        self._record_audit("start_review", task_id, reviewer)
        return task

    def submit_verdict(self, verdict: ReviewVerdict) -> AnnotationTaskRecord:
        task = self._get_task(verdict.task_id)
        if task.status != AnnotationStatus.IN_REVIEW:
            raise ValueError(f"Cannot submit verdict: task {verdict.task_id} is {task.status.value}")
        self._verdicts[verdict.task_id].append(verdict)
        task.status = AnnotationStatus.RESOLVED
        task.metadata["verdict"] = verdict.verdict
        task.metadata["resolved_at"] = verdict.reviewed_at.isoformat()
        if verdict.notes:
            task.metadata["notes"] = verdict.notes
        if verdict.corrected_status:
            task.metadata["corrected_status"] = verdict.corrected_status.value
        self._record_audit("submit_verdict", verdict.task_id, verdict.reviewer)
        return task

    def reopen_task(self, task_id: str, reason: str = "") -> AnnotationTaskRecord:
        task = self._get_task(task_id)
        task.status = AnnotationStatus.PENDING
        task.metadata["reopen_reason"] = reason
        self._record_audit("reopen_task", task_id, "system")
        return task

    def batch_start_review(
        self,
        task_ids: list[str],
        reviewer: str,
    ) -> list[AnnotationTaskRecord]:
        return [self.start_review(tid, reviewer) for tid in task_ids]

    def batch_submit_verdicts(
        self,
        verdicts: list[ReviewVerdict],
    ) -> list[AnnotationTaskRecord]:
        return [self.submit_verdict(v) for v in verdicts]

    def get_task(self, task_id: str) -> AnnotationTaskRecord | None:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        *,
        status: AnnotationStatus | None = None,
        role: Role | None = None,
        priority: int | None = None,
        limit: int = 50,
    ) -> list[AnnotationTaskRecord]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if role is not None:
            tasks = [t for t in tasks if t.requested_role == role]
        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]
        tasks.sort(key=lambda t: (t.priority, t.id))
        return tasks[:limit]

    def get_verdicts(self, task_id: str) -> list[ReviewVerdict]:
        return self._verdicts.get(task_id, [])

    def get_stats(self) -> ReviewStats:
        tasks = list(self._tasks.values())
        if not tasks:
            return ReviewStats()

        total = len(tasks)
        pending = sum(1 for t in tasks if t.status == AnnotationStatus.PENDING)
        in_review = sum(1 for t in tasks if t.status == AnnotationStatus.IN_REVIEW)
        resolved = sum(1 for t in tasks if t.status == AnnotationStatus.RESOLVED)

        agree_count = 0
        resolved_with_verdicts = 0
        for tid, verdicts in self._verdicts.items():
            if verdicts:
                resolved_with_verdicts += 1
                if any(v.verdict == "agree" for v in verdicts):
                    agree_count += 1

        agreement_rate = (
            agree_count / resolved_with_verdicts
            if resolved_with_verdicts > 0
            else 0.0
        )

        return ReviewStats(
            total=total,
            pending=pending,
            in_review=in_review,
            resolved=resolved,
            agreement_rate=round(agreement_rate, 3),
        )

    def _get_task(self, task_id: str) -> AnnotationTaskRecord:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        return task

    def _record_audit(self, action: str, task_id: str, actor: str) -> None:
        self._audit_events.append(AuditEventRecord(
            id=f"audit_{uuid4().hex[:10]}",
            action=action,
            actor_role=Role.ENGINEER,
            resource_type="annotation_task",
            resource_id=task_id,
            details={"actor": actor},
        ))
