"""Tests for human review workflow."""

import pytest

from agentlens.core.models import AnnotationStatus, Role, TraceStatus
from agentlens.core.review import ReviewVerdict, ReviewWorkflow


class TestReviewWorkflow:
    def setup_method(self):
        self.workflow = ReviewWorkflow()

    def test_create_task(self):
        task = self.workflow.create_task(
            title="Review scenario",
            reason="Failed evaluation",
        )
        assert task.status == AnnotationStatus.PENDING
        assert task.priority == 2

    def test_full_lifecycle(self):
        task = self.workflow.create_task(
            title="Review",
            reason="Test",
        )
        assert task.status == AnnotationStatus.PENDING

        task = self.workflow.start_review(task.id, "alice")
        assert task.status == AnnotationStatus.IN_REVIEW

        verdict = ReviewVerdict(
            task_id=task.id,
            reviewer="alice",
            reviewer_role=Role.QA,
            verdict="agree",
            notes="Looks correct",
        )
        task = self.workflow.submit_verdict(verdict)
        assert task.status == AnnotationStatus.RESOLVED

    def test_cannot_start_review_twice(self):
        task = self.workflow.create_task(title="T", reason="R")
        self.workflow.start_review(task.id, "alice")
        with pytest.raises(ValueError, match="Cannot start review"):
            self.workflow.start_review(task.id, "bob")

    def test_cannot_submit_verdict_before_review(self):
        task = self.workflow.create_task(title="T", reason="R")
        verdict = ReviewVerdict(
            task_id=task.id,
            reviewer="alice",
            reviewer_role=Role.QA,
            verdict="agree",
        )
        with pytest.raises(ValueError, match="Cannot submit verdict"):
            self.workflow.submit_verdict(verdict)

    def test_reopen_task(self):
        task = self.workflow.create_task(title="T", reason="R")
        self.workflow.start_review(task.id, "alice")
        self.workflow.submit_verdict(ReviewVerdict(
            task_id=task.id, reviewer="alice",
            reviewer_role=Role.QA, verdict="agree",
        ))
        task = self.workflow.reopen_task(task.id, reason="Need second opinion")
        assert task.status == AnnotationStatus.PENDING

    def test_list_tasks_filtered(self):
        self.workflow.create_task(title="T1", reason="R1", priority=1)
        self.workflow.create_task(title="T2", reason="R2", priority=2)
        self.workflow.create_task(title="T3", reason="R3", priority=1)

        high_priority = self.workflow.list_tasks(priority=1)
        assert len(high_priority) == 2

        pending = self.workflow.list_tasks(status=AnnotationStatus.PENDING)
        assert len(pending) == 3

    def test_batch_operations(self):
        tasks = [
            self.workflow.create_task(title=f"T{i}", reason="R")
            for i in range(3)
        ]
        task_ids = [t.id for t in tasks]

        started = self.workflow.batch_start_review(task_ids, "alice")
        assert all(t.status == AnnotationStatus.IN_REVIEW for t in started)

        verdicts = [
            ReviewVerdict(
                task_id=tid, reviewer="alice",
                reviewer_role=Role.QA, verdict="agree",
            )
            for tid in task_ids
        ]
        resolved = self.workflow.batch_submit_verdicts(verdicts)
        assert all(t.status == AnnotationStatus.RESOLVED for t in resolved)

    def test_stats(self):
        t1 = self.workflow.create_task(title="T1", reason="R")
        t2 = self.workflow.create_task(title="T2", reason="R")
        self.workflow.create_task(title="T3", reason="R")

        self.workflow.start_review(t1.id, "alice")
        self.workflow.submit_verdict(ReviewVerdict(
            task_id=t1.id, reviewer="alice",
            reviewer_role=Role.QA, verdict="agree",
        ))

        self.workflow.start_review(t2.id, "bob")

        stats = self.workflow.get_stats()
        assert stats.total == 3
        assert stats.pending == 1
        assert stats.in_review == 1
        assert stats.resolved == 1
        assert stats.agreement_rate == 1.0

    def test_verdict_with_corrected_status(self):
        task = self.workflow.create_task(title="T", reason="R")
        self.workflow.start_review(task.id, "alice")
        verdict = ReviewVerdict(
            task_id=task.id,
            reviewer="alice",
            reviewer_role=Role.QA,
            verdict="relabel",
            corrected_status=TraceStatus.PARTIAL_SUCCESS,
            notes="Actually this is a partial success",
        )
        task = self.workflow.submit_verdict(verdict)
        assert task.metadata["corrected_status"] == "partial_success"
