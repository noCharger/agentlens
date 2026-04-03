"""Local repository primitives for storing platform records by project."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agentlens.core.exporters import (
    ClosedLoopSnapshot,
    snapshot_to_dict,
)
from agentlens.core.alerts import evaluate_alert_rules
from agentlens.core.models import (
    AlertEventRecord,
    AlertRuleRecord,
    AnnotationTaskRecord,
    AuditEventRecord,
    DatasetVersionRecord,
    EvalRunRecord,
    ProjectRecord,
    Role,
    TraceRecord,
)


def slugify_project_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "default"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class StoredSnapshotPaths:
    project_dir: Path
    project_path: Path
    snapshot_path: Path
    eval_run_path: Path
    dataset_version_path: Path
    trace_paths: tuple[Path, ...]
    annotation_paths: tuple[Path, ...]
    audit_event_path: Path


class FileCoreRepository:
    """A minimal local store that mirrors the future platform data model."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _projects_root(self) -> Path:
        return self.root / "projects"

    def _project_dir(self, slug: str) -> Path:
        return self._projects_root() / slug

    def _project_path(self, slug: str) -> Path:
        return self._project_dir(slug) / "project.json"

    def _records_dir(self, slug: str, record_type: str) -> Path:
        return self._project_dir(slug) / record_type

    def list_projects(
        self, limit: int | None = None, offset: int = 0
    ) -> list[ProjectRecord]:
        projects: list[ProjectRecord] = []
        for path in sorted(self._projects_root().glob("*/project.json")):
            projects.append(ProjectRecord.model_validate(_read_json(path)))
        return self._slice_records(projects, limit=limit, offset=offset)

    def load_project(self, slug: str) -> ProjectRecord | None:
        path = self._project_path(slug)
        if not path.exists():
            return None
        return ProjectRecord.model_validate(_read_json(path))

    def ensure_project(
        self,
        *,
        name: str,
        slug: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProjectRecord:
        resolved_slug = slugify_project_name(slug or name)
        path = self._project_path(resolved_slug)
        existing = self.load_project(resolved_slug)

        if existing is not None:
            merged_tags = sorted(set(existing.tags).union(tags or []))
            merged_metadata = {**existing.metadata, **(metadata or {})}
            project = existing.model_copy(
                update={
                    "name": name or existing.name,
                    "tags": merged_tags,
                    "metadata": merged_metadata,
                }
            )
        else:
            project = ProjectRecord(
                id=f"project_{uuid4().hex[:12]}",
                name=name,
                slug=resolved_slug,
                tags=tags or [],
                metadata=metadata or {},
            )

        _write_json(path, project.model_dump(mode="json"))
        return project

    def save_snapshot(
        self,
        *,
        project_name: str,
        snapshot: ClosedLoopSnapshot,
        project_slug: str | None = None,
        project_tags: list[str] | None = None,
        project_metadata: dict[str, object] | None = None,
        actor_role: Role = Role.ENGINEER,
        idempotency_key: str | None = None,
    ) -> StoredSnapshotPaths:
        del idempotency_key
        project = self.ensure_project(
            name=project_name,
            slug=project_slug,
            tags=project_tags,
            metadata=project_metadata,
        )
        project_dir = self._project_dir(project.slug)

        snapshot_path = self._records_dir(project.slug, "snapshots") / f"{snapshot.eval_run.id}.json"
        eval_run_path = self._records_dir(project.slug, "eval_runs") / f"{snapshot.eval_run.id}.json"
        dataset_version_path = (
            self._records_dir(project.slug, "dataset_versions")
            / f"{snapshot.dataset_version.id}.json"
        )

        _write_json(snapshot_path, snapshot_to_dict(snapshot))
        _write_json(eval_run_path, snapshot.eval_run.model_dump(mode="json"))
        _write_json(dataset_version_path, snapshot.dataset_version.model_dump(mode="json"))

        trace_paths: list[Path] = []
        for trace in snapshot.traces:
            trace_path = self._records_dir(project.slug, "traces") / f"{trace.id}.json"
            _write_json(trace_path, trace.model_dump(mode="json"))
            trace_paths.append(trace_path)

        annotation_paths: list[Path] = []
        for task in snapshot.annotation_tasks:
            annotation_path = self._records_dir(project.slug, "annotation_tasks") / f"{task.id}.json"
            _write_json(annotation_path, task.model_dump(mode="json"))
            annotation_paths.append(annotation_path)

        alert_rules = self.list_alert_rules(project.slug)
        alert_events = evaluate_alert_rules(
            project_slug=project.slug,
            eval_run=snapshot.eval_run,
            rules=alert_rules,
        )
        for event in alert_events:
            event_path = self._records_dir(project.slug, "alert_events") / f"{event.id}.json"
            _write_json(event_path, event.model_dump(mode="json"))

        audit_event = AuditEventRecord(
            id=f"audit_{uuid4().hex[:12]}",
            action="platform.snapshot.saved",
            actor_role=actor_role,
            resource_type="eval_run",
            resource_id=snapshot.eval_run.id,
            details={
                "dataset_version_id": snapshot.dataset_version.id,
                "trace_count": len(snapshot.traces),
                "annotation_count": len(snapshot.annotation_tasks),
            },
        )
        audit_event_path = self._records_dir(project.slug, "audit_events") / f"{audit_event.id}.json"
        _write_json(audit_event_path, audit_event.model_dump(mode="json"))

        return StoredSnapshotPaths(
            project_dir=project_dir,
            project_path=self._project_path(project.slug),
            snapshot_path=snapshot_path,
            eval_run_path=eval_run_path,
            dataset_version_path=dataset_version_path,
            trace_paths=tuple(trace_paths),
            annotation_paths=tuple(annotation_paths),
            audit_event_path=audit_event_path,
        )

    def load_snapshot(self, project_slug: str, eval_run_id: str) -> ClosedLoopSnapshot:
        path = self._records_dir(project_slug, "snapshots") / f"{eval_run_id}.json"
        payload = _read_json(path)
        return ClosedLoopSnapshot(
            traces=[TraceRecord.model_validate(item) for item in payload["traces"]],
            dataset_version=DatasetVersionRecord.model_validate(payload["dataset_version"]),
            eval_run=EvalRunRecord.model_validate(payload["eval_run"]),
            annotation_tasks=[
                AnnotationTaskRecord.model_validate(item)
                for item in payload["annotation_tasks"]
            ],
        )

    def load_eval_run(self, project_slug: str, run_id: str) -> EvalRunRecord | None:
        path = self._records_dir(project_slug, "eval_runs") / f"{run_id}.json"
        if not path.exists():
            return None
        return EvalRunRecord.model_validate(_read_json(path))

    def save_alert_rule(
        self,
        *,
        project_name: str,
        alert_rule: AlertRuleRecord,
        project_slug: str | None = None,
        project_tags: list[str] | None = None,
        project_metadata: dict[str, object] | None = None,
        actor_role: Role = Role.ENGINEER,
    ) -> Path:
        project = self.ensure_project(
            name=project_name,
            slug=project_slug,
            tags=project_tags,
            metadata=project_metadata,
        )
        path = self._records_dir(project.slug, "alert_rules") / f"{alert_rule.id}.json"
        _write_json(path, alert_rule.model_dump(mode="json"))
        audit_event = AuditEventRecord(
            id=f"audit_{uuid4().hex[:12]}",
            action="platform.alert_rule.saved",
            actor_role=actor_role,
            resource_type="alert_rule",
            resource_id=alert_rule.id,
            details={
                "metric_key": alert_rule.metric_key,
                "operator": alert_rule.operator,
                "threshold": alert_rule.threshold,
            },
        )
        audit_path = self._records_dir(project.slug, "audit_events") / f"{audit_event.id}.json"
        _write_json(audit_path, audit_event.model_dump(mode="json"))
        return path

    def load_alert_rule(self, project_slug: str, rule_id: str) -> AlertRuleRecord | None:
        path = self._records_dir(project_slug, "alert_rules") / f"{rule_id}.json"
        if not path.exists():
            return None
        return AlertRuleRecord.model_validate(_read_json(path))

    def save_dataset_version(
        self,
        *,
        project_name: str,
        dataset_version: DatasetVersionRecord,
        project_slug: str | None = None,
        project_tags: list[str] | None = None,
        project_metadata: dict[str, object] | None = None,
        actor_role: Role = Role.ENGINEER,
    ) -> Path:
        project = self.ensure_project(
            name=project_name,
            slug=project_slug,
            tags=project_tags,
            metadata=project_metadata,
        )

        dataset_version_path = (
            self._records_dir(project.slug, "dataset_versions")
            / f"{dataset_version.id}.json"
        )
        _write_json(dataset_version_path, dataset_version.model_dump(mode="json"))

        audit_event = AuditEventRecord(
            id=f"audit_{uuid4().hex[:12]}",
            action="platform.dataset_version.saved",
            actor_role=actor_role,
            resource_type="dataset_version",
            resource_id=dataset_version.id,
            details={
                "dataset_id": dataset_version.dataset_id,
                "item_count": dataset_version.item_count,
                "version": dataset_version.version,
            },
        )
        audit_event_path = (
            self._records_dir(project.slug, "audit_events")
            / f"{audit_event.id}.json"
        )
        _write_json(audit_event_path, audit_event.model_dump(mode="json"))
        return dataset_version_path

    def load_dataset_version(
        self,
        project_slug: str,
        dataset_version_id: str,
    ) -> DatasetVersionRecord | None:
        path = (
            self._records_dir(project_slug, "dataset_versions")
            / f"{dataset_version_id}.json"
        )
        if not path.exists():
            return None
        return DatasetVersionRecord.model_validate(_read_json(path))

    def list_eval_runs(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[EvalRunRecord]:
        return self._load_records(
            project_slug, "eval_runs", EvalRunRecord, limit=limit, offset=offset
        )

    def list_dataset_versions(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[DatasetVersionRecord]:
        return self._load_records(
            project_slug, "dataset_versions", DatasetVersionRecord, limit=limit, offset=offset
        )

    def list_traces(
        self,
        project_slug: str,
        limit: int | None = None,
        offset: int = 0,
        status: str | None = None,
    ) -> list[TraceRecord]:
        records = self._load_records(project_slug, "traces", TraceRecord, limit=None, offset=0)
        if status:
            records = [record for record in records if record.status.value == status]
        return self._slice_records(records, limit=limit, offset=offset)

    def list_annotation_tasks(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AnnotationTaskRecord]:
        return self._load_records(
            project_slug, "annotation_tasks", AnnotationTaskRecord, limit=limit, offset=offset
        )

    def list_audit_events(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AuditEventRecord]:
        return self._load_records(
            project_slug, "audit_events", AuditEventRecord, limit=limit, offset=offset
        )

    def list_alert_rules(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AlertRuleRecord]:
        return self._load_records(
            project_slug, "alert_rules", AlertRuleRecord, limit=limit, offset=offset
        )

    def list_alert_events(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AlertEventRecord]:
        return self._load_records(
            project_slug, "alert_events", AlertEventRecord, limit=limit, offset=offset
        )

    def _slice_records(
        self, records: list, *, limit: int | None = None, offset: int = 0
    ) -> list:
        start = max(0, offset)
        if limit is None:
            return records[start:]
        return records[start : start + max(0, limit)]

    def _load_records(
        self,
        project_slug: str,
        record_type: str,
        model,
        *,
        limit: int | None = None,
        offset: int = 0,
    ):
        record_dir = self._records_dir(project_slug, record_type)
        if not record_dir.exists():
            return []
        records = [
            model.model_validate(_read_json(path))
            for path in sorted(record_dir.glob("*.json"))
        ]
        return self._slice_records(records, limit=limit, offset=offset)
