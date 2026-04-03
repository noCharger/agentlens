"""SQLite-backed repository for platform records."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from agentlens.core.exporters import (
    ClosedLoopSnapshot,
    snapshot_from_dict,
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
from agentlens.core.repository import slugify_project_name

_RETRYABLE_SQLITE_MESSAGES = ("database is locked", "database is busy", "locked")


def _dump_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_payload(payload: str) -> dict[str, object]:
    return json.loads(payload)


def _is_retryable_error(exc: sqlite3.OperationalError) -> bool:
    lowered = str(exc).lower()
    return any(message in lowered for message in _RETRYABLE_SQLITE_MESSAGES)


def _bounded_slice(limit: int | None, offset: int) -> tuple[int | None, int]:
    resolved_offset = max(0, offset)
    if limit is None:
        return None, resolved_offset
    return max(0, limit), resolved_offset


class SQLiteCoreRepository:
    """A SQLite implementation tuned for local durability and concurrent reads."""

    def __init__(
        self,
        path: Path,
        *,
        max_retries: int = 4,
        base_retry_delay_s: float = 0.05,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.base_retry_delay_s = base_retry_delay_s
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _run_with_retry(self, operation):
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not _is_retryable_error(exc) or attempt == self.max_retries:
                    raise
                last_error = exc
                time.sleep(self.base_retry_delay_s * (2 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unreachable retry path")

    def _init_db(self) -> None:
        def operation() -> None:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                        slug TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS snapshots (
                        eval_run_id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS eval_runs (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        name TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        pass_rate REAL NOT NULL,
                        total_cases INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        item_count INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS traces (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        eval_run_id TEXT NOT NULL,
                        scenario_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS annotation_tasks (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        eval_run_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS audit_events (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        action TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS alert_rules (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        metric_key TEXT NOT NULL,
                        operator TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        severity TEXT NOT NULL,
                        enabled INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS alert_events (
                        id TEXT PRIMARY KEY,
                        project_slug TEXT NOT NULL,
                        rule_id TEXT NOT NULL,
                        eval_run_id TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        triggered_at TEXT NOT NULL,
                        acknowledged INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS ingest_requests (
                        project_slug TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        eval_run_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (project_slug, idempotency_key)
                    );

                    CREATE INDEX IF NOT EXISTS idx_eval_runs_project_created
                    ON eval_runs(project_slug, created_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_dataset_versions_project_created
                    ON dataset_versions(project_slug, created_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_traces_project_created
                    ON traces(project_slug, created_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_traces_project_status
                    ON traces(project_slug, status);

                    CREATE INDEX IF NOT EXISTS idx_annotation_tasks_project_priority
                    ON annotation_tasks(project_slug, status, priority);

                    CREATE INDEX IF NOT EXISTS idx_audit_events_project_occurred
                    ON audit_events(project_slug, occurred_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_alert_rules_project_enabled
                    ON alert_rules(project_slug, enabled);

                    CREATE INDEX IF NOT EXISTS idx_alert_events_project_triggered
                    ON alert_events(project_slug, triggered_at DESC);
                    """
                )

        self._run_with_retry(operation)

    def list_projects(self, limit: int | None = None, offset: int = 0) -> list[ProjectRecord]:
        resolved_limit, resolved_offset = _bounded_slice(limit, offset)

        def operation() -> list[ProjectRecord]:
            query = "SELECT payload_json FROM projects ORDER BY slug"
            params: list[object] = []
            if resolved_limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([resolved_limit, resolved_offset])
            elif resolved_offset:
                query += " LIMIT -1 OFFSET ?"
                params.append(resolved_offset)

            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
            return [
                ProjectRecord.model_validate(_load_payload(row["payload_json"]))
                for row in rows
            ]

        return self._run_with_retry(operation)

    def load_project(self, slug: str) -> ProjectRecord | None:
        def operation() -> ProjectRecord | None:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload_json FROM projects WHERE slug = ?",
                    (slug,),
                ).fetchone()
            if row is None:
                return None
            return ProjectRecord.model_validate(_load_payload(row["payload_json"]))

        return self._run_with_retry(operation)

    def ensure_project(
        self,
        *,
        name: str,
        slug: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProjectRecord:
        resolved_slug = slugify_project_name(slug or name)
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

        def operation() -> ProjectRecord:
            payload_json = _dump_payload(project.model_dump(mode="json"))
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO projects (slug, project_id, name, created_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project.slug,
                        project.id,
                        project.name,
                        project.created_at.isoformat(),
                        payload_json,
                    ),
                )
            return project

        return self._run_with_retry(operation)

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
    ) -> Path:
        project = self.ensure_project(
            name=project_name,
            slug=project_slug,
            tags=project_tags,
            metadata=project_metadata,
        )
        snapshot_payload = snapshot_to_dict(snapshot)

        # Stable audit event id prevents duplicate audit noise on retries.
        audit_event = AuditEventRecord(
            id=f"audit_snapshot_{project.slug}_{snapshot.eval_run.id}",
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

        def operation() -> Path:
            with self._connect() as conn:
                if idempotency_key:
                    existing = conn.execute(
                        """
                        SELECT eval_run_id FROM ingest_requests
                        WHERE project_slug = ? AND idempotency_key = ?
                        """,
                        (project.slug, idempotency_key),
                    ).fetchone()
                    if existing is not None:
                        if existing["eval_run_id"] != snapshot.eval_run.id:
                            raise ValueError(
                                "idempotency_key conflict: key is already associated with "
                                f"run {existing['eval_run_id']}"
                            )
                        return self.path
                    conn.execute(
                        """
                        INSERT INTO ingest_requests
                        (project_slug, idempotency_key, eval_run_id, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            project.slug,
                            idempotency_key,
                            snapshot.eval_run.id,
                            snapshot.eval_run.created_at.isoformat(),
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO snapshots (eval_run_id, project_slug, created_at, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        snapshot.eval_run.id,
                        project.slug,
                        snapshot.eval_run.created_at.isoformat(),
                        _dump_payload(snapshot_payload),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_runs
                    (id, project_slug, name, source, created_at, pass_rate, total_cases, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.eval_run.id,
                        project.slug,
                        snapshot.eval_run.name,
                        snapshot.eval_run.source,
                        snapshot.eval_run.created_at.isoformat(),
                        snapshot.eval_run.summary.pass_rate,
                        snapshot.eval_run.summary.total,
                        _dump_payload(snapshot.eval_run.model_dump(mode="json")),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dataset_versions
                    (id, project_slug, dataset_id, name, version, created_at, item_count, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.dataset_version.id,
                        project.slug,
                        snapshot.dataset_version.dataset_id,
                        snapshot.dataset_version.name,
                        snapshot.dataset_version.version,
                        snapshot.dataset_version.created_at.isoformat(),
                        snapshot.dataset_version.item_count,
                        _dump_payload(snapshot.dataset_version.model_dump(mode="json")),
                    ),
                )

                for trace in snapshot.traces:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO traces
                        (id, project_slug, eval_run_id, scenario_id, status, created_at, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trace.id,
                            project.slug,
                            snapshot.eval_run.id,
                            trace.scenario_id,
                            trace.status.value,
                            trace.created_at.isoformat(),
                            _dump_payload(trace.model_dump(mode="json")),
                        ),
                    )

                for task in snapshot.annotation_tasks:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO annotation_tasks
                        (id, project_slug, eval_run_id, status, priority, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task.id,
                            project.slug,
                            snapshot.eval_run.id,
                            task.status.value,
                            task.priority,
                            _dump_payload(task.model_dump(mode="json")),
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_events
                    (id, project_slug, action, occurred_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        audit_event.id,
                        project.slug,
                        audit_event.action,
                        audit_event.occurred_at.isoformat(),
                        _dump_payload(audit_event.model_dump(mode="json")),
                    ),
                )

                alert_rules = self._load_alert_rules_for_project(conn, project.slug)
                alert_events = evaluate_alert_rules(
                    project_slug=project.slug,
                    eval_run=snapshot.eval_run,
                    rules=alert_rules,
                )
                for event in alert_events:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO alert_events
                        (id, project_slug, rule_id, eval_run_id, severity, triggered_at, acknowledged, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.id,
                            project.slug,
                            event.rule_id,
                            event.eval_run_id,
                            event.severity.value,
                            event.triggered_at.isoformat(),
                            1 if event.acknowledged else 0,
                            _dump_payload(event.model_dump(mode="json")),
                        ),
                    )

            return self.path

        return self._run_with_retry(operation)

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
        audit_event = AuditEventRecord(
            id=f"audit_alert_rule_{project.slug}_{alert_rule.id}",
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

        def operation() -> Path:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alert_rules
                    (id, project_slug, metric_key, operator, threshold, severity, enabled, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_rule.id,
                        project.slug,
                        alert_rule.metric_key,
                        alert_rule.operator,
                        float(alert_rule.threshold),
                        alert_rule.severity.value,
                        1 if alert_rule.enabled else 0,
                        _dump_payload(alert_rule.model_dump(mode="json")),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_events
                    (id, project_slug, action, occurred_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        audit_event.id,
                        project.slug,
                        audit_event.action,
                        audit_event.occurred_at.isoformat(),
                        _dump_payload(audit_event.model_dump(mode="json")),
                    ),
                )
            return self.path

        return self._run_with_retry(operation)

    def load_alert_rule(self, project_slug: str, rule_id: str) -> AlertRuleRecord | None:
        def operation() -> AlertRuleRecord | None:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json FROM alert_rules
                    WHERE project_slug = ? AND id = ?
                    """,
                    (project_slug, rule_id),
                ).fetchone()
            if row is None:
                return None
            return AlertRuleRecord.model_validate(_load_payload(row["payload_json"]))

        return self._run_with_retry(operation)

    def load_snapshot(self, project_slug: str, eval_run_id: str) -> ClosedLoopSnapshot:
        def operation() -> ClosedLoopSnapshot:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json FROM snapshots
                    WHERE project_slug = ? AND eval_run_id = ?
                    """,
                    (project_slug, eval_run_id),
                ).fetchone()
            if row is None:
                raise FileNotFoundError(
                    f"No snapshot for project '{project_slug}' and run '{eval_run_id}'"
                )
            return snapshot_from_dict(_load_payload(row["payload_json"]))

        return self._run_with_retry(operation)

    def load_eval_run(self, project_slug: str, run_id: str) -> EvalRunRecord | None:
        def operation() -> EvalRunRecord | None:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json FROM eval_runs
                    WHERE project_slug = ? AND id = ?
                    """,
                    (project_slug, run_id),
                ).fetchone()
            if row is None:
                return None
            return EvalRunRecord.model_validate(_load_payload(row["payload_json"]))

        return self._run_with_retry(operation)

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
        audit_event = AuditEventRecord(
            id=f"audit_dataset_{project.slug}_{dataset_version.id}",
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

        def operation() -> Path:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dataset_versions
                    (id, project_slug, dataset_id, name, version, created_at, item_count, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_version.id,
                        project.slug,
                        dataset_version.dataset_id,
                        dataset_version.name,
                        dataset_version.version,
                        dataset_version.created_at.isoformat(),
                        dataset_version.item_count,
                        _dump_payload(dataset_version.model_dump(mode="json")),
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_events
                    (id, project_slug, action, occurred_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        audit_event.id,
                        project.slug,
                        audit_event.action,
                        audit_event.occurred_at.isoformat(),
                        _dump_payload(audit_event.model_dump(mode="json")),
                    ),
                )
            return self.path

        return self._run_with_retry(operation)

    def load_dataset_version(
        self,
        project_slug: str,
        dataset_version_id: str,
    ) -> DatasetVersionRecord | None:
        def operation() -> DatasetVersionRecord | None:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json FROM dataset_versions
                    WHERE project_slug = ? AND id = ?
                    """,
                    (project_slug, dataset_version_id),
                ).fetchone()
            if row is None:
                return None
            return DatasetVersionRecord.model_validate(
                _load_payload(row["payload_json"])
            )

        return self._run_with_retry(operation)

    def list_eval_runs(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[EvalRunRecord]:
        return self._load_records(
            project_slug,
            "SELECT payload_json FROM eval_runs WHERE project_slug = ? ORDER BY created_at DESC",
            EvalRunRecord,
            limit=limit,
            offset=offset,
        )

    def list_dataset_versions(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[DatasetVersionRecord]:
        return self._load_records(
            project_slug,
            """
            SELECT payload_json FROM dataset_versions
            WHERE project_slug = ? ORDER BY created_at DESC
            """,
            DatasetVersionRecord,
            limit=limit,
            offset=offset,
        )

    def list_traces(
        self,
        project_slug: str,
        limit: int | None = None,
        offset: int = 0,
        status: str | None = None,
    ) -> list[TraceRecord]:
        query = """
            SELECT payload_json FROM traces
            WHERE project_slug = ?
        """
        params: list[object] = [project_slug]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        return self._load_records_from_query(
            query,
            params,
            TraceRecord,
            limit=limit,
            offset=offset,
        )

    def list_annotation_tasks(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AnnotationTaskRecord]:
        return self._load_records(
            project_slug,
            """
            SELECT payload_json FROM annotation_tasks
            WHERE project_slug = ? ORDER BY priority ASC, id ASC
            """,
            AnnotationTaskRecord,
            limit=limit,
            offset=offset,
        )

    def list_audit_events(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AuditEventRecord]:
        return self._load_records(
            project_slug,
            """
            SELECT payload_json FROM audit_events
            WHERE project_slug = ? ORDER BY occurred_at DESC
            """,
            AuditEventRecord,
            limit=limit,
            offset=offset,
        )

    def list_alert_rules(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AlertRuleRecord]:
        return self._load_records(
            project_slug,
            """
            SELECT payload_json FROM alert_rules
            WHERE project_slug = ? ORDER BY id ASC
            """,
            AlertRuleRecord,
            limit=limit,
            offset=offset,
        )

    def list_alert_events(
        self, project_slug: str, limit: int | None = None, offset: int = 0
    ) -> list[AlertEventRecord]:
        return self._load_records(
            project_slug,
            """
            SELECT payload_json FROM alert_events
            WHERE project_slug = ? ORDER BY triggered_at DESC
            """,
            AlertEventRecord,
            limit=limit,
            offset=offset,
        )

    def _load_alert_rules_for_project(
        self,
        connection: sqlite3.Connection,
        project_slug: str,
    ) -> list[AlertRuleRecord]:
        rows = connection.execute(
            """
            SELECT payload_json FROM alert_rules
            WHERE project_slug = ? AND enabled = 1
            ORDER BY id ASC
            """,
            (project_slug,),
        ).fetchall()
        return [
            AlertRuleRecord.model_validate(_load_payload(row["payload_json"]))
            for row in rows
        ]

    def _load_records(
        self,
        project_slug: str,
        query: str,
        model,
        *,
        limit: int | None = None,
        offset: int = 0,
    ):
        return self._load_records_from_query(
            query,
            [project_slug],
            model,
            limit=limit,
            offset=offset,
        )

    def _load_records_from_query(
        self,
        query: str,
        params: list[object],
        model,
        *,
        limit: int | None = None,
        offset: int = 0,
    ):
        resolved_limit, resolved_offset = _bounded_slice(limit, offset)
        query_with_page = query
        query_params = list(params)
        if resolved_limit is not None:
            query_with_page += " LIMIT ? OFFSET ?"
            query_params.extend([resolved_limit, resolved_offset])
        elif resolved_offset:
            query_with_page += " LIMIT -1 OFFSET ?"
            query_params.append(resolved_offset)

        def operation():
            with self._connect() as conn:
                rows = conn.execute(query_with_page, query_params).fetchall()
            return [
                model.model_validate(_load_payload(row["payload_json"]))
                for row in rows
            ]

        return self._run_with_retry(operation)
