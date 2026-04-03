"""Service helpers for exposing platform records to HTTP and other UIs."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from agentlens.core.exporters import snapshot_from_dict, snapshot_to_dict
from agentlens.core.models import AlertRuleRecord


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    status_code: int
    payload: dict[str, object]


def _parse_pagination(query: dict[str, list[str]]) -> tuple[int | None, int]:
    raw_limit = (query.get("limit") or [None])[0]
    raw_offset = (query.get("offset") or ["0"])[0]

    limit: int | None = None
    if raw_limit is not None:
        limit = max(0, min(500, int(raw_limit)))
    offset = max(0, int(raw_offset))
    return limit, offset


class CoreApiService:
    def __init__(self, repository):
        self.repository = repository

    def handle(
        self,
        method: str,
        raw_path: str,
        *,
        body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ServiceResponse:
        parsed = urlparse(raw_path)
        path = parsed.path.rstrip("/") or "/"
        parts = [part for part in path.split("/") if part]
        query = parse_qs(parsed.query)
        headers = headers or {}

        try:
            if method == "GET":
                return self._handle_get(path, parts, query)
            if method == "POST":
                return self._handle_post(path, parts, body=body, headers=headers)
            return ServiceResponse(405, {"error": "method_not_allowed"})
        except ValueError as exc:
            return ServiceResponse(400, {"error": "invalid_request", "message": str(exc)})

    def _handle_get(
        self,
        path: str,
        parts: list[str],
        query: dict[str, list[str]],
    ) -> ServiceResponse:
        if path == "/health":
            return ServiceResponse(200, {"status": "ok"})

        limit, offset = _parse_pagination(query)

        if path == "/projects":
            projects = [
                project.model_dump(mode="json")
                for project in self.repository.list_projects(limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "projects": projects,
                    "pagination": {"limit": limit, "offset": offset, "count": len(projects)},
                },
            )

        if len(parts) == 2 and parts[0] == "projects":
            project = self.repository.load_project(parts[1])
            if project is None:
                return ServiceResponse(404, {"error": "project_not_found"})
            return ServiceResponse(200, {"project": project.model_dump(mode="json")})

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "eval-runs":
            eval_runs = [
                run.model_dump(mode="json")
                for run in self.repository.list_eval_runs(parts[1], limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "eval_runs": eval_runs,
                    "pagination": {"limit": limit, "offset": offset, "count": len(eval_runs)},
                },
            )

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "dataset-versions":
            datasets = [
                dataset.model_dump(mode="json")
                for dataset in self.repository.list_dataset_versions(parts[1], limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "dataset_versions": datasets,
                    "pagination": {"limit": limit, "offset": offset, "count": len(datasets)},
                },
            )

        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "dataset-versions":
            dataset = self.repository.load_dataset_version(parts[1], parts[3])
            if dataset is None:
                return ServiceResponse(404, {"error": "dataset_version_not_found"})
            return ServiceResponse(200, {"dataset_version": dataset.model_dump(mode="json")})

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "annotation-tasks":
            tasks = [
                task.model_dump(mode="json")
                for task in self.repository.list_annotation_tasks(parts[1], limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "annotation_tasks": tasks,
                    "pagination": {"limit": limit, "offset": offset, "count": len(tasks)},
                },
            )

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "alert-rules":
            rules = [
                rule.model_dump(mode="json")
                for rule in self.repository.list_alert_rules(parts[1], limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "alert_rules": rules,
                    "pagination": {"limit": limit, "offset": offset, "count": len(rules)},
                },
            )

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "alert-events":
            events = [
                event.model_dump(mode="json")
                for event in self.repository.list_alert_events(parts[1], limit=limit, offset=offset)
            ]
            return ServiceResponse(
                200,
                {
                    "alert_events": events,
                    "pagination": {"limit": limit, "offset": offset, "count": len(events)},
                },
            )

        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "alert-rules":
            rule = self.repository.load_alert_rule(parts[1], parts[3])
            if rule is None:
                return ServiceResponse(404, {"error": "alert_rule_not_found"})
            return ServiceResponse(200, {"alert_rule": rule.model_dump(mode="json")})

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "traces":
            status = (query.get("status") or [None])[0]
            traces = [
                trace.model_dump(mode="json")
                for trace in self.repository.list_traces(
                    parts[1], limit=limit, offset=offset, status=status
                )
            ]
            return ServiceResponse(
                200,
                {
                    "traces": traces,
                    "pagination": {"limit": limit, "offset": offset, "count": len(traces)},
                },
            )

        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "snapshots":
            try:
                snapshot = self.repository.load_snapshot(parts[1], parts[3])
            except FileNotFoundError:
                return ServiceResponse(404, {"error": "snapshot_not_found"})
            return ServiceResponse(200, snapshot_to_dict(snapshot))

        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "eval-runs":
            run = self.repository.load_eval_run(parts[1], parts[3])
            if run is None:
                return ServiceResponse(404, {"error": "eval_run_not_found"})
            return ServiceResponse(200, {"eval_run": run.model_dump(mode="json")})

        if len(parts) == 4 and parts[0] == "projects" and parts[2] == "experiments" and parts[3] == "compare":
            return self._compare_runs(parts[1], query)

        return ServiceResponse(404, {"error": "not_found"})

    def _handle_post(
        self,
        path: str,
        parts: list[str],
        *,
        body: dict[str, object] | None,
        headers: dict[str, str],
    ) -> ServiceResponse:
        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "snapshots":
            if body is None:
                raise ValueError("request body is required")
            payload = body
            snapshot_payload = payload.get("snapshot", payload)
            if not isinstance(snapshot_payload, dict):
                raise ValueError("snapshot payload must be an object")

            snapshot = snapshot_from_dict(snapshot_payload)
            project_slug = parts[1]
            project_name = str(payload.get("project_name") or project_slug)
            idempotency_key = (
                headers.get("X-Idempotency-Key")
                or headers.get("x-idempotency-key")
                or payload.get("idempotency_key")
            )
            if idempotency_key is not None:
                idempotency_key = str(idempotency_key)

            try:
                self.repository.save_snapshot(
                    project_name=project_name,
                    project_slug=project_slug,
                    snapshot=snapshot,
                    idempotency_key=idempotency_key,
                )
            except ValueError as exc:
                return ServiceResponse(409, {"error": "idempotency_conflict", "message": str(exc)})

            return ServiceResponse(
                201,
                {
                    "project_slug": project_slug,
                    "eval_run_id": snapshot.eval_run.id,
                    "dataset_version_id": snapshot.dataset_version.id,
                },
            )

        if len(parts) == 3 and parts[0] == "projects" and parts[2] == "alert-rules":
            if body is None:
                raise ValueError("request body is required")
            return self._create_alert_rule(parts[1], body)

        return ServiceResponse(405, {"error": "method_not_allowed"})

    def _create_alert_rule(
        self,
        project_slug: str,
        payload: dict[str, object],
    ) -> ServiceResponse:
        raw_rule = payload.get("alert_rule", payload)
        if not isinstance(raw_rule, dict):
            raise ValueError("alert_rule payload must be an object")

        rule_data = dict(raw_rule)
        rule_data.setdefault("id", f"alert_rule_{uuid4().hex[:10]}")
        rule = AlertRuleRecord.model_validate(rule_data)
        self._validate_alert_rule(rule)

        project_name = str(payload.get("project_name") or project_slug)
        self.repository.save_alert_rule(
            project_name=project_name,
            project_slug=project_slug,
            alert_rule=rule,
        )
        return ServiceResponse(
            201,
            {
                "project_slug": project_slug,
                "alert_rule": rule.model_dump(mode="json"),
            },
        )

    def _validate_alert_rule(self, rule: AlertRuleRecord) -> None:
        allowed_metrics = {"pass_rate", "failed_cases", "passed_cases", "total_cases"}
        allowed_operators = {">", ">=", "<", "<=", "=="}
        if rule.metric_key not in allowed_metrics:
            raise ValueError(
                f"unsupported metric_key '{rule.metric_key}', expected one of {sorted(allowed_metrics)}"
            )
        if rule.operator not in allowed_operators:
            raise ValueError(
                f"unsupported operator '{rule.operator}', expected one of {sorted(allowed_operators)}"
            )

    def _compare_runs(
        self,
        project_slug: str,
        query: dict[str, list[str]],
    ) -> ServiceResponse:
        baseline_run_id = (query.get("baseline_run_id") or [None])[0]
        candidate_run_id = (query.get("candidate_run_id") or [None])[0]
        if not baseline_run_id or not candidate_run_id:
            return ServiceResponse(
                400,
                {
                    "error": "missing_query_params",
                    "message": "baseline_run_id and candidate_run_id are required",
                },
            )

        try:
            baseline = self.repository.load_snapshot(project_slug, baseline_run_id)
            candidate = self.repository.load_snapshot(project_slug, candidate_run_id)
        except FileNotFoundError:
            return ServiceResponse(404, {"error": "snapshot_not_found"})

        baseline_by_scenario = {trace.scenario_id: trace for trace in baseline.traces}
        candidate_by_scenario = {trace.scenario_id: trace for trace in candidate.traces}
        scenarios = sorted(set(baseline_by_scenario).union(candidate_by_scenario))

        regressions: list[str] = []
        improvements: list[str] = []
        unchanged: list[str] = []
        new_failures: list[str] = []

        for scenario_id in scenarios:
            base_trace = baseline_by_scenario.get(scenario_id)
            cand_trace = candidate_by_scenario.get(scenario_id)
            base_passed = bool(base_trace and base_trace.status.value == "passed")
            cand_passed = bool(cand_trace and cand_trace.status.value == "passed")

            if base_trace is None or cand_trace is None:
                new_failures.append(scenario_id)
                continue
            if base_passed and not cand_passed:
                regressions.append(scenario_id)
            elif not base_passed and cand_passed:
                improvements.append(scenario_id)
            else:
                unchanged.append(scenario_id)

        return ServiceResponse(
            200,
            {
                "baseline_run_id": baseline_run_id,
                "candidate_run_id": candidate_run_id,
                "baseline_pass_rate": baseline.eval_run.summary.pass_rate,
                "candidate_pass_rate": candidate.eval_run.summary.pass_rate,
                "delta_pass_rate": round(
                    candidate.eval_run.summary.pass_rate - baseline.eval_run.summary.pass_rate,
                    1,
                ),
                "regressions": regressions,
                "improvements": improvements,
                "unchanged": unchanged,
                "new_or_missing": new_failures,
            },
        )
