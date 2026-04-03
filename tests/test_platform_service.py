from agentlens.core.exporters import build_closed_loop_snapshot, snapshot_to_dict
from agentlens.core.models import AlertSeverity
from agentlens.core.service import CoreApiService
from agentlens.core.sqlite_repository import SQLiteCoreRepository

from tests.test_platform_exporters import _make_result


def test_platform_service_lists_projects_and_runs(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )
    repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot,
    )
    service = CoreApiService(repository)

    projects_response = service.handle("GET", "/projects")
    runs_response = service.handle("GET", "/projects/qa-project/eval-runs?limit=10&offset=0")
    snapshot_response = service.handle(
        "GET",
        f"/projects/qa-project/snapshots/{snapshot.eval_run.id}",
    )
    dataset_response = service.handle(
        "GET",
        f"/projects/qa-project/dataset-versions/{snapshot.dataset_version.id}",
    )
    traces_response = service.handle("GET", "/projects/qa-project/traces?status=passed")

    assert projects_response.status_code == 200
    assert projects_response.payload["projects"][0]["slug"] == "qa-project"
    assert projects_response.payload["pagination"]["count"] == 1
    assert runs_response.status_code == 200
    assert runs_response.payload["eval_runs"][0]["name"] == "nightly"
    assert snapshot_response.status_code == 200
    assert snapshot_response.payload["dataset_version"]["name"] == "triage"
    assert dataset_response.status_code == 200
    assert dataset_response.payload["dataset_version"]["id"] == snapshot.dataset_version.id
    assert traces_response.status_code == 200
    assert traces_response.payload["pagination"]["count"] == 1


def test_platform_service_handles_missing_resources(tmp_path):
    service = CoreApiService(SQLiteCoreRepository(tmp_path / "platform.db"))

    missing_project = service.handle("GET", "/projects/missing")
    missing_snapshot = service.handle("GET", "/projects/missing/snapshots/run-1")
    invalid_method = service.handle("POST", "/projects")

    assert missing_project.status_code == 404
    assert missing_project.payload["error"] == "project_not_found"
    assert missing_snapshot.status_code == 404
    assert missing_snapshot.payload["error"] == "snapshot_not_found"
    assert invalid_method.status_code == 405


def test_platform_service_creates_alert_rule_and_lists_alerts(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    service = CoreApiService(repository)

    create = service.handle(
        "POST",
        "/projects/qa-project/alert-rules",
        body={
            "alert_rule": {
                "id": "rule-pass-rate",
                "name": "Pass rate below 90",
                "metric_key": "pass_rate",
                "operator": "<",
                "threshold": 90.0,
                "severity": AlertSeverity.WARNING.value,
            }
        },
    )
    snapshot = build_closed_loop_snapshot(
        [
            _make_result(passed=True),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-002", "name": "Broken"}),
                passed=False,
                error="boom",
            ),
        ],
        dataset_name="triage",
        run_name="nightly",
    )
    ingest = service.handle(
        "POST",
        "/projects/qa-project/snapshots",
        body={"snapshot": snapshot_to_dict(snapshot)},
    )
    rules = service.handle("GET", "/projects/qa-project/alert-rules")
    events = service.handle("GET", "/projects/qa-project/alert-events")

    assert create.status_code == 201
    assert ingest.status_code == 201
    assert rules.status_code == 200
    assert rules.payload["pagination"]["count"] == 1
    assert events.status_code == 200
    assert events.payload["pagination"]["count"] == 1
    assert events.payload["alert_events"][0]["rule_id"] == "rule-pass-rate"


def test_platform_service_ingest_snapshot_with_idempotency(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    service = CoreApiService(repository)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )

    payload = {
        "project_name": "QA Project",
        "idempotency_key": "ingest-1",
        "snapshot": snapshot_to_dict(snapshot),
    }
    first = service.handle("POST", "/projects/qa-project/snapshots", body=payload)
    second = service.handle("POST", "/projects/qa-project/snapshots", body=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert len(repository.list_eval_runs("qa-project")) == 1


def test_platform_service_ingest_snapshot_idempotency_conflict(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    service = CoreApiService(repository)
    first_snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly-a",
    )
    second_snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly-b",
    )

    key = "same-key"
    first = service.handle(
        "POST",
        "/projects/qa-project/snapshots",
        body={"idempotency_key": key, "snapshot": snapshot_to_dict(first_snapshot)},
    )
    second = service.handle(
        "POST",
        "/projects/qa-project/snapshots",
        body={"idempotency_key": key, "snapshot": snapshot_to_dict(second_snapshot)},
    )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.payload["error"] == "idempotency_conflict"


def test_platform_service_compares_runs(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    service = CoreApiService(repository)

    baseline = build_closed_loop_snapshot(
        [
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-001", "name": "A"}),
                passed=True,
            ),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-002", "name": "B"}),
                passed=False,
                error="boom",
            ),
        ],
        dataset_name="triage",
        run_name="baseline",
    )
    candidate = build_closed_loop_snapshot(
        [
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-001", "name": "A"}),
                passed=False,
                error="regressed",
            ),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-002", "name": "B"}),
                passed=True,
            ),
        ],
        dataset_name="triage",
        run_name="candidate",
    )

    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=baseline)
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=candidate)

    response = service.handle(
        "GET",
        (
            "/projects/qa-project/experiments/compare"
            f"?baseline_run_id={baseline.eval_run.id}&candidate_run_id={candidate.eval_run.id}"
        ),
    )

    assert response.status_code == 200
    assert response.payload["delta_pass_rate"] == 0.0
    assert response.payload["regressions"] == ["tc-001"]
    assert response.payload["improvements"] == ["tc-002"]
