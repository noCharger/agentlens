from agentlens.core.exporters import build_closed_loop_snapshot
from agentlens.core.models import AlertRuleRecord, AlertSeverity
from agentlens.core.sqlite_repository import SQLiteCoreRepository

from tests.test_platform_exporters import _make_result


def test_sqlite_repository_saves_and_loads_snapshot(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )

    database_path = repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot,
    )

    assert database_path.exists()

    loaded = repository.load_snapshot("qa-project", snapshot.eval_run.id)
    assert loaded.eval_run.id == snapshot.eval_run.id
    assert loaded.dataset_version.id == snapshot.dataset_version.id
    assert loaded.traces[0].scenario_id == "tc-001"


def test_sqlite_repository_lists_records(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot = build_closed_loop_snapshot(
        [
            _make_result(),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-002", "name": "Broken"}),
                passed=False,
                error="boom",
            ),
        ],
        dataset_name="triage",
        run_name="nightly",
    )

    repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot,
    )

    assert [project.slug for project in repository.list_projects()] == ["qa-project"]
    assert len(repository.list_eval_runs("qa-project")) == 1
    assert len(repository.list_dataset_versions("qa-project")) == 1
    assert len(repository.list_traces("qa-project")) == 2
    assert len(repository.list_annotation_tasks("qa-project")) == 1
    assert len(repository.list_audit_events("qa-project")) == 1


def test_sqlite_repository_supports_pagination_and_filter(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")

    for idx in range(3):
        scenario = _make_result().scenario.model_copy(update={"id": f"tc-{idx:03d}", "name": f"Case {idx}"})
        snapshot = build_closed_loop_snapshot(
            [_make_result(scenario=scenario, passed=(idx % 2 == 0), error=None)],
            dataset_name="triage",
            run_name=f"nightly-{idx}",
        )
        repository.save_snapshot(
            project_name="QA Project",
            project_slug="qa-project",
            snapshot=snapshot,
        )

    paged_runs = repository.list_eval_runs("qa-project", limit=1, offset=1)
    failed_traces = repository.list_traces("qa-project", status="failed")

    assert len(paged_runs) == 1
    assert len(failed_traces) == 1
    assert failed_traces[0].status.value == "failed"


def test_sqlite_repository_enforces_idempotency_key(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot_a = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly-a",
    )
    snapshot_b = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly-b",
    )

    repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot_a,
        idempotency_key="same-key",
    )
    repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot_a,
        idempotency_key="same-key",
    )

    try:
        repository.save_snapshot(
            project_name="QA Project",
            project_slug="qa-project",
            snapshot=snapshot_b,
            idempotency_key="same-key",
        )
    except ValueError as exc:
        assert "idempotency_key conflict" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected idempotency key conflict")


def test_sqlite_repository_saves_and_loads_dataset_version(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )

    database_path = repository.save_dataset_version(
        project_name="QA Project",
        project_slug="qa-project",
        dataset_version=snapshot.dataset_version,
    )
    loaded = repository.load_dataset_version("qa-project", snapshot.dataset_version.id)

    assert database_path.exists()
    assert loaded is not None
    assert loaded.id == snapshot.dataset_version.id
    assert loaded.item_count == 1


def test_sqlite_repository_saves_alert_rule_and_triggers_event(tmp_path):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    repository.save_alert_rule(
        project_name="QA Project",
        project_slug="qa-project",
        alert_rule=AlertRuleRecord(
            id="rule-pass-rate",
            name="Pass rate below 90",
            metric_key="pass_rate",
            operator="<",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
        ),
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
    repository.save_snapshot(
        project_name="QA Project",
        project_slug="qa-project",
        snapshot=snapshot,
    )

    loaded_rule = repository.load_alert_rule("qa-project", "rule-pass-rate")
    events = repository.list_alert_events("qa-project")

    assert loaded_rule is not None
    assert loaded_rule.metric_key == "pass_rate"
    assert len(events) == 1
    assert events[0].rule_id == "rule-pass-rate"
    assert events[0].eval_run_id == snapshot.eval_run.id
