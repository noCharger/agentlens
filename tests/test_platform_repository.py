from agentlens.core.exporters import build_closed_loop_snapshot
from agentlens.core.models import AlertRuleRecord, AlertSeverity
from agentlens.core.repository import FileCoreRepository, slugify_project_name

from tests.test_platform_exporters import _make_result


def test_slugify_project_name_normalizes_input():
    assert slugify_project_name("GDPval Project") == "gdpval-project"
    assert slugify_project_name("  ") == "default"


def test_file_repository_saves_and_loads_snapshot(tmp_path):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )

    stored = repository.save_snapshot(
        project_name="Local QA Project",
        snapshot=snapshot,
    )

    assert stored.project_path.exists()
    assert stored.snapshot_path.exists()
    assert stored.eval_run_path.exists()
    assert stored.dataset_version_path.exists()
    assert len(stored.trace_paths) == 1
    assert len(stored.annotation_paths) == 0

    loaded = repository.load_snapshot("local-qa-project", snapshot.eval_run.id)
    assert loaded.eval_run.id == snapshot.eval_run.id
    assert loaded.dataset_version.id == snapshot.dataset_version.id
    assert loaded.traces[0].scenario_id == "tc-001"


def test_file_repository_lists_records_by_project(tmp_path):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [
            _make_result(),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-404", "name": "Broken"}),
                passed=False,
                error="boom",
            ),
        ],
        dataset_name="regression",
        run_name="nightly",
    )

    repository.save_snapshot(
        project_name="QA Workspace",
        project_slug="qa-workspace",
        snapshot=snapshot,
    )

    projects = repository.list_projects()
    eval_runs = repository.list_eval_runs("qa-workspace")
    datasets = repository.list_dataset_versions("qa-workspace")
    traces = repository.list_traces("qa-workspace")
    annotations = repository.list_annotation_tasks("qa-workspace")
    audits = repository.list_audit_events("qa-workspace")

    assert [project.slug for project in projects] == ["qa-workspace"]
    assert len(eval_runs) == 1
    assert len(datasets) == 1
    assert len(traces) == 2
    assert len(annotations) == 1
    assert len(audits) == 1
    assert audits[0].action == "platform.snapshot.saved"


def test_file_repository_load_eval_run_and_pagination(tmp_path):
    repository = FileCoreRepository(tmp_path)
    first = build_closed_loop_snapshot(
        [_make_result(scenario=_make_result().scenario.model_copy(update={"id": "tc-100", "name": "First"}))],
        dataset_name="triage",
        run_name="nightly-1",
    )
    second = build_closed_loop_snapshot(
        [_make_result(scenario=_make_result().scenario.model_copy(update={"id": "tc-101", "name": "Second"}))],
        dataset_name="triage",
        run_name="nightly-2",
    )
    repository.save_snapshot(project_name="QA", project_slug="qa", snapshot=first)
    repository.save_snapshot(project_name="QA", project_slug="qa", snapshot=second)

    loaded = repository.load_eval_run("qa", second.eval_run.id)
    paged = repository.list_eval_runs("qa", limit=1, offset=1)

    assert loaded is not None
    assert loaded.id == second.eval_run.id
    assert len(paged) == 1


def test_file_repository_saves_and_loads_dataset_version(tmp_path):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )

    path = repository.save_dataset_version(
        project_name="QA",
        project_slug="qa",
        dataset_version=snapshot.dataset_version,
    )
    loaded = repository.load_dataset_version("qa", snapshot.dataset_version.id)

    assert path.exists()
    assert loaded is not None
    assert loaded.id == snapshot.dataset_version.id
    assert loaded.item_count == 1


def test_file_repository_saves_alert_rule_and_triggers_event(tmp_path):
    repository = FileCoreRepository(tmp_path)
    rule = AlertRuleRecord(
        id="rule-pass-rate",
        name="Pass rate below 90",
        metric_key="pass_rate",
        operator="<",
        threshold=90.0,
        severity=AlertSeverity.WARNING,
    )
    repository.save_alert_rule(
        project_name="QA Project",
        project_slug="qa-project",
        alert_rule=rule,
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
