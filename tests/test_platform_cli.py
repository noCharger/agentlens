from agentlens.core.__main__ import main
from agentlens.core.exporters import build_closed_loop_snapshot
from agentlens.core.models import AlertRuleRecord, AlertSeverity
from agentlens.core.repository import FileCoreRepository
from agentlens.core.sqlite_repository import SQLiteCoreRepository

from tests.test_platform_exporters import _make_result


def test_platform_cli_lists_projects(tmp_path, capsys, monkeypatch):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=snapshot)

    monkeypatch.setattr(
        "sys.argv",
        ["agentlens.core", "--store", str(tmp_path), "--list-projects"],
    )
    main()
    output = capsys.readouterr().out

    assert "qa-project" in output
    assert "QA Project" in output


def test_platform_cli_lists_eval_runs(tmp_path, capsys, monkeypatch):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=snapshot)

    monkeypatch.setattr(
        "sys.argv",
        ["agentlens.core", "--store", str(tmp_path), "--project", "qa-project", "--list-eval-runs"],
    )
    main()
    output = capsys.readouterr().out

    assert "nightly" in output
    assert snapshot.eval_run.id in output


def test_platform_cli_lists_projects_from_sqlite(tmp_path, capsys, monkeypatch):
    repository = SQLiteCoreRepository(tmp_path / "platform.db")
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=snapshot)

    monkeypatch.setattr(
        "sys.argv",
        ["agentlens.core", "--sqlite", str(tmp_path / "platform.db"), "--list-projects"],
    )
    main()
    output = capsys.readouterr().out

    assert "qa-project" in output


def test_platform_cli_lists_dataset_versions(tmp_path, capsys, monkeypatch):
    repository = FileCoreRepository(tmp_path)
    snapshot = build_closed_loop_snapshot(
        [_make_result()],
        dataset_name="triage",
        run_name="nightly",
    )
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=snapshot)

    monkeypatch.setattr(
        "sys.argv",
        [
            "agentlens.core",
            "--store",
            str(tmp_path),
            "--project",
            "qa-project",
            "--list-dataset-versions",
        ],
    )
    main()
    output = capsys.readouterr().out

    assert snapshot.dataset_version.id in output
    assert "triage" in output


def test_platform_cli_lists_alert_rules_and_events(tmp_path, capsys, monkeypatch):
    repository = FileCoreRepository(tmp_path)
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
    repository.save_snapshot(project_name="QA Project", project_slug="qa-project", snapshot=snapshot)

    monkeypatch.setattr(
        "sys.argv",
        [
            "agentlens.core",
            "--store",
            str(tmp_path),
            "--project",
            "qa-project",
            "--list-alert-rules",
        ],
    )
    main()
    rules_output = capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv",
        [
            "agentlens.core",
            "--store",
            str(tmp_path),
            "--project",
            "qa-project",
            "--list-alert-events",
        ],
    )
    main()
    events_output = capsys.readouterr().out

    assert "Alert Rules for qa-project" in rules_output
    assert "Pass rate" in rules_output
    assert "warning" in rules_output
    assert "Alert Events for qa-project" in events_output
    assert "warning" in events_output
