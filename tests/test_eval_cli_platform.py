import json
from types import SimpleNamespace

from agentlens.dataset.builder import build_dataset_version_from_scenarios
from agentlens.eval.__main__ import _handle_platform_outputs

from tests.test_platform_exporters import _make_result


def test_handle_platform_outputs_writes_export_and_store(tmp_path, monkeypatch):
    export_path = tmp_path / "snapshot.json"
    store_path = tmp_path / "store"
    sqlite_path = tmp_path / "platform.db"
    args = SimpleNamespace(
        benchmark=["gdpval-aa"],
        level2=False,
        platform_export=export_path,
        platform_dataset_name="triage",
        platform_run_name="nightly",
        platform_store=store_path,
        platform_sqlite=sqlite_path,
        platform_project="QA Project",
        platform_project_slug="qa-project",
        platform_idempotency_key=None,
    )
    settings = SimpleNamespace(
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    messages: list[str] = []

    class DummyConsole:
        def print(self, message):
            messages.append(str(message))

    monkeypatch.setattr("agentlens.eval.__main__.console", DummyConsole())

    _handle_platform_outputs(args, [_make_result()], settings)

    assert export_path.exists()
    assert (store_path / "projects" / "qa-project" / "project.json").exists()
    assert sqlite_path.exists()
    assert any("Platform snapshot saved to" in message for message in messages)
    assert any("Platform records stored in" in message for message in messages)


def test_handle_platform_outputs_reuses_dataset_version_record(tmp_path):
    export_path = tmp_path / "snapshot.json"
    result = _make_result()
    dataset_version = build_dataset_version_from_scenarios(
        [result.scenario],
        name="golden",
    )
    args = SimpleNamespace(
        benchmark=[],
        level2=False,
        platform_export=export_path,
        platform_dataset_name=None,
        platform_run_name="nightly",
        platform_store=None,
        platform_sqlite=None,
        platform_project=None,
        platform_project_slug=None,
        platform_idempotency_key=None,
    )
    settings = SimpleNamespace(
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    _handle_platform_outputs(
        args,
        [result],
        settings,
        dataset_version=dataset_version,
    )

    payload = json.loads(export_path.read_text())
    assert payload["dataset_version"]["id"] == dataset_version.id
    assert payload["eval_run"]["dataset_version_id"] == dataset_version.id
