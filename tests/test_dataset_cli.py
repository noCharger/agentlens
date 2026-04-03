import json

from agentlens.dataset.__main__ import main
from agentlens.eval.scenarios import ExpectedResult, Scenario


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="tc-001",
        name="Case One",
        category="tool_calling",
        benchmark="gdpval-aa",
        evaluation_mode="deterministic",
        input="Do a thing",
        setup=[],
        expected=ExpectedResult(
            tools_called=["read_file"],
            max_steps=5,
            output_contains=["done"],
        ),
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_dataset_cli_builds_and_saves_output(tmp_path, monkeypatch):
    output_path = tmp_path / "dataset-version.json"
    store_path = tmp_path / "store"

    monkeypatch.setattr(
        "agentlens.dataset.__main__.load_runtime_scenarios",
        lambda *args, **kwargs: [_make_scenario()],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "agentlens.dataset",
            "--name",
            "triage",
            "--output",
            str(output_path),
            "--store",
            str(store_path),
            "--project",
            "QA Project",
            "--project-slug",
            "qa-project",
        ],
    )

    main()

    payload = json.loads(output_path.read_text())
    assert payload["name"] == "triage"
    dataset_id = payload["id"]
    assert (
        store_path
        / "projects"
        / "qa-project"
        / "dataset_versions"
        / f"{dataset_id}.json"
    ).exists()
