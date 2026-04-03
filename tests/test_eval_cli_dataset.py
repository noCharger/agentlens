from types import SimpleNamespace

from agentlens.dataset.builder import (
    build_dataset_version_from_scenarios,
    compute_dataset_fingerprint,
    make_deterministic_id_factory,
    write_dataset_version,
)
from agentlens.eval.__main__ import (
    _persist_runtime_dataset_version,
    _resolve_eval_dataset_and_scenarios,
)
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.core.repository import FileCoreRepository


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


def test_resolve_eval_dataset_and_scenarios_from_dataset_file(tmp_path):
    dataset_version = build_dataset_version_from_scenarios(
        [_make_scenario(), _make_scenario(id="tc-002", name="Case Two")],
        name="triage",
    )
    dataset_path = tmp_path / "dataset-version.json"
    write_dataset_version(dataset_version, dataset_path)

    args = SimpleNamespace(
        dataset_version_file=dataset_path,
        dataset_version_id=None,
        scenario_id="tc-002",
        benchmark=[],
        platform_store=None,
        platform_sqlite=None,
        platform_project=None,
        platform_project_slug=None,
        scenarios=tmp_path / "unused",
        benchmark_data_root=tmp_path / "unused",
        platform_dataset_name=None,
    )

    resolved_dataset, scenarios = _resolve_eval_dataset_and_scenarios(args)

    assert resolved_dataset.id == dataset_version.id
    assert len(scenarios) == 1
    assert scenarios[0].id == "tc-002"


def test_resolve_eval_dataset_and_scenarios_from_platform_store(tmp_path):
    repository = FileCoreRepository(tmp_path / "store")
    dataset_version = build_dataset_version_from_scenarios(
        [_make_scenario(), _make_scenario(id="tc-002", name="Case Two")],
        name="triage",
    )
    repository.save_dataset_version(
        project_name="QA Project",
        project_slug="qa-project",
        dataset_version=dataset_version,
    )

    args = SimpleNamespace(
        dataset_version_file=None,
        dataset_version_id=dataset_version.id,
        scenario_id=None,
        benchmark=[],
        platform_store=tmp_path / "store",
        platform_sqlite=None,
        platform_project="QA Project",
        platform_project_slug="qa-project",
        scenarios=tmp_path / "unused",
        benchmark_data_root=tmp_path / "unused",
        platform_dataset_name=None,
    )

    resolved_dataset, scenarios = _resolve_eval_dataset_and_scenarios(args)

    assert resolved_dataset.id == dataset_version.id
    assert len(scenarios) == 2


def test_resolve_eval_dataset_and_scenarios_reuses_existing_runtime_dataset(tmp_path, monkeypatch):
    scenarios_source = [_make_scenario(), _make_scenario(id="tc-002", name="Case Two")]
    fingerprint = compute_dataset_fingerprint(scenarios_source)
    deterministic = build_dataset_version_from_scenarios(
        scenarios_source,
        name="gdpval-aa-dataset",
        id_factory=make_deterministic_id_factory(fingerprint),
        dataset_fingerprint=fingerprint,
    )

    repository = FileCoreRepository(tmp_path / "store")
    repository.save_dataset_version(
        project_name="gdpval-aa project",
        project_slug="gdpval-aa-project",
        dataset_version=deterministic,
    )

    monkeypatch.setattr(
        "agentlens.eval.__main__.load_runtime_scenarios",
        lambda *args, **kwargs: list(scenarios_source),
    )

    args = SimpleNamespace(
        dataset_version_file=None,
        dataset_version_id=None,
        scenario_id=None,
        benchmark=["gdpval-aa"],
        platform_store=tmp_path / "store",
        platform_sqlite=None,
        platform_project=None,
        platform_project_slug=None,
        scenarios=tmp_path / "unused",
        benchmark_data_root=tmp_path / "unused",
        platform_dataset_name=None,
    )

    resolved_dataset, scenarios = _resolve_eval_dataset_and_scenarios(args)

    assert resolved_dataset.id == deterministic.id
    assert len(scenarios) == 2


def test_persist_runtime_dataset_version_saves_when_missing(tmp_path):
    scenarios_source = [_make_scenario()]
    fingerprint = compute_dataset_fingerprint(scenarios_source)
    dataset_version = build_dataset_version_from_scenarios(
        scenarios_source,
        name="triage",
        id_factory=make_deterministic_id_factory(fingerprint),
        dataset_fingerprint=fingerprint,
    )

    args = SimpleNamespace(
        dataset_version_file=None,
        dataset_version_id=None,
        benchmark=[],
        platform_store=tmp_path / "store",
        platform_sqlite=None,
        platform_project="QA Project",
        platform_project_slug="qa-project",
    )

    repository = FileCoreRepository(tmp_path / "store")
    assert repository.load_dataset_version("qa-project", dataset_version.id) is None
    _persist_runtime_dataset_version(args, dataset_version)
    loaded = repository.load_dataset_version("qa-project", dataset_version.id)
    assert loaded is not None
    assert loaded.id == dataset_version.id
