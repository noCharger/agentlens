import json

from agentlens.dataset.builder import (
    build_dataset_version_from_scenarios,
    compute_dataset_fingerprint,
    dataset_version_to_scenarios,
    load_dataset_version_from_path,
    make_deterministic_id_factory,
    write_dataset_version,
)
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.core.models import DatasetSource


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="tc-001",
        name="Case One",
        category="tool_calling",
        benchmark="gdpval-aa",
        evaluation_mode="llm_judge",
        input="Solve task one",
        setup=["echo prep"],
        expected=ExpectedResult(
            tools_called=["read_file"],
            max_steps=7,
            output_contains=["done"],
            max_tokens=2048,
        ),
        judge_rubric="accuracy",
        judge_rubric_text="Score by correctness",
        judge_threshold=3.5,
        reference_answer="gold answer",
        metadata={"owner": "qa"},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_build_dataset_version_from_scenarios_roundtrip():
    scenarios = [
        _make_scenario(),
        _make_scenario(id="tc-002", name="Case Two", input="Solve task two"),
    ]
    dataset = build_dataset_version_from_scenarios(
        scenarios,
        name="nightly",
        source=DatasetSource.BENCHMARK_IMPORT,
    )

    rebuilt = dataset_version_to_scenarios(dataset)

    assert dataset.name == "nightly"
    assert dataset.source == DatasetSource.BENCHMARK_IMPORT
    assert dataset.item_count == 2
    assert rebuilt[0].expected.max_steps == 7
    assert rebuilt[0].expected.max_tokens == 2048
    assert rebuilt[0].setup_commands == ["echo prep"]
    assert rebuilt[0].judge_threshold == 3.5
    assert rebuilt[1].id == "tc-002"


def test_write_and_load_dataset_version_file(tmp_path):
    dataset = build_dataset_version_from_scenarios([_make_scenario()], name="triage")
    path = tmp_path / "dataset-version.json"
    write_dataset_version(dataset, path)

    loaded = load_dataset_version_from_path(path)
    assert loaded.id == dataset.id
    assert loaded.items[0].scenario_id == "tc-001"

    snapshot_like = tmp_path / "snapshot-like.json"
    snapshot_like.write_text(
        json.dumps({"dataset_version": dataset.model_dump(mode="json")}),
        encoding="utf-8",
    )
    loaded_from_snapshot_like = load_dataset_version_from_path(snapshot_like)
    assert loaded_from_snapshot_like.id == dataset.id


def test_dataset_fingerprint_and_deterministic_ids_are_stable():
    scenarios = [
        _make_scenario(),
        _make_scenario(id="tc-002", name="Case Two", input="Solve task two"),
    ]
    fingerprint_a = compute_dataset_fingerprint(scenarios)
    fingerprint_b = compute_dataset_fingerprint(scenarios)

    first = build_dataset_version_from_scenarios(
        scenarios,
        name="nightly",
        id_factory=make_deterministic_id_factory(fingerprint_a),
        dataset_fingerprint=fingerprint_a,
    )
    second = build_dataset_version_from_scenarios(
        scenarios,
        name="nightly",
        id_factory=make_deterministic_id_factory(fingerprint_b),
        dataset_fingerprint=fingerprint_b,
    )

    assert fingerprint_a == fingerprint_b
    assert first.id == second.id
    assert first.dataset_id == second.dataset_id
    assert [item.id for item in first.items] == [item.id for item in second.items]
    assert first.metadata["dataset_fingerprint"] == fingerprint_a
