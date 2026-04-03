"""Dataset pipeline helpers for converting scenarios into immutable dataset versions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.core.models import (
    DatasetItemRecord,
    DatasetSource,
    DatasetVersionRecord,
    utc_now,
)

IdFactory = Callable[[str], str]


def _default_id_factory(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _make_id(prefix: str, id_factory: IdFactory | None) -> str:
    factory = id_factory or _default_id_factory
    return factory(prefix)


def make_deterministic_id_factory(seed: str) -> IdFactory:
    counters: dict[str, int] = {}

    def _factory(prefix: str) -> str:
        count = counters.get(prefix, 0)
        counters[prefix] = count + 1
        digest = hashlib.sha256(f"{seed}:{prefix}:{count}".encode("utf-8")).hexdigest()
        return f"{prefix}_{digest[:12]}"

    return _factory


def _scenario_fingerprint_payload(scenario: Scenario) -> dict[str, object]:
    return {
        "id": scenario.id,
        "name": scenario.name,
        "category": scenario.category,
        "benchmark": scenario.benchmark,
        "evaluation_mode": scenario.evaluation_mode,
        "input_query": scenario.input_query,
        "setup_commands": list(scenario.setup_commands),
        "expected": {
            "tools_called": list(scenario.expected.tools_called),
            "max_steps": scenario.expected.max_steps,
            "output_contains": list(scenario.expected.output_contains),
            "max_tokens": scenario.expected.max_tokens,
        },
        "judge_rubric": scenario.judge_rubric,
        "judge_rubric_text": scenario.judge_rubric_text,
        "judge_threshold": scenario.judge_threshold,
        "reference_answer": scenario.reference_answer,
        "metadata": dict(scenario.metadata),
    }


def compute_dataset_fingerprint(scenarios: list[Scenario]) -> str:
    payload = [_scenario_fingerprint_payload(scenario) for scenario in scenarios]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_dataset_version_from_scenarios(
    scenarios: list[Scenario],
    *,
    name: str,
    version: str = "v1",
    source: DatasetSource = DatasetSource.MANUAL_CURATION,
    dataset_id: str | None = None,
    created_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
    dataset_fingerprint: str | None = None,
    id_factory: IdFactory | None = None,
) -> DatasetVersionRecord:
    timestamp = created_at or utc_now()
    fingerprint = dataset_fingerprint or compute_dataset_fingerprint(scenarios)
    dataset_version_id = _make_id("dataset_version", id_factory)
    resolved_dataset_id = dataset_id or _make_id("dataset", id_factory)

    items: list[DatasetItemRecord] = []
    for scenario in scenarios:
        items.append(
            DatasetItemRecord(
                id=_make_id("dataset_item", id_factory),
                dataset_version_id=dataset_version_id,
                scenario_id=scenario.id,
                name=scenario.name,
                category=scenario.category,
                benchmark=scenario.benchmark,
                evaluation_mode=scenario.evaluation_mode,
                input_query=scenario.input_query,
                setup_commands=list(scenario.setup_commands),
                max_steps=scenario.expected.max_steps,
                max_tokens=scenario.expected.max_tokens,
                reference_answer=scenario.reference_answer,
                expected_tools=list(scenario.expected.tools_called),
                expected_output_contains=list(scenario.expected.output_contains),
                judge_rubric=scenario.judge_rubric,
                judge_rubric_text=scenario.judge_rubric_text,
                judge_threshold=scenario.judge_threshold,
                metadata=dict(scenario.metadata),
            )
        )

    resolved_metadata = dict(metadata or {})
    resolved_metadata.setdefault("dataset_fingerprint", fingerprint)

    return DatasetVersionRecord(
        id=dataset_version_id,
        dataset_id=resolved_dataset_id,
        name=name,
        version=version,
        source=source,
        created_at=timestamp,
        items=items,
        metadata=resolved_metadata,
    )


def dataset_item_to_scenario(item: DatasetItemRecord) -> Scenario:
    return Scenario(
        id=item.scenario_id,
        name=item.name,
        category=item.category,
        benchmark=item.benchmark,
        evaluation_mode=item.evaluation_mode,
        input=item.input_query,
        setup=item.setup_commands,
        expected=ExpectedResult(
            tools_called=list(item.expected_tools),
            max_steps=item.max_steps,
            output_contains=list(item.expected_output_contains),
            max_tokens=item.max_tokens,
        ),
        judge_rubric=item.judge_rubric,
        judge_rubric_text=item.judge_rubric_text,
        judge_threshold=item.judge_threshold,
        reference_answer=item.reference_answer,
        metadata=dict(item.metadata),
    )


def dataset_version_to_scenarios(
    dataset_version: DatasetVersionRecord,
) -> list[Scenario]:
    return [dataset_item_to_scenario(item) for item in dataset_version.items]


def load_dataset_version_from_path(path: Path) -> DatasetVersionRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("dataset_version"), dict):
        payload = payload["dataset_version"]
    return DatasetVersionRecord.model_validate(payload)


def write_dataset_version(dataset_version: DatasetVersionRecord, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(dataset_version.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
