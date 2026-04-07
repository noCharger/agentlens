"""Scenario data model and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from agentlens.eval.benchmarks import (
    display_benchmark_name,
    infer_benchmark_from_path,
    normalize_benchmark_name,
)

EVALUATION_MODES = {"deterministic", "llm_judge", "external"}


class ExpectedToolParam(BaseModel):
    """Expected tool parameter specification for validation."""
    tool_name: str
    param_name: str
    required: bool = True
    expected_value: str | None = None
    forbidden_values: list[str] = Field(default_factory=list)


class ExpectedResult(BaseModel):
    tools_called: list[str] = Field(default_factory=list)
    tool_params: list[ExpectedToolParam] = Field(default_factory=list)
    max_steps: int = Field(default=10)
    min_steps: int = Field(default=0)
    output_contains: list[str] = Field(default_factory=list)
    max_tokens: int | None = None
    expected_escalation: bool = False
    max_steps_after_answer: int = 1
    safety_checks: bool = True
    forbidden_patterns: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    name: str
    category: str
    benchmark: str = ""
    evaluation_mode: str = "deterministic"
    input_query: str = Field(alias="input")
    setup_commands: list[str] = Field(default_factory=list, alias="setup")
    expected: ExpectedResult
    judge_rubric: str = ""
    judge_rubric_text: str = ""
    judge_threshold: float = 4.0
    reference_answer: str = ""
    context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @field_validator("benchmark", mode="before")
    @classmethod
    def _normalize_benchmark(cls, value: object) -> str:
        if value in (None, ""):
            return ""
        return normalize_benchmark_name(str(value))

    @field_validator("evaluation_mode", mode="before")
    @classmethod
    def _normalize_evaluation_mode(cls, value: object) -> str:
        if value in (None, ""):
            return "deterministic"
        mode = str(value).strip().casefold()
        if mode not in EVALUATION_MODES:
            raise ValueError(
                f"Unsupported evaluation_mode '{value}'. Expected one of {sorted(EVALUATION_MODES)}"
            )
        return mode

    @property
    def benchmark_name(self) -> str:
        return display_benchmark_name(self.benchmark)

    @classmethod
    def from_dict(cls, data: dict, path: Path | None = None) -> Scenario:
        input_data = data.get("input", {})
        query = input_data if isinstance(input_data, str) else input_data.get("query", "")
        setup = [] if isinstance(input_data, str) else input_data.get("setup", [])
        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            benchmark=data.get("benchmark") or infer_benchmark_from_path(path),
            evaluation_mode=data.get("evaluation_mode", "deterministic"),
            input=query,
            setup=setup,
            expected=ExpectedResult(**data.get("expected", {})),
            judge_rubric=data.get("judge_rubric", ""),
            judge_rubric_text=data.get("judge_rubric_text", ""),
            judge_threshold=data.get("judge_threshold", 4.0),
            reference_answer=data.get("reference_answer", ""),
            context=data.get("context", []),
            metadata=data.get("metadata", {}),
        )


def load_scenario(path: Path) -> Scenario:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Scenario.from_dict(data, path=path)


def load_scenarios_from_dir(
    directory: Path,
    *,
    exclude_dir_names: set[str] | None = None,
) -> list[Scenario]:
    scenarios = []
    excluded = exclude_dir_names or set()
    for yaml_file in sorted(directory.rglob("*.yaml")):
        if any(part in excluded for part in yaml_file.parts):
            continue
        scenarios.append(load_scenario(yaml_file))
    return scenarios


def load_runtime_scenarios(
    scenarios_dir: Path,
    *,
    benchmark_data_root: Path | None = None,
    benchmarks: list[str] | None = None,
) -> list[Scenario]:
    scenarios = load_scenarios_from_dir(
        scenarios_dir,
        exclude_dir_names={"benchmarks"},
    )
    if not benchmark_data_root or not benchmark_data_root.exists():
        return scenarios

    from agentlens.eval.importers import load_downloaded_benchmark_scenarios

    dynamic_scenarios = load_downloaded_benchmark_scenarios(
        benchmark_data_root,
        benchmarks=benchmarks,
    )
    yaml_ids = {scenario.id for scenario in scenarios}
    overlap = yaml_ids.intersection(scenario.id for scenario in dynamic_scenarios)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Duplicate scenario ids between YAML and downloaded benchmarks: {sample}")
    return scenarios + dynamic_scenarios
