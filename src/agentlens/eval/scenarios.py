"""Scenario data model and YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ExpectedResult(BaseModel):
    tools_called: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10)
    output_contains: list[str] = Field(default_factory=list)
    max_tokens: int | None = None


class Scenario(BaseModel):
    id: str
    name: str
    category: str
    input_query: str = Field(alias="input")
    setup_commands: list[str] = Field(default_factory=list, alias="setup")
    expected: ExpectedResult
    judge_rubric: str = ""
    reference_answer: str = ""

    model_config = {"populate_by_name": True}

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        input_data = data.get("input", {})
        query = input_data if isinstance(input_data, str) else input_data.get("query", "")
        setup = [] if isinstance(input_data, str) else input_data.get("setup", [])
        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            input=query,
            setup=setup,
            expected=ExpectedResult(**data.get("expected", {})),
            judge_rubric=data.get("judge_rubric", ""),
            reference_answer=data.get("reference_answer", ""),
        )


def load_scenario(path: Path) -> Scenario:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Scenario.from_dict(data)


def load_scenarios_from_dir(directory: Path) -> list[Scenario]:
    scenarios = []
    for yaml_file in sorted(directory.rglob("*.yaml")):
        scenarios.append(load_scenario(yaml_file))
    return scenarios
