import tempfile
from pathlib import Path

import pytest
import yaml

from agentlens.eval.scenarios import Scenario, ExpectedResult, load_scenario, load_scenarios_from_dir


def _write_yaml(path: Path, data: dict) -> Path:
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _sample_data():
    return {
        "id": "tc-001",
        "name": "Test Scenario",
        "category": "tool_calling",
        "input": {
            "query": "Read /tmp/test.txt",
            "setup": ["echo hello > /tmp/test.txt"],
        },
        "expected": {
            "tools_called": ["read_file"],
            "max_steps": 3,
            "output_contains": ["hello"],
        },
        "judge_rubric": "accuracy",
        "reference_answer": "hello",
    }


def test_scenario_from_dict():
    data = _sample_data()
    s = Scenario.from_dict(data)
    assert s.id == "tc-001"
    assert s.name == "Test Scenario"
    assert s.category == "tool_calling"
    assert s.input_query == "Read /tmp/test.txt"
    assert s.setup_commands == ["echo hello > /tmp/test.txt"]
    assert s.expected.tools_called == ["read_file"]
    assert s.expected.max_steps == 3
    assert s.expected.output_contains == ["hello"]
    assert s.judge_rubric == "accuracy"
    assert s.reference_answer == "hello"


def test_scenario_defaults():
    data = {
        "id": "min",
        "name": "Minimal",
        "category": "test",
        "input": {"query": "hi"},
        "expected": {},
    }
    s = Scenario.from_dict(data)
    assert s.setup_commands == []
    assert s.expected.tools_called == []
    assert s.expected.max_steps == 10
    assert s.expected.output_contains == []
    assert s.expected.max_tokens is None
    assert s.judge_rubric == ""
    assert s.reference_answer == ""


def test_scenario_string_input():
    data = {
        "id": "str",
        "name": "String input",
        "category": "test",
        "input": "just a string query",
        "expected": {"tools_called": ["shell"]},
    }
    s = Scenario.from_dict(data)
    assert s.input_query == "just a string query"
    assert s.setup_commands == []


def test_load_scenario_from_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        _write_yaml(path, _sample_data())
        s = load_scenario(path)
        assert s.id == "tc-001"
        assert s.expected.tools_called == ["read_file"]


def test_load_scenarios_from_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _write_yaml(d / "a.yaml", {**_sample_data(), "id": "a"})
        _write_yaml(d / "b.yaml", {**_sample_data(), "id": "b"})
        scenarios = load_scenarios_from_dir(d)
        assert len(scenarios) == 2
        ids = {s.id for s in scenarios}
        assert ids == {"a", "b"}


def test_load_scenarios_from_nested_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        sub = d / "sub"
        sub.mkdir()
        _write_yaml(sub / "c.yaml", {**_sample_data(), "id": "c"})
        scenarios = load_scenarios_from_dir(d)
        assert len(scenarios) == 1
        assert scenarios[0].id == "c"


def test_load_real_scenarios():
    """Load the actual scenario files shipped with the project."""
    scenarios_dir = Path(__file__).parent.parent / "src" / "agentlens" / "scenarios"
    if not scenarios_dir.exists():
        pytest.skip("Scenarios directory not found")
    scenarios = load_scenarios_from_dir(scenarios_dir)
    assert len(scenarios) >= 7
    ids = {s.id for s in scenarios}
    assert "tc-001" in ids
    assert "rc-001" in ids
