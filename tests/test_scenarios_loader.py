import tempfile
from pathlib import Path

import pytest
import yaml

from agentlens.eval.scenarios import (
    Scenario,
    load_runtime_scenarios,
    load_scenario,
    load_scenarios_from_dir,
)


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
    assert s.evaluation_mode == "deterministic"
    assert s.judge_rubric == "accuracy"
    assert s.judge_rubric_text == ""
    assert s.judge_threshold == 4.0
    assert s.reference_answer == "hello"
    assert s.metadata == {}


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
    assert s.evaluation_mode == "deterministic"
    assert s.judge_rubric == ""
    assert s.judge_rubric_text == ""
    assert s.judge_threshold == 4.0
    assert s.reference_answer == ""
    assert s.metadata == {}


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


def test_scenario_normalizes_benchmark_name():
    data = {**_sample_data(), "benchmark": "SWE Bench Pro"}
    s = Scenario.from_dict(data)
    assert s.benchmark == "swe-bench-pro"
    assert s.benchmark_name == "SWE Bench Pro"


def test_scenario_accepts_llm_judge_mode_and_metadata():
    data = {
        **_sample_data(),
        "evaluation_mode": "llm_judge",
        "judge_rubric_text": "Score task completion from 1-5.",
        "judge_threshold": 3.5,
        "metadata": {"source": "import"},
    }
    s = Scenario.from_dict(data)
    assert s.evaluation_mode == "llm_judge"
    assert s.judge_rubric_text == "Score task completion from 1-5."
    assert s.judge_threshold == 3.5
    assert s.metadata == {"source": "import"}


def test_scenario_rejects_unknown_evaluation_mode():
    data = {**_sample_data(), "evaluation_mode": "totally_custom"}
    with pytest.raises(ValueError, match="Unsupported evaluation_mode"):
        Scenario.from_dict(data)


def test_load_scenario_from_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        _write_yaml(path, _sample_data())
        s = load_scenario(path)
        assert s.id == "tc-001"
        assert s.expected.tools_called == ["read_file"]


def test_load_scenario_infers_benchmark_from_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "benchmarks" / "multi-swe-bench" / "test.yaml"
        path.parent.mkdir(parents=True)
        _write_yaml(path, _sample_data())

        s = load_scenario(path)
        assert s.benchmark == "multi-swe-bench"
        assert s.benchmark_name == "Multi-SWE Bench"


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


def test_load_scenarios_from_dir_can_exclude_benchmarks_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        bench = d / "benchmarks" / "gdpval-aa"
        bench.mkdir(parents=True)
        _write_yaml(d / "root.yaml", {**_sample_data(), "id": "root"})
        _write_yaml(bench / "bench.yaml", {**_sample_data(), "id": "bench"})

        scenarios = load_scenarios_from_dir(d, exclude_dir_names={"benchmarks"})

        assert [scenario.id for scenario in scenarios] == ["root"]


def test_load_runtime_scenarios_merges_downloaded_benchmarks():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        scenarios_dir = root / "scenarios"
        data_root = root / "data" / "benchmarks"
        scenarios_dir.mkdir(parents=True)
        (data_root / "gdpval-aa").mkdir(parents=True)

        _write_yaml(scenarios_dir / "a.yaml", {**_sample_data(), "id": "a"})
        (data_root / "gdpval-aa" / "sample.yaml").write_text(
            yaml.safe_dump(
                [
                    {
                        "task_id": "task-001",
                        "prompt": "Analyze the document.",
                        "rubric_pretty": "[+5] Good analysis",
                        "reference_files": ["brief.md"],
                    }
                ]
            )
        )

        scenarios = load_runtime_scenarios(
            scenarios_dir,
            benchmark_data_root=data_root,
        )

        ids = {scenario.id for scenario in scenarios}
        assert "a" in ids
        assert "gdpval-aa-task-001" in ids


def test_load_real_scenarios():
    """Load the actual scenario files shipped with the project."""
    scenarios_dir = Path(__file__).parent.parent / "src" / "agentlens" / "scenarios"
    if not scenarios_dir.exists():
        pytest.skip("Scenarios directory not found")
    scenarios = load_scenarios_from_dir(
        scenarios_dir,
        exclude_dir_names={"benchmarks"},
    )
    assert len(scenarios) >= 7
    ids = {s.id for s in scenarios}
    assert "tc-001" in ids
    assert "rc-001" in ids
