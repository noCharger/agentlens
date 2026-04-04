import pytest
import json

from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.sandbox import (
    BENCHMARK_SANDBOX_PROFILES,
    BenchmarkSandboxProfile,
    GuardedShellTool,
    build_shell_sandbox_policy,
    prepare_benchmark_environment,
    validate_shell_commands,
    SandboxViolationError,
)


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="gdpval-aa-case-001",
        name="GDPval case",
        category="analysis",
        benchmark="gdpval-aa",
        evaluation_mode="llm_judge",
        input="Analyze spreadsheet",
        setup=[],
        expected=ExpectedResult(max_steps=10),
        judge_rubric_text="Score quality from 1-5.",
        metadata={},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_build_shell_sandbox_policy_disabled_without_benchmark():
    scenario = _make_scenario(benchmark="")
    policy = build_shell_sandbox_policy(scenario)
    assert policy.enabled is False


def test_validate_shell_commands_blocks_pip_for_benchmark(tmp_path):
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    with pytest.raises(SandboxViolationError, match="blocked"):
        validate_shell_commands("pip install pandas", policy)


def test_validate_shell_commands_blocks_python_module_pip_for_benchmark(tmp_path):
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    with pytest.raises(SandboxViolationError, match="installer commands are blocked"):
        validate_shell_commands("python3 -m pip install pandas", policy)


def test_validate_shell_commands_allows_python_c_with_semicolon(tmp_path):
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    validate_shell_commands(
        'python3 -c "import pandas as pd; print(pd.__version__)"',
        policy,
    )


def test_validate_shell_commands_blocks_out_of_root_absolute_path(tmp_path):
    scenario = _make_scenario(
        metadata={
            "resolved_reference_files": [str((tmp_path / "data/benchmarks/gdpval-aa/ref.xlsx").resolve())],
            "resolved_deliverable_files": [str((tmp_path / "data/benchmarks/gdpval-aa/out.xlsx").resolve())],
        }
    )
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    with pytest.raises(SandboxViolationError, match="outside sandbox roots"):
        validate_shell_commands("ls /etc", policy)


def test_validate_shell_commands_allows_benchmark_paths(tmp_path):
    benchmark_root = (tmp_path / "data" / "benchmarks" / "gdpval-aa")
    target = benchmark_root / "reference_files" / "foo.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    scenario = _make_scenario(
        metadata={
            "resolved_reference_files": [str(target.resolve())],
            "resolved_deliverable_files": [str((benchmark_root / "deliverable_files" / "bar.xlsx").resolve())],
        }
    )
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    validate_shell_commands(f"ls {target}", policy)


def test_prepare_benchmark_environment_reports_missing_modules(monkeypatch):
    scenario = _make_scenario(benchmark="custombench")
    monkeypatch.setitem(
        BENCHMARK_SANDBOX_PROFILES,
        "custombench",
        BenchmarkSandboxProfile(
            allowed_commands=("python",),
            blocked_commands=("pip",),
            required_python_modules=("module_that_does_not_exist_abcxyz",),
        ),
    )

    message = prepare_benchmark_environment(scenario)
    assert message is not None
    assert "missing Python modules" in message


def test_sandbox_profile_override_can_allow_pip(tmp_path):
    benchmark_root = tmp_path / "data" / "benchmarks" / "gdpval-aa"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "sandbox_profile.json").write_text(
        json.dumps(
            {
                "allowed_commands": ["python", "pip"],
                "blocked_commands": [],
            }
        )
    )
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    validate_shell_commands("pip install openpyxl", policy)


def test_sandbox_profile_override_validates_json_shape(tmp_path):
    benchmark_root = tmp_path / "data" / "benchmarks" / "gdpval-aa"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "sandbox_profile.json").write_text('{"allowed_commands": "pip"}')
    scenario = _make_scenario()

    with pytest.raises(SandboxViolationError, match="must be a list of strings"):
        build_shell_sandbox_policy(scenario, workspace_root=tmp_path)


def test_validate_shell_commands_still_blocks_pip_in_compound_segment(tmp_path):
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    with pytest.raises(SandboxViolationError, match="blocked"):
        validate_shell_commands("python3 -c \"print('ok')\"; pip install pandas", policy)


def test_guarded_shell_tool_returns_violation_message_for_blocked_command(tmp_path):
    scenario = _make_scenario()
    policy = build_shell_sandbox_policy(scenario, workspace_root=tmp_path)
    tool = GuardedShellTool(shell_policy=policy)

    result = tool._run("open /tmp/demo.txt")

    assert "SandboxViolation" in result
    assert "blocked in benchmark sandbox mode" in result
