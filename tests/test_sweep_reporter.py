"""Tests for sweep HTML report generation."""

from __future__ import annotations

from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.sweep import ModelRun, SweepResult
from agentlens.eval.level3_human.sweep_reporter import (
    build_scenario_grid,
    generate_sweep_report,
)


def _make_scenario(sid: str, benchmark: str = "") -> Scenario:
    return Scenario(
        id=sid,
        name=f"Scenario {sid}",
        category="test",
        benchmark=benchmark,
        input="query",
        setup=[],
        expected=ExpectedResult(tools_called=[], output_contains=[]),
    )


def _make_eval_result(scenario: Scenario, *, passed: bool = True) -> EvalResult:
    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(passed, [], [], [], []),
            output_format=OutputFormatResult(passed, "ok" if passed else "", [], []),
            trajectory=TrajectoryResult(True, 2, 10, False, 100, 50, None, []),
        ),
        level2_scores={},
        feature_flags={},
    )


def _make_sweep(models: list[str], *, pass_matrix: list[list[bool]], benchmarks: list[str] | None = None) -> SweepResult:
    """Build a SweepResult from a matrix of pass/fail values (rows=models, cols=scenarios)."""
    n_scenarios = max(len(row) for row in pass_matrix)
    scenarios = [
        _make_scenario(f"s{i}", benchmark=benchmarks[i] if benchmarks else "")
        for i in range(n_scenarios)
    ]
    model_runs = []
    for model, pass_vals in zip(models, pass_matrix):
        results = [_make_eval_result(scenarios[i], passed=p) for i, p in enumerate(pass_vals)]
        model_runs.append(ModelRun(agent_model=model, results=results))
    return SweepResult(sweep_id="test-sweep-id", model_runs=model_runs)


class TestBuildScenarioGrid:
    def test_one_row_per_unique_scenario(self):
        sweep = _make_sweep(["gemini", "deepseek"], pass_matrix=[[True, False], [True, True]])
        grid = build_scenario_grid(sweep)
        assert len(grid) == 2

    def test_cells_contain_all_models(self):
        sweep = _make_sweep(["gemini", "deepseek"], pass_matrix=[[True, False], [False, True]])
        grid = build_scenario_grid(sweep)
        for row in grid:
            assert "gemini" in row.cells
            assert "deepseek" in row.cells

    def test_pass_mapped_correctly(self):
        sweep = _make_sweep(["m1"], pass_matrix=[[True, False]])
        grid = build_scenario_grid(sweep)
        assert grid[0].cells["m1"] == "PASS"
        assert grid[1].cells["m1"] == "FAIL"

    def test_missing_result_mapped_to_dash(self):
        # Model A has 2 scenarios, model B only ran 1 (quota)
        s1 = _make_scenario("s1")
        s2 = _make_scenario("s2")
        run_a = ModelRun(agent_model="a", results=[_make_eval_result(s1), _make_eval_result(s2)])
        run_b = ModelRun(agent_model="b", results=[_make_eval_result(s1)])
        sweep = SweepResult(sweep_id="x", model_runs=[run_a, run_b])
        grid = build_scenario_grid(sweep)
        s2_row = next(r for r in grid if r.scenario_id == "s2")
        assert s2_row.cells["b"] == "—"

    def test_benchmarked_scenarios_sorted_before_unassigned(self):
        benchmarks = ["toolathlon", "", "swe-bench-pro"]
        sweep = _make_sweep(["m"], pass_matrix=[[True, True, True]], benchmarks=benchmarks * 1)
        # We only have 1 row per scenario. Let's use 3 scenarios.
        scenarios = [
            _make_scenario("s0", benchmark="toolathlon"),
            _make_scenario("s1", benchmark=""),
            _make_scenario("s2", benchmark="swe-bench-pro"),
        ]
        run = ModelRun(agent_model="m", results=[_make_eval_result(s) for s in scenarios])
        sweep = SweepResult(sweep_id="x", model_runs=[run])
        grid = build_scenario_grid(sweep)
        # Unassigned (benchmark="") should sort last
        assert grid[-1].benchmark == ""


class TestGenerateSweepReport:
    def test_returns_html_string(self):
        sweep = _make_sweep(["gemini", "deepseek"], pass_matrix=[[True], [False]])
        html = generate_sweep_report(sweep)
        assert "<html" in html
        assert "AgentLens Sweep Report" in html

    def test_contains_all_model_names(self):
        sweep = _make_sweep(
            ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"],
            pass_matrix=[[True], [True]],
        )
        html = generate_sweep_report(sweep)
        assert "gemini:gemini-2.5-flash" in html
        assert "deepseek:deepseek-chat" in html

    def test_contains_sweep_id(self):
        sweep = _make_sweep(["m1"], pass_matrix=[[True]])
        html = generate_sweep_report(sweep)
        assert "test-sweep-id" in html

    def test_scenario_grid_contains_all_scenario_ids(self):
        sweep = _make_sweep(["m1", "m2"], pass_matrix=[[True, False], [False, True]])
        html = generate_sweep_report(sweep)
        assert "s0" in html
        assert "s1" in html

    def test_benchmark_breakdown_section_present_when_benchmarks_exist(self):
        scenarios = [
            _make_scenario("s0", benchmark="toolathlon"),
            _make_scenario("s1", benchmark="toolathlon"),
        ]
        run = ModelRun(
            agent_model="m",
            results=[_make_eval_result(s) for s in scenarios],
        )
        sweep = SweepResult(sweep_id="x", model_runs=[run])
        html = generate_sweep_report(sweep)
        assert "Per-Benchmark Breakdown" in html
        assert "toolathlon" in html.lower() or "Toolathlon" in html

    def test_no_benchmark_section_when_all_unassigned(self):
        sweep = _make_sweep(["m1"], pass_matrix=[[True]])
        html = generate_sweep_report(sweep)
        assert "Per-Benchmark Breakdown" not in html

    def test_pairwise_regression_section_present_for_two_models(self, monkeypatch):
        scenarios = [_make_scenario("s1"), _make_scenario("s2")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]
        from types import SimpleNamespace
        from agentlens.eval.sweep import run_sweep

        def fake_execute(scenario, settings, **kwargs):
            passed = settings.agent_model == "deepseek:deepseek-chat" or scenario.id == "s1"
            return _make_eval_result(scenario, passed=passed)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        def factory(model):
            return SimpleNamespace(agent_model=model)

        sweep = run_sweep(models, scenarios, factory)
        html = generate_sweep_report(sweep)
        assert "Pairwise Comparison" in html

    def test_no_pairwise_section_for_three_models(self):
        sweep = _make_sweep(["m1", "m2", "m3"], pass_matrix=[[True], [True], [True]])
        html = generate_sweep_report(sweep)
        assert "Pairwise Comparison" not in html

    def test_writes_file(self, tmp_path):
        output = tmp_path / "sweep.html"
        sweep = _make_sweep(["m1"], pass_matrix=[[True]])
        generate_sweep_report(sweep, output_path=output)
        assert output.exists()
        assert "<html" in output.read_text()

    def test_creates_parent_directory(self, tmp_path):
        output = tmp_path / "reports" / "nested" / "sweep.html"
        sweep = _make_sweep(["m1"], pass_matrix=[[True]])
        generate_sweep_report(sweep, output_path=output)
        assert output.exists()

    def test_pass_badge_in_grid(self):
        sweep = _make_sweep(["m1"], pass_matrix=[[True]])
        html = generate_sweep_report(sweep)
        assert "PASS" in html

    def test_fail_badge_in_grid(self):
        sweep = _make_sweep(["m1"], pass_matrix=[[False]])
        html = generate_sweep_report(sweep)
        assert "FAIL" in html
