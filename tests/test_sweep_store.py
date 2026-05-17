"""Tests for sweep snapshot persistence and trend comparison."""

from __future__ import annotations

import json

from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.sweep import ModelRun, SweepResult
from agentlens.eval.sweep_store import (
    SweepSnapshot,
    compare_sweeps,
    load_sweep,
    save_sweep,
    snapshot_from_sweep,
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


def _make_sweep(models: list[str], pass_matrix: list[list[bool]]) -> SweepResult:
    n = max(len(row) for row in pass_matrix)
    scenarios = [_make_scenario(f"s{i}") for i in range(n)]
    model_runs = []
    for model, pass_vals in zip(models, pass_matrix):
        results = [_make_eval_result(scenarios[i], passed=p) for i, p in enumerate(pass_vals)]
        model_runs.append(ModelRun(agent_model=model, results=results))
    return SweepResult(sweep_id="test-sweep", model_runs=model_runs)


class TestSnapshotFromSweep:
    def test_snapshot_has_correct_sweep_id(self):
        sweep = _make_sweep(["a"], [[True]])
        snap = snapshot_from_sweep(sweep)
        assert snap.sweep_id == "test-sweep"

    def test_snapshot_has_timestamp(self):
        sweep = _make_sweep(["a"], [[True]])
        snap = snapshot_from_sweep(sweep)
        assert snap.timestamp  # non-empty ISO string

    def test_one_model_run_snapshot_per_model(self):
        sweep = _make_sweep(["a", "b"], [[True], [False]])
        snap = snapshot_from_sweep(sweep)
        assert len(snap.model_run_snapshots) == 2

    def test_scenario_passed_correctly_extracted(self):
        sweep = _make_sweep(["a"], [[True, False]])
        snap = snapshot_from_sweep(sweep)
        run = snap.model_run_snapshots[0]
        pass_map = run.scenario_pass_map()
        assert pass_map["s0"] is True
        assert pass_map["s1"] is False

    def test_pass_rate_preserved(self):
        sweep = _make_sweep(["a"], [[True, True, False, False]])
        snap = snapshot_from_sweep(sweep)
        assert snap.model_run_snapshots[0].pass_rate == 50.0

    def test_retention_score_none_when_no_memory(self):
        sweep = _make_sweep(["a"], [[True]])
        snap = snapshot_from_sweep(sweep)
        assert snap.model_run_snapshots[0].scenario_snapshots[0].retention_score is None


class TestSaveAndLoadSweep:
    def test_round_trip_json(self, tmp_path):
        sweep = _make_sweep(["gemini", "deepseek"], [[True, False], [False, True]])
        path = tmp_path / "sweep.json"
        save_sweep(sweep, path)
        loaded = load_sweep(path)
        assert loaded.sweep_id == sweep.sweep_id
        assert len(loaded.model_run_snapshots) == 2

    def test_creates_parent_directories(self, tmp_path):
        sweep = _make_sweep(["a"], [[True]])
        path = tmp_path / "nested" / "dir" / "sweep.json"
        save_sweep(sweep, path)
        assert path.exists()

    def test_file_is_valid_json(self, tmp_path):
        sweep = _make_sweep(["a"], [[True]])
        path = tmp_path / "sweep.json"
        save_sweep(sweep, path)
        data = json.loads(path.read_text())
        assert "sweep_id" in data
        assert "model_run_snapshots" in data

    def test_scenario_snapshots_survive_round_trip(self, tmp_path):
        sweep = _make_sweep(["a"], [[True, False]])
        path = tmp_path / "sweep.json"
        save_sweep(sweep, path)
        loaded = load_sweep(path)
        run = loaded.model_run_snapshots[0]
        assert len(run.scenario_snapshots) == 2
        pass_map = run.scenario_pass_map()
        assert pass_map["s0"] is True
        assert pass_map["s1"] is False

    def test_save_returns_snapshot(self, tmp_path):
        sweep = _make_sweep(["a"], [[True]])
        snap = save_sweep(sweep, tmp_path / "sweep.json")
        assert isinstance(snap, SweepSnapshot)
        assert snap.sweep_id == sweep.sweep_id


class TestCompareSweeps:
    def _make_snapshot(self, models_pass: dict[str, list[bool]], sweep_id: str = "old") -> SweepSnapshot:
        models = list(models_pass)
        pass_matrix = list(models_pass.values())
        sweep = _make_sweep(models, pass_matrix)
        sweep.sweep_id = sweep_id
        return snapshot_from_sweep(sweep)

    def test_delta_pass_rate_positive_when_candidate_improves(self):
        baseline = self._make_snapshot({"gemini": [True, False]})   # 50%
        candidate = self._make_snapshot({"gemini": [True, True]}, sweep_id="new")  # 100%
        trend = compare_sweeps(baseline, candidate)
        gemini_trend = trend.model_trends[0]
        assert gemini_trend.delta_pass_rate == 50.0

    def test_new_regression_detected(self):
        baseline = self._make_snapshot({"a": [True, True]})
        candidate = self._make_snapshot({"a": [True, False]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        assert "s1" in trend.model_trends[0].new_regressions

    def test_new_improvement_detected(self):
        baseline = self._make_snapshot({"a": [False, True]})
        candidate = self._make_snapshot({"a": [True, True]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        assert "s0" in trend.model_trends[0].new_improvements

    def test_new_models_detected(self):
        baseline = self._make_snapshot({"a": [True]})
        candidate = self._make_snapshot({"a": [True], "b": [False]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        assert "b" in trend.new_models

    def test_dropped_models_detected(self):
        baseline = self._make_snapshot({"a": [True], "b": [True]})
        candidate = self._make_snapshot({"a": [True]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        assert "b" in trend.dropped_models

    def test_sweep_ids_and_timestamps_in_trend(self):
        baseline = self._make_snapshot({"a": [True]}, sweep_id="old-sweep")
        candidate = self._make_snapshot({"a": [True]}, sweep_id="new-sweep")
        trend = compare_sweeps(baseline, candidate)
        assert trend.baseline_sweep_id == "old-sweep"
        assert trend.candidate_sweep_id == "new-sweep"

    def test_no_trend_entry_for_model_only_in_baseline(self):
        baseline = self._make_snapshot({"a": [True], "b": [True]})
        candidate = self._make_snapshot({"a": [False]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        model_names = [t.agent_model for t in trend.model_trends]
        assert "b" not in model_names
        assert "a" in model_names

    def test_no_regressions_when_unchanged(self):
        baseline = self._make_snapshot({"a": [True, False]})
        candidate = self._make_snapshot({"a": [True, False]}, sweep_id="new")
        trend = compare_sweeps(baseline, candidate)
        mt = trend.model_trends[0]
        assert mt.new_regressions == []
        assert mt.new_improvements == []
        assert mt.delta_pass_rate == 0.0
