"""Tests for multi-model sweep orchestration."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.sweep import ModelRun, SweepResult, run_sweep


def _make_scenario(sid: str) -> Scenario:
    return Scenario(
        id=sid,
        name=f"Scenario {sid}",
        category="test",
        benchmark="",
        input="query",
        setup=[],
        expected=ExpectedResult(tools_called=[], output_contains=[]),
    )


def _make_eval_result(scenario: Scenario, *, passed: bool = True, steps: int = 2, tokens: int = 100) -> EvalResult:
    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(passed, [], [], [], []),
            output_format=OutputFormatResult(passed, "ok" if passed else "", [], []),
            trajectory=TrajectoryResult(True, steps, 10, False, tokens, tokens // 2, None, []),
        ),
        level2_scores={},
        feature_flags={},
    )


def _make_model_run(model: str, *, pass_vals: list[bool], steps: int = 2, tokens: int = 100) -> ModelRun:
    scenarios = [_make_scenario(f"s{i}") for i in range(len(pass_vals))]
    results = [_make_eval_result(s, passed=p, steps=steps, tokens=tokens) for s, p in zip(scenarios, pass_vals)]
    return ModelRun(agent_model=model, results=results)


def _settings_factory(model: str):
    return SimpleNamespace(agent_model=model)


class TestModelRun:
    def test_pass_rate_half(self):
        run = _make_model_run("gemini:gemini-2.5-flash", pass_vals=[True, False, True, False])
        assert run.pass_rate == 50.0

    def test_pass_rate_all_pass(self):
        run = _make_model_run("m", pass_vals=[True, True])
        assert run.pass_rate == 100.0

    def test_pass_rate_empty(self):
        run = ModelRun(agent_model="m")
        assert run.pass_rate == 0.0

    def test_avg_steps(self):
        run = _make_model_run("m", pass_vals=[True, True], steps=4)
        assert run.avg_steps == 4.0

    def test_avg_tokens(self):
        # tokens=200 → prompt=200, completion=100 → total per result=300 → avg=300
        run = _make_model_run("m", pass_vals=[True, True], tokens=200)
        assert run.avg_tokens == 300.0

    def test_avg_tokens_empty(self):
        run = ModelRun(agent_model="m")
        assert run.avg_tokens == 0.0

    def test_avg_steps_empty(self):
        run = ModelRun(agent_model="m")
        assert run.avg_steps == 0.0


class TestSweepResult:
    def test_ranked_models_sorted_descending(self):
        run_a = _make_model_run("gemini", pass_vals=[True, False])   # 50%
        run_b = _make_model_run("deepseek", pass_vals=[True, True])  # 100%
        sweep = SweepResult(sweep_id="abc", model_runs=[run_a, run_b])
        ranked = sweep.ranked_models
        assert ranked[0].agent_model == "deepseek"
        assert ranked[1].agent_model == "gemini"

    def test_ranked_models_preserves_equal_rates(self):
        run_a = _make_model_run("a", pass_vals=[True])
        run_b = _make_model_run("b", pass_vals=[True])
        sweep = SweepResult(sweep_id="x", model_runs=[run_a, run_b])
        assert len(sweep.ranked_models) == 2

    def test_pairwise_comparison_defaults_none(self):
        sweep = SweepResult(sweep_id="x", model_runs=[])
        assert sweep.pairwise_comparison is None


class TestRunSweep:
    def test_submits_all_model_scenario_combinations(self, monkeypatch):
        scenarios = [_make_scenario("s1"), _make_scenario("s2")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]
        calls: list[tuple[str, str]] = []

        def fake_execute(scenario, settings, **kwargs):
            calls.append((settings.agent_model, scenario.id))
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        assert len(calls) == 4
        assert {(m, s) for m, s in calls} == {
            ("gemini:gemini-2.5-flash", "s1"),
            ("gemini:gemini-2.5-flash", "s2"),
            ("deepseek:deepseek-chat", "s1"),
            ("deepseek:deepseek-chat", "s2"),
        }
        assert len(sweep.model_runs) == 2

    def test_no_scenario_barrier_across_models(self, monkeypatch):
        """Model A should start scenario 2 without waiting for model B to finish scenario 1."""
        scenarios = [_make_scenario("s1"), _make_scenario("s2")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]

        started: dict[str, threading.Event] = {m: threading.Event() for m in models}
        errors: list[str] = []

        def fake_execute(scenario, settings, **kwargs):
            model = settings.agent_model
            if scenario.id == "s1":
                started[model].set()
                other = [m for m in models if m != model][0]
                if not started[other].wait(timeout=5.0):
                    errors.append(f"{model} waited more than 5s for {other}")
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        run_sweep(models, scenarios, _settings_factory)

        assert errors == [], f"Scenario barrier detected: {errors}"

    def test_preserves_scenario_order_per_model(self, monkeypatch):
        scenarios = [_make_scenario(f"s{i}") for i in range(5)]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        for run in sweep.model_runs:
            ids = [r.scenario.id for r in run.results]
            assert ids == [f"s{i}" for i in range(5)]

    def test_two_models_produces_pairwise_comparison(self, monkeypatch):
        scenarios = [_make_scenario("s1"), _make_scenario("s2")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]
        pass_map = {
            ("gemini:gemini-2.5-flash", "s1"): True,
            ("gemini:gemini-2.5-flash", "s2"): False,
            ("deepseek:deepseek-chat", "s1"): True,
            ("deepseek:deepseek-chat", "s2"): True,
        }

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario, passed=pass_map[(settings.agent_model, scenario.id)])

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        assert sweep.pairwise_comparison is not None
        assert sweep.pairwise_comparison.delta_pass_rate == 50.0
        assert len(sweep.pairwise_comparison.improvements) == 1

    def test_three_models_no_pairwise_comparison(self, monkeypatch):
        scenarios = [_make_scenario("s1")]
        models = ["m1", "m2", "m3"]

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        assert sweep.pairwise_comparison is None

    def test_quota_error_stops_only_that_model(self, monkeypatch):
        from agentlens.eval.runner import QuotaExhaustedError

        scenarios = [_make_scenario("s1"), _make_scenario("s2"), _make_scenario("s3")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]

        def fake_execute(scenario, settings, **kwargs):
            if settings.agent_model == "gemini:gemini-2.5-flash" and scenario.id == "s2":
                raise QuotaExhaustedError("quota")
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        gemini_run = next(r for r in sweep.model_runs if r.agent_model == "gemini:gemini-2.5-flash")
        deepseek_run = next(r for r in sweep.model_runs if r.agent_model == "deepseek:deepseek-chat")

        assert len(gemini_run.results) == 1   # only s1; quota hit on s2 stops gemini
        assert len(deepseek_run.results) == 3  # all scenarios completed unaffected

    def test_on_scenario_complete_called_for_every_result(self, monkeypatch):
        scenarios = [_make_scenario("s1"), _make_scenario("s2")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        completed: list[tuple[str, str]] = []
        lock = threading.Lock()

        def on_complete(model: str, result) -> None:
            with lock:
                completed.append((model, result.scenario.id))

        run_sweep(models, scenarios, _settings_factory, on_scenario_complete=on_complete)

        assert len(completed) == 4
        assert set(m for m, _ in completed) == set(models)

    def test_sweep_id_is_unique_per_call(self, monkeypatch):
        scenarios = [_make_scenario("s1")]
        models = ["m1"]

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        s1 = run_sweep(models, scenarios, _settings_factory)
        s2 = run_sweep(models, scenarios, _settings_factory)

        assert s1.sweep_id != s2.sweep_id

    def test_model_order_preserved_in_model_runs(self, monkeypatch):
        scenarios = [_make_scenario("s1")]
        models = ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat", "openrouter:openai/gpt-4o"]

        def fake_execute(scenario, settings, **kwargs):
            return _make_eval_result(scenario)

        monkeypatch.setattr("agentlens.eval.runner.execute_and_eval", fake_execute)

        sweep = run_sweep(models, scenarios, _settings_factory)

        assert [r.agent_model for r in sweep.model_runs] == models
