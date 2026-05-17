"""Integration tests for multi-model sweep CLI path."""

from __future__ import annotations

from types import SimpleNamespace

from agentlens.eval.__main__ import main
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.sweep import ModelRun, SweepResult


def _make_scenario(sid: str) -> Scenario:
    return Scenario(
        id=sid,
        name=f"Case {sid}",
        category="tool_calling",
        benchmark="",
        input="Do a thing",
        setup=[],
        expected=ExpectedResult(tools_called=[], output_contains=[]),
    )


def _make_settings(**overrides):
    defaults = dict(
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
        agent_framework="langgraph",
        judge_use_geval=False,
        judge_task_completion=False,
        judge_answer_relevancy=False,
        judge_hallucination=False,
        judge_faithfulness=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _install_common_mocks(monkeypatch, scenarios, settings):
    dataset_version = SimpleNamespace(id="ds-1", items=[])
    monkeypatch.setattr(
        "agentlens.eval.__main__._resolve_eval_dataset_and_scenarios",
        lambda args: (dataset_version, scenarios),
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__._persist_runtime_dataset_version",
        lambda args, dv: None,
    )
    monkeypatch.setattr("agentlens.eval.__main__._init_metrics", lambda s: None)
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_deepseek_preflight",
        lambda s, require_judge=False: None,
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_openrouter_preflight",
        lambda s, require_judge=False: None,
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_zhipu_preflight",
        lambda s, require_judge=False: None,
    )
    monkeypatch.setattr("agentlens.config.get_settings", lambda **overrides: settings)
    return dataset_version


def test_two_agent_models_routes_to_sweep_path(monkeypatch):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    sweep_calls: list[dict] = []

    def fake_run_sweep(models, scenarios, settings_factory, **kwargs):
        sweep_calls.append({"models": models, "n_scenarios": len(scenarios)})
        model_runs = [ModelRun(agent_model=m, results=[]) for m in models]
        return SweepResult(sweep_id="test-id", model_runs=model_runs)

    # run_sweep is imported lazily inside _run_sweep_path — patch at source module
    monkeypatch.setattr("agentlens.eval.sweep.run_sweep", fake_run_sweep)
    monkeypatch.setattr("agentlens.eval.__main__._print_sweep_results", lambda sweep: None)
    monkeypatch.setattr("agentlens.eval.__main__._handle_platform_outputs_for_model_run", lambda *a, **kw: None)

    monkeypatch.setattr("sys.argv", [
        "agentlens.eval",
        "--agent-model", "gemini:gemini-2.5-flash",
        "--agent-model", "deepseek:deepseek-chat",
    ])

    main()

    assert len(sweep_calls) == 1
    assert sweep_calls[0]["models"] == ["gemini:gemini-2.5-flash", "deepseek:deepseek-chat"]


def test_sweep_cli_with_output_calls_generate_sweep_report(monkeypatch, tmp_path):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    report_calls: list[dict] = []

    def fake_run_sweep(models, scenarios, settings_factory, **kwargs):
        model_runs = [ModelRun(agent_model=m, results=[]) for m in models]
        return SweepResult(sweep_id="abc", model_runs=model_runs)

    def fake_generate_sweep_report(sweep, output_path=None, trend_comparison=None):
        report_calls.append({"output_path": output_path})
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("<html>sweep</html>")
        return "<html>sweep</html>"

    # Both imported lazily inside _run_sweep_path — patch at their source modules
    monkeypatch.setattr("agentlens.eval.sweep.run_sweep", fake_run_sweep)
    monkeypatch.setattr("agentlens.eval.__main__._print_sweep_results", lambda sweep: None)
    monkeypatch.setattr("agentlens.eval.__main__._handle_platform_outputs_for_model_run", lambda *a, **kw: None)
    monkeypatch.setattr("agentlens.eval.level3_human.sweep_reporter.generate_sweep_report", fake_generate_sweep_report)

    output = tmp_path / "sweep.html"
    monkeypatch.setattr("sys.argv", [
        "agentlens.eval",
        "--agent-model", "gemini:gemini-2.5-flash",
        "--agent-model", "deepseek:deepseek-chat",
        "--output", str(output),
    ])

    main()

    assert len(report_calls) == 1
    assert report_calls[0]["output_path"] == output


def test_single_agent_model_uses_original_path(monkeypatch):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    single_model_calls: list[int] = []

    def fake_single_model_path(args, scenarios, settings, meter_provider, dataset_version):
        single_model_calls.append(1)

    sweep_calls: list[int] = []

    def fake_sweep_path(*args, **kwargs):
        sweep_calls.append(1)

    monkeypatch.setattr("agentlens.eval.__main__._run_single_model_path", fake_single_model_path)
    monkeypatch.setattr("agentlens.eval.__main__._run_sweep_path", fake_sweep_path)

    monkeypatch.setattr("sys.argv", [
        "agentlens.eval",
        "--agent-model", "gemini:gemini-2.5-flash",
    ])

    main()

    assert len(single_model_calls) == 1
    assert len(sweep_calls) == 0


def test_no_agent_model_flag_uses_original_path(monkeypatch):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    single_model_calls: list[int] = []

    def fake_single_model_path(args, scenarios, settings, meter_provider, dataset_version):
        single_model_calls.append(1)

    sweep_calls: list[int] = []

    def fake_sweep_path(*args, **kwargs):
        sweep_calls.append(1)

    monkeypatch.setattr("agentlens.eval.__main__._run_single_model_path", fake_single_model_path)
    monkeypatch.setattr("agentlens.eval.__main__._run_sweep_path", fake_sweep_path)

    monkeypatch.setattr("sys.argv", ["agentlens.eval"])

    main()

    assert len(single_model_calls) == 1
    assert len(sweep_calls) == 0


def test_sweep_stores_sweep_id_in_platform_export(monkeypatch, tmp_path):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    platform_calls: list[dict] = []

    def fake_run_sweep(models, scenarios, settings_factory, **kwargs):
        from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
        from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
        from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
        from agentlens.eval.runner import EvalResult, Level1Result

        model_runs = []
        for m in models:
            scenario = scenarios[0]
            result = EvalResult(
                scenario=scenario,
                level1=Level1Result(
                    tool_usage=ToolUsageResult(True, [], [], [], []),
                    output_format=OutputFormatResult(True, "ok", [], []),
                    trajectory=TrajectoryResult(True, 1, 5, False, 10, 5, None, []),
                ),
                level2_scores={},
                feature_flags={},
            )
            model_runs.append(ModelRun(agent_model=m, results=[result]))
        return SweepResult(sweep_id="sweep-xyz", model_runs=model_runs)

    def fake_handle_platform(args, model_run, settings, sweep_id, *, dataset_version=None):
        platform_calls.append({"sweep_id": sweep_id, "model": model_run.agent_model})

    monkeypatch.setattr("agentlens.eval.sweep.run_sweep", fake_run_sweep)
    monkeypatch.setattr("agentlens.eval.__main__._print_sweep_results", lambda sweep: None)
    monkeypatch.setattr("agentlens.eval.__main__._handle_platform_outputs_for_model_run", fake_handle_platform)

    monkeypatch.setattr("sys.argv", [
        "agentlens.eval",
        "--agent-model", "gemini:gemini-2.5-flash",
        "--agent-model", "deepseek:deepseek-chat",
    ])

    main()

    assert len(platform_calls) == 2
    assert all(c["sweep_id"] == "sweep-xyz" for c in platform_calls)
    models_called = {c["model"] for c in platform_calls}
    assert models_called == {"gemini:gemini-2.5-flash", "deepseek:deepseek-chat"}


def test_three_models_all_routed_to_sweep(monkeypatch):
    scenarios = [_make_scenario("tc-001")]
    settings = _make_settings()
    _install_common_mocks(monkeypatch, scenarios, settings)

    captured: list[list[str]] = []

    def fake_run_sweep(models, scenarios, settings_factory, **kwargs):
        captured.append(list(models))
        model_runs = [ModelRun(agent_model=m, results=[]) for m in models]
        return SweepResult(sweep_id="x", model_runs=model_runs)

    monkeypatch.setattr("agentlens.eval.sweep.run_sweep", fake_run_sweep)
    monkeypatch.setattr("agentlens.eval.__main__._print_sweep_results", lambda sweep: None)
    monkeypatch.setattr("agentlens.eval.__main__._handle_platform_outputs_for_model_run", lambda *a, **kw: None)

    monkeypatch.setattr("sys.argv", [
        "agentlens.eval",
        "--agent-model", "gemini:gemini-2.5-flash",
        "--agent-model", "deepseek:deepseek-chat",
        "--agent-model", "openrouter:openai/gpt-4o-mini",
    ])

    main()

    assert captured == [["gemini:gemini-2.5-flash", "deepseek:deepseek-chat", "openrouter:openai/gpt-4o-mini"]]
