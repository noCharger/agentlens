from __future__ import annotations

from types import SimpleNamespace

from agentlens.eval.__main__ import main
from agentlens.eval.scenarios import ExpectedResult, Scenario


def _make_scenario(case_id: str) -> Scenario:
    return Scenario(
        id=case_id,
        name=f"Case {case_id}",
        category="tool_calling",
        benchmark="",
        evaluation_mode="deterministic",
        input="Do a thing",
        setup=[],
        expected=ExpectedResult(
            tools_called=[],
            max_steps=5,
            output_contains=[],
        ),
    )


def _make_eval_result(scenario: Scenario):
    return SimpleNamespace(
        scenario=scenario,
        level1=SimpleNamespace(
            tool_usage=SimpleNamespace(passed=True),
            output_format=SimpleNamespace(passed=True),
            trajectory=SimpleNamespace(passed=True),
        ),
        level2_scores={},
        error=None,
        passed=True,
    )


def test_eval_cli_ctrl_c_keeps_completed_results_for_report(tmp_path, monkeypatch):
    report_path = tmp_path / "partial-report.html"
    dataset_version = SimpleNamespace(id="dataset-1", items=[])
    scenarios = [_make_scenario("tc-001"), _make_scenario("tc-002")]
    settings = SimpleNamespace(
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    monkeypatch.setattr(
        "agentlens.eval.__main__._resolve_eval_dataset_and_scenarios",
        lambda args: (dataset_version, scenarios),
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__._persist_runtime_dataset_version",
        lambda args, dataset: None,
    )
    monkeypatch.setattr("agentlens.eval.__main__._init_metrics", lambda settings: None)
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_deepseek_preflight",
        lambda settings, require_judge=False: None,
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_openrouter_preflight",
        lambda settings, require_judge=False: None,
    )
    monkeypatch.setattr(
        "agentlens.eval.__main__.validate_zhipu_preflight",
        lambda settings, require_judge=False: None,
    )
    monkeypatch.setattr("agentlens.config.get_settings", lambda **overrides: settings)

    calls = {"count": 0}

    def fake_execute_and_eval(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _make_eval_result(scenarios[0])
        raise KeyboardInterrupt

    monkeypatch.setattr("agentlens.eval.__main__.execute_and_eval", fake_execute_and_eval)
    monkeypatch.setattr("agentlens.eval.__main__._print_results", lambda results: None)

    generated = {}

    def fake_generate_report(results, output_path=None):
        generated["count"] = len(results)
        generated["output_path"] = output_path
        if output_path:
            output_path.write_text("partial")
        return "partial"

    monkeypatch.setattr("agentlens.eval.__main__.generate_report", fake_generate_report)

    handled = {}

    def fake_handle_platform_outputs(args, results, settings, *, dataset_version=None):
        handled["count"] = len(results)
        handled["dataset_id"] = dataset_version.id if dataset_version is not None else None

    monkeypatch.setattr(
        "agentlens.eval.__main__._handle_platform_outputs",
        fake_handle_platform_outputs,
    )

    monkeypatch.setattr(
        "sys.argv",
        ["agentlens.eval", "--output", str(report_path)],
    )

    main()

    assert calls["count"] == 2
    assert generated["count"] == 1
    assert generated["output_path"] == report_path
    assert report_path.exists()
    assert handled["count"] == 1
    assert handled["dataset_id"] == "dataset-1"
