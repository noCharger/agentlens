from agentlens.eval.benchmarks import (
    UNASSIGNED_BENCHMARK,
    collect_benchmark_inventory,
    filter_scenarios_by_benchmark,
    normalize_benchmark_name,
    summarize_results_by_benchmark,
)
from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario


def _make_scenario(scenario_id: str, benchmark: str = "") -> Scenario:
    return Scenario(
        id=scenario_id,
        name=f"Scenario {scenario_id}",
        category="test",
        benchmark=benchmark,
        input="query",
        setup=[],
        expected=ExpectedResult(),
    )


def _make_result(passed: bool, scenario_id: str, benchmark: str = "") -> EvalResult:
    return EvalResult(
        scenario=_make_scenario(scenario_id, benchmark=benchmark),
        level1=Level1Result(
            tool_usage=ToolUsageResult(
                passed=passed,
                expected_tools=[],
                actual_tools=[],
                missing_tools=[],
                unexpected_tools=[],
            ),
            output_format=OutputFormatResult(
                passed=passed,
                output_text="ok" if passed else "",
                expected_substrings=[],
                missing_substrings=[],
            ),
            trajectory=TrajectoryResult(
                passed=True,
                total_steps=1,
                max_steps=5,
                has_loop=False,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                max_tokens=None,
                reasons=[],
            ),
        ),
    )


def test_normalize_benchmark_name_supports_aliases():
    assert normalize_benchmark_name("SWE Bench Pro") == "swe-bench-pro"
    assert normalize_benchmark_name("MLE-Bench lite") == "mle-bench-lite"
    assert normalize_benchmark_name("MM ClawBench") == "mm-clawbench"


def test_filter_scenarios_by_benchmark_accepts_display_name():
    scenarios = [
        _make_scenario("a", benchmark="swe-bench-pro"),
        _make_scenario("b", benchmark="toolathlon"),
    ]

    filtered = filter_scenarios_by_benchmark(scenarios, ["SWE Bench Pro"])

    assert [scenario.id for scenario in filtered] == ["a"]


def test_collect_benchmark_inventory_counts_built_in_and_unassigned():
    inventory = collect_benchmark_inventory(
        [
            _make_scenario("a", benchmark="toolathlon"),
            _make_scenario("b"),
        ]
    )
    by_slug = {row.slug: row for row in inventory}

    assert by_slug["toolathlon"].scenario_count == 1
    assert by_slug[UNASSIGNED_BENCHMARK].scenario_count == 1


def test_summarize_results_by_benchmark_groups_pass_rates():
    summaries = summarize_results_by_benchmark(
        [
            _make_result(True, "a", benchmark="toolathlon"),
            _make_result(False, "b", benchmark="toolathlon"),
            _make_result(True, "c", benchmark="swe-bench-pro"),
        ]
    )
    by_slug = {summary.slug: summary for summary in summaries}

    assert by_slug["toolathlon"].total == 2
    assert by_slug["toolathlon"].pass_rate == 50.0
    assert by_slug["swe-bench-pro"].passed == 1
