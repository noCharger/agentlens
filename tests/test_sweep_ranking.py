"""Tests for N-model head-to-head ranking within a SweepResult."""

from __future__ import annotations

from agentlens.eval.level1_deterministic.output_format import OutputFormatResult
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult
from agentlens.eval.runner import EvalResult, Level1Result
from agentlens.eval.scenarios import ExpectedResult, Scenario
from agentlens.eval.sweep import ModelRun, SweepResult


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


def _make_sweep(models: list[str], pass_matrix: list[list[bool]], benchmarks: list[str] | None = None) -> SweepResult:
    n = max(len(row) for row in pass_matrix)
    scenarios = [
        _make_scenario(f"s{i}", benchmark=benchmarks[i] if benchmarks else "")
        for i in range(n)
    ]
    model_runs = []
    for model, pass_vals in zip(models, pass_matrix):
        results = [_make_eval_result(scenarios[i], passed=p) for i, p in enumerate(pass_vals)]
        model_runs.append(ModelRun(agent_model=model, results=results))
    return SweepResult(sweep_id="test-id", model_runs=model_runs)


class TestHeadToHead:
    def test_record_count_equals_n_choose_2(self):
        sweep = _make_sweep(["a", "b", "c"], [[True, False], [False, True], [True, True]])
        assert len(sweep.ranking.head_to_head) == 3  # 3 choose 2

    def test_two_models_one_record(self):
        sweep = _make_sweep(["a", "b"], [[True, False], [False, True]])
        assert len(sweep.ranking.head_to_head) == 1

    def test_wins_counted_correctly(self):
        # a wins s0, b wins s1
        sweep = _make_sweep(["a", "b"], [[True, False], [False, True]])
        rec = sweep.ranking.head_to_head[0]
        # model_a=a wins 1, model_b=b wins 1
        assert rec.wins_a == 1
        assert rec.wins_b == 1
        assert rec.ties == 0

    def test_ties_when_both_same_outcome(self):
        sweep = _make_sweep(["a", "b"], [[True, True], [True, True]])
        rec = sweep.ranking.head_to_head[0]
        assert rec.ties == 2
        assert rec.wins_a == 0
        assert rec.wins_b == 0

    def test_total_equals_common_scenarios(self):
        sweep = _make_sweep(["a", "b"], [[True, False, True], [False, True, True]])
        rec = sweep.ranking.head_to_head[0]
        assert rec.total == 3


class TestCondorcetRanking:
    def test_clear_winner_has_highest_condorcet_score(self):
        # a beats both b and c; c beats b
        # a: wins s0 and s1; b: wins nothing; c: wins s2
        # a > c > b by condorcet
        sweep = _make_sweep(
            ["a", "b", "c"],
            [
                [True, True, True],   # a: all pass
                [False, False, True], # b: only s2 pass
                [False, True, True],  # c: s1 s2 pass
            ],
        )
        ranking = sweep.ranking
        assert ranking.rankings[0].agent_model == "a"
        assert ranking.rankings[0].rank == 1

    def test_rank_field_is_1_indexed(self):
        sweep = _make_sweep(["a", "b", "c"], [[True], [False], [False]])
        ranks = [r.rank for r in sweep.ranking.rankings]
        assert sorted(ranks) == [1, 2, 3]

    def test_condorcet_score_for_dominant_model(self):
        # model a beats both b and c
        sweep = _make_sweep(
            ["a", "b", "c"],
            [[True, True], [False, False], [False, False]],
        )
        a_rank = next(r for r in sweep.ranking.rankings if r.agent_model == "a")
        assert a_rank.condorcet_score == 2  # beats b and c

    def test_two_models_ranking_mirrors_pass_rate(self):
        sweep = _make_sweep(["a", "b"], [[True, True], [True, False]])
        rankings = sweep.ranking.rankings
        assert rankings[0].agent_model == "a"  # higher pass rate

    def test_pass_rate_in_ranking(self):
        sweep = _make_sweep(["a", "b"], [[True, False], [True, True]])
        b_rank = next(r for r in sweep.ranking.rankings if r.agent_model == "b")
        assert b_rank.pass_rate == 100.0


class TestBenchmarkWinners:
    def test_benchmark_winner_is_best_model(self):
        # a wins s0 (toolathlon), b wins s1 (toolathlon)
        # a pass rate on toolathlon = 50%, b pass rate = 50% — tie goes to max(), so first
        # let's make a clear winner: a passes both, b passes only s1
        scenarios = [
            _make_scenario("s0", benchmark="toolathlon"),
            _make_scenario("s1", benchmark="toolathlon"),
        ]
        run_a = ModelRun("a", [_make_eval_result(s, passed=True) for s in scenarios])
        run_b = ModelRun("b", [_make_eval_result(scenarios[0], passed=False),
                                _make_eval_result(scenarios[1], passed=True)])
        sweep = SweepResult(sweep_id="x", model_runs=[run_a, run_b])
        assert sweep.ranking.benchmark_winners.get("Toolathlon") == "a"

    def test_no_benchmark_winners_when_unassigned(self):
        sweep = _make_sweep(["a", "b"], [[True], [False]])
        assert sweep.ranking.benchmark_winners == {}
