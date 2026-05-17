"""Multi-model sweep orchestration.

Each model runs in its own thread, advancing through scenarios independently
with no cross-model barrier at scenario boundaries.  max_workers=len(models)
ensures at most one concurrent API call per provider.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

if TYPE_CHECKING:
    from agentlens.config import AgentLensSettings
    from agentlens.eval.experiment import ExperimentComparison
    from agentlens.eval.runner import EvalResult
    from agentlens.eval.scenarios import Scenario


@dataclass
class ModelRun:
    """Results for one model within a sweep."""

    agent_model: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(1 for r in self.results if r.passed) / len(self.results) * 100, 1)

    @property
    def avg_steps(self) -> float:
        if not self.results:
            return 0.0
        return round(
            sum(r.level1.trajectory.total_steps for r in self.results) / len(self.results), 2
        )

    @property
    def avg_tokens(self) -> float:
        if not self.results:
            return 0.0
        total = sum(
            r.level1.trajectory.total_prompt_tokens + r.level1.trajectory.total_completion_tokens
            for r in self.results
        )
        return round(total / len(self.results), 2)


@dataclass
class HeadToHeadRecord:
    """Win/loss/tie counts for one (model_a, model_b) pair over shared scenarios."""
    model_a: str
    model_b: str
    wins_a: int   # model_a passed, model_b didn't
    wins_b: int   # model_b passed, model_a didn't
    ties: int     # both same pass/fail outcome
    total: int    # scenarios both ran


@dataclass
class ModelRanking:
    """Final ranking for one model using Condorcet scoring."""
    agent_model: str
    pass_rate: float
    condorcet_score: int  # number of other models beaten head-to-head
    rank: int


@dataclass
class SweepRanking:
    """N-model ranking computed from all pairwise head-to-head records."""
    rankings: list[ModelRanking]           # sorted by condorcet_score desc, then pass_rate
    head_to_head: list[HeadToHeadRecord]   # all (n choose 2) pairs
    benchmark_winners: dict[str, str]      # benchmark name -> agent_model with best pass rate


def _compute_ranking(model_runs: list[ModelRun]) -> SweepRanking:
    from agentlens.eval.benchmarks import UNASSIGNED_BENCHMARK, summarize_results_by_benchmark

    models = [run.agent_model for run in model_runs]
    results_by_model = {
        run.agent_model: {r.scenario.id: r for r in run.results}
        for run in model_runs
    }

    head_to_head: list[HeadToHeadRecord] = []
    for run_a, run_b in combinations(model_runs, 2):
        a, b = run_a.agent_model, run_b.agent_model
        common_ids = set(results_by_model[a]) & set(results_by_model[b])
        wins_a = wins_b = ties = 0
        for sid in common_ids:
            pa = results_by_model[a][sid].passed
            pb = results_by_model[b][sid].passed
            if pa and not pb:
                wins_a += 1
            elif pb and not pa:
                wins_b += 1
            else:
                ties += 1
        head_to_head.append(HeadToHeadRecord(
            model_a=a, model_b=b,
            wins_a=wins_a, wins_b=wins_b, ties=ties,
            total=len(common_ids),
        ))

    condorcet: dict[str, int] = {m: 0 for m in models}
    for rec in head_to_head:
        if rec.wins_a > rec.wins_b:
            condorcet[rec.model_a] += 1
        elif rec.wins_b > rec.wins_a:
            condorcet[rec.model_b] += 1

    pass_rates = {run.agent_model: run.pass_rate for run in model_runs}
    sorted_models = sorted(models, key=lambda m: (condorcet[m], pass_rates[m]), reverse=True)
    rankings = [
        ModelRanking(
            agent_model=m,
            pass_rate=pass_rates[m],
            condorcet_score=condorcet[m],
            rank=i + 1,
        )
        for i, m in enumerate(sorted_models)
    ]

    benchmark_summaries_by_model: dict[str, dict[str, float]] = {}
    all_benchmark_names: set[str] = set()
    for run in model_runs:
        summaries = {
            s.name: s.pass_rate
            for s in summarize_results_by_benchmark(run.results)
            if s.slug != UNASSIGNED_BENCHMARK
        }
        benchmark_summaries_by_model[run.agent_model] = summaries
        all_benchmark_names.update(summaries)

    benchmark_winners: dict[str, str] = {}
    for name in sorted(all_benchmark_names):
        winner = max(models, key=lambda m: benchmark_summaries_by_model.get(m, {}).get(name, -1.0))
        benchmark_winners[name] = winner

    return SweepRanking(
        rankings=rankings,
        head_to_head=head_to_head,
        benchmark_winners=benchmark_winners,
    )


@dataclass
class SweepResult:
    """Aggregated results for a multi-model sweep."""

    sweep_id: str
    model_runs: list[ModelRun]
    pairwise_comparison: ExperimentComparison | None = None

    @property
    def ranked_models(self) -> list[ModelRun]:
        return sorted(self.model_runs, key=lambda r: r.pass_rate, reverse=True)

    @property
    def ranking(self) -> SweepRanking:
        return _compute_ranking(self.model_runs)


def _run_model(
    model: str,
    scenarios: list[Scenario],
    settings_factory: Callable[[str], AgentLensSettings],
    eval_kwargs: dict,
    on_scenario_complete: Callable[[str, EvalResult], None] | None,
) -> ModelRun:
    from agentlens.eval.runner import QuotaExhaustedError, execute_and_eval

    model_settings = settings_factory(model)
    results: list[EvalResult] = []
    for scenario in scenarios:
        try:
            result = execute_and_eval(scenario, model_settings, **eval_kwargs)
        except QuotaExhaustedError:
            break
        results.append(result)
        if on_scenario_complete:
            on_scenario_complete(model, result)
    return ModelRun(agent_model=model, results=results)


def run_sweep(
    models: list[str],
    scenarios: list[Scenario],
    settings_factory: Callable[[str], AgentLensSettings],
    *,
    preset: str = "full",
    with_level2: bool = False,
    use_geval: bool = False,
    task_completion: bool = False,
    answer_relevancy: bool = False,
    hallucination: bool = False,
    faithfulness: bool = False,
    on_scenario_complete: Callable[[str, EvalResult], None] | None = None,
) -> SweepResult:
    """Run all scenarios against each model concurrently and return a SweepResult.

    Models run in separate threads with no cross-model barrier: model A advances
    from scenario N to N+1 immediately without waiting for model B.
    """
    eval_kwargs: dict = {
        "preset": preset,
        "with_level2": with_level2,
        "use_geval": use_geval,
        "task_completion": task_completion,
        "answer_relevancy": answer_relevancy,
        "hallucination": hallucination,
        "faithfulness": faithfulness,
    }

    model_index = {model: i for i, model in enumerate(models)}
    ordered_runs: list[ModelRun | None] = [None] * len(models)

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                _run_model, model, scenarios, settings_factory, eval_kwargs, on_scenario_complete
            ): model
            for model in models
        }
        for future in as_completed(futures):
            model = futures[future]
            ordered_runs[model_index[model]] = future.result()

    model_runs = [r for r in ordered_runs if r is not None]

    pairwise = None
    if len(model_runs) == 2:
        from agentlens.eval.experiment import VersionedConfig, compare_experiments

        pairwise = compare_experiments(
            model_runs[0].results,
            model_runs[1].results,
            baseline_config=VersionedConfig(agent_model=model_runs[0].agent_model),
            candidate_config=VersionedConfig(agent_model=model_runs[1].agent_model),
        )

    return SweepResult(
        sweep_id=uuid4().hex[:12],
        model_runs=model_runs,
        pairwise_comparison=pairwise,
    )
