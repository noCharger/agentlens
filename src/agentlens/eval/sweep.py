"""Multi-model sweep orchestration.

Each model runs in its own thread, advancing through scenarios independently
with no cross-model barrier at scenario boundaries.  max_workers=len(models)
ensures at most one concurrent API call per provider.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
class SweepResult:
    """Aggregated results for a multi-model sweep."""

    sweep_id: str
    model_runs: list[ModelRun]
    pairwise_comparison: ExperimentComparison | None = None

    @property
    def ranked_models(self) -> list[ModelRun]:
        return sorted(self.model_runs, key=lambda r: r.pass_rate, reverse=True)


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
