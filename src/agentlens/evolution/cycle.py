"""Evolution cycle: orchestrates baseline eval → signal analysis → prompt evolution → candidate eval.

Design: GEPA/TextGrad prompt-space optimization using delta_pass_rate as the outcome
signal (analogous to Memory-R1's QA exact-match reward — no intermediate op labels).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from agentlens.agents.runner_interface import AgentRunner, EmbeddedAgentRunner
from agentlens.evolution.signal_analyzer import SignalSummary, analyze_signals
from agentlens.evolution.prompt_evolver import EvolutionProposal, evolve_prompt

log = logging.getLogger("agentlens.evolution")


@dataclass
class EvolutionConfig:
    max_cycles: int = 3
    min_improvement: float = 0.05
    target_pass_rate: float = 0.90
    scenarios_dir: Path | None = None
    project_name: str = "agentlens"
    project_slug: str | None = None


@dataclass
class CycleResult:
    cycle: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    accepted: bool
    proposal: EvolutionProposal
    signal_summary: SignalSummary
    baseline_run_id: str
    candidate_run_id: str


class EvolutionCycle:
    """Runs iterative prompt evolution cycles using eval pass-rate as the reward signal."""

    def __init__(self, settings, *, repository=None, agent_runner: AgentRunner | None = None):
        self.settings = settings
        self._repo = repository
        self._agent_runner = agent_runner

    def run(self, config: EvolutionConfig) -> list[CycleResult]:
        from agentlens.eval.runner import execute_and_eval, load_and_summarize
        from agentlens.agents.runtime import _AG2_SYSTEM_MESSAGE
        from agentlens.core.models import EvolutionRecord

        scenarios_dir = config.scenarios_dir or Path("src/agentlens/scenarios")
        scenarios = load_and_summarize(scenarios_dir)
        if not scenarios:
            log.warning("No scenarios found in %s — evolution cycle aborted.", scenarios_dir)
            return []

        results: list[CycleResult] = []
        current_prompt: str | None = None

        if self._repo:
            current_prompt = self._repo.load_active_prompt(
                config.project_slug or config.project_name
            )

        for cycle in range(1, config.max_cycles + 1):
            log.info("Evolution cycle %d / %d", cycle, config.max_cycles)

            baseline_evals = self._run_eval(scenarios, system_prompt=current_prompt)
            baseline_pass_rate = (
                sum(1 for r in baseline_evals if r.passed) / len(baseline_evals)
                if baseline_evals else 0.0
            )
            baseline_run_id = f"baseline_cycle{cycle}_{uuid4().hex[:8]}"

            if baseline_pass_rate >= config.target_pass_rate:
                log.info(
                    "Target pass rate %.0f%% reached (%.0f%%). Stopping.",
                    config.target_pass_rate * 100,
                    baseline_pass_rate * 100,
                )
                break

            summary = analyze_signals(baseline_evals)
            log.info(
                "Pass rate: %.0f%% | top failure: %s | memory retention: %s",
                baseline_pass_rate * 100,
                summary.dominant_failure_patterns[0][0] if summary.dominant_failure_patterns else "none",
                f"{summary.memory_retention_score:.2f}" if summary.memory_retention_score else "N/A",
            )

            proposal = evolve_prompt(
                summary,
                current_prompt or _AG2_SYSTEM_MESSAGE,
                model=self.settings.judge_model,
                settings=self.settings,
            )

            candidate_evals = self._run_eval(scenarios, system_prompt=proposal.evolved_prompt)
            candidate_pass_rate = (
                sum(1 for r in candidate_evals if r.passed) / len(candidate_evals)
                if candidate_evals else 0.0
            )
            candidate_run_id = f"candidate_cycle{cycle}_{uuid4().hex[:8]}"

            delta = candidate_pass_rate - baseline_pass_rate
            accepted = delta >= config.min_improvement
            if accepted:
                current_prompt = proposal.evolved_prompt
                log.info(
                    "Cycle %d accepted: +%.1f%% pass rate", cycle, delta * 100
                )
            else:
                log.info(
                    "Cycle %d rejected: delta %.1f%% < threshold %.1f%%",
                    cycle, delta * 100, config.min_improvement * 100,
                )

            cycle_result = CycleResult(
                cycle=cycle,
                baseline_pass_rate=baseline_pass_rate,
                candidate_pass_rate=candidate_pass_rate,
                delta=delta,
                accepted=accepted,
                proposal=proposal,
                signal_summary=summary,
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
            )
            results.append(cycle_result)

            if self._repo:
                evo_record = EvolutionRecord(
                    id=f"evo_{uuid4().hex[:12]}",
                    cycle=cycle,
                    baseline_run_id=baseline_run_id,
                    candidate_run_id=candidate_run_id,
                    signal_summary=self._summary_to_dict(summary),
                    original_prompt=proposal.original_prompt,
                    evolved_prompt=proposal.evolved_prompt,
                    rationale=proposal.rationale,
                    targeted_patterns=proposal.targeted_patterns,
                    delta_pass_rate=delta,
                    accepted=accepted,
                )
                self._repo.save_evolution_record(
                    project_name=config.project_name,
                    record=evo_record,
                    project_slug=config.project_slug,
                )

        return results

    def _run_eval(self, scenarios, *, system_prompt: str | None) -> list:
        if self._agent_runner is not None:
            return self._run_eval_with_runner(scenarios, system_prompt=system_prompt)

        from agentlens.eval.runner import execute_and_eval

        evals = []
        for scenario in scenarios:
            try:
                result = execute_and_eval(
                    scenario,
                    self.settings,
                    with_level2=bool(scenario.judge_rubric),
                    system_prompt=system_prompt,
                )
                evals.append(result)
            except Exception as exc:
                log.warning("Scenario %s failed: %s", scenario.id, exc)
        return evals

    def _run_eval_with_runner(self, scenarios, *, system_prompt: str | None) -> list:
        import subprocess
        from agentlens.eval.runner import (
            _error_result,
            _run_level2,
            evaluate_scenario,
            run_setup_commands,
        )

        evals = []
        for scenario in scenarios:
            if scenario.setup_commands:
                try:
                    run_setup_commands(scenario.setup_commands)
                except subprocess.CalledProcessError as exc:
                    evals.append(_error_result(scenario, f"Setup failed: {exc}"))
                    continue

            runner = (
                EmbeddedAgentRunner(self.settings, scenario=scenario)
                if self._agent_runner is None
                else self._agent_runner
            )
            try:
                run_result = runner.run(scenario.input_query, system_prompt=system_prompt)
            except Exception as exc:
                log.warning("Scenario %s runner error: %s", scenario.id, exc)
                evals.append(_error_result(scenario, str(exc)))
                continue

            if run_result.error:
                evals.append(_error_result(scenario, run_result.error))
                continue

            result = evaluate_scenario(
                scenario, run_result.trace, output_text=run_result.output
            )
            if scenario.judge_rubric:
                _run_level2(result, run_result.trace, scenario, self.settings)
            evals.append(result)

        return evals

    @staticmethod
    def _summary_to_dict(summary: SignalSummary) -> dict:
        return {
            "pass_rate": summary.pass_rate,
            "dominant_failure_patterns": summary.dominant_failure_patterns,
            "weak_dimensions": summary.weak_dimensions,
            "frequent_risk_signals": summary.frequent_risk_signals,
            "memory_retention_score": summary.memory_retention_score,
        }
