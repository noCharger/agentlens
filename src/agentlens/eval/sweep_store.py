"""Sweep snapshot persistence and cross-sweep trend comparison.

SweepSnapshot is the serializable (JSON) summary of a SweepResult.  It
captures per-scenario pass/fail and key metrics but omits span-level data
that is only useful for in-process analysis.

Typical workflow:
    # After a sweep run:
    snapshot = save_sweep(sweep_result, Path("sweeps/2026-05-17.json"))

    # Next day, compare with yesterday's snapshot:
    old = load_sweep(Path("sweeps/2026-05-16.json"))
    trend = compare_sweeps(old, snapshot)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.sweep import SweepResult


@dataclass
class ScenarioSnapshot:
    scenario_id: str
    scenario_name: str
    benchmark: str
    passed: bool
    status: str
    judge_score: float | None
    total_steps: int
    total_tokens: int
    retention_score: float | None


@dataclass
class ModelRunSnapshot:
    agent_model: str
    pass_rate: float
    scenario_snapshots: list[ScenarioSnapshot] = field(default_factory=list)

    def scenario_pass_map(self) -> dict[str, bool]:
        """Return {scenario_id: passed} for fast lookup."""
        return {s.scenario_id: s.passed for s in self.scenario_snapshots}


@dataclass
class SweepSnapshot:
    sweep_id: str
    timestamp: str  # ISO 8601 UTC
    model_run_snapshots: list[ModelRunSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SweepSnapshot:
        return cls(
            sweep_id=d["sweep_id"],
            timestamp=d["timestamp"],
            model_run_snapshots=[
                ModelRunSnapshot(
                    agent_model=r["agent_model"],
                    pass_rate=r["pass_rate"],
                    scenario_snapshots=[
                        ScenarioSnapshot(**s) for s in r["scenario_snapshots"]
                    ],
                )
                for r in d["model_run_snapshots"]
            ],
        )


@dataclass
class ModelTrend:
    agent_model: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta_pass_rate: float
    new_regressions: list[str]   # scenario IDs: passed in baseline, failed in candidate
    new_improvements: list[str]  # scenario IDs: failed in baseline, passed in candidate


@dataclass
class SweepTrendComparison:
    baseline_sweep_id: str
    candidate_sweep_id: str
    baseline_timestamp: str
    candidate_timestamp: str
    model_trends: list[ModelTrend]   # only models present in both snapshots
    new_models: list[str]            # in candidate but not baseline
    dropped_models: list[str]        # in baseline but not candidate


def snapshot_from_sweep(sweep: SweepResult) -> SweepSnapshot:
    """Build an in-memory SweepSnapshot from a live SweepResult (no file I/O)."""
    run_snapshots = []
    for run in sweep.model_runs:
        scenario_snapshots = []
        for r in run.results:
            retention = (
                r.level1.memory_retention.retention_score
                if r.level1.memory_retention is not None
                else None
            )
            scenario_snapshots.append(ScenarioSnapshot(
                scenario_id=r.scenario.id,
                scenario_name=r.scenario.name,
                benchmark=r.scenario.benchmark or "",
                passed=r.passed,
                status=r.status.value,
                judge_score=r.judge_overall_score,
                total_steps=r.level1.trajectory.total_steps,
                total_tokens=(
                    r.level1.trajectory.total_prompt_tokens
                    + r.level1.trajectory.total_completion_tokens
                ),
                retention_score=retention,
            ))
        run_snapshots.append(ModelRunSnapshot(
            agent_model=run.agent_model,
            pass_rate=run.pass_rate,
            scenario_snapshots=scenario_snapshots,
        ))

    return SweepSnapshot(
        sweep_id=sweep.sweep_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_run_snapshots=run_snapshots,
    )


def save_sweep(sweep: SweepResult, path: Path) -> SweepSnapshot:
    """Serialize sweep results to JSON and return the snapshot."""
    snapshot = snapshot_from_sweep(sweep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2))
    return snapshot


def load_sweep(path: Path) -> SweepSnapshot:
    """Load a previously saved SweepSnapshot from a JSON file."""
    return SweepSnapshot.from_dict(json.loads(path.read_text()))


def compare_sweeps(baseline: SweepSnapshot, candidate: SweepSnapshot) -> SweepTrendComparison:
    """Compute per-model trends between two sweep snapshots."""
    baseline_models = {r.agent_model: r for r in baseline.model_run_snapshots}
    candidate_models = {r.agent_model: r for r in candidate.model_run_snapshots}

    shared = sorted(set(baseline_models) & set(candidate_models))
    new_models = sorted(set(candidate_models) - set(baseline_models))
    dropped_models = sorted(set(baseline_models) - set(candidate_models))

    model_trends: list[ModelTrend] = []
    for model in shared:
        base_run = baseline_models[model]
        cand_run = candidate_models[model]

        base_pass = base_run.scenario_pass_map()
        cand_pass = cand_run.scenario_pass_map()
        common_ids = set(base_pass) & set(cand_pass)

        new_regressions = sorted(
            sid for sid in common_ids if base_pass[sid] and not cand_pass[sid]
        )
        new_improvements = sorted(
            sid for sid in common_ids if not base_pass[sid] and cand_pass[sid]
        )

        delta = round(cand_run.pass_rate - base_run.pass_rate, 1)
        model_trends.append(ModelTrend(
            agent_model=model,
            baseline_pass_rate=base_run.pass_rate,
            candidate_pass_rate=cand_run.pass_rate,
            delta_pass_rate=delta,
            new_regressions=new_regressions,
            new_improvements=new_improvements,
        ))

    return SweepTrendComparison(
        baseline_sweep_id=baseline.sweep_id,
        candidate_sweep_id=candidate.sweep_id,
        baseline_timestamp=baseline.timestamp,
        candidate_timestamp=candidate.timestamp,
        model_trends=model_trends,
        new_models=new_models,
        dropped_models=dropped_models,
    )
