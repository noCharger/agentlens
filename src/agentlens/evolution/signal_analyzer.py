"""Signal analyzer: extracts actionable patterns from a batch of eval results.

Outcome signal is delta_pass_rate (analogous to Memory-R1's QA exact-match reward).
No intermediate operation labels are required — only the final pass/fail status and
the failure patterns already detected by L1/L2.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from agentlens.eval.runner import EvalResult


@dataclass
class SignalSummary:
    pass_rate: float
    dominant_failure_patterns: list[tuple[str, int]]  # (pattern_type, count) sorted desc
    weak_dimensions: list[tuple[str, float]]           # (dimension, avg_score) where avg < 3.5
    frequent_risk_signals: list[tuple[str, int]]       # (signal_type, count) sorted desc
    memory_retention_score: float | None               # None if no memory scenarios ran
    failure_evidence: list[str]                        # sample output excerpts from failures


_PATTERN_TO_STRATEGY: dict[str, str] = {
    "context_forgetting": (
        "Maintain a working memory: after reading the task, explicitly list all key facts "
        "and constraints before taking any action."
    ),
    "tool_confusion": (
        "Before each tool call, state in one sentence: which tool you are using and why it "
        "is the correct choice over alternatives."
    ),
    "loop_trap": (
        "If you call the same tool three or more times with nearly identical arguments, stop "
        "and explain why the previous calls were insufficient before retrying."
    ),
    "confabulation": (
        "Only state facts that are directly derivable from tool outputs or the original task "
        "description. Do not infer or interpolate unstated values."
    ),
    "fuzzy_guessing": (
        "Never guess numerical values, identifiers, or status fields. Use tools to verify "
        "before stating any fact."
    ),
    "missing_confirmation": (
        "Before writing your final answer, verify the key result against the task requirements "
        "by re-reading the task description."
    ),
}

_MEMORY_STRATEGY = (
    "Re-read all facts and constraints from the task description before writing your final "
    "answer. If a tool returned updated information, use that value instead of the originally "
    "stated value."
)


def _collect_failure_evidence(results: list[EvalResult]) -> list[str]:
    evidence: list[str] = []
    for r in results:
        if r.passed:
            continue
        parts: list[str] = []
        if r.level1.failure_reasons:
            parts.append("; ".join(r.level1.failure_reasons[:3]))
        for dim, explanation in r.level2_explanations.items():
            if r.level2_scores.get(dim, 5.0) < 3.5 and explanation:
                parts.append(f"{dim}: {explanation[:150].strip()}")
                break
        if parts:
            evidence.append(f"[{r.scenario.id}] " + " | ".join(parts))
        if len(evidence) >= 5:
            break
    return evidence


def analyze_signals(results: list[EvalResult]) -> SignalSummary:
    """Aggregate eval results into an actionable signal summary."""
    if not results:
        return SignalSummary(
            pass_rate=0.0,
            dominant_failure_patterns=[],
            weak_dimensions=[],
            frequent_risk_signals=[],
            memory_retention_score=None,
            failure_evidence=[],
        )

    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / len(results)

    pattern_counter: Counter[str] = Counter()
    for r in results:
        if r.level1.trajectory_analysis:
            for p in r.level1.trajectory_analysis.failure_map.patterns:
                pattern_counter[p.pattern_type] += 1

    risk_counter: Counter[str] = Counter()
    for r in results:
        for sig in r.risk_signals:
            risk_type = sig.split(":", 1)[0].strip()
            risk_counter[risk_type] += 1

    dimension_scores: dict[str, list[float]] = {}
    for r in results:
        for dim, score in r.level2_scores.items():
            if score >= 0:
                dimension_scores.setdefault(dim, []).append(score)
    weak: list[tuple[str, float]] = []
    for dim, scores in dimension_scores.items():
        avg = sum(scores) / len(scores)
        if avg < 3.5:
            weak.append((dim, round(avg, 2)))
    weak.sort(key=lambda x: x[1])

    retention_scores: list[float] = []
    for r in results:
        if r.level1.memory_retention is not None:
            retention_scores.append(r.level1.memory_retention.retention_score)
    memory_score = (
        sum(retention_scores) / len(retention_scores) if retention_scores else None
    )

    evidence = _collect_failure_evidence(results)

    return SignalSummary(
        pass_rate=pass_rate,
        dominant_failure_patterns=pattern_counter.most_common(5),
        weak_dimensions=weak[:5],
        frequent_risk_signals=risk_counter.most_common(5),
        memory_retention_score=memory_score,
        failure_evidence=evidence,
    )


def build_strategy_hints(summary: SignalSummary) -> list[str]:
    """Map detected patterns to concrete prompt strategy hints (rule-based pre-filter).

    These hints are passed to the LLM optimizer (GEPA/TextGrad-style) to guide
    prompt evolution. The LLM integrates them rather than copying verbatim.
    """
    hints: list[str] = []
    seen: set[str] = set()

    for pattern_type, _ in summary.dominant_failure_patterns:
        strategy = _PATTERN_TO_STRATEGY.get(pattern_type)
        if strategy and strategy not in seen:
            hints.append(strategy)
            seen.add(strategy)

    if summary.memory_retention_score is not None and summary.memory_retention_score < 0.8:
        if _MEMORY_STRATEGY not in seen:
            hints.append(_MEMORY_STRATEGY)

    for dim, _ in summary.weak_dimensions:
        if dim == "memory_fidelity" and _MEMORY_STRATEGY not in seen:
            hints.append(_MEMORY_STRATEGY)
            seen.add(_MEMORY_STRATEGY)

    return hints
