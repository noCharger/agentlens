"""Alert rule evaluation helpers for monitoring eval run health."""

from __future__ import annotations

import hashlib
from typing import Callable

from agentlens.core.models import AlertEventRecord, AlertRuleRecord, EvalRunRecord

Operator = Callable[[float, float], bool]

_OPERATORS: dict[str, Operator] = {
    ">": lambda observed, threshold: observed > threshold,
    ">=": lambda observed, threshold: observed >= threshold,
    "<": lambda observed, threshold: observed < threshold,
    "<=": lambda observed, threshold: observed <= threshold,
    "==": lambda observed, threshold: observed == threshold,
}


def _metric_value(eval_run: EvalRunRecord, metric_key: str) -> float:
    if metric_key == "pass_rate":
        return float(eval_run.summary.pass_rate)
    if metric_key == "failed_cases":
        return float(eval_run.summary.failed)
    if metric_key == "passed_cases":
        return float(eval_run.summary.passed)
    if metric_key == "total_cases":
        return float(eval_run.summary.total)
    raise ValueError(f"Unsupported alert metric_key '{metric_key}'")


def _event_id(project_slug: str, eval_run_id: str, rule_id: str) -> str:
    digest = hashlib.sha256(
        f"{project_slug}:{eval_run_id}:{rule_id}".encode("utf-8")
    ).hexdigest()
    return f"alert_event_{digest[:16]}"


def evaluate_alert_rules(
    *,
    project_slug: str,
    eval_run: EvalRunRecord,
    rules: list[AlertRuleRecord],
) -> list[AlertEventRecord]:
    events: list[AlertEventRecord] = []
    for rule in rules:
        if not rule.enabled:
            continue
        operator = _OPERATORS.get(rule.operator)
        if operator is None:
            continue
        observed = _metric_value(eval_run, rule.metric_key)
        if not operator(observed, float(rule.threshold)):
            continue

        events.append(
            AlertEventRecord(
                id=_event_id(project_slug, eval_run.id, rule.id),
                rule_id=rule.id,
                eval_run_id=eval_run.id,
                metric_key=rule.metric_key,
                operator=rule.operator,
                threshold=float(rule.threshold),
                observed_value=observed,
                severity=rule.severity,
                message=(
                    f"Rule '{rule.name}' triggered: "
                    f"{rule.metric_key} {rule.operator} {rule.threshold} "
                    f"(observed={observed:.2f})"
                ),
                metadata={
                    "rule_name": rule.name,
                    "eval_run_name": eval_run.name,
                    "eval_run_source": eval_run.source,
                },
            )
        )
    return events
