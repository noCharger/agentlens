from agentlens.core.alerts import evaluate_alert_rules
from agentlens.core.exporters import build_eval_run_record
from agentlens.core.models import AlertRuleRecord, AlertSeverity

from tests.test_platform_exporters import _make_result


def test_evaluate_alert_rules_triggers_pass_rate_rule():
    eval_run = build_eval_run_record(
        [
            _make_result(passed=True),
            _make_result(
                scenario=_make_result().scenario.model_copy(update={"id": "tc-002", "name": "Broken"}),
                passed=False,
                error="boom",
            ),
        ],
        name="nightly",
    )
    rule = AlertRuleRecord(
        id="rule-pass-rate",
        name="Pass rate below 80",
        metric_key="pass_rate",
        operator="<",
        threshold=80.0,
        severity=AlertSeverity.WARNING,
    )

    events = evaluate_alert_rules(
        project_slug="qa-project",
        eval_run=eval_run,
        rules=[rule],
    )

    assert len(events) == 1
    assert events[0].rule_id == "rule-pass-rate"
    assert events[0].eval_run_id == eval_run.id
    assert events[0].observed_value == 50.0
    assert events[0].severity == AlertSeverity.WARNING


def test_evaluate_alert_rules_skips_disabled_or_non_matching():
    eval_run = build_eval_run_record([_make_result(passed=True)], name="nightly")
    disabled = AlertRuleRecord(
        id="disabled-rule",
        name="Disabled",
        metric_key="pass_rate",
        operator="<",
        threshold=99.0,
        severity=AlertSeverity.WARNING,
        enabled=False,
    )
    non_matching = AlertRuleRecord(
        id="non-matching",
        name="Pass rate under 50",
        metric_key="pass_rate",
        operator="<",
        threshold=50.0,
        severity=AlertSeverity.WARNING,
    )

    events = evaluate_alert_rules(
        project_slug="qa-project",
        eval_run=eval_run,
        rules=[disabled, non_matching],
    )

    assert events == []
