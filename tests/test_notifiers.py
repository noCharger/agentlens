"""Tests for alert notification channels."""

from agentlens.core.models import AlertEventRecord, AlertSeverity
from agentlens.core.notifiers import (
    ConsoleNotifier,
    NotificationDispatcher,
)


def _make_event() -> AlertEventRecord:
    return AlertEventRecord(
        id="test_event_1",
        rule_id="rule_1",
        eval_run_id="run_1",
        metric_key="pass_rate",
        operator="<",
        threshold=80.0,
        observed_value=65.0,
        severity=AlertSeverity.WARNING,
        message="Pass rate below threshold",
    )


class TestConsoleNotifier:
    def test_send_returns_true(self):
        notifier = ConsoleNotifier()
        result = notifier.send(_make_event())
        assert result is True

    def test_batch_send(self):
        notifier = ConsoleNotifier()
        results = notifier.send_batch([_make_event(), _make_event()])
        assert all(results)


class TestDispatcher:
    def test_dispatch_to_channel(self):
        dispatcher = NotificationDispatcher()
        dispatcher.add_channel(ConsoleNotifier())

        results = dispatcher.dispatch(_make_event())
        assert len(results) == 1
        assert results["ConsoleNotifier"] is True

    def test_dispatch_batch(self):
        dispatcher = NotificationDispatcher()
        dispatcher.add_channel(ConsoleNotifier())
        dispatcher.dispatch_batch([_make_event(), _make_event()])

    def test_empty_dispatcher(self):
        dispatcher = NotificationDispatcher()
        results = dispatcher.dispatch(_make_event())
        assert results == {}
