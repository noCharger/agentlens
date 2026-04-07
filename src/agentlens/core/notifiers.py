"""Alert notification channels.

Dispatches alert events to external systems:
- Webhook (generic HTTP POST)
- Slack (incoming webhook)
- Console (logging, for development)

Extensible via the Notifier protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agentlens.core.models import AlertEventRecord

log = logging.getLogger("agentlens.notifiers")


class Notifier(ABC):
    """Base class for alert notification channels."""

    @abstractmethod
    def send(self, event: AlertEventRecord) -> bool:
        """Send an alert notification. Returns True if successful."""
        ...

    def send_batch(self, events: list[AlertEventRecord]) -> list[bool]:
        return [self.send(e) for e in events]


class ConsoleNotifier(Notifier):
    """Logs alerts to console (development/testing)."""

    def send(self, event: AlertEventRecord) -> bool:
        log.warning(
            "[ALERT %s] %s | severity=%s | metric=%s %s %s (observed=%.2f)",
            event.id,
            event.message,
            event.severity.value,
            event.metric_key,
            event.operator,
            event.threshold,
            event.observed_value,
        )
        return True


class WebhookNotifier(Notifier):
    """Sends alerts via HTTP POST to a webhook URL."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, event: AlertEventRecord) -> bool:
        import httpx

        payload = {
            "alert_id": event.id,
            "rule_id": event.rule_id,
            "eval_run_id": event.eval_run_id,
            "severity": event.severity.value,
            "message": event.message,
            "metric_key": event.metric_key,
            "observed_value": event.observed_value,
            "threshold": event.threshold,
            "triggered_at": event.triggered_at.isoformat(),
        }
        try:
            response = httpx.post(self.url, json=payload, headers=self.headers, timeout=10)
            return response.status_code < 400
        except Exception as e:
            log.error("Webhook notification failed: %s", e)
            return False


class SlackNotifier(Notifier):
    """Sends alerts to Slack via incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send(self, event: AlertEventRecord) -> bool:
        import httpx

        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        emoji = severity_emoji.get(event.severity.value, "📢")

        payload = {
            "text": f"{emoji} *AgentLens Alert* [{event.severity.value.upper()}]",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} AgentLens Alert",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Severity:* `{event.severity.value}`\n"
                            f"*Message:* {event.message}\n"
                            f"*Metric:* `{event.metric_key}` {event.operator} {event.threshold}\n"
                            f"*Observed:* {event.observed_value:.2f}\n"
                            f"*Run:* `{event.eval_run_id}`"
                        ),
                    },
                },
            ],
        }
        try:
            response = httpx.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            return response.status_code < 400
        except Exception as e:
            log.error("Slack notification failed: %s", e)
            return False


@dataclass
class NotificationDispatcher:
    """Dispatches alerts to multiple notification channels."""

    channels: list[Notifier] = field(default_factory=list)

    def add_channel(self, notifier: Notifier) -> None:
        self.channels.append(notifier)

    def dispatch(self, event: AlertEventRecord) -> dict[str, bool]:
        """Send alert to all channels. Returns success status per channel."""
        results = {}
        for channel in self.channels:
            name = type(channel).__name__
            results[name] = channel.send(event)
        return results

    def dispatch_batch(self, events: list[AlertEventRecord]) -> None:
        """Send multiple alerts to all channels."""
        for event in events:
            self.dispatch(event)
