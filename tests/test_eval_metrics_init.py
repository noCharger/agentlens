from types import SimpleNamespace

from agentlens.eval.__main__ import _init_metrics


def test_init_metrics_skips_unreachable_collector(monkeypatch):
    monkeypatch.setattr("agentlens.eval.__main__._is_endpoint_reachable", lambda endpoint: False)

    provider = _init_metrics(
        SimpleNamespace(otel_exporter_otlp_endpoint="http://localhost:4317")
    )

    assert provider is None
