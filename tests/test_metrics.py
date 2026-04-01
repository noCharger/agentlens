from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from agentlens.observability.metrics import AgentMetrics


def _make_metrics() -> tuple[AgentMetrics, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    return AgentMetrics(meter=meter), reader


def _get_metric_value(reader: InMemoryMetricReader, name: str) -> int | float | None:
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        return dp.value
    return None


def _get_metric_attributes(reader: InMemoryMetricReader, name: str) -> dict | None:
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        return dict(dp.attributes)
    return None


def test_record_agent_run_success():
    m, reader = _make_metrics()
    m.record_agent_run(success=True, scenario_id="tc-001")

    total = _get_metric_value(reader, "agent.runs.total")
    success = _get_metric_value(reader, "agent.runs.success")
    assert total == 1
    assert success == 1


def test_record_agent_run_failure():
    m, reader = _make_metrics()
    m.record_agent_run(success=False)

    total = _get_metric_value(reader, "agent.runs.total")
    success = _get_metric_value(reader, "agent.runs.success")
    assert total == 1
    assert success is None  # counter never incremented


def test_record_agent_run_includes_benchmark_attribute():
    m, reader = _make_metrics()
    m.record_agent_run(success=True, scenario_id="tc-001", benchmark="swe-bench-pro")

    attrs = _get_metric_attributes(reader, "agent.runs.total")
    assert attrs == {"scenario_id": "tc-001", "benchmark": "swe-bench-pro"}


def test_record_tool_call():
    m, reader = _make_metrics()
    m.record_tool_call(tool_name="shell", latency_s=0.5)

    total = _get_metric_value(reader, "tool.calls.total")
    assert total == 1


def test_record_tool_error():
    m, reader = _make_metrics()
    m.record_tool_call(tool_name="read_file", latency_s=0.1, error=True, error_type="ENOENT")

    total = _get_metric_value(reader, "tool.calls.total")
    errors = _get_metric_value(reader, "tool.errors.total")
    assert total == 1
    assert errors == 1


def test_record_llm_call():
    m, reader = _make_metrics()
    m.record_llm_call(latency_s=1.2, prompt_tokens=100, completion_tokens=50, model="flash")

    # Histograms expose sum, count, etc. Just verify no errors.
    data = reader.get_metrics_data()
    metric_names = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                metric_names.add(metric.name)
    assert "llm.latency_seconds" in metric_names
    assert "llm.tokens.prompt" in metric_names
    assert "llm.tokens.completion" in metric_names
