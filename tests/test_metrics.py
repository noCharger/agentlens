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
    m.record_agent_run(success=True, benchmark="swe-bench-pro", category="qa")

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
    m.record_agent_run(
        success=True,
        benchmark="swe-bench-pro",
        category="qa",
        evaluation_mode="deterministic",
    )

    attrs = _get_metric_attributes(reader, "agent.runs.total")
    assert attrs == {
        "benchmark": "swe-bench-pro",
        "category": "qa",
        "evaluation_mode": "deterministic",
    }


def test_record_eval_outcome():
    m, reader = _make_metrics()
    m.record_eval_outcome(
        "risky_success",
        benchmark="swe-bench-pro",
        category="qa",
        evaluation_mode="llm_judge",
    )

    total = _get_metric_value(reader, "eval.outcomes.total")
    attrs = _get_metric_attributes(reader, "eval.outcomes.total")
    assert total == 1
    assert attrs == {
        "eval.status": "risky_success",
        "benchmark": "swe-bench-pro",
        "category": "qa",
        "evaluation_mode": "llm_judge",
    }


def test_record_risk_signal_and_failure_pattern():
    m, reader = _make_metrics()
    m.record_risk_signal("unexpected_privileged_tool", benchmark="bench", category="ops")
    m.record_failure_pattern("loop_trap", severity="high", benchmark="bench", category="ops")

    risk_total = _get_metric_value(reader, "eval.risk_signals.total")
    failure_total = _get_metric_value(reader, "eval.failure_patterns.total")
    assert risk_total == 1
    assert failure_total == 1


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


def test_record_eval_judge_score_and_risk_count():
    m, reader = _make_metrics()
    m.record_risk_signal_count(2, benchmark="bench")
    m.record_judge_score(4.5, dimension="overall", benchmark="bench")

    data = reader.get_metrics_data()
    metric_names = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                metric_names.add(metric.name)
    assert "eval.risk_signal.count" in metric_names
    assert "eval.judge.score" in metric_names
