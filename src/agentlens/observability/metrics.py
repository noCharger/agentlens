"""Custom OTEL metrics for agent observability.

These metrics feed into Grafana dashboards and eval trend analysis:
- agent.runs.total / agent.runs.success: pass rate over time
- tool.calls.total / tool.errors.total: tool reliability
- llm.latency_seconds / llm.tokens.*: cost and performance tracking
"""

from __future__ import annotations

from opentelemetry import metrics

METER_NAME = "agentlens.metrics"


class AgentMetrics:
    def __init__(self, meter: metrics.Meter | None = None):
        self._meter = meter or metrics.get_meter(METER_NAME)

        self.agent_runs_total = self._meter.create_counter(
            "agent.runs.total",
            description="Total number of agent runs",
        )
        self.agent_runs_success = self._meter.create_counter(
            "agent.runs.success",
            description="Number of successful agent runs",
        )
        self.tool_calls_total = self._meter.create_counter(
            "tool.calls.total",
            description="Total number of tool calls",
        )
        self.tool_errors_total = self._meter.create_counter(
            "tool.errors.total",
            description="Total number of tool errors",
        )
        self.llm_latency = self._meter.create_histogram(
            "llm.latency_seconds",
            description="LLM call latency in seconds",
            unit="s",
        )
        self.tool_latency = self._meter.create_histogram(
            "tool.latency_seconds",
            description="Tool call latency in seconds",
            unit="s",
        )
        self.llm_tokens_prompt = self._meter.create_histogram(
            "llm.tokens.prompt",
            description="Number of prompt tokens per LLM call",
        )
        self.llm_tokens_completion = self._meter.create_histogram(
            "llm.tokens.completion",
            description="Number of completion tokens per LLM call",
        )

    def record_agent_run(self, success: bool, scenario_id: str = "") -> None:
        attrs = {"scenario_id": scenario_id} if scenario_id else {}
        self.agent_runs_total.add(1, attrs)
        if success:
            self.agent_runs_success.add(1, attrs)

    def record_tool_call(
        self, tool_name: str, latency_s: float, error: bool = False, error_type: str = ""
    ) -> None:
        attrs = {"tool.name": tool_name}
        self.tool_calls_total.add(1, attrs)
        self.tool_latency.record(latency_s, attrs)
        if error:
            error_attrs = {**attrs, "error.type": error_type}
            self.tool_errors_total.add(1, error_attrs)

    def record_llm_call(
        self, latency_s: float, prompt_tokens: int, completion_tokens: int, model: str = ""
    ) -> None:
        attrs = {"llm.model": model} if model else {}
        self.llm_latency.record(latency_s, attrs)
        self.llm_tokens_prompt.record(prompt_tokens, attrs)
        self.llm_tokens_completion.record(completion_tokens, attrs)
