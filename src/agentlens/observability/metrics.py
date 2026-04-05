"""Custom OTEL metrics for agent observability.

These metrics feed into Grafana dashboards and eval trend analysis:
- agent.runs.*: raw run volume and pass rate
- eval.*: semantic outcomes, risk signals, failure patterns, judge scores
- tool.* / llm.*: tool reliability, cost, and performance
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
        self.eval_outcomes_total = self._meter.create_counter(
            "eval.outcomes.total",
            description="Semantic eval outcomes by status",
        )
        self.eval_risk_signals_total = self._meter.create_counter(
            "eval.risk_signals.total",
            description="Risk signals emitted by eval",
        )
        self.eval_failure_patterns_total = self._meter.create_counter(
            "eval.failure_patterns.total",
            description="Failure patterns emitted by eval",
        )
        self.eval_risk_signal_count = self._meter.create_histogram(
            "eval.risk_signal.count",
            description="Number of risk signals emitted per evaluated run",
        )
        self.eval_judge_score = self._meter.create_histogram(
            "eval.judge.score",
            description="Judge score by dimension",
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

    def record_agent_run(
        self,
        success: bool,
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {}
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.agent_runs_total.add(1, attrs)
        if success:
            self.agent_runs_success.add(1, attrs)

    def record_eval_outcome(
        self,
        status: str,
        *,
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {"eval.status": status}
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.eval_outcomes_total.add(1, attrs)

    def record_risk_signal(
        self,
        signal_type: str,
        *,
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {"risk.type": signal_type}
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.eval_risk_signals_total.add(1, attrs)

    def record_failure_pattern(
        self,
        pattern_type: str,
        *,
        severity: str = "",
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {"pattern.type": pattern_type}
        if severity:
            attrs["pattern.severity"] = severity
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.eval_failure_patterns_total.add(1, attrs)

    def record_risk_signal_count(
        self,
        count: int,
        *,
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {}
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.eval_risk_signal_count.record(count, attrs)

    def record_judge_score(
        self,
        score: float,
        *,
        dimension: str,
        benchmark: str = "",
        category: str = "",
        evaluation_mode: str = "",
    ) -> None:
        attrs = {"judge.dimension": dimension}
        if benchmark:
            attrs["benchmark"] = benchmark
        if category:
            attrs["category"] = category
        if evaluation_mode:
            attrs["evaluation_mode"] = evaluation_mode
        self.eval_judge_score.record(score, attrs)

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
