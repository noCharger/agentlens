# AgentLens Observability Stack

## Quick Start

```bash
docker compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Tempo | http://localhost:3200 | — |
| OTEL Collector (gRPC) | localhost:4317 | — |
| OTEL Collector (HTTP) | localhost:4318 | — |

## Recommended Split

Use the dashboard as a layered view:

- Runtime health: LLM latency, tool failures, throughput, span distribution
- Eval semantics: outcome mix, risky runs, failure patterns, judge scores
- Trace view: per-request detail, including raw spans plus eval-derived events on `agent.run`

## Prometheus Sample Queries

### Agent Runs

```promql
# Total evaluated runs
sum(traces_span_metrics_calls_total{span_name="agent.run"})

# Eval pass rate (%)
sum(traces_span_metrics_calls_total{span_name="agent.run", eval_status=~"passed|risky_success"})
/ sum(traces_span_metrics_calls_total{span_name="agent.run"}) * 100

# Evaluated runs per minute
sum(rate(traces_span_metrics_calls_total{span_name="agent.run"}[5m])) * 60
```

### Eval Signals

```promql
# Outcome mix
sum by (eval_status) (traces_span_metrics_calls_total{span_name="agent.run", eval_status!=""})

# Outcome rate over time
sum(rate(traces_span_metrics_calls_total{span_name="agent.run", eval_status!=""}[10m])) by (eval_status)

# Risk signals by type
sum by (risk_type) (eval_risk_signals_total)

# Failure patterns by type
sum by (pattern_type) (eval_failure_patterns_total)

# Average judge score by dimension
sum(rate(eval_judge_score_sum[30m])) by (judge_dimension)
/ sum(rate(eval_judge_score_count[30m])) by (judge_dimension)
```

### LLM Metrics

```promql
# Total LLM calls (all providers: Gemini, OpenAI/OpenRouter/Zhipu, DeepSeek)
sum(traces_span_metrics_calls_total{span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic"})

# LLM calls by provider
sum by (span_name) (traces_span_metrics_calls_total{span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic"})

# LLM p50 latency (ms)
histogram_quantile(0.5,
  sum(rate(traces_span_metrics_duration_milliseconds_bucket{
    span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic"
  }[10m])) by (le)
)

# LLM p95 latency (ms)
histogram_quantile(0.95,
  sum(rate(traces_span_metrics_duration_milliseconds_bucket{
    span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic"
  }[10m])) by (le)
)

# LLM p95 latency by provider
histogram_quantile(0.95,
  sum(rate(traces_span_metrics_duration_milliseconds_bucket{
    span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic"
  }[10m])) by (le, span_name)
)

# LLM error rate
sum(rate(traces_span_metrics_calls_total{
  span_name=~"ChatGoogleGenerativeAI|ChatOpenAI|ChatDeepSeek|ChatAnthropic",
  status_code="STATUS_CODE_ERROR"
}[10m]))
```

### Tool Metrics

```promql
# Total tool calls
sum(traces_span_metrics_calls_total{tool_name!=""})

# Tool calls by tool name
sum by (tool_name) (traces_span_metrics_calls_total{tool_name!=""})

# Tool error rate (%)
sum(rate(traces_span_metrics_calls_total{tool_name!="", status_code="STATUS_CODE_ERROR"}[10m]))
/ sum(rate(traces_span_metrics_calls_total{tool_name!=""}[10m])) * 100

# Tool p95 latency by tool
histogram_quantile(0.95,
  sum(rate(traces_span_metrics_duration_milliseconds_bucket{tool_name!=""}[10m])) by (le, tool_name)
)
```

### Operations

```promql
# All operations by call count
sum by (span_name) (traces_span_metrics_calls_total{service_name="agentlens-eval"})

# Slowest operations (p95 ms)
histogram_quantile(0.95,
  sum(rate(traces_span_metrics_duration_milliseconds_bucket{service_name="agentlens-eval"}[30m])) by (le, span_name)
)

# Span throughput over time
sum(rate(traces_span_metrics_calls_total{service_name="agentlens-eval"}[10m])) by (span_name)
```

## Alert Rules

Prometheus alert rules are defined in `prometheus-alerts.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| AgentSuccessRateLow | Success rate < 50% for 5m | critical |
| LLMLatencyHigh | LLM p95 > 30s for 5m | warning |
| ToolErrorRateHigh | Tool error rate > 20% for 5m | warning |
| AgentRunsStalled | No runs in 30m | info |
| LLMErrorSpike | LLM errors > 0.1/sec for 5m | warning |

View active alerts at http://localhost:9090/alerts

## Tempo Trace Queries

In Grafana Explore (Tempo datasource):

```
# Find all traces for a service
{ service.name = "agentlens-eval" }

# Find traces with errors
{ service.name = "agentlens-eval" && status = error }

# Find traces by span name
{ span.name = "agent.run" }

# Find risky evals
{ span.name = "agent.run" && span.eval.status = "risky_success" }

# Find failed or partial evals
{ span.name = "agent.run" && span.eval.status =~ "failed|partial_success|error" }

# Find traces using a specific tool
{ span.tool.name = "shell" }
```

In the trace waterfall, use the root `agent.run` span for semantic context:

- Attributes: `eval.status`, `eval.risk_signal_count`, `eval.judge.overall_score`, L1 pass/fail flags
- Events: `eval.risk_signal`, `eval.failure_pattern`, `eval.judge_score`

Click any trace ID in the Recent Traces table to view the full trace waterfall with both runtime and eval context.

## Architecture

```
Agent Eval Runner
      │
      ▼ (OTLP gRPC :4317)
OTEL Collector
      │
      ├──▶ Tempo (:3200)           ← raw spans + eval-enriched root traces
      ├──▶ spanmetrics connector   ← derives metrics from spans
      │         │
      │         ▼
      └──▶ Prometheus (:9090)      ← raw + semantic metrics
                │
                ▼
           Grafana (:3001)         ← runtime health + eval semantics
```
