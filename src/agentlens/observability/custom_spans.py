"""Custom spans for eval-driven observability.

These spans supplement what OpenInference auto-captures, providing data
that the eval harness needs but OpenInference doesn't track:
- agent.step: ReAct step-level spans (thought, action, step_index)
- recovery events: error recovery tracking for eval Level 2
- agent.run: root span with scenario metadata for eval correlation
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.trace import StatusCode, Tracer, TracerProvider

TRACER_NAME = "agentlens.custom"

_tracer_provider: TracerProvider | None = None


def set_custom_tracer_provider(provider: TracerProvider) -> None:
    global _tracer_provider
    _tracer_provider = provider


def get_tracer() -> Tracer:
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(TRACER_NAME)
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def agent_run_span(
    scenario_id: str | None = None,
    query: str = "",
    benchmark: str = "",
    category: str = "",
    evaluation_mode: str = "",
) -> Generator[trace.Span, None, None]:
    tracer = get_tracer()
    with tracer.start_as_current_span("agent.run") as span:
        if scenario_id:
            span.set_attribute("agent.scenario_id", scenario_id)
        if benchmark:
            span.set_attribute("agent.benchmark", benchmark)
        if category:
            span.set_attribute("agent.category", category)
        if evaluation_mode:
            span.set_attribute("eval.evaluation_mode", evaluation_mode)
        span.set_attribute("agent.input_query", query)
        span.set_attribute("agent.total_steps", 0)
        span.set_attribute("agent.success", False)
        yield span


@contextmanager
def agent_step_span(
    step_index: int,
    thought: str = "",
    action: str = "",
) -> Generator[trace.Span, None, None]:
    tracer = get_tracer()
    with tracer.start_as_current_span("agent.step") as span:
        span.set_attribute("step.index", step_index)
        span.set_attribute("step.thought", thought)
        span.set_attribute("step.action", action)
        yield span


def record_recovery_event(
    span: trace.Span,
    error_type: str,
    retry_count: int,
    fallback_action: str = "",
) -> None:
    span.add_event(
        "recovery",
        attributes={
            "recovery.error_type": error_type,
            "recovery.retry_count": retry_count,
            "recovery.fallback_action": fallback_action,
        },
    )


def finalize_run_span(
    span: trace.Span,
    total_steps: int,
    success: bool,
    output: str = "",
    error: str | None = None,
) -> None:
    span.set_attribute("agent.total_steps", total_steps)
    span.set_attribute("agent.success", success)
    if output:
        span.set_attribute("agent.output", output)
    if error:
        span.set_status(StatusCode.ERROR, error)
    else:
        span.set_status(StatusCode.OK)
