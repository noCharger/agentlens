"""AgentRunner abstraction: decouples the evolution cycle from agent creation.

Two implementations:
  EmbeddedAgentRunner — runs the agent in-process via the configured AgentRuntime.
  HttpAgentRunner     — delegates to a remote endpoint (POST /run).

Both satisfy the AgentRunner protocol, so EvolutionCycle works unchanged
regardless of where the agent actually executes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agentlens.agents.runtime import SpanView

log = logging.getLogger("agentlens.agents.runner")


@dataclass
class AgentRunResult:
    output: str
    trace: list["SpanView"] = field(default_factory=list)
    error: str | None = None


class AgentRunner(Protocol):
    def run(self, query: str, *, system_prompt: str | None = None) -> AgentRunResult: ...


class EmbeddedAgentRunner:
    """Runs the agent in-process using the configured AgentRuntime (LangGraph or AG2).

    Creates its own OTEL provider per invocation so span collection is
    independent of any outer eval context.
    """

    def __init__(self, settings, *, preset: str = "full", scenario=None):
        self._settings = settings
        self._preset = preset
        self._scenario = scenario

    def run(self, query: str, *, system_prompt: str | None = None) -> AgentRunResult:
        from agentlens.agents.runtime import create_agent_runtime
        from agentlens.eval.runner import (
            MAX_RETRIES,
            QuotaExhaustedError,
            _create_provider,
            _resolve_preset,
            _teardown,
            classify_error,
        )
        from agentlens.observability.custom_spans import set_custom_tracer_provider

        scenario_tools = (
            set(self._scenario.expected.tools_called) if self._scenario else set()
        )
        actual_preset = _resolve_preset(self._preset, scenario_tools)
        max_steps = (
            self._scenario.expected.max_steps
            if self._scenario
            else getattr(self._settings, "agent_max_steps", 10)
        )

        provider, mem_exporter = _create_provider(
            self._settings.otel_exporter_otlp_endpoint
        )
        runtime = create_agent_runtime(
            self._settings,
            preset=actual_preset,
            scenario=self._scenario,
            system_prompt=system_prompt,
        )
        instrumentor = runtime.instrument(provider)
        set_custom_tracer_provider(provider)

        output_text = ""
        error_msg: str | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                invoke_result = runtime.invoke(query, max_steps=max_steps)
                output_text = invoke_result.output_text
                break
            except Exception as exc:
                kind = classify_error(exc)
                if kind.should_stop_run:
                    error_msg = f"Quota exhausted: {exc}"
                    break
                if kind.is_retryable and attempt < MAX_RETRIES:
                    wait = 5.0 * (attempt + 1)
                    log.warning(
                        "[%s] attempt %d failed, retrying in %.0fs",
                        kind.value, attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                error_msg = f"Agent failed [{kind.value}]: {exc}"
                break
        else:
            error_msg = f"Agent did not complete after {MAX_RETRIES + 1} attempts"

        provider.force_flush()
        spans = runtime.normalize_spans(list(mem_exporter.get_finished_spans()))
        _teardown(provider, instrumentor)

        return AgentRunResult(output=output_text, trace=spans, error=error_msg)


class HttpAgentRunner:
    """Delegates agent execution to a remote HTTP endpoint.

    The endpoint must implement:

        POST /run
        Body:    {"query": str, "system_prompt": str | null}
        Response: {"output": str, "error": str | null}

    Spans are not available from remote agents, so L1 tool-usage and trajectory
    checks will be skipped. Output-based L1 checks and L2 scoring remain fully
    functional.
    """

    def __init__(self, endpoint: str, *, timeout: float = 120.0):
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def run(self, query: str, *, system_prompt: str | None = None) -> AgentRunResult:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "HttpAgentRunner requires the 'httpx' package: pip install httpx"
            ) from exc

        try:
            resp = httpx.post(
                f"{self._endpoint}/run",
                json={"query": query, "system_prompt": system_prompt},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return AgentRunResult(output="", error=str(exc))

        return AgentRunResult(
            output=data.get("output", ""),
            trace=[],
            error=data.get("error"),
        )
