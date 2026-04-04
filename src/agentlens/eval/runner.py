"""Eval runner: orchestrates scenario execution and three-level evaluation."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from agentlens.eval.scenarios import Scenario, load_runtime_scenarios
from agentlens.eval.level1_deterministic.tool_usage import ToolUsageResult, evaluate_tool_usage
from agentlens.eval.level1_deterministic.output_format import (
    OutputFormatResult,
    evaluate_output_format,
)
from agentlens.eval.level1_deterministic.trajectory import TrajectoryResult, evaluate_trajectory
from agentlens.sandbox import prepare_benchmark_environment

log = logging.getLogger("agentlens.eval")

_PRESET_MAP: dict[frozenset[str], str] = {
    frozenset({"read_file", "write_file"}): "file_ops",
    frozenset({"read_file"}): "file_ops",
    frozenset({"shell"}): "shell",
    frozenset({"terminal"}): "shell",
    frozenset({"duckduckgo_search"}): "search",
    frozenset({"shell", "write_file", "read_file"}): "shell_file",
    frozenset({"terminal", "write_file", "read_file"}): "shell_file",
}

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class ErrorKind(str, Enum):
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    SSL_ERROR = "ssl_error"
    NETWORK_ERROR = "network_error"
    SETUP_FAILED = "setup_failed"
    AGENT_ERROR = "agent_error"

    @property
    def is_retryable(self) -> bool:
        return self in (ErrorKind.SSL_ERROR, ErrorKind.NETWORK_ERROR, ErrorKind.RATE_LIMITED)

    @property
    def should_stop_run(self) -> bool:
        return self == ErrorKind.QUOTA_EXHAUSTED


class QuotaExhaustedError(Exception):
    def __init__(self, retry_after: float = 0, message: str = ""):
        self.retry_after = retry_after
        super().__init__(message or f"Quota exhausted. Retry after {retry_after}s")


def classify_error(error: Exception) -> ErrorKind:
    msg = str(error).lower()
    if "resource_exhausted" in msg or "429" in msg:
        if any(kw in msg for kw in ("per_day", "quotavalue", "retrydelay", "retry")):
            return ErrorKind.QUOTA_EXHAUSTED
        return ErrorKind.RATE_LIMITED
    if "ssl" in msg or "eof occurred" in msg or "unexpected_eof" in msg:
        return ErrorKind.SSL_ERROR
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return ErrorKind.NETWORK_ERROR
    return ErrorKind.AGENT_ERROR


def _extract_retry_delay(error: Exception) -> float:
    msg = str(error)
    for pattern in (r"retryDelay['\"]:\s*['\"](\d+)", r"retry in (\d+(?:\.\d+)?)"):
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 60.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Level1Result:
    tool_usage: ToolUsageResult
    output_format: OutputFormatResult
    trajectory: TrajectoryResult

    @property
    def passed(self) -> bool:
        return self.tool_usage.passed and self.output_format.passed and self.trajectory.passed


@dataclass
class EvalResult:
    scenario: Scenario
    level1: Level1Result
    level2_scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def judge_overall_score(self) -> float | None:
        scores = [score for score in self.level2_scores.values() if score >= 0]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _llm_judge_level1_passed(self) -> bool:
        has_explicit_expectations = bool(
            self.scenario.expected.tools_called or self.scenario.expected.output_contains
        )
        if has_explicit_expectations:
            return self.level1.passed
        return self.level1.trajectory.passed

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False

        if self.scenario.evaluation_mode == "llm_judge":
            score = self.judge_overall_score
            return (
                self._llm_judge_level1_passed()
                and score is not None
                and score >= self.scenario.judge_threshold
            )

        if self.scenario.evaluation_mode == "external":
            return False

        return self.level1.passed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_setup_commands(commands: list[str]) -> None:
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)


def run_level1_eval(scenario: Scenario, spans: list[ReadableSpan]) -> Level1Result:
    return Level1Result(
        tool_usage=evaluate_tool_usage(spans, scenario.expected.tools_called),
        output_format=evaluate_output_format(spans, scenario.expected.output_contains),
        trajectory=evaluate_trajectory(
            spans,
            max_steps=scenario.expected.max_steps,
            max_tokens=scenario.expected.max_tokens,
        ),
    )


def evaluate_scenario(scenario: Scenario, spans: list[ReadableSpan]) -> EvalResult:
    try:
        return EvalResult(scenario=scenario, level1=run_level1_eval(scenario, spans))
    except Exception as e:
        return _error_result(scenario, str(e))


def _error_result(scenario: Scenario, error_msg: str) -> EvalResult:
    return EvalResult(
        scenario=scenario,
        level1=Level1Result(
            tool_usage=ToolUsageResult(
                passed=False,
                expected_tools=scenario.expected.tools_called,
                actual_tools=[],
                missing_tools=scenario.expected.tools_called,
                unexpected_tools=[],
            ),
            output_format=OutputFormatResult(
                passed=False, output_text="",
                expected_substrings=scenario.expected.output_contains,
                missing_substrings=scenario.expected.output_contains,
            ),
            trajectory=TrajectoryResult(
                passed=False, total_steps=0, max_steps=scenario.expected.max_steps,
                has_loop=False, total_prompt_tokens=0, total_completion_tokens=0,
                max_tokens=scenario.expected.max_tokens, reasons=[f"Error: {error_msg}"],
            ),
        ),
        error=error_msg,
    )


def _resolve_preset(preset: str, scenario_tools: set[str]) -> str:
    if preset != "full" or not scenario_tools:
        return preset
    return _PRESET_MAP.get(frozenset(scenario_tools), preset)


def _has_level2_rubric(scenario: Scenario) -> bool:
    return bool(
        (scenario.judge_rubric or "").strip()
        or (scenario.judge_rubric_text or "").strip()
    )


def _normalize_content(raw: object) -> str:
    """Normalize Gemini response content to a plain string.

    Gemini sometimes returns a list of dicts (structured content blocks)
    instead of a plain string; OTEL span attributes require primitives.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw]
        return "\n".join(parts)
    return str(raw)


def _create_provider(otlp_endpoint: str) -> tuple[TracerProvider, InMemorySpanExporter]:
    mem_exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "agentlens-eval"}))
    provider.add_span_processor(SimpleSpanProcessor(mem_exporter))
    try:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
    except Exception:
        pass
    return provider, mem_exporter


def _teardown(provider: TracerProvider, instrumentor) -> None:
    from agentlens.observability.instrument import uninstrument_langchain

    uninstrument_langchain(instrumentor)
    provider.force_flush()
    provider.shutdown()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def execute_and_eval(
    scenario: Scenario,
    settings: "AgentLensSettings",  # noqa: F821
    preset: str = "full",
    with_level2: bool = False,
    rate_limit_delay: float = 6.0,
) -> EvalResult:
    if scenario.evaluation_mode == "external":
        return _error_result(
            scenario,
            "This scenario requires an external benchmark harness and cannot be scored "
            "by the built-in AgentLens runner.",
        )

    if scenario.evaluation_mode == "llm_judge" and not with_level2:
        return _error_result(
            scenario,
            "This scenario requires LLM-as-Judge scoring. Re-run with --level2 enabled.",
        )

    from agentlens.observability.instrument import instrument_langchain

    if scenario.setup_commands:
        try:
            run_setup_commands(scenario.setup_commands)
        except subprocess.CalledProcessError as e:
            return _error_result(scenario, f"Setup failed: {e}")

    prepare_error = prepare_benchmark_environment(scenario)
    if prepare_error:
        return _error_result(scenario, prepare_error)

    provider, mem_exporter = _create_provider(settings.otel_exporter_otlp_endpoint)
    actual_preset = _resolve_preset(preset, set(scenario.expected.tools_called))
    instrumentor = instrument_langchain(provider)

    # Run agent with retries for transient errors
    result = _invoke_agent_with_retries(
        settings, actual_preset, scenario, provider, instrumentor,
    )
    if isinstance(result, EvalResult):
        return result

    # Collect spans and inject agent output for L1 evaluation
    provider.force_flush()
    final_messages = result.get("messages", [])
    if final_messages:
        last_msg = final_messages[-1]
        content = _normalize_content(
            last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        )
        tracer = provider.get_tracer("agentlens.eval")
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.output", content)
            span.set_attribute("agent.scenario_id", scenario.id)
            if scenario.benchmark:
                span.set_attribute("agent.benchmark", scenario.benchmark)
        provider.force_flush()

    from agentlens.observability.instrument import uninstrument_langchain
    uninstrument_langchain(instrumentor)

    spans = list(mem_exporter.get_finished_spans())
    eval_result = evaluate_scenario(scenario, spans)

    _record_metrics_best_effort(spans, scenario, eval_result)

    if with_level2 and _has_level2_rubric(scenario):
        _run_level2(eval_result, spans, scenario, settings)
        time.sleep(rate_limit_delay)

    provider.shutdown()
    time.sleep(rate_limit_delay)
    return eval_result


def _invoke_agent_with_retries(settings, preset, scenario, provider, instrumentor):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            from agentlens.agents.factory import create_agent

            agent = create_agent(settings, preset=preset, scenario=scenario)
            return agent.invoke(
                {"messages": [("user", scenario.input_query)]},
                config={"recursion_limit": scenario.expected.max_steps * 2},
            )
        except Exception as e:
            last_error = e
            kind = classify_error(e)

            if kind.should_stop_run:
                _teardown(provider, instrumentor)
                raise QuotaExhaustedError(
                    retry_after=_extract_retry_delay(e),
                    message=f"Gemini quota exhausted. Retry after {_extract_retry_delay(e)}s",
                )

            if kind.is_retryable and attempt < MAX_RETRIES:
                wait = 5.0 * (attempt + 1)
                log.warning(f"[{kind.value}] Attempt {attempt + 1} failed, retrying in {wait}s")
                time.sleep(wait)
                continue

            _teardown(provider, instrumentor)
            return _error_result(scenario, f"Agent failed [{kind.value}]: {e}")

    _teardown(provider, instrumentor)
    return _error_result(
        scenario,
        f"Agent failed after {MAX_RETRIES + 1} attempts: {last_error}",
    )


def _run_level2(eval_result: EvalResult, spans, scenario, settings) -> None:
    try:
        from agentlens.eval.level2_llm_judge.judge import judge_scenario, create_judge_llm

        judge_result = judge_scenario(
            llm=create_judge_llm(settings),
            spans=spans,
            query=scenario.input_query,
            reference_answer=scenario.reference_answer,
            rubric_name=scenario.judge_rubric,
            rubric_text=scenario.judge_rubric_text,
        )
        eval_result.level2_scores = {s.dimension: s.score for s in judge_result.scores}
        if not eval_result.level2_scores:
            eval_result.level2_scores = {"error": -1}
            eval_result.error = (
                "L2 judge returned no scores. Check rubric data and judge model response format."
            )
    except Exception as exc:
        eval_result.level2_scores = {"error": -1}
        eval_result.error = f"L2 judge failed: {exc}"


def _record_metrics_best_effort(
    spans: list[ReadableSpan], scenario: Scenario, result: EvalResult,
) -> None:
    try:
        from agentlens.observability.metrics import AgentMetrics

        m = AgentMetrics()
        m.record_agent_run(
            success=result.passed,
            scenario_id=scenario.id,
            benchmark=scenario.benchmark,
        )

        for span in spans:
            attrs = dict(span.attributes or {})
            duration = (
                (span.end_time - span.start_time) / 1e9
                if span.end_time and span.start_time else 0
            )

            tool_name = attrs.get("tool.name") or attrs.get("openinference.tool.name")
            if tool_name:
                is_error = span.status.status_code.name == "ERROR" if span.status else False
                m.record_tool_call(tool_name=str(tool_name), latency_s=duration, error=is_error)

            prompt_tokens = attrs.get("llm.token_count.prompt") or attrs.get(
                "llm.usage.prompt_tokens"
            )
            if prompt_tokens is not None:
                completion_tokens = (
                    attrs.get("llm.token_count.completion")
                    or attrs.get("llm.usage.completion_tokens")
                    or 0
                )
                model = str(attrs.get("llm.model_name", attrs.get("llm.model", "")))
                m.record_llm_call(
                    latency_s=duration,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    model=model,
                )
    except Exception:
        pass


def load_and_summarize(
    scenarios_dir: Path,
    benchmark_data_root: Path | None = None,
) -> list[Scenario]:
    return load_runtime_scenarios(
        scenarios_dir,
        benchmark_data_root=benchmark_data_root,
    )
