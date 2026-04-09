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
from agentlens.eval.level1_deterministic.trajectory import (
    TrajectoryAnalysis,
    TrajectoryResult,
    analyze_trajectory,
)
from agentlens.eval.level1_deterministic.tool_params import (
    ToolParamsResult,
    ExpectedToolParam as ToolParamSpec,
    evaluate_tool_params,
)
from agentlens.eval.level1_deterministic.termination import TerminationResult, evaluate_termination
from agentlens.eval.level1_deterministic.safety import SafetyResult, evaluate_safety
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
_FEATURE_FLAG_NAMES = (
    "geval",
    "task_completion",
    "answer_relevancy",
    "hallucination",
    "faithfulness",
)


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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@dataclass
class Level1Result:
    tool_usage: ToolUsageResult
    output_format: OutputFormatResult
    trajectory: TrajectoryResult
    trajectory_analysis: TrajectoryAnalysis | None = None
    tool_params: ToolParamsResult | None = None
    termination: TerminationResult | None = None
    safety: SafetyResult | None = None

    @property
    def passed(self) -> bool:
        base = self.tool_usage.passed and self.output_format.passed and self.trajectory.passed
        if not base:
            return False
        if self.tool_params is not None and not self.tool_params.passed:
            return False
        if self.termination is not None and not self.termination.passed:
            return False
        if self.safety is not None and not self.safety.passed:
            return False
        return True

    @property
    def supplemental_checks(self) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        if self.tool_params is not None:
            checks["params"] = self.tool_params.passed
        if self.termination is not None:
            checks["termination"] = self.termination.passed
        if self.safety is not None:
            checks["safety"] = self.safety.passed
        return checks

    @property
    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []

        if not self.tool_usage.passed:
            if self.tool_usage.missing_tools:
                reasons.append(f"Missing tools: {', '.join(self.tool_usage.missing_tools)}")
            if self.tool_usage.unexpected_tools:
                reasons.append(f"Unexpected tools: {', '.join(self.tool_usage.unexpected_tools)}")
            if not self.tool_usage.missing_tools and not self.tool_usage.unexpected_tools:
                reasons.append("Tool usage check failed.")

        if not self.output_format.passed:
            if self.output_format.missing_substrings:
                reasons.append(
                    f"Missing output: {', '.join(self.output_format.missing_substrings)}"
                )
            else:
                reasons.append("Output format check failed.")

        if not self.trajectory.passed:
            if self.trajectory.reasons:
                reasons.extend(self.trajectory.reasons)
            else:
                reasons.append("Trajectory check failed.")

        if self.tool_params is not None and not self.tool_params.passed:
            for violation in self.tool_params.violations:
                reasons.append(
                    "Tool params "
                    f"{violation.tool_name}.{violation.param_name}: {violation.reason}"
                )

        if self.termination is not None and not self.termination.passed:
            if self.termination.reasons:
                reasons.extend(self.termination.reasons)
            else:
                reasons.append("Termination check failed.")

        if self.safety is not None and not self.safety.passed:
            for violation in self.safety.violations:
                reasons.append(f"Safety {violation.violation_type}: {violation.description}")

        return _dedupe_preserve_order(reasons)


@dataclass
class EvalResult:
    scenario: Scenario
    level1: Level1Result
    level2_scores: dict[str, float] = field(default_factory=dict)
    level2_explanations: dict[str, str] = field(default_factory=dict)
    feature_flags: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    risk_signals: list[str] = field(default_factory=list)

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
        from agentlens.core.models import TraceStatus
        return self.status in (TraceStatus.PASSED, TraceStatus.RISKY_SUCCESS)

    @property
    def status(self):  # -> TraceStatus
        """Four-category evaluation status.

        - PASSED: all checks pass, no risk signals
        - PARTIAL_SUCCESS: core goal achieved but some checks failed
        - RISKY_SUCCESS: all checks pass but risk signals detected
        - FAILED: core checks failed
        - ERROR: execution error
        """
        from agentlens.core.models import TraceStatus

        if self.error is not None:
            return TraceStatus.ERROR

        if self.scenario.evaluation_mode == "external":
            return TraceStatus.FAILED

        if self.scenario.evaluation_mode == "llm_judge":
            score = self.judge_overall_score
            core_passed = (
                self._llm_judge_level1_passed()
                and score is not None
                and score >= self.scenario.judge_threshold
            )
        else:
            core_passed = self.level1.passed

        if not core_passed:
            if self._is_partial_success():
                return TraceStatus.PARTIAL_SUCCESS
            return TraceStatus.FAILED

        if self.risk_signals:
            return TraceStatus.RISKY_SUCCESS

        return TraceStatus.PASSED

    def _is_partial_success(self) -> bool:
        """Partial success: output is correct but tool usage or trajectory has issues."""
        if self.error is not None:
            return False
        output_ok = self.level1.output_format.passed
        tool_ok = self.level1.tool_usage.passed
        trajectory_ok = self.level1.trajectory.passed
        if output_ok and (not tool_ok or not trajectory_ok):
            return True
        if tool_ok and not output_ok and self.level1.output_format.missing_substrings:
            total = len(self.level1.output_format.expected_substrings)
            missing = len(self.level1.output_format.missing_substrings)
            if total > 0 and missing < total:
                return True
        return False

    @property
    def level2_reason_lines(self) -> list[str]:
        reasons: list[str] = []
        for dimension, score in self.level2_scores.items():
            if score < 0:
                continue
            explanation = self.level2_explanations.get(dimension, "").strip()
            if explanation:
                reasons.append(f"{dimension} ({score}/5): {explanation}")
            else:
                reasons.append(f"{dimension} ({score}/5)")
        return reasons


def run_setup_commands(commands: list[str]) -> None:
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)


def detect_risk_signals(spans: list[ReadableSpan], scenario: Scenario) -> list[str]:
    """Detect risk signals in agent execution trajectory.

    Risk signals indicate the agent achieved its goal but exhibited
    potentially dangerous behavior during execution.
    """
    signals: list[str] = []

    tool_names = []
    for span in spans:
        attrs = dict(span.attributes or {})
        name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
        if name:
            tool_names.append(str(name))

    from collections import Counter
    tool_counts = Counter(tool_names)
    for tool, count in tool_counts.items():
        if count >= 5:
            signals.append(f"excessive_retries:{tool}({count} calls)")

    privileged_tools = {"shell", "terminal", "write_file"}
    expected_set = set(scenario.expected.tools_called)
    for tool in tool_names:
        if tool in privileged_tools and tool not in expected_set:
            signals.append(f"unexpected_privileged_tool:{tool}")
            break

    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("sandbox.blocked"):
            signals.append(f"blocked_command_attempt:{attrs.get('sandbox.blocked_command', 'unknown')}")

    if scenario.expected.max_tokens:
        prompt_tokens = 0
        completion_tokens = 0
        for span in spans:
            attrs = dict(span.attributes or {})
            pt = attrs.get("llm.token_count.prompt") or attrs.get("llm.usage.prompt_tokens") or 0
            ct = attrs.get("llm.token_count.completion") or attrs.get("llm.usage.completion_tokens") or 0
            prompt_tokens += int(pt)
            completion_tokens += int(ct)
        total = prompt_tokens + completion_tokens
        if total > scenario.expected.max_tokens * 0.9:
            signals.append(f"near_token_limit:{total}/{scenario.expected.max_tokens}")

    return signals


def run_level1_eval(scenario: Scenario, spans: list[ReadableSpan]) -> Level1Result:
    trajectory_analysis = analyze_trajectory(
        spans,
        max_steps=scenario.expected.max_steps,
        max_tokens=scenario.expected.max_tokens,
        available_tool_count=len(set(scenario.expected.tools_called)),
    )
    result = Level1Result(
        tool_usage=evaluate_tool_usage(spans, scenario.expected.tools_called),
        output_format=evaluate_output_format(spans, scenario.expected.output_contains),
        trajectory=trajectory_analysis.basic,
        trajectory_analysis=trajectory_analysis,
    )

    if scenario.expected.tool_params:
        param_specs = [
            ToolParamSpec(
                tool_name=p.tool_name,
                param_name=p.param_name,
                required=p.required,
                expected_value=p.expected_value,
                forbidden_values=p.forbidden_values,
            )
            for p in scenario.expected.tool_params
        ]
        result.tool_params = evaluate_tool_params(spans, param_specs)

    if scenario.expected.min_steps > 0 or scenario.expected.expected_escalation:
        result.termination = evaluate_termination(
            spans,
            expected_min_steps=scenario.expected.min_steps,
            expected_escalation=scenario.expected.expected_escalation,
            max_steps_after_answer=scenario.expected.max_steps_after_answer,
        )

    if scenario.expected.safety_checks:
        result.safety = evaluate_safety(
            spans,
            extra_forbidden_patterns=scenario.expected.forbidden_patterns or None,
        )

    return result


def evaluate_scenario(scenario: Scenario, spans: list[ReadableSpan]) -> EvalResult:
    try:
        level1 = run_level1_eval(scenario, spans)
        risk_signals = detect_risk_signals(spans, scenario)
        if level1.safety and level1.safety.violations:
            for v in level1.safety.violations:
                risk_signals.append(f"safety_{v.violation_type}:{v.description}")
        return EvalResult(scenario=scenario, level1=level1, risk_signals=risk_signals)
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


def _risk_signal_type(signal: str) -> str:
    return signal.split(":", 1)[0].strip()


def _annotate_eval_span(span, scenario: Scenario, result: EvalResult) -> None:
    span.set_attribute("agent.scenario_name", scenario.name)
    span.set_attribute("eval.status", result.status.value)
    span.set_attribute("eval.passed", result.passed)
    span.set_attribute("eval.has_risk_signal", bool(result.risk_signals))
    span.set_attribute("eval.risk_signal_count", len(result.risk_signals))
    span.set_attribute("eval.level1.passed", result.level1.passed)
    span.set_attribute("eval.level1.tool_usage_passed", result.level1.tool_usage.passed)
    span.set_attribute("eval.level1.output_format_passed", result.level1.output_format.passed)
    span.set_attribute("eval.level1.trajectory_passed", result.level1.trajectory.passed)
    if result.level1.tool_params is not None:
        span.set_attribute("eval.level1.tool_params_passed", result.level1.tool_params.passed)
    if result.level1.termination is not None:
        span.set_attribute("eval.level1.termination_passed", result.level1.termination.passed)
    if result.level1.safety is not None:
        span.set_attribute("eval.level1.safety_passed", result.level1.safety.passed)
        span.set_attribute("eval.level1.safety_violation_count", len(result.level1.safety.violations))
    if result.error:
        span.set_attribute("eval.error", result.error)
    if result.judge_overall_score is not None:
        span.set_attribute("eval.judge.overall_score", result.judge_overall_score)
    if scenario.evaluation_mode == "llm_judge":
        span.set_attribute("eval.judge.threshold", scenario.judge_threshold)
    for flag_name in _FEATURE_FLAG_NAMES:
        span.set_attribute(
            f"eval.flags.{flag_name}",
            bool(result.feature_flags.get(flag_name, False)),
        )

    analysis = result.level1.trajectory_analysis
    if analysis is not None:
        failure_map = analysis.failure_map
        span.set_attribute("eval.failure_pattern_count", len(failure_map.patterns))
        span.set_attribute("eval.has_failure_pattern", bool(failure_map.patterns))
        span.set_attribute("eval.failure_map.risk_score", failure_map.risk_score)
        if failure_map.dominant_pattern:
            span.set_attribute("eval.failure_pattern_dominant", failure_map.dominant_pattern)

    for signal in result.risk_signals:
        span.add_event(
            "eval.risk_signal",
            {
                "risk.type": _risk_signal_type(signal),
                "risk.signal": signal,
            },
        )

    if analysis is not None:
        for pattern in analysis.failure_map.patterns:
            span.add_event(
                "eval.failure_pattern",
                {
                    "pattern.type": pattern.pattern_type,
                    "pattern.severity": pattern.severity,
                    "pattern.description": pattern.description,
                },
            )

    for dimension, score in result.level2_scores.items():
        if score >= 0:
            span.add_event(
                "eval.judge_score",
                {
                    "judge.dimension": dimension,
                    "judge.score": float(score),
                },
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


def _is_endpoint_reachable(endpoint: str, timeout: float = 1.0) -> bool:
    """Quick TCP probe to check if the OTLP collector is listening."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


def _create_provider(otlp_endpoint: str) -> tuple[TracerProvider, InMemorySpanExporter]:
    mem_exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "agentlens-eval"}))
    provider.add_span_processor(SimpleSpanProcessor(mem_exporter))
    if _is_endpoint_reachable(otlp_endpoint):
        try:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        except Exception:
            pass
    else:
        log.debug("OTLP collector at %s not reachable, skipping remote export", otlp_endpoint)
    return provider, mem_exporter


def _teardown(provider: TracerProvider, instrumentor) -> None:
    from agentlens.observability.instrument import uninstrument_runtime

    uninstrument_runtime(instrumentor)
    provider.force_flush()
    provider.shutdown()


def execute_and_eval(
    scenario: Scenario,
    settings: "AgentLensSettings",  # noqa: F821
    preset: str = "full",
    with_level2: bool = False,
    rate_limit_delay: float = 6.0,
    *,
    use_geval: bool = False,
    task_completion: bool = False,
    answer_relevancy: bool = False,
    hallucination: bool = False,
    faithfulness: bool = False,
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

    from agentlens.observability.custom_spans import (
        agent_run_span,
        finalize_run_span,
        set_custom_tracer_provider,
    )
    from agentlens.agents.runtime import create_agent_runtime

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
    agent_framework = getattr(settings, "agent_framework", "langgraph")
    runtime = create_agent_runtime(settings, preset=actual_preset, scenario=scenario)
    instrumentor = runtime.instrument(provider)
    set_custom_tracer_provider(provider)

    final_result: EvalResult | None = None
    quota_error: QuotaExhaustedError | None = None
    resolved_feature_flags = {
        "geval": use_geval,
        "task_completion": task_completion,
        "answer_relevancy": answer_relevancy,
        "hallucination": hallucination,
        "faithfulness": faithfulness,
    }

    try:
        with agent_run_span(
            scenario_id=scenario.id,
            query=scenario.input_query,
            benchmark=scenario.benchmark,
            category=scenario.category,
            evaluation_mode=scenario.evaluation_mode,
        ) as run_span:
            run_span.set_attribute("agent.preset", actual_preset)
            run_span.set_attribute("agent.framework", agent_framework)

            try:
                invoke_result = _invoke_agent_with_retries(runtime, scenario)
            except KeyboardInterrupt:
                finalize_run_span(run_span, total_steps=0, success=False, error="Interrupted")
                raise
            except QuotaExhaustedError as exc:
                final_result = _error_result(scenario, str(exc))
                final_result.feature_flags = dict(resolved_feature_flags)
                _annotate_eval_span(run_span, scenario, final_result)
                _record_metrics_best_effort([], scenario, final_result)
                finalize_run_span(run_span, total_steps=0, success=False, error=str(exc))
                quota_error = exc
            else:
                if isinstance(invoke_result, EvalResult):
                    final_result = invoke_result
                    final_result.feature_flags = dict(resolved_feature_flags)
                    _annotate_eval_span(run_span, scenario, final_result)
                    _record_metrics_best_effort([], scenario, final_result)
                    finalize_run_span(
                        run_span,
                        total_steps=final_result.level1.trajectory.total_steps,
                        success=final_result.passed,
                        error=final_result.error,
                    )
                else:
                    output_text = invoke_result.output_text
                    if output_text:
                        tracer = provider.get_tracer("agentlens.eval")
                        with tracer.start_as_current_span("agent.output") as output_span:
                            output_span.set_attribute("agent.output", output_text)
                            output_span.set_attribute("agent.scenario_id", scenario.id)
                            output_span.set_attribute("agent.framework", agent_framework)

                    provider.force_flush()
                    spans = runtime.normalize_spans(list(mem_exporter.get_finished_spans()))
                    final_result = evaluate_scenario(scenario, spans)

                    _feature_flags = dict(
                        use_geval=use_geval,
                        task_completion=task_completion,
                        answer_relevancy=answer_relevancy,
                        hallucination=hallucination,
                        faithfulness=faithfulness,
                    )
                    _has_extensions = any(_feature_flags.values())
                    if with_level2 and (_has_level2_rubric(scenario) or _has_extensions):
                        _run_level2(
                            final_result, spans, scenario, settings,
                            **_feature_flags,
                        )
                        provider.force_flush()
                        spans = runtime.normalize_spans(list(mem_exporter.get_finished_spans()))

                    final_result.feature_flags = dict(resolved_feature_flags)
                    _annotate_eval_span(run_span, scenario, final_result)
                    _record_metrics_best_effort(spans, scenario, final_result)
                    finalize_run_span(
                        run_span,
                        total_steps=final_result.level1.trajectory.total_steps,
                        success=final_result.passed,
                        output=output_text,
                        error=final_result.error,
                    )
    finally:
        set_custom_tracer_provider(None)
        _teardown(provider, instrumentor)

    if quota_error is not None:
        raise quota_error
    if final_result is None:
        final_result = _error_result(scenario, "Eval run did not produce a result.")
    final_result.feature_flags = dict(resolved_feature_flags)
    time.sleep(rate_limit_delay)
    return final_result


def _invoke_agent_with_retries(runtime, scenario):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return runtime.invoke(
                scenario.input_query,
                max_steps=scenario.expected.max_steps,
            )
        except Exception as e:
            last_error = e
            kind = classify_error(e)

            if kind.should_stop_run:
                raise QuotaExhaustedError(
                    retry_after=_extract_retry_delay(e),
                    message=f"Gemini quota exhausted. Retry after {_extract_retry_delay(e)}s",
                )

            if kind.is_retryable and attempt < MAX_RETRIES:
                wait = 5.0 * (attempt + 1)
                log.warning(f"[{kind.value}] Attempt {attempt + 1} failed, retrying in {wait}s")
                time.sleep(wait)
                continue

            return _error_result(scenario, f"Agent failed [{kind.value}]: {e}")

    return _error_result(
        scenario,
        f"Agent failed after {MAX_RETRIES + 1} attempts: {last_error}",
    )


def _run_level2(
    eval_result: EvalResult, spans, scenario, settings,
    *,
    use_geval: bool = False,
    task_completion: bool = False,
    answer_relevancy: bool = False,
    hallucination: bool = False,
    faithfulness: bool = False,
) -> None:
    try:
        from agentlens.eval.level2_llm_judge.judge import create_judge_llm

        llm = create_judge_llm(settings)
        if use_geval:
            from agentlens.eval.level2_llm_judge.geval import clear_step_cache

            clear_step_cache()
        eval_result.level2_scores = {}
        eval_result.level2_explanations = {}

        if _has_level2_rubric(scenario):
            if use_geval:
                from agentlens.eval.level2_llm_judge.geval import geval_judge_scenario

                judge_result = geval_judge_scenario(
                    llm=llm,
                    spans=spans,
                    query=scenario.input_query,
                    reference_answer=scenario.reference_answer,
                    rubric_name=scenario.judge_rubric,
                    rubric_text=scenario.judge_rubric_text,
                )
            else:
                from agentlens.eval.level2_llm_judge.judge import judge_scenario

                judge_result = judge_scenario(
                    llm=llm,
                    spans=spans,
                    query=scenario.input_query,
                    reference_answer=scenario.reference_answer,
                    rubric_name=scenario.judge_rubric,
                    rubric_text=scenario.judge_rubric_text,
                )
            eval_result.level2_scores = {s.dimension: s.score for s in judge_result.scores}
            eval_result.level2_explanations = {
                s.dimension: s.explanation for s in judge_result.scores if s.explanation
            }
            if not eval_result.level2_scores:
                eval_result.level2_scores = {"error": -1}
                eval_result.error = (
                    "L2 judge returned no scores. Check rubric data and judge model response format."
                )
        else:
            eval_result.level2_scores = {}
            eval_result.level2_explanations = {}

        if task_completion:
            try:
                from agentlens.eval.level2_llm_judge.task_completion import evaluate_task_completion

                score = evaluate_task_completion(llm, spans, scenario.input_query)
                eval_result.level2_scores[score.dimension] = score.score
                if score.explanation:
                    eval_result.level2_explanations[score.dimension] = score.explanation
            except Exception as exc:
                log.warning("Task completion metric failed: %s", exc)

        if answer_relevancy:
            try:
                from agentlens.eval.level2_llm_judge.answer_relevancy import evaluate_answer_relevancy

                score = evaluate_answer_relevancy(llm, spans, scenario.input_query)
                eval_result.level2_scores[score.dimension] = score.score
                if score.explanation:
                    eval_result.level2_explanations[score.dimension] = score.explanation
            except Exception as exc:
                log.warning("Answer relevancy metric failed: %s", exc)

        if hallucination:
            try:
                from agentlens.eval.level2_llm_judge.hallucination import evaluate_hallucination

                context = getattr(scenario, "context", None) or []
                score = evaluate_hallucination(llm, spans, scenario.input_query, context=context or None)
                eval_result.level2_scores[score.dimension] = score.score
                if score.explanation:
                    eval_result.level2_explanations[score.dimension] = score.explanation
            except Exception as exc:
                log.warning("Hallucination metric failed: %s", exc)

        if faithfulness:
            try:
                from agentlens.eval.level2_llm_judge.faithfulness import evaluate_faithfulness

                context = getattr(scenario, "context", None) or []
                score = evaluate_faithfulness(llm, spans, scenario.input_query, context=context or None)
                eval_result.level2_scores[score.dimension] = score.score
                if score.explanation:
                    eval_result.level2_explanations[score.dimension] = score.explanation
            except Exception as exc:
                log.warning("Faithfulness metric failed: %s", exc)

    except Exception as exc:
        eval_result.level2_scores = {"error": -1}
        eval_result.level2_explanations = {}
        eval_result.error = f"L2 judge failed: {exc}"


def _record_metrics_best_effort(
    spans: list[ReadableSpan], scenario: Scenario, result: EvalResult,
) -> None:
    try:
        from agentlens.observability.metrics import AgentMetrics

        m = AgentMetrics()
        m.record_agent_run(
            success=result.passed,
            benchmark=scenario.benchmark,
            category=scenario.category,
            evaluation_mode=scenario.evaluation_mode,
        )
        m.record_eval_outcome(
            result.status.value,
            benchmark=scenario.benchmark,
            category=scenario.category,
            evaluation_mode=scenario.evaluation_mode,
        )
        m.record_risk_signal_count(
            len(result.risk_signals),
            benchmark=scenario.benchmark,
            category=scenario.category,
            evaluation_mode=scenario.evaluation_mode,
        )

        for signal in result.risk_signals:
            m.record_risk_signal(
                _risk_signal_type(signal),
                benchmark=scenario.benchmark,
                category=scenario.category,
                evaluation_mode=scenario.evaluation_mode,
            )

        if result.level1.trajectory_analysis is not None:
            for pattern in result.level1.trajectory_analysis.failure_map.patterns:
                m.record_failure_pattern(
                    pattern.pattern_type,
                    severity=pattern.severity,
                    benchmark=scenario.benchmark,
                    category=scenario.category,
                    evaluation_mode=scenario.evaluation_mode,
                )

        for dimension, score in result.level2_scores.items():
            if score >= 0:
                m.record_judge_score(
                    float(score),
                    dimension=dimension,
                    benchmark=scenario.benchmark,
                    category=scenario.category,
                    evaluation_mode=scenario.evaluation_mode,
                )
        if result.judge_overall_score is not None:
            m.record_judge_score(
                result.judge_overall_score,
                dimension="overall",
                benchmark=scenario.benchmark,
                category=scenario.category,
                evaluation_mode=scenario.evaluation_mode,
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
