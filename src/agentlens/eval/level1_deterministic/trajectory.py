"""Trajectory analysis and failure diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class FailurePattern:
    """A detected failure pattern in the trajectory."""
    pattern_type: str
    description: str
    severity: str
    evidence: list[str] = field(default_factory=list)

@dataclass
class TrajectoryResult:
    """Basic trajectory metrics."""
    passed: bool
    total_steps: int
    max_steps: int
    has_loop: bool
    total_prompt_tokens: int
    total_completion_tokens: int
    max_tokens: int | None
    reasons: list[str]


@dataclass
class StructuralAnalysis:
    """Structural trajectory analysis."""
    tool_sequence: list[str]
    unique_tools_used: int
    tool_switch_count: int
    strategy_drift_score: float  # 0.0 = consistent, 1.0 = chaotic
    subtask_switches: int
    avg_tools_per_subtask: float


@dataclass
class FailureMapResult:
    """Failure pattern diagnosis output."""
    patterns: list[FailurePattern]
    dominant_pattern: str | None
    risk_score: float

    @property
    def has_failures(self) -> bool:
        return len(self.patterns) > 0


@dataclass
class TrajectoryAnalysis:
    """Combined trajectory analysis result."""
    basic: TrajectoryResult
    structural: StructuralAnalysis
    failure_map: FailureMapResult

def count_steps(spans: list[ReadableSpan]) -> int:
    """Count ReAct loop iterations.

    OpenInference emits one AGENT span per LangGraph node invocation.
    Each `agent` node call = one ReAct step (think -> act -> observe).
    Falls back to counting our custom `agent.step` spans if present.
    """
    agent_steps = sum(
        1 for s in spans
        if dict(s.attributes or {}).get("openinference.span.kind") == "AGENT"
    )
    if agent_steps > 0:
        return agent_steps

    return sum(1 for s in spans if s.name == "agent.step")


def detect_loop(spans: list[ReadableSpan], max_repeats: int = 3) -> bool:
    """Detect repeated identical tool calls (indicates stuck agent).

    Checks both OpenInference TOOL spans and custom agent.step spans.
    """
    actions = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("openinference.span.kind") == "TOOL":
            tool = attrs.get("tool.name", "")
            params = attrs.get("input.value", "")
            actions.append(f"{tool}:{params}")
        elif span.name == "agent.step":
            action = attrs.get("step.action", "")
            if action:
                actions.append(str(action))

    if len(actions) < max_repeats:
        return False

    for i in range(len(actions) - max_repeats + 1):
        window = actions[i: i + max_repeats]
        if len(set(window)) == 1 and window[0]:
            return True
    return False


def sum_tokens(spans: list[ReadableSpan]) -> tuple[int, int]:
    """Sum prompt and completion tokens from LLM spans."""
    prompt_total = 0
    completion_total = 0
    for span in spans:
        attrs = dict(span.attributes or {})
        pt = attrs.get("llm.token_count.prompt") or attrs.get("llm.usage.prompt_tokens") or 0
        ct = (
            attrs.get("llm.token_count.completion")
            or attrs.get("llm.usage.completion_tokens")
            or 0
        )
        prompt_total += int(pt)
        completion_total += int(ct)
    return prompt_total, completion_total


def evaluate_trajectory(
    spans: list[ReadableSpan],
    max_steps: int = 10,
    max_tokens: int | None = None,
) -> TrajectoryResult:
    steps = count_steps(spans)
    has_loop = detect_loop(spans)
    prompt_tokens, completion_tokens = sum_tokens(spans)
    total_tokens = prompt_tokens + completion_tokens

    reasons = []
    passed = True

    if steps > max_steps:
        passed = False
        reasons.append(f"Step count {steps} exceeds max {max_steps}")

    if has_loop:
        passed = False
        reasons.append("Detected repeated identical actions (loop)")

    if max_tokens is not None and total_tokens > max_tokens:
        passed = False
        reasons.append(f"Token usage {total_tokens} exceeds max {max_tokens}")

    return TrajectoryResult(
        passed=passed,
        total_steps=steps,
        max_steps=max_steps,
        has_loop=has_loop,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        max_tokens=max_tokens,
        reasons=reasons,
    )


def _extract_tool_sequence(spans: list[ReadableSpan]) -> list[str]:
    """Extract ordered tool names from spans."""
    tools = []
    for span in spans:
        attrs = dict(span.attributes or {})
        name = (
            attrs.get("tool.name")
            or attrs.get("tool_call.function.name")
        )
        if name:
            tools.append((span.start_time or 0, str(name)))
        elif attrs.get("openinference.span.kind") == "TOOL":
            name = attrs.get("tool.name", "unknown")
            tools.append((span.start_time or 0, str(name)))

    tools.sort(key=lambda t: t[0])
    return [name for _, name in tools]


def _count_tool_switches(sequence: list[str]) -> int:
    """Count how many times the agent switched between different tools."""
    if len(sequence) < 2:
        return 0
    return sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])


def _compute_strategy_drift(sequence: list[str]) -> float:
    """Measure how chaotic the tool usage pattern is.

    Strategy drift = ratio of unique tool transitions to total transitions.
    Low drift (near 0) = consistent strategy; high drift (near 1) = chaotic.
    """
    if len(sequence) < 2:
        return 0.0
    transitions = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
    unique_transitions = len(set(transitions))
    return round(unique_transitions / len(transitions), 3)


def _count_subtask_switches(spans: list[ReadableSpan]) -> int:
    """Count subtask boundary switches from agent.step spans."""
    steps = []
    for span in spans:
        attrs = dict(span.attributes or {})
        if span.name == "agent.step":
            thought = str(attrs.get("step.thought", ""))
            steps.append(thought)
        elif attrs.get("openinference.span.kind") == "AGENT":
            steps.append(span.name)

    if len(steps) < 2:
        return 0

    switches = 0
    for i in range(1, len(steps)):
        if steps[i] and steps[i - 1] and steps[i] != steps[i - 1]:
            words_prev = set(steps[i - 1].lower().split())
            words_curr = set(steps[i].lower().split())
            if words_prev and words_curr:
                overlap = len(words_prev & words_curr) / max(len(words_prev), len(words_curr))
                if overlap < 0.3:
                    switches += 1
    return switches


def analyze_structure(spans: list[ReadableSpan]) -> StructuralAnalysis:
    """Perform structural analysis of the trajectory."""
    tool_sequence = _extract_tool_sequence(spans)
    unique = len(set(tool_sequence))
    switches = _count_tool_switches(tool_sequence)
    drift = _compute_strategy_drift(tool_sequence)
    subtask_sw = _count_subtask_switches(spans)

    avg_tools = (
        len(tool_sequence) / max(subtask_sw + 1, 1)
        if tool_sequence else 0.0
    )

    return StructuralAnalysis(
        tool_sequence=tool_sequence,
        unique_tools_used=unique,
        tool_switch_count=switches,
        strategy_drift_score=drift,
        subtask_switches=subtask_sw,
        avg_tools_per_subtask=round(avg_tools, 2),
    )


def _detect_fuzzy_guessing(
    spans: list[ReadableSpan],
    structure: StructuralAnalysis,
) -> FailurePattern | None:
    """Detect high-drift, trial-and-error behavior."""
    if len(structure.tool_sequence) < 3:
        return None

    if structure.strategy_drift_score > 0.7 and structure.tool_switch_count >= 3:
        evidence = [
            f"strategy_drift={structure.strategy_drift_score}",
            f"tool_switches={structure.tool_switch_count}",
            f"unique_tools={structure.unique_tools_used}",
        ]
        return FailurePattern(
            pattern_type="fuzzy_guessing",
            description="Agent appears to be guessing — high strategy drift with frequent tool switches",
            severity="medium",
            evidence=evidence,
        )
    return None


def _detect_tool_confusion(
    spans: list[ReadableSpan],
    structure: StructuralAnalysis,
    available_tool_count: int = 0,
) -> FailurePattern | None:
    """Detect when the agent selects wrong tools due to too many options.

    Signals: many unexpected tools, tool calls don't match the task.
    """
    if len(structure.tool_sequence) < 2:
        return None

    counts = Counter(structure.tool_sequence)
    single_use = sum(1 for c in counts.values() if c == 1)

    if len(counts) >= 3 and single_use >= len(counts) * 0.6:
        evidence = [
            f"tools_tried={len(counts)}",
            f"single_use_tools={single_use}",
            f"available_tools={available_tool_count}" if available_tool_count else "",
        ]
        return FailurePattern(
            pattern_type="tool_confusion",
            description="Agent tried many tools without committing — possible tool selection confusion",
            severity="medium",
            evidence=[e for e in evidence if e],
        )
    return None


def _detect_context_forgetting(
    spans: list[ReadableSpan],
) -> FailurePattern | None:
    """Detect repeated distant tool+parameter calls."""
    tool_calls_with_params: list[str] = []
    for span in spans:
        attrs = dict(span.attributes or {})
        name = attrs.get("tool.name")
        params = attrs.get("input.value", "") or attrs.get("tool.params", "")
        if name:
            tool_calls_with_params.append(f"{name}:{params}")

    if len(tool_calls_with_params) < 4:
        return None

    seen: dict[str, list[int]] = {}
    for i, call in enumerate(tool_calls_with_params):
        if call not in seen:
            seen[call] = []
        seen[call].append(i)

    repeated = {k: v for k, v in seen.items() if len(v) > 1}
    distant_repeats = {
        k: v for k, v in repeated.items()
        if any(v[j + 1] - v[j] > 2 for j in range(len(v) - 1))
    }

    if len(distant_repeats) >= 2:
        evidence = [f"repeated_calls={list(distant_repeats.keys())[:3]}"]
        return FailurePattern(
            pattern_type="context_forgetting",
            description="Agent repeated earlier work — may have forgotten previous results",
            severity="medium",
            evidence=evidence,
        )
    return None


def _detect_loop_trap(
    spans: list[ReadableSpan],
    basic: TrajectoryResult,
    structure: StructuralAnalysis,
) -> FailurePattern | None:
    """Detect when the agent gets stuck in a loop during subtask transitions.

    Extends basic loop detection with subtask-aware analysis.
    """
    if not basic.has_loop:
        return None

    if structure.subtask_switches > 0:
        evidence = [
            f"loop_detected=True",
            f"subtask_switches={structure.subtask_switches}",
            f"steps={basic.total_steps}",
        ]
        return FailurePattern(
            pattern_type="loop_trap",
            description="Agent stuck in loop while switching between subtasks",
            severity="high",
            evidence=evidence,
        )

    return FailurePattern(
        pattern_type="loop_trap",
        description="Agent stuck repeating identical actions",
        severity="high",
        evidence=[f"steps={basic.total_steps}"],
    )


def _detect_missing_confirmation(
    spans: list[ReadableSpan],
) -> FailurePattern | None:
    """Detect privileged actions that skip read/verification steps."""
    privileged_tools = {"shell", "terminal", "write_file"}
    read_tools = {"read_file", "duckduckgo_search"}

    tool_sequence: list[str] = []
    for span in spans:
        attrs = dict(span.attributes or {})
        name = attrs.get("tool.name") or attrs.get("tool_call.function.name")
        if name:
            tool_sequence.append(str(name))

    if not tool_sequence:
        return None

    if tool_sequence[0] in privileged_tools and len(tool_sequence) > 1:
        return FailurePattern(
            pattern_type="missing_confirmation",
            description=f"Agent's first action was privileged tool '{tool_sequence[0]}' without prior read/check",
            severity="medium",
            evidence=[f"first_tool={tool_sequence[0]}"],
        )

    consecutive_privileged = 0
    for tool in tool_sequence:
        if tool in privileged_tools:
            consecutive_privileged += 1
            if consecutive_privileged >= 3:
                return FailurePattern(
                    pattern_type="missing_confirmation",
                    description=f"Agent executed {consecutive_privileged} consecutive privileged operations without verification",
                    severity="high",
                    evidence=[f"consecutive_privileged={consecutive_privileged}"],
                )
        elif tool in read_tools:
            consecutive_privileged = 0

    return None


def _detect_confabulation(
    spans: list[ReadableSpan],
) -> FailurePattern | None:
    """Detect detailed outputs with little or no tool grounding."""
    tool_count = 0
    output_text = ""

    for span in spans:
        attrs = dict(span.attributes or {})
        if attrs.get("tool.name") or attrs.get("tool_call.function.name"):
            tool_count += 1
        output = attrs.get("agent.output") or attrs.get("output.value")
        if output:
            output_text = str(output)

    if not output_text:
        return None

    output_words = len(output_text.split())
    if output_words > 50 and tool_count == 0:
        return FailurePattern(
            pattern_type="confabulation",
            description=f"Agent produced detailed output ({output_words} words) with zero tool calls",
            severity="high",
            evidence=[f"output_words={output_words}", f"tool_calls={tool_count}"],
        )

    if output_words > 100 and tool_count <= 1:
        return FailurePattern(
            pattern_type="confabulation",
            description=f"Agent produced very detailed output ({output_words} words) with only {tool_count} tool call",
            severity="medium",
            evidence=[f"output_words={output_words}", f"tool_calls={tool_count}"],
        )

    return None


def detect_failure_patterns(
    spans: list[ReadableSpan],
    basic: TrajectoryResult,
    structure: StructuralAnalysis,
    *,
    available_tool_count: int = 0,
) -> FailureMapResult:
    """Run all six failure pattern detectors."""
    patterns: list[FailurePattern] = []

    detectors = [
        lambda: _detect_fuzzy_guessing(spans, structure),
        lambda: _detect_tool_confusion(spans, structure, available_tool_count),
        lambda: _detect_context_forgetting(spans),
        lambda: _detect_loop_trap(spans, basic, structure),
        lambda: _detect_missing_confirmation(spans),
        lambda: _detect_confabulation(spans),
    ]

    for detect_fn in detectors:
        pattern = detect_fn()
        if pattern is not None:
            patterns.append(pattern)

    severity_weights = {"low": 0.2, "medium": 0.5, "high": 0.8}
    if patterns:
        risk_score = max(severity_weights.get(p.severity, 0) for p in patterns)
        dominant = max(patterns, key=lambda p: severity_weights.get(p.severity, 0))
        dominant_type = dominant.pattern_type
    else:
        risk_score = 0.0
        dominant_type = None

    return FailureMapResult(
        patterns=patterns,
        dominant_pattern=dominant_type,
        risk_score=risk_score,
    )


def analyze_trajectory(
    spans: list[ReadableSpan],
    *,
    max_steps: int = 10,
    max_tokens: int | None = None,
    available_tool_count: int = 0,
) -> TrajectoryAnalysis:
    """Run full trajectory analysis."""
    basic = evaluate_trajectory(spans, max_steps=max_steps, max_tokens=max_tokens)
    structure = analyze_structure(spans)
    failure_map = detect_failure_patterns(
        spans, basic, structure,
        available_tool_count=available_tool_count,
    )
    return TrajectoryAnalysis(
        basic=basic,
        structural=structure,
        failure_map=failure_map,
    )
