"""Prompt evolver: uses an LLM to generate improved system prompts from failure signals.

Implements the GEPA/TextGrad pattern:
- Outcome signal: delta_pass_rate from the eval run (no intermediate operation labels)
- Mechanism: LLM reads structured failure evidence + strategy hints and rewrites the prompt
- Constraint: evolved prompt must not exceed 130% of original length, must retain safety rules
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field

from agentlens.evolution.signal_analyzer import SignalSummary, build_strategy_hints


@dataclass
class EvolutionProposal:
    original_prompt: str
    evolved_prompt: str
    rationale: str
    targeted_patterns: list[str] = field(default_factory=list)


_EVOLVER_SYSTEM = textwrap.dedent("""\
    You are an expert AI agent prompt engineer. Your task is to improve a system prompt
    based on observed failure patterns from evaluation runs.

    Rules:
    1. The evolved prompt must address the listed failure patterns and strategy hints.
    2. The evolved prompt must NOT exceed 130% of the original prompt's word count.
    3. All safety instructions and tool-usage rules in the original must be preserved.
    4. Do not mention the evaluation process or these instructions in the evolved prompt.
    5. Return ONLY the evolved prompt text — no preamble, no explanation.
""")

_EVOLVER_USER_TEMPLATE = textwrap.dedent("""\
    ## Current system prompt
    {original_prompt}

    ## Evaluation results (pass rate: {pass_rate:.0%})

    ### Dominant failure patterns
    {failure_patterns}

    ### Weak evaluation dimensions (avg score < 3.5 / 5)
    {weak_dimensions}

    ### Risk signals
    {risk_signals}

    ### Memory retention score
    {memory_score}

    ### Failure evidence (L1 reasons + L2 judge explanations)
    {failure_evidence}

    ### Strategy hints (incorporate these specific behaviors)
    {strategy_hints}

    ---
    Rewrite the system prompt to address the above issues. Preserve all existing safety
    rules. Do not exceed {max_words} words (original is {orig_words} words).
""")

# Matches sentences containing safety-critical keywords.
_SAFETY_SENTENCE_RE = re.compile(
    r"[^.\n]*\b(never|must not|do not|always|shall not|prohibited|forbidden|"
    r"under no circumstances|must never|do NOT)\b[^.\n]*",
    re.IGNORECASE,
)


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_safety_rules(prompt: str) -> list[str]:
    return [m.group().strip() for m in _SAFETY_SENTENCE_RE.finditer(prompt)]


def _enforce_safety_rules(evolved: str, safety_rules: list[str]) -> str:
    if not safety_rules:
        return evolved
    missing = [
        rule for rule in safety_rules
        if rule.lower() not in evolved.lower()
    ]
    if not missing:
        return evolved
    return evolved.rstrip() + "\n\n" + "\n".join(missing)


def _truncate_to_word_budget(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Prefer a clean sentence boundary if one falls in the last 20% of the budget.
    last_stop = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_stop > len(truncated) * 0.8:
        return truncated[:last_stop + 1].strip()
    return truncated.strip()


def _render_list(items: list[tuple[str, int | float]], label: str) -> str:
    if not items:
        return f"No {label} detected."
    return "\n".join(f"- {name} (count/score: {val})" for name, val in items)


def evolve_prompt(
    summary: SignalSummary,
    current_prompt: str,
    *,
    model: str,
    settings,
) -> EvolutionProposal:
    """Generate an evolved system prompt using an LLM optimizer.

    Args:
        summary: Signal summary from analyze_signals().
        current_prompt: The agent's current system prompt.
        model: Model identifier for the optimizer LLM.
        settings: AgentLensSettings for LLM instantiation.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from agentlens.llms import create_chat_llm

    strategy_hints = build_strategy_hints(summary)
    targeted_patterns = [p for p, _ in summary.dominant_failure_patterns]

    memory_score_str = (
        f"{summary.memory_retention_score:.2f}" if summary.memory_retention_score is not None
        else "N/A (no memory scenarios)"
    )

    orig_words = _count_words(current_prompt)
    max_words = int(orig_words * 1.3)
    safety_rules = _extract_safety_rules(current_prompt)

    user_text_base = _EVOLVER_USER_TEMPLATE.format(
        original_prompt=current_prompt,
        pass_rate=summary.pass_rate,
        failure_patterns=_render_list(summary.dominant_failure_patterns, "failure patterns"),
        weak_dimensions=_render_list(summary.weak_dimensions, "weak dimensions"),
        risk_signals=_render_list(summary.frequent_risk_signals, "risk signals"),
        memory_score=memory_score_str,
        failure_evidence="\n".join(
            f"- {e}" for e in summary.failure_evidence
        ) or "No failure evidence captured.",
        strategy_hints="\n".join(
            f"- {h}" for h in strategy_hints
        ) or "No specific hints.",
        max_words=max_words,
        orig_words=orig_words,
    )

    llm = create_chat_llm(settings, model)
    evolved = current_prompt  # safe fallback

    for attempt in range(3):
        user_text = user_text_base
        if attempt > 0:
            over_by = _count_words(evolved) - max_words
            user_text += (
                f"\n\nPREVIOUS ATTEMPT WAS {over_by} WORDS OVER BUDGET. "
                f"You must produce a response of at most {max_words} words. "
                "Be more concise while keeping all safety rules."
            )

        response = llm.invoke([
            SystemMessage(content=_EVOLVER_SYSTEM),
            HumanMessage(content=user_text),
        ])
        candidate = response.content.strip() if hasattr(response, "content") else str(response).strip()

        if not candidate or candidate == current_prompt:
            continue

        candidate = _enforce_safety_rules(candidate, safety_rules)

        if _count_words(candidate) <= max_words:
            evolved = candidate
            break

        evolved = candidate  # keep for next retry's over-budget message
    else:
        # All attempts over budget — truncate at sentence boundary.
        evolved = _truncate_to_word_budget(evolved, max_words)
        evolved = _enforce_safety_rules(evolved, safety_rules)

    rationale = (
        f"Targeted patterns: {', '.join(targeted_patterns) or 'none'}. "
        f"Weak dimensions: {', '.join(d for d, _ in summary.weak_dimensions) or 'none'}. "
        f"Memory retention: {memory_score_str}."
    )

    return EvolutionProposal(
        original_prompt=current_prompt,
        evolved_prompt=evolved,
        rationale=rationale,
        targeted_patterns=targeted_patterns,
    )
