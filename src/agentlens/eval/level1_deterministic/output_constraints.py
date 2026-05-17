"""Level 1 evaluator: output constraint checking (IFEval-style).

Machine-checkable format and content constraints that verify without an LLM judge.
Covers 20+ instruction types matching Google's IFEval benchmark design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.scenarios import OutputConstraints


@dataclass
class ConstraintViolation:
    constraint: str
    expected: str
    actual: str


@dataclass
class OutputConstraintResult:
    passed: bool
    violations: list[ConstraintViolation] = field(default_factory=list)


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return sum(1 for p in parts if p.strip())


def _count_paragraphs(text: str) -> int:
    parts = re.split(r"\n\s*\n", text.strip())
    return sum(1 for p in parts if p.strip())


def _count_bullet_points(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*[*\-]\s", line))


def _count_highlighted_sections(text: str) -> int:
    return len(re.findall(r"\*[^*\n]+\*", text))


def _detect_language(text: str) -> str | None:
    try:
        from langdetect import detect  # type: ignore[import]
        return detect(text)
    except Exception:
        return None


def evaluate_output_constraints(
    output_text: str,
    constraints: OutputConstraints,
    *,
    prompt: str = "",
) -> OutputConstraintResult:
    """Check all configured output constraints against the agent's response."""
    violations: list[ConstraintViolation] = []

    # --- Keywords ---
    output_lower = output_text.lower()
    for kw in constraints.required_keywords:
        if kw.lower() not in output_lower:
            violations.append(ConstraintViolation(
                constraint="required_keyword",
                expected=f"contains '{kw}'",
                actual="not found",
            ))
    for fw in constraints.forbidden_words:
        if fw.lower() in output_lower:
            violations.append(ConstraintViolation(
                constraint="forbidden_word",
                expected=f"not '{fw}'",
                actual="found",
            ))

    # --- Language ---
    if constraints.response_language:
        detected = _detect_language(output_text)
        if detected is not None and detected != constraints.response_language:
            violations.append(ConstraintViolation(
                constraint="response_language",
                expected=constraints.response_language,
                actual=detected,
            ))

    # --- Length: words ---
    word_count = _count_words(output_text)
    if constraints.word_count_min is not None and word_count < constraints.word_count_min:
        violations.append(ConstraintViolation(
            constraint="word_count_min",
            expected=f">= {constraints.word_count_min}",
            actual=str(word_count),
        ))
    if constraints.word_count_max is not None and word_count > constraints.word_count_max:
        violations.append(ConstraintViolation(
            constraint="word_count_max",
            expected=f"<= {constraints.word_count_max}",
            actual=str(word_count),
        ))

    # --- Length: sentences ---
    sentence_count = _count_sentences(output_text)
    if constraints.sentence_count_min is not None and sentence_count < constraints.sentence_count_min:
        violations.append(ConstraintViolation(
            constraint="sentence_count_min",
            expected=f">= {constraints.sentence_count_min}",
            actual=str(sentence_count),
        ))
    if constraints.sentence_count_max is not None and sentence_count > constraints.sentence_count_max:
        violations.append(ConstraintViolation(
            constraint="sentence_count_max",
            expected=f"<= {constraints.sentence_count_max}",
            actual=str(sentence_count),
        ))

    # --- Length: paragraphs ---
    paragraph_count = _count_paragraphs(output_text)
    if constraints.paragraph_count_min is not None and paragraph_count < constraints.paragraph_count_min:
        violations.append(ConstraintViolation(
            constraint="paragraph_count_min",
            expected=f">= {constraints.paragraph_count_min}",
            actual=str(paragraph_count),
        ))
    if constraints.paragraph_count_max is not None and paragraph_count > constraints.paragraph_count_max:
        violations.append(ConstraintViolation(
            constraint="paragraph_count_max",
            expected=f"<= {constraints.paragraph_count_max}",
            actual=str(paragraph_count),
        ))

    # --- Format ---
    if constraints.json_format:
        try:
            json.loads(output_text)
        except (json.JSONDecodeError, ValueError):
            violations.append(ConstraintViolation(
                constraint="json_format",
                expected="valid JSON",
                actual="invalid JSON",
            ))

    if constraints.no_comma and "," in output_text:
        violations.append(ConstraintViolation(
            constraint="no_comma",
            expected="no commas",
            actual="commas found",
        ))

    if constraints.all_uppercase and output_text != output_text.upper():
        violations.append(ConstraintViolation(
            constraint="all_uppercase",
            expected="all uppercase",
            actual="not all uppercase",
        ))

    if constraints.all_lowercase and output_text != output_text.lower():
        violations.append(ConstraintViolation(
            constraint="all_lowercase",
            expected="all lowercase",
            actual="not all lowercase",
        ))

    if constraints.starts_with and not output_text.startswith(constraints.starts_with):
        prefix = output_text[: len(constraints.starts_with)]
        violations.append(ConstraintViolation(
            constraint="starts_with",
            expected=f"starts with '{constraints.starts_with}'",
            actual=f"starts with '{prefix}'",
        ))

    if constraints.ends_with and not output_text.rstrip().endswith(constraints.ends_with):
        violations.append(ConstraintViolation(
            constraint="ends_with",
            expected=f"ends with '{constraints.ends_with}'",
            actual="does not end with expected string",
        ))

    # --- Structure ---
    if constraints.bullet_list_count is not None:
        actual_bullets = _count_bullet_points(output_text)
        if actual_bullets != constraints.bullet_list_count:
            violations.append(ConstraintViolation(
                constraint="bullet_list_count",
                expected=str(constraints.bullet_list_count),
                actual=str(actual_bullets),
            ))

    if constraints.num_highlighted_sections is not None:
        actual_highlights = _count_highlighted_sections(output_text)
        if actual_highlights != constraints.num_highlighted_sections:
            violations.append(ConstraintViolation(
                constraint="num_highlighted_sections",
                expected=str(constraints.num_highlighted_sections),
                actual=str(actual_highlights),
            ))

    if constraints.has_postscript and not re.search(r"P\.S\.", output_text):
        violations.append(ConstraintViolation(
            constraint="has_postscript",
            expected="contains 'P.S.'",
            actual="not found",
        ))

    if constraints.title_format and not re.search(r"<<[^<>]+>>", output_text):
        violations.append(ConstraintViolation(
            constraint="title_format",
            expected="contains <<title>> format",
            actual="no <<...>> found",
        ))

    if constraints.two_responses and "******" not in output_text:
        violations.append(ConstraintViolation(
            constraint="two_responses",
            expected="contains '******' separator",
            actual="separator not found",
        ))

    if constraints.repeat_prompt and prompt and prompt.strip() not in output_text:
        violations.append(ConstraintViolation(
            constraint="repeat_prompt",
            expected="output contains the original prompt",
            actual="prompt not found in output",
        ))

    if constraints.constrained_to:
        normalized = output_text.strip().lower()
        allowed = [v.lower() for v in constraints.constrained_to]
        if normalized not in allowed:
            violations.append(ConstraintViolation(
                constraint="constrained_to",
                expected=f"one of {constraints.constrained_to}",
                actual=output_text.strip()[:50],
            ))

    return OutputConstraintResult(passed=len(violations) == 0, violations=violations)
