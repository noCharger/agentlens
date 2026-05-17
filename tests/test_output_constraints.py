"""Tests for IFEval-style output constraint checking."""

from __future__ import annotations

from agentlens.eval.level1_deterministic.output_constraints import (
    evaluate_output_constraints,
)
from agentlens.eval.scenarios import OutputConstraints


def _check(text: str, **kwargs) -> bool:
    c = OutputConstraints(**kwargs)
    return evaluate_output_constraints(text, c).passed


def _violations(text: str, **kwargs) -> list[str]:
    c = OutputConstraints(**kwargs)
    result = evaluate_output_constraints(text, c)
    return [v.constraint for v in result.violations]


class TestRequiredKeywords:
    def test_present_passes(self):
        assert _check("Hello world", required_keywords=["world"])

    def test_absent_fails(self):
        assert not _check("Hello world", required_keywords=["python"])

    def test_case_insensitive(self):
        assert _check("Hello WORLD", required_keywords=["world"])

    def test_multiple_all_present(self):
        assert _check("foo bar baz", required_keywords=["foo", "bar"])

    def test_multiple_one_missing(self):
        assert not _check("foo bar", required_keywords=["foo", "qux"])

    def test_violation_constraint_name(self):
        assert "required_keyword" in _violations("hi", required_keywords=["missing"])


class TestForbiddenWords:
    def test_absent_passes(self):
        assert _check("Hello world", forbidden_words=["bad"])

    def test_present_fails(self):
        assert not _check("This is bad", forbidden_words=["bad"])

    def test_case_insensitive(self):
        assert not _check("This is BAD", forbidden_words=["bad"])

    def test_violation_constraint_name(self):
        assert "forbidden_word" in _violations("bad word", forbidden_words=["bad"])


class TestWordCount:
    def test_min_passes_when_enough_words(self):
        assert _check("one two three", word_count_min=3)

    def test_min_fails_when_too_few(self):
        assert not _check("one two", word_count_min=3)

    def test_max_passes_when_within_limit(self):
        assert _check("one two three", word_count_max=5)

    def test_max_fails_when_too_many(self):
        assert not _check("one two three four five six", word_count_max=5)

    def test_both_min_and_max_pass(self):
        assert _check("one two three", word_count_min=2, word_count_max=5)

    def test_min_violation_name(self):
        assert "word_count_min" in _violations("hi", word_count_min=10)

    def test_max_violation_name(self):
        assert "word_count_max" in _violations("a b c d e f", word_count_max=3)


class TestSentenceCount:
    def test_min_passes(self):
        assert _check("Hello. World.", sentence_count_min=2)

    def test_min_fails(self):
        assert not _check("Hello.", sentence_count_min=3)

    def test_max_passes(self):
        assert _check("Hello.", sentence_count_max=2)

    def test_max_fails(self):
        assert not _check("One. Two. Three. Four.", sentence_count_max=2)

    def test_question_mark_counted(self):
        assert _check("Really? Yes!", sentence_count_min=2)


class TestParagraphCount:
    def test_min_passes(self):
        text = "Para one.\n\nPara two."
        assert _check(text, paragraph_count_min=2)

    def test_min_fails(self):
        assert not _check("Single paragraph.", paragraph_count_min=2)

    def test_max_passes(self):
        assert _check("Single paragraph.", paragraph_count_max=2)

    def test_max_fails(self):
        text = "A.\n\nB.\n\nC."
        assert not _check(text, paragraph_count_max=2)


class TestJsonFormat:
    def test_valid_json_passes(self):
        assert _check('{"key": "value"}', json_format=True)

    def test_invalid_json_fails(self):
        assert not _check("not json", json_format=True)

    def test_json_array_passes(self):
        assert _check('[1, 2, 3]', json_format=True)

    def test_violation_name(self):
        assert "json_format" in _violations("bad", json_format=True)


class TestNoComma:
    def test_no_comma_passes(self):
        assert _check("Hello world", no_comma=True)

    def test_comma_fails(self):
        assert not _check("Hello, world", no_comma=True)

    def test_violation_name(self):
        assert "no_comma" in _violations("a, b", no_comma=True)


class TestCase:
    def test_all_uppercase_passes(self):
        assert _check("HELLO WORLD", all_uppercase=True)

    def test_all_uppercase_fails_mixed(self):
        assert not _check("Hello World", all_uppercase=True)

    def test_all_lowercase_passes(self):
        assert _check("hello world", all_lowercase=True)

    def test_all_lowercase_fails_mixed(self):
        assert not _check("Hello World", all_lowercase=True)

    def test_violation_names(self):
        assert "all_uppercase" in _violations("hello", all_uppercase=True)
        assert "all_lowercase" in _violations("HELLO", all_lowercase=True)


class TestStartsEndsWith:
    def test_starts_with_passes(self):
        assert _check("Hello world", starts_with="Hello")

    def test_starts_with_fails(self):
        assert not _check("World hello", starts_with="Hello")

    def test_ends_with_passes(self):
        assert _check("Hello world", ends_with="world")

    def test_ends_with_trailing_whitespace_ignored(self):
        assert _check("Hello world  ", ends_with="world")

    def test_ends_with_fails(self):
        assert not _check("Hello world", ends_with="Hello")

    def test_violation_names(self):
        assert "starts_with" in _violations("nope", starts_with="yes")
        assert "ends_with" in _violations("nope", ends_with="yes")


class TestBulletListCount:
    def test_exact_count_passes(self):
        text = "- item one\n- item two\n- item three"
        assert _check(text, bullet_list_count=3)

    def test_wrong_count_fails(self):
        text = "- item one\n- item two"
        assert not _check(text, bullet_list_count=3)

    def test_asterisk_bullets_counted(self):
        text = "* a\n* b"
        assert _check(text, bullet_list_count=2)

    def test_zero_bullets_when_none(self):
        assert _check("no bullets here", bullet_list_count=0)

    def test_violation_name(self):
        assert "bullet_list_count" in _violations("- one", bullet_list_count=2)


class TestHighlightedSections:
    def test_exact_count_passes(self):
        text = "see *important* and *also this*"
        assert _check(text, num_highlighted_sections=2)

    def test_wrong_count_fails(self):
        text = "see *important*"
        assert not _check(text, num_highlighted_sections=2)

    def test_zero_when_none(self):
        assert _check("no highlights", num_highlighted_sections=0)

    def test_violation_name(self):
        assert "num_highlighted_sections" in _violations("*one*", num_highlighted_sections=2)


class TestPostscript:
    def test_has_ps_passes(self):
        assert _check("Main body.\n\nP.S. Extra note.", has_postscript=True)

    def test_missing_ps_fails(self):
        assert not _check("No postscript here.", has_postscript=True)

    def test_violation_name(self):
        assert "has_postscript" in _violations("no ps", has_postscript=True)


class TestTitleFormat:
    def test_title_present_passes(self):
        assert _check("<<My Title>>\nBody text.", title_format=True)

    def test_title_absent_fails(self):
        assert not _check("Body text only.", title_format=True)

    def test_violation_name(self):
        assert "title_format" in _violations("no title", title_format=True)


class TestTwoResponses:
    def test_separator_present_passes(self):
        assert _check("First response.\n\n******\n\nSecond response.", two_responses=True)

    def test_separator_absent_fails(self):
        assert not _check("Only one response.", two_responses=True)

    def test_violation_name(self):
        assert "two_responses" in _violations("no sep", two_responses=True)


class TestRepeatPrompt:
    def test_prompt_in_output_passes(self):
        prompt = "What is 2+2?"
        output = "What is 2+2? The answer is 4."
        c = OutputConstraints(repeat_prompt=True)
        result = evaluate_output_constraints(output, c, prompt=prompt)
        assert result.passed

    def test_prompt_missing_fails(self):
        prompt = "What is 2+2?"
        output = "The answer is 4."
        c = OutputConstraints(repeat_prompt=True)
        result = evaluate_output_constraints(output, c, prompt=prompt)
        assert not result.passed

    def test_violation_name(self):
        c = OutputConstraints(repeat_prompt=True)
        result = evaluate_output_constraints("answer only", c, prompt="original prompt")
        assert "repeat_prompt" in [v.constraint for v in result.violations]

    def test_no_violation_when_prompt_empty(self):
        c = OutputConstraints(repeat_prompt=True)
        result = evaluate_output_constraints("answer", c, prompt="")
        assert result.passed


class TestConstrainedTo:
    def test_exact_match_passes(self):
        assert _check("yes", constrained_to=["yes", "no", "maybe"])

    def test_case_insensitive(self):
        assert _check("YES", constrained_to=["yes", "no"])

    def test_non_member_fails(self):
        assert not _check("absolutely", constrained_to=["yes", "no", "maybe"])

    def test_violation_name(self):
        assert "constrained_to" in _violations("maybe not", constrained_to=["yes", "no"])


class TestMultipleConstraints:
    def test_all_pass(self):
        text = "hello world foo"
        assert _check(text, required_keywords=["hello"], word_count_min=2, word_count_max=5)

    def test_one_of_many_fails_collects_all_violations(self):
        text = "hi"
        c = OutputConstraints(required_keywords=["missing"], word_count_min=10)
        result = evaluate_output_constraints(text, c)
        assert not result.passed
        constraint_names = [v.constraint for v in result.violations]
        assert "required_keyword" in constraint_names
        assert "word_count_min" in constraint_names

    def test_no_constraint_configured_passes_empty_text(self):
        assert _check("", )


class TestLevel1Integration:
    """Verify output_constraints wires into Level1Result via run_level1_eval."""

    def _make_spans_with_output(self, output: str):
        from unittest.mock import MagicMock
        span = MagicMock()
        span.attributes = {"agent.output": output}
        return [span]

    def test_constraint_failure_causes_level1_to_fail(self):
        from agentlens.eval.runner import run_level1_eval
        from agentlens.eval.scenarios import ExpectedResult, OutputConstraints, Scenario

        scenario = Scenario(
            id="t1",
            name="T1",
            category="test",
            input="query",
            setup=[],
            expected=ExpectedResult(
                tools_called=[],
                output_contains=[],
                safety_checks=False,
                constraints=OutputConstraints(required_keywords=["secret"]),
            ),
        )
        spans = self._make_spans_with_output("no keyword here")
        result = run_level1_eval(scenario, spans)
        assert result.output_constraints is not None
        assert not result.output_constraints.passed
        assert not result.passed

    def test_constraint_pass_does_not_affect_level1(self):
        from agentlens.eval.runner import run_level1_eval
        from agentlens.eval.scenarios import ExpectedResult, OutputConstraints, Scenario

        scenario = Scenario(
            id="t2",
            name="T2",
            category="test",
            input="query",
            setup=[],
            expected=ExpectedResult(
                tools_called=[],
                output_contains=[],
                safety_checks=False,
                constraints=OutputConstraints(required_keywords=["hello"]),
            ),
        )
        spans = self._make_spans_with_output("hello world")
        result = run_level1_eval(scenario, spans)
        assert result.output_constraints is not None
        assert result.output_constraints.passed

    def test_no_constraint_skips_check(self):
        from agentlens.eval.runner import run_level1_eval
        from agentlens.eval.scenarios import ExpectedResult, Scenario

        scenario = Scenario(
            id="t3",
            name="T3",
            category="test",
            input="query",
            setup=[],
            expected=ExpectedResult(tools_called=[], output_contains=[], safety_checks=False),
        )
        spans = self._make_spans_with_output("anything")
        result = run_level1_eval(scenario, spans)
        assert result.output_constraints is None

    def test_constraint_failure_appears_in_failure_reasons(self):
        from agentlens.eval.runner import run_level1_eval
        from agentlens.eval.scenarios import ExpectedResult, OutputConstraints, Scenario

        scenario = Scenario(
            id="t4",
            name="T4",
            category="test",
            input="query",
            setup=[],
            expected=ExpectedResult(
                tools_called=[],
                output_contains=[],
                safety_checks=False,
                constraints=OutputConstraints(no_comma=True),
            ),
        )
        spans = self._make_spans_with_output("a, b, c")
        result = run_level1_eval(scenario, spans)
        reasons = result.failure_reasons
        assert any("no_comma" in r for r in reasons)

    def test_supplemental_checks_includes_output_constraints(self):
        from agentlens.eval.runner import run_level1_eval
        from agentlens.eval.scenarios import ExpectedResult, OutputConstraints, Scenario

        scenario = Scenario(
            id="t5",
            name="T5",
            category="test",
            input="query",
            setup=[],
            expected=ExpectedResult(
                tools_called=[],
                output_contains=[],
                safety_checks=False,
                constraints=OutputConstraints(word_count_min=1),
            ),
        )
        spans = self._make_spans_with_output("hello")
        result = run_level1_eval(scenario, spans)
        assert "output_constraints" in result.supplemental_checks
