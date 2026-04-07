# Contributing to AgentLens

Thanks for your interest in contributing. This document covers how to get set up, what kinds of contributions are most useful, and how to submit a pull request.

## Getting started

```bash
git clone https://github.com/noCharger/agentlens.git
cd agentlens
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify your setup:

```bash
python -m pytest          # all tests should pass
python -m ruff check src tests  # no lint errors
```

## What to work on

Check the [open issues](https://github.com/noCharger/agentlens/issues) for things labeled `good first issue` or `help wanted`. If you have an idea that isn't tracked yet, open a feature request first so we can discuss scope before you invest time in an implementation.

Good areas for contribution:

- **New L1 checks** — deterministic checks for edge cases not yet covered (e.g. specific tool parameter validation patterns)
- **New L2 metrics** — research-backed evaluation metrics that produce a `JudgeScore` and integrate cleanly with the existing runner
- **New benchmark adapters** — importers for public benchmarks not yet supported
- **Model provider support** — additional LLM providers beyond the current four
- **Documentation and examples** — scenario YAML examples, tutorials, usage patterns

## Project layout

```text
src/agentlens/
├── eval/
│   ├── level1_deterministic/   # Add new L1 checks here
│   ├── level2_llm_judge/       # Add new L2 metrics here
│   └── level3_human/           # HTML reporter
├── core/                       # Local records, alerts, API
├── observability/              # OTEL instrumentation
└── scenarios/                  # Built-in YAML scenarios
tests/                          # One test file per module
```

## Adding an L2 metric

New metrics should follow the pattern in `src/agentlens/eval/level2_llm_judge/`:

1. Create `your_metric.py` — implement an `evaluate_your_metric(llm, spans, query, ...) -> JudgeScore` function
2. Add prompts to `prompts.py`
3. Add a rubric definition to `rubrics.py` if applicable
4. Add a feature flag field to `config.py` and a CLI flag in `__main__.py`
5. Wire the flag into `_run_level2` in `runner.py`
6. Write tests in `tests/test_your_metric.py`

Every metric must produce a `JudgeScore(dimension, score, explanation)`. This ensures it flows through OTEL, HTML reporting, and experiment comparison automatically.

## Pull request checklist

- [ ] `python -m pytest` passes
- [ ] `python -m ruff check src tests` passes
- [ ] New behavior has test coverage
- [ ] Public functions have docstrings
- [ ] If adding a new CLI flag, it's documented in the README

## Commit style

Use conventional commits:

```
feat(eval): add new metric for ...
fix(runner): handle edge case when spans is empty
docs: update CLI reference in README
test(geval): add cache hit coverage
```

## Code style

- Ruff is the linter and formatter — run `ruff check` before committing
- Type annotations on all public functions
- No `print()` in library code — use `logging.getLogger(__name__)`
- Keep prompt strings in `prompts.py`, not inline in metric files

## Questions

Open an [issue](https://github.com/noCharger/agentlens/issues) if you're unsure where a change belongs or want to discuss scope before writing code. Use bug reports for defects and feature requests for new ideas.
