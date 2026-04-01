# AgentLens

[English](README.md) | [简体中文](README.zh-CN.md)

AgentLens is a lightweight evaluation and observability toolkit for AI agents.
It currently covers four main areas:

- Agent execution with LangGraph + LangChain
- Selectable model providers for both agent and judge
- Multi-level evaluation with deterministic checks, LLM-as-Judge, and HTML reports
- OpenTelemetry traces and metrics

The repository now uses a runtime benchmark-loading architecture:

- Handwritten scenarios live under `src/agentlens/scenarios/`
- Raw benchmark files live under `data/benchmarks/<slug>/`
- Benchmark tasks are loaded dynamically at runtime instead of being persisted as generated YAML

## Architecture

```mermaid
flowchart TD
    A["src/agentlens/scenarios/*.yaml<br/>static scenarios"] --> C["load_runtime_scenarios"]
    B["data/benchmarks/<slug>/*<br/>raw benchmark files"] --> D["benchmark adapters<br/>eval/importers.py"]
    D --> C
    C --> E["execute_and_eval"]
    E --> F["L1 deterministic<br/>tool / output / trajectory"]
    E --> G["L2 llm_judge<br/>rubric + reference"]
    E --> H["L3 report<br/>HTML summary"]
    E --> I["OTEL traces / metrics"]
```

Core modules:

- `src/agentlens/agents/factory.py`
  Creates the runtime agent and tool preset.
- `src/agentlens/model_selection.py`
  Resolves `provider:model` selections such as `gemini:gemini-2.5-flash` and `deepseek:deepseek-chat`.
- `src/agentlens/llms.py`
  Builds provider-specific chat models for Gemini and DeepSeek.
- `src/agentlens/deepseek.py`
  Runs DeepSeek-specific preflight checks, including balance validation.
- `src/agentlens/eval/scenarios.py`
  Defines the `Scenario` model and merges static YAML scenarios with dynamically discovered benchmark tasks.
- `src/agentlens/eval/importers.py`
  Maps parquet / jsonl / manifest / task-directory benchmark inputs into runtime scenarios.
- `src/agentlens/eval/runner.py`
  Executes the agent, captures spans, performs evaluation, and handles retries / external benchmark behavior.
- `src/agentlens/eval/level3_human/reporter.py`
  Generates the HTML report.
- `src/agentlens/observability/`
  Initializes telemetry and records metrics.

## Repository Layout

```text
agentlens/
├── src/agentlens/
│   ├── agents/
│   ├── eval/
│   │   ├── level1_deterministic/
│   │   ├── level2_llm_judge/
│   │   ├── level3_human/
│   │   ├── benchmarks.py
│   │   ├── importers.py
│   │   ├── runner.py
│   │   └── scenarios.py
│   ├── observability/
│   └── scenarios/
│       ├── reasoning/
│       ├── recovery/
│       └── tool_calling/
├── data/benchmarks/
│   └── <slug>/
├── tests/
└── pyproject.toml
```

## Environment Setup

Requirements:

- Python `3.11+`
- A local virtual environment in `.venv` is recommended

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 2. Activate it

For `zsh` / `bash`:

```bash
source .venv/bin/activate
```

To leave the virtual environment:

```bash
deactivate
```

### 3. Install dependencies

For normal development:

```bash
pip install -e ".[dev]"
```

If you want parquet support or benchmark downloads:

```bash
pip install -e ".[dev,benchmarks]"
```

## Configuration

Settings are loaded via `pydantic-settings` from `.env`.
A minimal example:

```bash
GOOGLE_API_KEY=your_google_ai_studio_key
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com
AGENT_MODEL=gemini:gemini-2.5-flash
JUDGE_MODEL=gemini:gemini-2.5-flash-lite
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=agentlens
AGENT_MAX_STEPS=10
```

Notes:

- `GOOGLE_API_KEY` is only required when you select a Gemini model.
- `DEEPSEEK_API_KEY` is only required when you select a DeepSeek model.
- `JUDGE_MODEL` is only used when `--level2` is enabled.
- OTEL is optional. If no collector is available, the runner will degrade as gracefully as possible.

## Model Selection

`AGENT_MODEL` and `JUDGE_MODEL` both support:

- Explicit provider syntax: `gemini:gemini-2.5-flash`
- Bare model names: `deepseek-chat`

Explicit provider syntax is recommended because it is clearer and avoids ambiguity.

Common examples:

```bash
AGENT_MODEL=gemini:gemini-2.5-flash
JUDGE_MODEL=gemini:gemini-2.5-flash-lite
```

```bash
AGENT_MODEL=deepseek:deepseek-chat
JUDGE_MODEL=deepseek:deepseek-chat
```

Mixed setup:

```bash
AGENT_MODEL=deepseek:deepseek-chat
JUDGE_MODEL=gemini:gemini-2.5-flash-lite
```

Temporary CLI override:

```bash
./.venv/bin/python -m agentlens.eval \
  --agent-model deepseek:deepseek-chat \
  --judge-model deepseek:deepseek-chat \
  --scenario-id tc-001
```

Notes:

- `deepseek:deepseek-chat` is a good default for general tool-using agent runs.
- The judge can use either Gemini or DeepSeek.
- When a DeepSeek model is selected, AgentLens performs a balance preflight check before running scenarios. If the account has insufficient balance, the command fails early with a clear error.

## Local Development Commands

Run tests:

```bash
./.venv/bin/python -m pytest
```

Run lint:

```bash
./.venv/bin/python -m ruff check src tests
```

Show CLI help:

```bash
./.venv/bin/python -m agentlens.eval --help
./.venv/bin/python -m agentlens.eval.importers --help
```

## Running Built-In YAML Scenarios

List loaded benchmarks and scenario counts:

```bash
./.venv/bin/python -m agentlens.eval --list-benchmarks
```

Dry run without calling the model:

```bash
./.venv/bin/python -m agentlens.eval --dry-run
```

Run a single scenario:

```bash
./.venv/bin/python -m agentlens.eval --scenario-id tc-001
```

Run a single scenario with DeepSeek:

```bash
./.venv/bin/python -m agentlens.eval \
  --scenario-id tc-001 \
  --agent-model deepseek:deepseek-chat
```

Generate an HTML report:

```bash
./.venv/bin/python -m agentlens.eval --output report.html
```

## Running Benchmarks

### Key Principle

- Put raw benchmark files under `data/benchmarks/<slug>/`
- AgentLens discovers and loads them dynamically at runtime
- Generated benchmark YAML files are no longer part of the workflow

### Supported Benchmarks

| Benchmark | slug | Expected input | Default evaluation mode | Can the built-in runner score it directly? |
| --- | --- | --- | --- | --- |
| SWE Bench Pro | `swe-bench-pro` | `data/*.parquet` | `external` | No, requires an external harness |
| Multi-SWE Bench | `multi-swe-bench` | `**/*.jsonl` | `external` | No, requires an external harness |
| GDPval-AA | `gdpval-aa` | `data/*.parquet` or record files | `llm_judge` | Yes, with `--level2` |
| Toolathlon | `toolathlon` | task directories | `external` | No, requires an external harness |
| VIBE-Pro | `vibe-pro` | manifest | `external` | Usually needs an external harness |
| MLE-Bench lite | `mle-bench-lite` | manifest | `external` | Usually needs an external harness |
| MM-ClawBench | `mm-clawbench` | manifest | `external` | Usually needs an external harness |
| Artificial Analysis | `artificial-analysis` | manifest | `external` | Depends on the manifest |

Evaluation mode summary:

- `deterministic`
  Uses tool usage, output content, and trajectory checks.
- `llm_judge`
  Requires `--level2` and uses a rubric plus a reference answer.
- `external`
  AgentLens can load, filter, inventory, and report these tasks, but does not score them as passing with the built-in runner.

### Benchmark Data Layout

Example layout:

```text
data/benchmarks/
├── gdpval-aa/
│   └── data/
│       └── train-00000-of-00001.parquet
├── multi-swe-bench/
│   ├── python/
│   │   └── multi_swe_bench_python.jsonl
│   └── rust/
│       └── tokio-rs__tokio_dataset.jsonl
└── swe-bench-pro/
    └── data/
        └── test-00000-of-00001.parquet
```

### Preview How Benchmark Data Maps to Runtime Scenarios

List available adapters:

```bash
./.venv/bin/python -m agentlens.eval.importers --list-benchmarks
```

Preview a record-style benchmark file:

```bash
./.venv/bin/python -m agentlens.eval.importers \
  --benchmark gdpval-aa \
  --input data/benchmarks/gdpval-aa/data/train-00000-of-00001.parquet \
  --limit 3
```

Preview a directory-style benchmark:

```bash
./.venv/bin/python -m agentlens.eval.importers \
  --benchmark multi-swe-bench \
  --input data/benchmarks/multi-swe-bench \
  --limit 3
```

### Run a Benchmark

1. Confirm that AgentLens can discover the benchmark:

```bash
./.venv/bin/python -m agentlens.eval --list-benchmarks
```

2. Dry run the benchmark selection:

```bash
./.venv/bin/python -m agentlens.eval --benchmark gdpval-aa --dry-run
```

3. Run a benchmark that the built-in runner can score:

```bash
./.venv/bin/python -m agentlens.eval --benchmark gdpval-aa --level2 --output gdpval.html
```

4. For `external` benchmarks, use dry-run and inventory mode first:

```bash
./.venv/bin/python -m agentlens.eval --benchmark swe-bench-pro --dry-run
./.venv/bin/python -m agentlens.eval --benchmark multi-swe-bench --dry-run
```

If your benchmark root is elsewhere:

```bash
./.venv/bin/python -m agentlens.eval \
  --benchmark gdpval-aa \
  --benchmark-data-root /absolute/path/to/benchmarks \
  --dry-run
```

## Downloading Benchmark Data with the Hugging Face CLI

If the benchmark extra dependencies are installed and `hf` is available, you can download files directly into the layout expected by AgentLens.

GDPval-AA example:

```bash
./.venv/bin/hf download openai/gdpval \
  --repo-type dataset \
  --include "data/*.parquet" \
  --local-dir data/benchmarks/gdpval-aa
```

Multi-SWE Bench example:

```bash
./.venv/bin/hf download bytedance-research/Multi-SWE-Bench \
  --repo-type dataset \
  --include "*.jsonl" \
  --local-dir data/benchmarks/multi-swe-bench
```

For other benchmarks, any download method is fine as long as the final layout matches the adapter expectations.

## Reports and Observability

The CLI output includes:

- PASS / FAIL for each scenario
- Benchmark-level summary
- Error messages

If you pass `--output report.html`, the generated report includes:

- Overall pass rate
- Benchmark summary
- L1 / L2 details for each scenario

If an OTEL collector is available, AgentLens also exports:

- Agent run metrics
- Tool call metrics
- LLM token and latency metrics

## FAQ

### 1. Why does a benchmark say it requires an external harness?

That is expected for benchmarks such as `swe-bench-pro` and `multi-swe-bench`.
Their real scoring depends on their own evaluation harnesses.
AgentLens currently handles:

- Runtime task loading
- Filtering and inventory
- Unified reporting
- Avoiding false PASS results for externally scored tasks

### 2. Why does GDPval-AA require `--level2`?

Its main scoring signal comes from rubric text and a reference answer, so it runs in `llm_judge` mode.

### 3. Why do I see `sysctlbyname failed` warnings from `pyarrow` on macOS?

Those warnings are common in restricted environments and usually do not affect parquet loading.
