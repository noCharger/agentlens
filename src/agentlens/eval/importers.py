"""Benchmark dataset adapters for mapping raw benchmark data to runtime scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from rich.console import Console
from rich.table import Table

from agentlens.eval.benchmarks import display_benchmark_name, normalize_benchmark_name
from agentlens.eval.scenarios import ExpectedResult, Scenario

console = Console()


@dataclass(frozen=True, slots=True)
class BenchmarkImporterInfo:
    slug: str
    source_kind: str
    visibility: str
    default_evaluation_mode: str
    description: str

    @property
    def name(self) -> str:
        return display_benchmark_name(self.slug) or self.slug


@dataclass(frozen=True, slots=True)
class BenchmarkLoadResult:
    benchmark: str
    scenarios: list[Scenario]
    input_path: Path


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return sanitized or "scenario"


def _safe_name(value: str, fallback: str) -> str:
    clean = " ".join(str(value).split())
    return clean[:120] if clean else fallback


def _first_nonempty(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _ensure_list(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        return [stripped]
    return [value]


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_sanitize_value(v) for v in value)
    return value


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        records = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    if suffix == ".json":
        raw_text = path.read_text()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = yaml.safe_load(raw_text)
    elif suffix in {".yaml", ".yml"}:
        raw_text = path.read_text()
        payload = yaml.safe_load(raw_text)
    elif suffix == ".csv":
        with path.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Reading parquet benchmark data requires optional dependency 'pyarrow'."
            ) from exc
        table = pq.read_table(path)
        return list(table.to_pylist())
    else:
        raise ValueError(f"Unsupported input format for benchmark import: {path}")

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Could not parse records from {path}")


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            yield json.loads(stripped)


def _format_rubric_text(raw: Any) -> str:
    if raw in (None, "", []):
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        lines = []
        for item in raw:
            if isinstance(item, dict):
                score = item.get("score")
                criterion = item.get("criterion") or item.get("description") or item.get("text")
                prefix = f"[{score}] " if score not in (None, "") else ""
                if criterion:
                    lines.append(f"- {prefix}{criterion}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)
    if isinstance(raw, dict):
        lines = []
        for key, value in raw.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    return str(raw)


class BenchmarkImporter:
    info: BenchmarkImporterInfo
    default_category: str

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return [local_root] if local_root.exists() else []

    def load_scenarios(self, input_path: Path, limit: int | None = None) -> list[Scenario]:
        scenarios = []
        for index, item in enumerate(self.iter_items(input_path), start=1):
            scenarios.append(self.make_scenario(item, index, input_path))
            if limit is not None and len(scenarios) >= limit:
                break
        return scenarios

    def iter_items(self, input_path: Path) -> Iterable[Any]:
        raise NotImplementedError

    def make_scenario(self, item: Any, index: int, input_path: Path) -> Scenario:
        raise NotImplementedError

    def _compose_scenario_id(self, source_id: str, index: int) -> str:
        base = _sanitize_id(source_id or f"item-{index:04d}")
        if base.startswith(f"{self.info.slug}-"):
            return base
        return f"{self.info.slug}-{base}"


class RecordBenchmarkImporter(BenchmarkImporter):
    def iter_items(self, input_path: Path) -> Iterable[dict[str, Any]]:
        if input_path.is_dir():
            raise ValueError(
                f"{self.info.name} expects a file input (json/jsonl/yaml/csv/parquet), got directory {input_path}"
            )
        return _load_records(input_path)


class ManifestBenchmarkImporter(RecordBenchmarkImporter):
    def make_scenario(self, item: dict[str, Any], index: int, input_path: Path) -> Scenario:
        source_id = str(
            _first_nonempty(item, "id", "scenario_id", "task_id", "instance_id", "name")
            or f"item-{index:04d}"
        )
        query = self._extract_query(item)
        if not query:
            raise ValueError(
                f"Manifest record {source_id} in {input_path} is missing a prompt/query/input field."
            )

        input_block = item.get("input", {})
        setup_commands = []
        if isinstance(input_block, dict):
            setup_commands = _ensure_list(input_block.get("setup"))
        if not setup_commands:
            setup_commands = _ensure_list(item.get("setup") or item.get("setup_commands"))

        judge_rubric = str(item.get("judge_rubric", "")).strip()
        judge_rubric_text = _format_rubric_text(
            _first_nonempty(item, "judge_rubric_text", "rubric_text", "rubric")
        )
        reference_answer = str(
            _first_nonempty(item, "reference_answer", "reference", "answer", "gold_answer") or ""
        ).strip()

        expected = item.get("expected", {}) if isinstance(item.get("expected"), dict) else {}
        tools_called = _ensure_list(
            _first_nonempty(item, "expected_tools", "tools_called") or expected.get("tools_called")
        )
        output_contains = _ensure_list(
            _first_nonempty(item, "output_contains", "expected_output") or expected.get("output_contains")
        )
        max_steps = int(
            _first_nonempty(item, "max_steps") or expected.get("max_steps") or 10
        )
        max_tokens = _first_nonempty(item, "max_tokens") or expected.get("max_tokens")
        judge_threshold = float(item.get("judge_threshold", 4.0))

        has_deterministic_signal = bool(tools_called or output_contains)
        has_judge_signal = bool(judge_rubric or judge_rubric_text or reference_answer)
        evaluation_mode = str(item.get("evaluation_mode", "")).strip() or (
            "deterministic"
            if has_deterministic_signal
            else "llm_judge"
            if has_judge_signal
            else self.info.default_evaluation_mode
        )
        if evaluation_mode == "llm_judge" and not judge_rubric and not judge_rubric_text:
            judge_rubric = "accuracy" if reference_answer else ""

        metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "id",
                "scenario_id",
                "task_id",
                "instance_id",
                "name",
                "title",
                "category",
                "input",
                "prompt",
                "query",
                "instruction",
                "task",
                "problem_statement",
                "setup",
                "setup_commands",
                "expected",
                "expected_tools",
                "tools_called",
                "output_contains",
                "expected_output",
                "max_steps",
                "max_tokens",
                "judge_rubric",
                "judge_rubric_text",
                "rubric_text",
                "rubric",
                "judge_threshold",
                "reference_answer",
                "reference",
                "answer",
                "gold_answer",
                "evaluation_mode",
            }
        }

        return Scenario(
            id=self._compose_scenario_id(source_id, index),
            name=_safe_name(
                str(_first_nonempty(item, "name", "title") or source_id),
                fallback=f"{self.info.name} Task {index}",
            ),
            category=str(item.get("category") or self.default_category),
            benchmark=self.info.slug,
            evaluation_mode=evaluation_mode,
            input=query,
            setup=setup_commands,
            expected=ExpectedResult(
                tools_called=[str(tool) for tool in tools_called],
                max_steps=max_steps,
                output_contains=[str(v) for v in output_contains],
                max_tokens=int(max_tokens) if max_tokens is not None else None,
            ),
            judge_rubric=judge_rubric,
            judge_rubric_text=judge_rubric_text,
            judge_threshold=judge_threshold,
            reference_answer=reference_answer,
            metadata=_sanitize_value(metadata),
        )

    def _extract_query(self, item: dict[str, Any]) -> str:
        input_value = item.get("input")
        if isinstance(input_value, dict):
            nested = _first_nonempty(input_value, "query", "prompt", "instruction", "task")
            if nested:
                return str(nested)
        elif input_value not in (None, ""):
            return str(input_value)

        query = _first_nonempty(
            item, "prompt", "query", "instruction", "task", "problem_statement"
        )
        return str(query or "")


class SWEBenchProImporter(RecordBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="swe-bench-pro",
        source_kind="records",
        visibility="public",
        default_evaluation_mode="external",
        description="Imports SWE-bench style issue-resolution records into external-eval scenarios.",
    )
    default_category = "software_engineering"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        candidates = sorted((local_root / "data").glob("*.parquet"))
        if candidates:
            return candidates
        return sorted(local_root.glob("*.parquet"))

    def make_scenario(self, item: dict[str, Any], index: int, input_path: Path) -> Scenario:
        source_id = str(_first_nonempty(item, "instance_id", "id") or f"item-{index:04d}")
        repo = str(_first_nonempty(item, "repo", "repository") or "unknown/repo")
        base_commit = str(_first_nonempty(item, "base_commit", "commit") or "")
        problem_statement = str(
            _first_nonempty(item, "problem_statement", "problem", "issue", "body") or ""
        ).strip()
        if not problem_statement:
            raise ValueError(f"SWE-bench record {source_id} is missing problem_statement.")

        query = (
            f"You are working in repository `{repo}`"
            f"{f' at base commit `{base_commit}`' if base_commit else ''}.\n\n"
            f"Resolve the following issue:\n{problem_statement}\n\n"
            "Make the minimal code changes required to fix the issue while preserving existing behavior."
        )

        title = problem_statement.splitlines()[0]
        metadata = {
            "source_path": str(input_path),
            "repo": repo,
            "base_commit": base_commit,
            "environment_setup_commit": item.get("environment_setup_commit"),
            "version": item.get("version"),
            "hints_text": item.get("hints_text"),
            "patch": item.get("patch"),
            "test_patch": item.get("test_patch"),
            "fail_to_pass": item.get("FAIL_TO_PASS"),
            "pass_to_pass": item.get("PASS_TO_PASS"),
            "created_at": item.get("created_at"),
        }

        return Scenario(
            id=self._compose_scenario_id(source_id, index),
            name=_safe_name(f"{repo}: {title}", fallback=f"SWE Bench Pro {index}"),
            category=self.default_category,
            benchmark=self.info.slug,
            evaluation_mode="external",
            input=query,
            setup=[],
            expected=ExpectedResult(),
            metadata=_sanitize_value(metadata),
        )


class MultiSWEBenchImporter(RecordBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="multi-swe-bench",
        source_kind="directory",
        visibility="public",
        default_evaluation_mode="external",
        description="Imports Multi-SWE-bench issue-resolution records from the official JSONL directory layout.",
    )
    default_category = "software_engineering"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return [local_root] if any(local_root.rglob("*.jsonl")) else []

    def iter_items(self, input_path: Path) -> Iterable[dict[str, Any]]:
        if input_path.is_file():
            records = _iter_jsonl_records(input_path) if input_path.suffix == ".jsonl" else _load_records(input_path)
            for record in records:
                enriched = dict(record)
                enriched["__source_file"] = input_path.name
                yield enriched
            return

        jsonl_files = sorted(input_path.rglob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No JSONL files found under {input_path}")

        for jsonl_file in jsonl_files:
            for record in _iter_jsonl_records(jsonl_file):
                enriched = dict(record)
                enriched["__source_file"] = str(jsonl_file.relative_to(input_path))
                yield enriched

    def make_scenario(self, item: dict[str, Any], index: int, input_path: Path) -> Scenario:
        source_id = str(_first_nonempty(item, "instance_id", "id") or f"item-{index:04d}")
        org = str(item.get("org", "")).strip()
        repo_name = str(item.get("repo", "")).strip()
        repo = "/".join(part for part in (org, repo_name) if part) or "unknown/repo"
        title = str(_first_nonempty(item, "title") or "").strip()
        body = str(_first_nonempty(item, "body") or "").strip()
        resolved_issues = [str(issue) for issue in _ensure_list(item.get("resolved_issues"))]
        issue_block = "\n".join(f"- {issue}" for issue in resolved_issues)
        problem_statement = "\n\n".join(part for part in (title, body) if part)
        if issue_block:
            problem_statement = f"{problem_statement}\n\nResolved issues:\n{issue_block}".strip()
        if not problem_statement:
            raise ValueError(f"Multi-SWE record {source_id} is missing title/body.")

        base = item.get("base")
        base_ref = ""
        if isinstance(base, dict):
            base_ref = str(_first_nonempty(base, "ref", "sha") or "")

        query = (
            f"You are working in repository `{repo}`"
            f"{f' on base `{base_ref}`' if base_ref else ''}.\n\n"
            f"Resolve the following issue or pull request context:\n{problem_statement}\n\n"
            "Produce the code changes needed to resolve the issue."
        )

        metadata = {
            "source_path": str(input_path),
            "source_file": item.get("__source_file"),
            "org": org,
            "repo": repo_name,
            "number": item.get("number"),
            "state": item.get("state"),
            "base": item.get("base"),
            "resolved_issues": resolved_issues,
            "fix_patch": item.get("fix_patch"),
            "test_patch": item.get("test_patch"),
            "fixed_tests": item.get("fixed_tests"),
            "p2p_tests": item.get("p2p_tests"),
            "f2p_tests": item.get("f2p_tests"),
            "s2p_tests": item.get("s2p_tests"),
            "n2p_tests": item.get("n2p_tests"),
            "run_result": item.get("run_result"),
            "test_patch_result": item.get("test_patch_result"),
            "fix_patch_result": item.get("fix_patch_result"),
        }

        return Scenario(
            id=self._compose_scenario_id(source_id, index),
            name=_safe_name(f"{repo}: {title or source_id}", fallback=f"Multi-SWE Bench {index}"),
            category=self.default_category,
            benchmark=self.info.slug,
            evaluation_mode="external",
            input=query,
            setup=[],
            expected=ExpectedResult(),
            metadata=_sanitize_value(metadata),
        )


class GDPValAAImporter(RecordBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="gdpval-aa",
        source_kind="records",
        visibility="public",
        default_evaluation_mode="llm_judge",
        description="Imports OpenAI GDPval analytical tasks with custom rubric text for judge-only scoring.",
    )
    default_category = "analysis"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        parquet_candidates = sorted((local_root / "data").glob("*.parquet"))
        if parquet_candidates:
            return parquet_candidates

        for pattern in ("*.json", "*.jsonl", "*.yaml", "*.yml", "*.csv", "*.parquet"):
            matches = sorted(local_root.glob(pattern))
            if matches:
                return [matches[0]]
        return []

    def make_scenario(self, item: dict[str, Any], index: int, input_path: Path) -> Scenario:
        source_id = str(_first_nonempty(item, "task_id", "id") or f"item-{index:04d}")
        prompt = str(_first_nonempty(item, "prompt", "query", "task") or "").strip()
        if not prompt:
            raise ValueError(f"GDPval record {source_id} is missing prompt.")

        reference_files = [str(value) for value in _ensure_list(item.get("reference_files"))]
        deliverable_files = [str(value) for value in _ensure_list(item.get("deliverable_files"))]
        rubric_text = _format_rubric_text(
            _first_nonempty(item, "rubric_pretty", "rubric", "rubric_json")
        )
        if reference_files:
            prompt = (
                f"{prompt}\n\nReference files available to the agent:\n"
                + "\n".join(f"- {path}" for path in reference_files)
            )
        if deliverable_files:
            prompt = (
                f"{prompt}\n\nExpected deliverable files:\n"
                + "\n".join(f"- {path}" for path in deliverable_files)
            )

        metadata = {
            "source_path": str(input_path),
            "task_id": source_id,
            "sector": item.get("sector"),
            "occupation": item.get("occupation"),
            "reference_files": reference_files,
            "reference_file_urls": item.get("reference_file_urls"),
            "reference_file_hf_uris": item.get("reference_file_hf_uris"),
            "deliverable_files": deliverable_files,
            "deliverable_text": item.get("deliverable_text"),
        }

        return Scenario(
            id=self._compose_scenario_id(source_id, index),
            name=_safe_name(
                str(_first_nonempty(item, "title", "task_name") or f"GDPval {source_id}"),
                fallback=f"GDPval-AA {index}",
            ),
            category=self.default_category,
            benchmark=self.info.slug,
            evaluation_mode="llm_judge",
            input=prompt,
            setup=[],
            expected=ExpectedResult(max_steps=20),
            judge_rubric_text=rubric_text,
            judge_threshold=float(item.get("judge_threshold", 4.0)),
            reference_answer=str(item.get("deliverable_text", "") or ""),
            metadata=_sanitize_value(metadata),
        )


class ToolathlonImporter(BenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="toolathlon",
        source_kind="directory",
        visibility="public",
        default_evaluation_mode="external",
        description="Imports Toolathlon task directories and keeps task/evaluator metadata for external harnesses.",
    )
    default_category = "tool_use"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        if (local_root / "task_config.json").exists() or any(local_root.rglob("task_config.json")):
            return [local_root]
        return []

    def iter_items(self, input_path: Path) -> Iterable[Path]:
        if input_path.is_file():
            raise ValueError(f"Toolathlon importer expects a task directory, got file {input_path}")

        if (input_path / "task_config.json").exists():
            return [input_path]

        tasks = sorted(path.parent for path in input_path.rglob("task_config.json"))
        if not tasks:
            raise ValueError(f"No Toolathlon task_config.json files found under {input_path}")
        return tasks

    def make_scenario(self, item: Path, index: int, input_path: Path) -> Scenario:
        config_path = item / "task_config.json"
        task_path = item / "docs" / "task.md"
        if not task_path.exists():
            raise ValueError(f"Toolathlon task {item} is missing docs/task.md")

        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        task_text = task_path.read_text().strip()
        system_prompt_path = item / "docs" / "agent_system_prompt.md"
        system_prompt = system_prompt_path.read_text().strip() if system_prompt_path.exists() else ""

        meta = config.get("meta", {}) if isinstance(config.get("meta"), dict) else {}
        source_id = str(_first_nonempty(meta, "task_id", "id", "slug") or item.name)
        name = str(_first_nonempty(meta, "title", "name") or item.name.replace("-", " ").title())
        query = task_text
        if system_prompt:
            query = f"{task_text}\n\nSystem prompt:\n{system_prompt}"

        metadata = {
            "source_path": str(item),
            "needed_mcp_servers": config.get("needed_mcp_servers", []),
            "needed_local_tools": config.get("needed_local_tools", []),
            "meta": meta,
            "has_preprocess": (item / "preprocess").exists(),
            "has_initial_workspace": (item / "initial_workspace").exists(),
            "has_groundtruth_workspace": (item / "groundtruth_workspace").exists(),
            "has_evaluation": (item / "evaluation").exists(),
        }

        return Scenario(
            id=self._compose_scenario_id(source_id, index),
            name=_safe_name(name, fallback=f"Toolathlon {index}"),
            category=self.default_category,
            benchmark=self.info.slug,
            evaluation_mode="external",
            input=query,
            setup=[],
            expected=ExpectedResult(max_steps=40),
            metadata=_sanitize_value(metadata),
        )


class MLEBenchLiteImporter(ManifestBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="mle-bench-lite",
        source_kind="manifest",
        visibility="public",
        default_evaluation_mode="external",
        description=(
            "Imports user-exported MLE-Bench lite manifests. "
            "The official benchmark still requires MLE-Bench's own preparation and grading harness."
        ),
    )
    default_category = "ml_engineering"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return _discover_manifest_inputs(local_root)


class VIBEProImporter(ManifestBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="vibe-pro",
        source_kind="manifest",
        visibility="private_or_internal",
        default_evaluation_mode="external",
        description="Imports private or internal VIBE-Pro manifests into AgentLens scenarios.",
    )
    default_category = "agent_workflows"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return _discover_manifest_inputs(local_root)


class MMClawBenchImporter(ManifestBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="mm-clawbench",
        source_kind="manifest",
        visibility="private_or_internal",
        default_evaluation_mode="external",
        description="Imports multimodal MM-ClawBench manifests with asset references stored in metadata.",
    )
    default_category = "multimodal"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return _discover_manifest_inputs(local_root)


class ArtificialAnalysisImporter(ManifestBenchmarkImporter):
    info = BenchmarkImporterInfo(
        slug="artificial-analysis",
        source_kind="manifest",
        visibility="private_or_internal",
        default_evaluation_mode="external",
        description="Imports Artificial Analysis benchmark manifests with optional custom judge rubrics.",
    )
    default_category = "analysis"

    def discover_input_paths(self, local_root: Path) -> list[Path]:
        return _discover_manifest_inputs(local_root)


IMPORTERS: dict[str, BenchmarkImporter] = {
    importer.info.slug: importer
    for importer in (
        SWEBenchProImporter(),
        MultiSWEBenchImporter(),
        VIBEProImporter(),
        MLEBenchLiteImporter(),
        GDPValAAImporter(),
        ToolathlonImporter(),
        MMClawBenchImporter(),
        ArtificialAnalysisImporter(),
    )
}


def get_importer(benchmark: str) -> BenchmarkImporter:
    slug = normalize_benchmark_name(benchmark)
    importer = IMPORTERS.get(slug)
    if importer is None:
        raise ValueError(
            f"Unsupported benchmark importer '{benchmark}'. Available: {', '.join(sorted(IMPORTERS))}"
        )
    return importer


def list_importers() -> list[BenchmarkImporterInfo]:
    return [IMPORTERS[key].info for key in sorted(IMPORTERS)]


def _discover_manifest_inputs(local_root: Path) -> list[Path]:
    for pattern in ("*.parquet", "*.jsonl", "*.json", "*.yaml", "*.yml", "*.csv"):
        matches = sorted(local_root.glob(pattern))
        if matches:
            return matches
    return []


def load_downloaded_benchmark_scenarios(
    data_root: Path,
    benchmarks: Iterable[str] | None = None,
) -> list[Scenario]:
    selected = (
        {normalize_benchmark_name(benchmark) for benchmark in benchmarks}
        if benchmarks
        else set(IMPORTERS)
    )

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()

    for slug in sorted(selected):
        importer = IMPORTERS.get(slug)
        if importer is None:
            continue

        local_root = data_root / slug
        if not local_root.exists():
            continue

        for input_path in importer.discover_input_paths(local_root):
            for scenario in importer.load_scenarios(input_path):
                if scenario.id in seen_ids:
                    raise ValueError(f"Duplicate scenario id while loading downloaded benchmarks: {scenario.id}")
                seen_ids.add(scenario.id)
                scenarios.append(scenario)

    return scenarios


def _ensure_unique_scenario_ids(scenarios: list[Scenario], *, context: str) -> None:
    seen_ids: set[str] = set()
    for scenario in scenarios:
        if scenario.id in seen_ids:
            raise ValueError(f"Duplicate scenario id {context}: {scenario.id}")
        seen_ids.add(scenario.id)


def load_benchmark_dataset(
    benchmark: str,
    input_path: Path,
    limit: int | None = None,
) -> BenchmarkLoadResult:
    importer = get_importer(benchmark)
    scenarios = importer.load_scenarios(input_path, limit=limit)
    _ensure_unique_scenario_ids(scenarios, context="generated while loading benchmark data")
    return BenchmarkLoadResult(
        benchmark=importer.info.slug,
        scenarios=scenarios,
        input_path=input_path,
    )


def import_benchmark_dataset(
    benchmark: str,
    input_path: Path,
    output_root: Path = Path("src/agentlens/scenarios/benchmarks"),
    limit: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> BenchmarkLoadResult:
    del output_root, overwrite, dry_run
    raise RuntimeError(
        "Benchmark YAML persistence has been removed. "
        "Keep raw benchmark files under data/benchmarks and load them dynamically at runtime, "
        "or use load_benchmark_dataset(...) to preview the in-memory scenario mapping."
    )


def _print_importers() -> None:
    table = Table(title="AgentLens Benchmark Importers")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Slug")
    table.add_column("Input")
    table.add_column("Eval Mode")
    table.add_column("Visibility")
    table.add_column("Description")

    for info in list_importers():
        table.add_row(
            info.name,
            info.slug,
            info.source_kind,
            info.default_evaluation_mode,
            info.visibility,
            info.description,
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview how raw benchmark data maps into runtime AgentLens scenarios"
    )
    parser.add_argument("--list-benchmarks", action="store_true", help="List supported benchmark importers")
    parser.add_argument("--benchmark", type=str, help="Benchmark slug or display name to import")
    parser.add_argument("--input", type=Path, help="Input dataset file or task directory")
    parser.add_argument("--limit", type=int, help="Limit the number of imported scenarios")
    args = parser.parse_args()

    if args.list_benchmarks:
        _print_importers()
        return

    if not args.benchmark or not args.input:
        parser.error("--benchmark and --input are required unless --list-benchmarks is used")

    result = load_benchmark_dataset(
        benchmark=args.benchmark,
        input_path=args.input,
        limit=args.limit,
    )

    console.print(
        f"Mapped [bold]{len(result.scenarios)}[/bold] runtime scenario(s) for "
        f"[cyan]{display_benchmark_name(result.benchmark)}[/cyan]"
    )
    console.print(f"Input path: [green]{result.input_path}[/green]")
    for scenario in result.scenarios[:10]:
        console.print(f"  [{scenario.evaluation_mode}] {scenario.id}: {scenario.name}")
    if len(result.scenarios) > 10:
        console.print(f"  ... and {len(result.scenarios) - 10} more")
    console.print()
    console.print(
        "[dim]Place raw files under data/benchmarks/<slug>/ to have them loaded by "
        "`python -m agentlens.eval` at runtime.[/dim]"
    )


if __name__ == "__main__":
    main()
