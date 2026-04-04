import json
import tempfile
from pathlib import Path

import pytest

from agentlens.eval.importers import (
    get_importer,
    import_benchmark_dataset,
    list_importers,
    load_benchmark_dataset,
)


def test_list_importers_includes_requested_benchmarks():
    slugs = {info.slug for info in list_importers()}
    assert "swe-bench-pro" in slugs
    assert "gdpval-aa" in slugs
    assert "toolathlon" in slugs


def test_swe_bench_importer_maps_issue_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "swe.jsonl"
        path.write_text(
            json.dumps(
                {
                    "instance_id": "psf__requests-123",
                    "repo": "psf/requests",
                    "base_commit": "abc123",
                    "problem_statement": "Fix session cookie regression",
                    "patch": "diff --git a/foo b/foo",
                }
            )
            + "\n"
        )

        result = load_benchmark_dataset("SWE Bench Pro", path)

    scenario = result.scenarios[0]
    assert scenario.benchmark == "swe-bench-pro"
    assert scenario.evaluation_mode == "external"
    assert "psf/requests" in scenario.input_query
    assert scenario.metadata["patch"] == "diff --git a/foo b/foo"


def test_gdpval_importer_builds_llm_judge_scenarios():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "gdpval-aa"
        root.mkdir()
        (root / "reference_files").mkdir()
        (root / "deliverable_files").mkdir()
        (root / "reference_files" / "brief.xlsx").write_text("placeholder")
        path = root / "gdpval.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "task_id": "task-001",
                        "prompt": "Analyze the market and provide a written memo.",
                            "rubric_pretty": [
                                {"score": 5, "criterion": "Insightful and complete"},
                                {"score": 3, "criterion": "Adequate but shallow"},
                            ],
                            "reference_files": ["reference_files/brief.xlsx"],
                            "deliverable_files": ["deliverable_files/memo.xlsx"],
                            "deliverable_text": "A strong memo.",
                        }
                    ]
                )
            )

        result = load_benchmark_dataset("gdpval-aa", path)

    scenario = result.scenarios[0]
    assert scenario.evaluation_mode == "llm_judge"
    assert "brief.xlsx" in scenario.input_query
    assert str((root / "reference_files" / "brief.xlsx").resolve()) in scenario.input_query
    assert "Insightful and complete" in scenario.judge_rubric_text
    assert scenario.reference_answer == "A strong memo."
    assert str((root / "deliverable_files" / "memo.xlsx").resolve()) in scenario.input_query
    assert "openpyxl" in scenario.input_query
    assert "Do not use `cat` or GUI `open` on .xlsx files" in scenario.input_query


def test_toolathlon_importer_reads_task_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / "task-alpha"
        docs_dir = task_dir / "docs"
        docs_dir.mkdir(parents=True)
        (docs_dir / "task.md").write_text("Use the available tools to produce report.txt")
        (docs_dir / "agent_system_prompt.md").write_text("Prefer using shell utilities.")
        (task_dir / "task_config.json").write_text(
            json.dumps(
                {
                    "meta": {"task_id": "alpha", "title": "Alpha Task"},
                    "needed_local_tools": ["python"],
                }
            )
        )
        (task_dir / "evaluation").mkdir()

        result = load_benchmark_dataset("toolathlon", task_dir)

    scenario = result.scenarios[0]
    assert scenario.id == "toolathlon-alpha"
    assert scenario.evaluation_mode == "external"
    assert "Prefer using shell utilities." in scenario.input_query
    assert scenario.metadata["has_evaluation"] is True


def test_multi_swe_importer_reads_jsonl_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "multi"
        root.mkdir()
        data_path = root / "python_dataset.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "instance_id": "django__django-1",
                    "org": "django",
                    "repo": "django",
                    "title": "Fix admin bug",
                    "body": "Admin page crashes for empty values.",
                    "resolved_issues": ["#123"],
                    "fix_patch": "diff --git a/a b/a",
                }
            )
            + "\n"
        )

        result = load_benchmark_dataset("multi-swe-bench", root)

    scenario = result.scenarios[0]
    assert scenario.benchmark == "multi-swe-bench"
    assert scenario.evaluation_mode == "external"
    assert "django/django" in scenario.input_query
    assert scenario.metadata["source_file"] == "python_dataset.jsonl"


def test_manifest_importer_builds_runtime_scenarios():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "manifest.json"
        input_path.write_text(
            json.dumps(
                [
                    {
                        "id": "case-001",
                        "name": "Judge task",
                        "prompt": "Summarize the incident.",
                        "judge_rubric_text": "Score completeness from 1-5.",
                        "reference_answer": "Complete summary.",
                    }
                ]
            )
        )

        result = load_benchmark_dataset("artificial-analysis", input_path)

        scenario = result.scenarios[0]
        assert scenario.benchmark == "artificial-analysis"
        assert scenario.evaluation_mode == "llm_judge"
        assert scenario.judge_rubric_text == "Score completeness from 1-5."
        assert scenario.reference_answer == "Complete summary."


def test_persisting_benchmark_yaml_files_is_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "manifest.json"
        input_path.write_text(
            json.dumps(
                [
                    {
                        "id": "case-001",
                        "prompt": "Summarize the incident.",
                    }
                ]
            )
        )

        with pytest.raises(RuntimeError, match="Benchmark YAML persistence has been removed"):
            import_benchmark_dataset("artificial-analysis", input_path)


def test_get_importer_rejects_unknown_benchmark():
    try:
        get_importer("unknown-suite")
    except ValueError as exc:
        assert "Unsupported benchmark importer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unknown importer lookup to fail")
