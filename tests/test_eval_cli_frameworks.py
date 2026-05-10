from agentlens.eval.__main__ import main
from agentlens.eval.scenarios import ExpectedResult, Scenario


def _make_scenario() -> Scenario:
    return Scenario(
        id="cli-001",
        name="CLI Framework",
        category="tool_calling",
        input="Read /tmp/data.txt",
        setup=[],
        expected=ExpectedResult(
            tools_called=["read_file"],
            max_steps=4,
            output_contains=["hello"],
        ),
    )


def test_eval_cli_accepts_claude_code_and_codex_frameworks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agentlens.eval.__main__.load_runtime_scenarios",
        lambda *args, **kwargs: [_make_scenario()],
    )

    for framework in ("claude-code", "codex"):
        monkeypatch.setattr(
            "sys.argv",
            [
                "agentlens.eval",
                "--dry-run",
                "--scenarios",
                str(tmp_path),
                "--agent-framework",
                framework,
            ],
        )

        main()
