import json
import subprocess
from types import SimpleNamespace

import pytest

from agentlens.agents.runtime import (
    ClaudeCodeRuntime,
    CodexRuntime,
    _parse_claude_code_events,
    _parse_codex_events,
    create_agent_runtime,
)
from agentlens.eval.level1_deterministic.output_format import extract_output
from agentlens.eval.level1_deterministic.tool_params import extract_tool_params
from agentlens.eval.level1_deterministic.tool_usage import extract_tool_names
from agentlens.eval.level1_deterministic.trajectory import count_steps, sum_tokens


CLAUDE_STREAM = "\n".join(
    [
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1",'
        '"name":"Read","input":{"file_path":"/tmp/agentlens_test/data.txt"}}]}}',
        '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_1",'
        '"content":"hello agentlens"}]}}',
        '{"type":"result","result":"The file says hello agentlens.",'
        '"model":"claude-sonnet-4-6","usage":{"input_tokens":12,"output_tokens":5}}',
    ]
)


CODEX_STREAM = "\n".join(
    [
        '{"type":"item.started","item":{"type":"tool_call","id":"call_1","name":"read_file",'
        '"arguments":{"file_path":"/tmp/agentlens_test/data.txt"}}}',
        '{"type":"item.completed","item":{"type":"tool_call","id":"call_1","name":"read_file",'
        '"arguments":{"file_path":"/tmp/agentlens_test/data.txt"},'
        '"output":"hello agentlens"}}',
        '{"type":"turn.completed","last_message":"The file says hello agentlens.",'
        '"model":"gpt-5.2","usage":{"input_tokens":9,"output_tokens":4}}',
    ]
)


CODEX_REAL_COMMAND_STREAM = "\n".join(
    json.dumps(event)
    for event in [
        {
            "type": "item.started",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'cat /tmp/agentlens_test/data.txt'",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'cat /tmp/agentlens_test/data.txt'",
                "aggregated_output": "hello agentlens\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "agent_message",
                "text": "hello agentlens",
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 42,
                "cached_input_tokens": 10,
                "output_tokens": 7,
                "reasoning_output_tokens": 0,
            },
        },
    ]
)


def _settings(framework: str, model: str = "raw-model") -> SimpleNamespace:
    return SimpleNamespace(
        agent_framework=framework,
        agent_model=model,
        agent_max_tokens=128,
    )


def test_parse_claude_code_events_into_agentlens_spans():
    output, spans = _parse_claude_code_events(CLAUDE_STREAM)
    params = extract_tool_params(spans)[0]

    assert output == "The file says hello agentlens."
    assert extract_output(spans) == "The file says hello agentlens."
    assert extract_tool_names(spans) == ["read_file"]
    assert params["params_raw"] == '{"file_path": "/tmp/agentlens_test/data.txt"}'
    assert count_steps(spans) == 1
    assert sum_tokens(spans) == (12, 5)


def test_parse_codex_events_into_agentlens_spans():
    output, spans = _parse_codex_events(CODEX_STREAM)
    params = extract_tool_params(spans)[0]

    assert output == "The file says hello agentlens."
    assert extract_output(spans) == "The file says hello agentlens."
    assert extract_tool_names(spans) == ["read_file"]
    assert params["params_raw"] == '{"file_path": "/tmp/agentlens_test/data.txt"}'
    assert count_steps(spans) == 1
    assert sum_tokens(spans) == (9, 4)


def test_parse_real_codex_command_execution_and_agent_message_events():
    output, spans = _parse_codex_events(CODEX_REAL_COMMAND_STREAM)
    params = extract_tool_params(spans)[0]

    assert output == "hello agentlens"
    assert extract_output(spans) == "hello agentlens"
    assert extract_tool_names(spans) == ["read_file"]
    assert "cat /tmp/agentlens_test/data.txt" in params["params_raw"]
    assert count_steps(spans) == 1
    assert sum_tokens(spans) == (42, 7)


def test_claude_code_runtime_invokes_cli_with_raw_model_and_restricted_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout=CLAUDE_STREAM, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runtime = ClaudeCodeRuntime(_settings("claude-code", model="sonnet"), preset="full")

    result = runtime.invoke("Read the file", max_steps=3)

    assert result.output_text == "The file says hello agentlens."
    assert captured["kwargs"]["input"] == "Read the file"
    assert captured["cmd"][:1] == ["/usr/local/bin/claude"]
    assert "--model" in captured["cmd"]
    assert "sonnet" in captured["cmd"]
    assert "--verbose" in captured["cmd"]
    assert "--permission-mode" in captured["cmd"]
    assert not any("dangerously" in part for part in captured["cmd"])


def test_codex_runtime_invokes_cli_with_raw_model_and_restricted_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout=CODEX_STREAM, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    runtime = CodexRuntime(_settings("codex", model="gpt-5.2"), preset="full")

    result = runtime.invoke("Read the file", max_steps=3)

    assert result.output_text == "The file says hello agentlens."
    assert captured["kwargs"]["input"] == "Read the file"
    assert captured["cmd"][:2] == ["/usr/local/bin/codex", "exec"]
    assert "--model" in captured["cmd"]
    assert "gpt-5.2" in captured["cmd"]
    assert "--sandbox" in captured["cmd"]
    assert "workspace-write" in captured["cmd"]
    assert "--ask-for-approval" not in captured["cmd"]
    assert "-c" in captured["cmd"]
    assert 'approval_policy="never"' in captured["cmd"]
    assert not any("bypass" in part for part in captured["cmd"])


def test_cli_runtime_reports_missing_executable(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="Claude Code CLI"):
        ClaudeCodeRuntime(_settings("claude-code"), preset="full")


def test_codex_runtime_falls_back_to_desktop_app_bundle(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(
        "pathlib.Path.is_file",
        lambda self: str(self) == "/Applications/Codex.app/Contents/Resources/codex",
    )

    runtime = CodexRuntime(_settings("codex"), preset="full")

    assert runtime._executable == "/Applications/Codex.app/Contents/Resources/codex"


def test_cli_runtime_honors_executable_path_env(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setenv("CODEX_CLI_PATH", "/opt/codex/bin/codex")
    monkeypatch.setattr(
        "pathlib.Path.is_file",
        lambda self: str(self) == "/opt/codex/bin/codex",
    )

    runtime = CodexRuntime(_settings("codex"), preset="full")

    assert runtime._executable == "/opt/codex/bin/codex"


def test_cli_runtime_reports_nonzero_exit(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="auth failed")

    monkeypatch.setattr("subprocess.run", fake_run)
    runtime = CodexRuntime(_settings("codex"), preset="full")

    with pytest.raises(RuntimeError, match="Codex CLI failed \\(2\\): auth failed"):
        runtime.invoke("Read the file", max_steps=3)


def test_create_agent_runtime_returns_cli_runtimes_without_provider_parsing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")

    claude_runtime = create_agent_runtime(
        _settings("claude-code", model="sonnet"),
        preset="full",
    )
    codex_runtime = create_agent_runtime(
        _settings("codex", model="gpt-5.2"),
        preset="full",
    )

    assert isinstance(claude_runtime, ClaudeCodeRuntime)
    assert isinstance(codex_runtime, CodexRuntime)
