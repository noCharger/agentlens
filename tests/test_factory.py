import pytest
from types import SimpleNamespace

from agentlens.agents.factory import (
    TOOL_PRESETS,
    _build_tools,
    create_agent,
    get_tool_names_for_preset,
)


def test_tool_presets_contains_expected_keys():
    assert "file_ops" in TOOL_PRESETS
    assert "shell" in TOOL_PRESETS
    assert "search" in TOOL_PRESETS
    assert "full" in TOOL_PRESETS


def test_get_tool_names_for_preset():
    names = get_tool_names_for_preset("file_ops")
    assert names == ["read_file", "write_file"]


def test_get_tool_names_for_preset_full():
    names = get_tool_names_for_preset("full")
    assert "read_file" in names
    assert "write_file" in names
    assert "shell" in names
    assert "duckduckgo_search" in names


def test_get_tool_names_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_tool_names_for_preset("nonexistent")


def test_build_tools_creates_correct_count():
    tools = _build_tools(["read_file", "write_file"])
    assert len(tools) == 2


def test_build_tools_shell():
    tools = _build_tools(["shell"])
    assert len(tools) == 1
    assert tools[0].name == "terminal"  # ShellTool's default name


def test_build_tools_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        _build_tools(["nonexistent_tool"])


def test_build_tools_empty_list():
    tools = _build_tools([])
    assert tools == []


def test_create_agent_passes_agent_max_tokens(monkeypatch):
    captured = {}

    def fake_create_chat_llm(settings, model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("agentlens.agents.factory.create_chat_llm", fake_create_chat_llm)
    monkeypatch.setattr(
        "agentlens.agents.factory._build_tools",
        lambda _, **kwargs: [],
    )
    monkeypatch.setattr("agentlens.agents.factory.create_react_agent", lambda llm, tools: llm)

    settings = SimpleNamespace(
        agent_model="openrouter:openai/gpt-4.1",
        agent_max_tokens=321,
    )

    _ = create_agent(settings, preset="full")

    assert captured["model"] == "openrouter:openai/gpt-4.1"
    assert captured["max_tokens"] == 321
