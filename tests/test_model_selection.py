import sys
from types import ModuleType

import pytest

from agentlens.config import AgentLensSettings
from agentlens.llms import create_chat_llm
from agentlens.model_selection import resolve_model_selection


def test_resolve_model_selection_prefixed_provider():
    selection = resolve_model_selection("deepseek:deepseek-chat")
    assert selection.provider == "deepseek"
    assert selection.model_name == "deepseek-chat"
    assert selection.label == "deepseek:deepseek-chat"


def test_resolve_model_selection_prefixed_openrouter_provider():
    selection = resolve_model_selection("openrouter:openai/gpt-4o-mini")
    assert selection.provider == "openrouter"
    assert selection.model_name == "openai/gpt-4o-mini"
    assert selection.label == "openrouter:openai/gpt-4o-mini"


def test_resolve_model_selection_prefixed_zhipu_provider():
    selection = resolve_model_selection("zhipu:glm-4-plus")
    assert selection.provider == "zhipu"
    assert selection.model_name == "glm-4-plus"
    assert selection.label == "zhipu:glm-4-plus"


def test_resolve_model_selection_infers_provider_from_bare_name():
    selection = resolve_model_selection("deepseek-chat")
    assert selection.provider == "deepseek"
    assert selection.model_name == "deepseek-chat"


def test_resolve_model_selection_infers_zhipu_from_glm_model_name():
    selection = resolve_model_selection("glm-4-plus")
    assert selection.provider == "zhipu"
    assert selection.model_name == "glm-4-plus"


def test_resolve_model_selection_infers_openrouter_for_namespace_model():
    selection = resolve_model_selection("openai/gpt-4o-mini")
    assert selection.provider == "openrouter"
    assert selection.model_name == "openai/gpt-4o-mini"


def test_resolve_model_selection_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported model provider"):
        resolve_model_selection("anthropic:claude-sonnet")


def test_create_chat_llm_requires_matching_provider_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="deepseek:deepseek-chat",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_chat_llm(settings, settings.agent_model)


def test_create_chat_llm_uses_deepseek_sdk(monkeypatch):
    captured = {}
    fake_module = ModuleType("langchain_deepseek")

    class FakeChatDeepSeek:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.ChatDeepSeek = FakeChatDeepSeek
    monkeypatch.setitem(sys.modules, "langchain_deepseek", fake_module)

    settings = AgentLensSettings(
        _env_file=None,
        deepseek_api_key="ds-test-key",
        deepseek_api_base="https://api.deepseek.com",
        agent_model="deepseek:deepseek-chat",
        judge_model="deepseek:deepseek-chat",
    )

    model = create_chat_llm(settings, settings.agent_model, temperature=0.2)

    assert isinstance(model, FakeChatDeepSeek)
    assert captured["model"] == "deepseek-chat"
    assert captured["api_key"] == "ds-test-key"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["temperature"] == 0.2


def test_create_chat_llm_requires_openrouter_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="openrouter:openai/gpt-4o-mini",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_chat_llm(settings, settings.agent_model)


def test_create_chat_llm_requires_zhipu_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="zhipu:glm-4-plus",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
        create_chat_llm(settings, settings.agent_model)


def test_create_chat_llm_uses_openrouter_sdk(monkeypatch):
    captured = {}
    fake_module = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    settings = AgentLensSettings(
        _env_file=None,
        openrouter_api_key="or-test-key",
        openrouter_api_base="https://openrouter.ai",
        openrouter_http_referer="https://agentlens.example",
        openrouter_x_title="AgentLens Test",
        agent_model="openrouter:openai/gpt-4o-mini",
        judge_model="openrouter:openai/gpt-4o-mini",
    )

    model = create_chat_llm(settings, settings.agent_model, temperature=0.4, max_tokens=256)

    assert isinstance(model, FakeChatOpenAI)
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["api_key"] == "or-test-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["temperature"] == 0.4
    assert captured["max_tokens"] == 256
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://agentlens.example",
        "X-Title": "AgentLens Test",
    }


def test_create_chat_llm_uses_zhipu_via_openai_compatible_sdk(monkeypatch):
    captured = {}
    fake_module = ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    settings = AgentLensSettings(
        _env_file=None,
        zhipu_api_key="zp-test-key",
        zhipu_api_base="https://open.bigmodel.cn",
        agent_model="zhipu:glm-4-plus",
        judge_model="zhipu:glm-4-plus",
    )

    model = create_chat_llm(settings, settings.agent_model, temperature=0.1, max_tokens=300)

    assert isinstance(model, FakeChatOpenAI)
    assert captured["model"] == "glm-4-plus"
    assert captured["api_key"] == "zp-test-key"
    assert captured["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
    assert captured["temperature"] == 0.1
    assert captured["max_tokens"] == 300
