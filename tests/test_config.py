from agentlens.config import AgentLensSettings, get_settings
from agentlens.model_selection import DEFAULT_AGENT_MODEL, DEFAULT_JUDGE_MODEL


def test_settings_with_explicit_values():
    s = AgentLensSettings(_env_file=None, google_api_key="test-key")
    assert s.google_api_key == "test-key"
    assert s.deepseek_api_key is None
    assert s.openrouter_api_key is None
    assert s.zhipu_api_key is None
    assert s.openrouter_api_base == "https://openrouter.ai/api/v1"
    assert s.zhipu_api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert s.agent_model == DEFAULT_AGENT_MODEL
    assert s.agent_max_tokens == 2048
    assert s.judge_model == DEFAULT_JUDGE_MODEL
    assert s.judge_max_tokens == 512
    assert s.otel_exporter_otlp_endpoint == "http://localhost:4317"
    assert s.otel_service_name == "agentlens"
    assert s.agent_max_steps == 10


def test_settings_override_defaults():
    s = AgentLensSettings(
        _env_file=None,
        google_api_key="k",
        agent_model="gemini:gemini-custom",
        agent_max_steps=5,
    )
    assert s.agent_model == "gemini:gemini-custom"
    assert s.agent_max_steps == 5


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://openrouter.ai")
    monkeypatch.setenv("ZHIPU_API_KEY", "zp-env-key")
    monkeypatch.setenv("ZHIPU_API_BASE", "https://open.bigmodel.cn")
    monkeypatch.setenv("AGENT_MODEL", "openrouter:openai/gpt-4o-mini")
    s = AgentLensSettings(_env_file=None)
    assert s.openrouter_api_key == "or-env-key"
    assert s.openrouter_api_base == "https://openrouter.ai"
    assert s.zhipu_api_key == "zp-env-key"
    assert s.zhipu_api_base == "https://open.bigmodel.cn"
    assert s.agent_model == "openrouter:openai/gpt-4o-mini"


def test_settings_allow_missing_keys_until_model_is_used(monkeypatch):
    for key in ["GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ZHIPU_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    settings = AgentLensSettings(_env_file=None)
    assert settings.google_api_key is None
    assert settings.deepseek_api_key is None
    assert settings.openrouter_api_key is None
    assert settings.zhipu_api_key is None


def test_get_settings_helper():
    s = get_settings(_env_file=None, google_api_key="helper-key", agent_max_steps=3)
    assert s.google_api_key == "helper-key"
    assert s.agent_max_steps == 3
