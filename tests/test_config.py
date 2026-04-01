from agentlens.config import AgentLensSettings, get_settings
from agentlens.model_selection import DEFAULT_AGENT_MODEL, DEFAULT_JUDGE_MODEL


def test_settings_with_explicit_values():
    s = AgentLensSettings(_env_file=None, google_api_key="test-key")
    assert s.google_api_key == "test-key"
    assert s.deepseek_api_key is None
    assert s.agent_model == DEFAULT_AGENT_MODEL
    assert s.judge_model == DEFAULT_JUDGE_MODEL
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
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.setenv("AGENT_MODEL", "deepseek:deepseek-chat")
    s = AgentLensSettings(_env_file=None)
    assert s.deepseek_api_key == "env-key"
    assert s.agent_model == "deepseek:deepseek-chat"


def test_settings_allow_missing_keys_until_model_is_used(monkeypatch):
    for key in ["GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    settings = AgentLensSettings(_env_file=None)
    assert settings.google_api_key is None
    assert settings.deepseek_api_key is None


def test_get_settings_helper():
    s = get_settings(_env_file=None, google_api_key="helper-key", agent_max_steps=3)
    assert s.google_api_key == "helper-key"
    assert s.agent_max_steps == 3
