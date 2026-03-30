import os

import pytest

from agentlens.config import AgentLensSettings, get_settings


def test_settings_with_explicit_values():
    s = AgentLensSettings(google_api_key="test-key")
    assert s.google_api_key == "test-key"
    assert s.agent_model == "gemini-2.5-flash"
    assert s.judge_model == "gemini-2.5-flash-lite"
    assert s.otel_exporter_otlp_endpoint == "http://localhost:4317"
    assert s.otel_service_name == "agentlens"
    assert s.agent_max_steps == 10


def test_settings_override_defaults():
    s = AgentLensSettings(
        google_api_key="k",
        agent_model="gemini-custom",
        agent_max_steps=5,
    )
    assert s.agent_model == "gemini-custom"
    assert s.agent_max_steps == 5


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    monkeypatch.setenv("AGENT_MODEL", "gemini-env")
    s = AgentLensSettings()
    assert s.google_api_key == "env-key"
    assert s.agent_model == "gemini-env"


def test_settings_missing_api_key():
    # Ensure no env var is set
    env = os.environ.copy()
    for k in ["GOOGLE_API_KEY"]:
        env.pop(k, None)
    with pytest.raises(Exception):
        AgentLensSettings(_env_file=None)


def test_get_settings_helper():
    s = get_settings(google_api_key="helper-key", agent_max_steps=3)
    assert s.google_api_key == "helper-key"
    assert s.agent_max_steps == 3
