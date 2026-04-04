from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from agentlens.config import AgentLensSettings
from agentlens.openrouter import (
    OpenRouterPreflightError,
    _normalize_openrouter_api_base,
    fetch_openrouter_key_info,
    validate_openrouter_preflight,
)


def test_normalize_openrouter_api_base():
    assert _normalize_openrouter_api_base("https://openrouter.ai") == "https://openrouter.ai/api/v1"
    assert _normalize_openrouter_api_base("https://openrouter.ai/api") == "https://openrouter.ai/api/v1"
    assert _normalize_openrouter_api_base("https://openrouter.ai/api/v1/") == (
        "https://openrouter.ai/api/v1"
    )


def test_fetch_openrouter_key_info_success(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": {
                    "label": "ci-key",
                    "usage": 1.5,
                    "limit": 20,
                    "limit_remaining": 18.5,
                    "is_free_tier": False,
                    "is_provisioning_key": False,
                }
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    summary = fetch_openrouter_key_info(
        api_key="or-key",
        base_url="https://openrouter.ai",
        timeout_s=2.0,
    )

    assert summary.is_available is True
    assert "remaining 18.5/20" in summary.formatted_status
    assert captured["url"] == "https://openrouter.ai/api/v1/key"
    assert captured["headers"]["Authorization"] == "Bearer or-key"
    assert captured["timeout"] == 2.0


def test_fetch_openrouter_key_info_raises_on_402(monkeypatch):
    def fake_get(url, headers, timeout):
        return SimpleNamespace(
            status_code=402,
            json=lambda: {
                "error": {
                    "message": "Insufficient credits",
                }
            },
            text="",
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(OpenRouterPreflightError, match="insufficient credits"):
        fetch_openrouter_key_info(api_key="or-key", base_url="https://openrouter.ai/api/v1")


def test_validate_openrouter_preflight_skips_non_openrouter_models(monkeypatch):
    settings = AgentLensSettings(
        _env_file=None,
        deepseek_api_key="ds-key",
        agent_model="deepseek:deepseek-chat",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    def unexpected_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("OpenRouter key endpoint should not be called")

    monkeypatch.setattr(httpx, "get", unexpected_call)
    assert validate_openrouter_preflight(settings, require_judge=True) is None


def test_validate_openrouter_preflight_requires_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="openrouter:openai/gpt-4o-mini",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(OpenRouterPreflightError, match="OPENROUTER_API_KEY"):
        validate_openrouter_preflight(settings)


def test_validate_openrouter_preflight_rejects_empty_remaining_limit(monkeypatch):
    settings = AgentLensSettings(
        _env_file=None,
        openrouter_api_key="or-key",
        openrouter_api_base="https://openrouter.ai",
        agent_model="openrouter:openai/gpt-4o-mini",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    def fake_get(url, headers, timeout):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": {
                    "limit_remaining": 0,
                    "limit": 10,
                    "usage": 10,
                }
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(OpenRouterPreflightError, match="not available for API calls"):
        validate_openrouter_preflight(settings)
