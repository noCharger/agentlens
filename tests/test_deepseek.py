from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from agentlens.config import AgentLensSettings
from agentlens.deepseek import (
    DeepSeekPreflightError,
    _normalize_deepseek_api_base,
    fetch_deepseek_balance,
    validate_deepseek_preflight,
)


def test_normalize_deepseek_api_base_strips_v1_suffix():
    assert _normalize_deepseek_api_base("https://api.deepseek.com/v1") == "https://api.deepseek.com"
    assert _normalize_deepseek_api_base("https://api.deepseek.com/") == "https://api.deepseek.com"


def test_fetch_deepseek_balance_success(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "is_available": True,
                "balance_infos": [
                    {
                        "currency": "USD",
                        "total_balance": "1.23",
                        "granted_balance": "0.00",
                        "topped_up_balance": "1.23",
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    summary = fetch_deepseek_balance(
        api_key="ds-key",
        base_url="https://api.deepseek.com/v1",
        timeout_s=3.0,
    )

    assert summary.is_available is True
    assert summary.formatted_totals == "USD 1.23"
    assert captured["url"] == "https://api.deepseek.com/user/balance"
    assert captured["headers"]["Authorization"] == "Bearer ds-key"
    assert captured["timeout"] == 3.0


def test_fetch_deepseek_balance_raises_on_402(monkeypatch):
    def fake_get(url, headers, timeout):
        return SimpleNamespace(
            status_code=402,
            json=lambda: {
                "error": {
                    "message": "Insufficient Balance",
                }
            },
            text="",
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(DeepSeekPreflightError, match="insufficient balance"):
        fetch_deepseek_balance(api_key="ds-key", base_url="https://api.deepseek.com")


def test_validate_deepseek_preflight_skips_non_deepseek_models(monkeypatch):
    settings = AgentLensSettings(
        _env_file=None,
        google_api_key="g-key",
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    def unexpected_call(*args, **kwargs):  # pragma: no cover
        raise AssertionError("DeepSeek balance endpoint should not be called")

    monkeypatch.setattr(httpx, "get", unexpected_call)
    assert validate_deepseek_preflight(settings, require_judge=True) is None


def test_validate_deepseek_preflight_requires_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="deepseek:deepseek-chat",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(DeepSeekPreflightError, match="DEEPSEEK_API_KEY"):
        validate_deepseek_preflight(settings)
