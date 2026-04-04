import pytest

from agentlens.config import AgentLensSettings
from agentlens.zhipu import (
    ZhipuPreflightError,
    _normalize_zhipu_api_base,
    validate_zhipu_preflight,
)


def test_normalize_zhipu_api_base():
    assert _normalize_zhipu_api_base("https://open.bigmodel.cn") == (
        "https://open.bigmodel.cn/api/paas/v4"
    )
    assert _normalize_zhipu_api_base("https://open.bigmodel.cn/api/paas") == (
        "https://open.bigmodel.cn/api/paas/v4"
    )
    assert _normalize_zhipu_api_base("https://open.bigmodel.cn/api/paas/v4/") == (
        "https://open.bigmodel.cn/api/paas/v4"
    )


def test_validate_zhipu_preflight_skips_non_zhipu_models():
    settings = AgentLensSettings(
        _env_file=None,
        deepseek_api_key="ds-key",
        agent_model="deepseek:deepseek-chat",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    assert validate_zhipu_preflight(settings, require_judge=True) is None


def test_validate_zhipu_preflight_requires_key():
    settings = AgentLensSettings(
        _env_file=None,
        agent_model="zhipu:glm-4-plus",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    with pytest.raises(ZhipuPreflightError, match="ZHIPU_API_KEY"):
        validate_zhipu_preflight(settings)


def test_validate_zhipu_preflight_returns_summary_when_key_present():
    settings = AgentLensSettings(
        _env_file=None,
        zhipu_api_key="zp-key",
        zhipu_api_base="https://open.bigmodel.cn",
        agent_model="zhipu:glm-4-plus",
        judge_model="gemini:gemini-2.5-flash-lite",
    )

    summary = validate_zhipu_preflight(settings)

    assert summary is not None
    assert summary.base_url == "https://open.bigmodel.cn/api/paas/v4"
