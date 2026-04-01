import pytest

from agentlens.config import AgentLensSettings


@pytest.fixture
def settings():
    return AgentLensSettings(
        _env_file=None,
        google_api_key="test-key-not-real",
        deepseek_api_key="test-deepseek-key-not-real",
        agent_model="gemini:gemini-2.5-flash",
        judge_model="gemini:gemini-2.5-flash-lite",
    )
