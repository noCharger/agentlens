import pytest

from agentlens.config import AgentLensSettings


@pytest.fixture
def settings():
    return AgentLensSettings(
        google_api_key="test-key-not-real",
        agent_model="gemini-2.5-flash",
        judge_model="gemini-2.5-flash-lite",
    )
