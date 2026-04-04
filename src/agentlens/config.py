from pydantic import Field
from pydantic_settings import BaseSettings

from agentlens.model_selection import DEFAULT_AGENT_MODEL, DEFAULT_JUDGE_MODEL


class AgentLensSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    google_api_key: str | None = Field(default=None, description="Google AI Studio API key")
    deepseek_api_key: str | None = Field(default=None, description="DeepSeek API key")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL",
    )
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL (OpenAI-compatible)",
    )
    openrouter_http_referer: str | None = Field(
        default=None,
        description="Optional HTTP-Referer header sent to OpenRouter",
    )
    openrouter_x_title: str | None = Field(
        default=None,
        description="Optional X-Title header sent to OpenRouter",
    )
    agent_model: str = Field(
        default=DEFAULT_AGENT_MODEL,
        description=(
            "Agent model selection, for example gemini:gemini-2.5-flash, "
            "deepseek:deepseek-chat, or openrouter:openai/gpt-4o-mini"
        ),
    )
    agent_max_tokens: int = Field(
        default=2048,
        ge=64,
        description="Maximum completion tokens for agent responses",
    )
    judge_model: str = Field(
        default=DEFAULT_JUDGE_MODEL,
        description=(
            "Judge model selection, for example gemini:gemini-2.5-flash-lite, "
            "deepseek:deepseek-chat, or openrouter:openai/gpt-4o-mini"
        ),
    )
    judge_max_tokens: int = Field(
        default=512,
        ge=64,
        description="Maximum completion tokens for L2 judge responses",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP gRPC endpoint"
    )
    otel_service_name: str = Field(default="agentlens", description="OTEL service name")
    agent_max_steps: int = Field(default=10, description="Default max steps for agent execution")


def get_settings(**overrides) -> AgentLensSettings:
    return AgentLensSettings(**overrides)
