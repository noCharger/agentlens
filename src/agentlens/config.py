from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

from agentlens.model_selection import DEFAULT_AGENT_MODEL, DEFAULT_JUDGE_MODEL


class AgentLensSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    google_api_key: str | None = Field(default=None, description="Google AI Studio API key")
    deepseek_api_key: str | None = Field(default=None, description="DeepSeek API key")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    zhipu_api_key: str | None = Field(default=None, description="Zhipu API key")
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL",
    )
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL (OpenAI-compatible)",
    )
    zhipu_api_base: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="Zhipu API base URL (OpenAI-compatible)",
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
            "deepseek:deepseek-chat, openrouter:openai/gpt-4o-mini, or zhipu:glm-4-plus"
        ),
    )
    agent_framework: Literal["langgraph", "ag2"] = Field(
        default="langgraph",
        description="Agent runtime framework to execute scenarios with.",
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
            "deepseek:deepseek-chat, openrouter:openai/gpt-4o-mini, or zhipu:glm-4-plus"
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

    judge_use_geval: bool = Field(
        default=False,
        description="Use G-Eval CoT meta-evaluation for L2 rubric scoring",
    )
    judge_task_completion: bool = Field(
        default=False,
        description="Enable trace-based task completion metric",
    )
    judge_answer_relevancy: bool = Field(
        default=False,
        description="Enable atomic statement-level answer relevancy metric",
    )
    judge_hallucination: bool = Field(
        default=False,
        description="Enable NLI-based hallucination detection metric",
    )
    judge_faithfulness: bool = Field(
        default=False,
        description="Enable context-support faithfulness metric",
    )


def get_settings(**overrides) -> AgentLensSettings:
    return AgentLensSettings(**overrides)
