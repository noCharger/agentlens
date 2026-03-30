from pydantic import Field
from pydantic_settings import BaseSettings


class AgentLensSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    google_api_key: str = Field(description="Google AI Studio API key")
    agent_model: str = Field(default="gemini-2.5-flash", description="Model for agent execution")
    judge_model: str = Field(
        default="gemini-2.5-flash-lite", description="Model for LLM-as-Judge evaluation"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP gRPC endpoint"
    )
    otel_service_name: str = Field(default="agentlens", description="OTEL service name")
    agent_max_steps: int = Field(default=10, description="Default max steps for agent execution")


def get_settings(**overrides) -> AgentLensSettings:
    return AgentLensSettings(**overrides)
