from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_MODEL_PROVIDERS = ("gemini", "deepseek", "openrouter")
DEFAULT_AGENT_MODEL = "gemini:gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "gemini:gemini-2.5-flash-lite"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    provider: str
    model_name: str
    raw: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_name}"


def _infer_provider(model_name: str, default_provider: str = "gemini") -> str:
    lowered = model_name.casefold()
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if "/" in lowered and not lowered.startswith(("http://", "https://")):
        # OpenRouter model identifiers are typically provider-prefixed
        # names like "openai/gpt-4o-mini".
        return "openrouter"
    return default_provider


def resolve_model_selection(
    model: str,
    *,
    default_provider: str = "gemini",
) -> ModelSelection:
    raw = str(model or "").strip()
    if not raw:
        raise ValueError("Model selection cannot be empty.")

    if ":" in raw:
        provider, model_name = raw.split(":", 1)
        provider = provider.strip().casefold()
        model_name = model_name.strip()
    else:
        provider = _infer_provider(raw, default_provider=default_provider)
        model_name = raw

    if provider not in SUPPORTED_MODEL_PROVIDERS:
        supported = ", ".join(SUPPORTED_MODEL_PROVIDERS)
        raise ValueError(f"Unsupported model provider '{provider}'. Supported: {supported}")
    if not model_name:
        raise ValueError(f"Model selection '{raw}' is missing a model name.")

    return ModelSelection(provider=provider, model_name=model_name, raw=raw)


def required_api_key_env(provider: str) -> str:
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"Unsupported model provider '{provider}'")
