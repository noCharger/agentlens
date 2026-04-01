from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentlens.model_selection import required_api_key_env, resolve_model_selection

if TYPE_CHECKING:
    from agentlens.config import AgentLensSettings


def create_chat_llm(
    settings: "AgentLensSettings",
    model: str,
    *,
    temperature: float = 0.0,
) -> Any:
    selection = resolve_model_selection(model)

    if selection.provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                f"{required_api_key_env(selection.provider)} is required for model '{selection.label}'."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=selection.model_name,
            google_api_key=settings.google_api_key,
            temperature=temperature,
        )

    if selection.provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError(
                f"{required_api_key_env(selection.provider)} is required for model '{selection.label}'."
            )
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "DeepSeek support requires the 'langchain-deepseek' package. "
                "Install project dependencies again after updating pyproject.toml."
            ) from exc

        return ChatDeepSeek(
            model=selection.model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_api_base,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported model provider '{selection.provider}'.")
