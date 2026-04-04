from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentlens.model_selection import required_api_key_env, resolve_model_selection
from agentlens.openrouter import _normalize_openrouter_api_base
from agentlens.zhipu import _normalize_zhipu_api_base

if TYPE_CHECKING:
    from agentlens.config import AgentLensSettings


def create_chat_llm(
    settings: "AgentLensSettings",
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
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

    if selection.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                f"{required_api_key_env(selection.provider)} is required for model '{selection.label}'."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "OpenRouter support requires the 'langchain-openai' package. "
                "Install project dependencies again after updating pyproject.toml."
            ) from exc

        extra_headers: dict[str, str] = {}
        if settings.openrouter_http_referer:
            extra_headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_x_title:
            extra_headers["X-Title"] = settings.openrouter_x_title

        kwargs: dict[str, Any] = {
            "model": selection.model_name,
            "api_key": settings.openrouter_api_key,
            "base_url": _normalize_openrouter_api_base(settings.openrouter_api_base),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        return ChatOpenAI(**kwargs)

    if selection.provider == "zhipu":
        if not settings.zhipu_api_key:
            raise ValueError(
                f"{required_api_key_env(selection.provider)} is required for model '{selection.label}'."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Zhipu support requires the 'langchain-openai' package. "
                "Install project dependencies again after updating pyproject.toml."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": selection.model_name,
            "api_key": settings.zhipu_api_key,
            "base_url": _normalize_zhipu_api_base(settings.zhipu_api_base),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unsupported model provider '{selection.provider}'.")
