from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from agentlens.agents.tool_registry import build_ag2_tools, build_langgraph_tools, get_tool_names_for_preset
from agentlens.config import AgentLensSettings
from agentlens.llms import create_chat_llm
from agentlens.model_selection import resolve_model_selection
from agentlens.openrouter import _normalize_openrouter_api_base
from agentlens.sandbox import build_shell_sandbox_policy
from agentlens.zhipu import _normalize_zhipu_api_base

_TERMINATE_RE = re.compile(r"\bTERMINATE\b\s*$", re.IGNORECASE)
_AG2_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Solve the task using the available tools when needed. "
    "Prefer tools over guessing. When the task is complete, provide the final answer and end "
    "your response with TERMINATE."
)


@dataclass
class AgentInvocationResult:
    raw_result: Any
    output_text: str


@dataclass
class SpanView:
    name: str
    attributes: dict[str, Any]
    start_time: int | None = None
    end_time: int | None = None
    status: Any = None


class RuntimeInstrumentor(Protocol):
    def uninstrument(self) -> None: ...


class AgentRuntime(Protocol):
    framework: str

    def instrument(self, tracer_provider: TracerProvider) -> RuntimeInstrumentor: ...

    def invoke(self, query: str, *, max_steps: int) -> AgentInvocationResult: ...

    def normalize_spans(self, spans: list[ReadableSpan]) -> list[ReadableSpan | SpanView]: ...


class LangGraphRuntime:
    framework = "langgraph"

    def __init__(
        self,
        settings: AgentLensSettings,
        *,
        preset: str,
        scenario=None,
    ):
        from langgraph.prebuilt import create_react_agent

        tool_names = get_tool_names_for_preset(preset)
        shell_policy = build_shell_sandbox_policy(scenario) if scenario is not None else None
        tools = build_langgraph_tools(tool_names, shell_policy=shell_policy)
        llm = create_chat_llm(
            settings,
            settings.agent_model,
            max_tokens=settings.agent_max_tokens,
        )
        self.agent = create_react_agent(llm, tools)

    def instrument(self, tracer_provider: TracerProvider) -> RuntimeInstrumentor:
        from agentlens.observability.instrument import instrument_runtime

        return instrument_runtime(self.framework, tracer_provider, target=self.agent)

    def invoke(self, query: str, *, max_steps: int) -> AgentInvocationResult:
        raw_result = self.agent.invoke(
            {"messages": [("user", query)]},
            config={"recursion_limit": max_steps * 2},
        )
        output_text = ""
        final_messages = raw_result.get("messages", [])
        if final_messages:
            last_msg = final_messages[-1]
            output_text = _normalize_output_content(
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )
        return AgentInvocationResult(raw_result=raw_result, output_text=output_text)

    def normalize_spans(self, spans: list[ReadableSpan]) -> list[ReadableSpan]:
        return spans


class AG2Runtime:
    framework = "ag2"

    def __init__(
        self,
        settings: AgentLensSettings,
        *,
        preset: str,
        scenario=None,
    ):
        try:
            from autogen import AssistantAgent
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "AG2 support requires the 'ag2' package. Install project dependencies again after "
                "updating pyproject.toml."
            ) from exc

        tool_names = get_tool_names_for_preset(preset)
        shell_policy = build_shell_sandbox_policy(scenario) if scenario is not None else None
        self.tools = build_ag2_tools(tool_names, shell_policy=shell_policy)
        self.agent = AssistantAgent(
            name="assistant",
            system_message=_AG2_SYSTEM_MESSAGE,
            llm_config=_build_ag2_llm_config(settings),
            human_input_mode="NEVER",
        )

    def instrument(self, tracer_provider: TracerProvider) -> RuntimeInstrumentor:
        from agentlens.observability.instrument import instrument_runtime

        return instrument_runtime(self.framework, tracer_provider, target=self.agent)

    def invoke(self, query: str, *, max_steps: int) -> AgentInvocationResult:
        response = self.agent.run(
            message=query,
            max_turns=max_steps,
            tools=self.tools,
            silent=True,
            user_input=False,
            executor_kwargs={"is_termination_msg": _should_terminate_ag2_executor_turn},
        )
        for _ in response.events:
            pass
        output_text = _clean_terminate_marker(response.summary or "")
        if not output_text:
            output_text = _extract_output_from_messages(list(response.messages))
        return AgentInvocationResult(raw_result=response, output_text=output_text)

    def normalize_spans(self, spans: list[ReadableSpan]) -> list[ReadableSpan | SpanView]:
        normalized: list[ReadableSpan | SpanView] = []
        for span in spans:
            attrs = dict(span.attributes or {})
            span_type = attrs.get("ag2.span.type")
            if not span_type:
                normalized.append(span)
                continue

            if span_type == "agent" and attrs.get("gen_ai.agent.name") != self.agent.name:
                normalized.append(SpanView(
                    name=span.name,
                    attributes=attrs,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    status=span.status,
                ))
                continue

            normalized.append(
                SpanView(
                    name=span.name,
                    attributes=_normalize_ag2_attributes(span_type, attrs),
                    start_time=span.start_time,
                    end_time=span.end_time,
                    status=span.status,
                )
            )
        return normalized


def create_agent_runtime(
    settings: AgentLensSettings,
    preset: str = "full",
    *,
    scenario=None,
) -> AgentRuntime:
    framework = getattr(settings, "agent_framework", "langgraph")
    if framework == "langgraph":
        return LangGraphRuntime(settings, preset=preset, scenario=scenario)
    if framework == "ag2":
        return AG2Runtime(settings, preset=preset, scenario=scenario)
    raise ValueError(f"Unsupported agent framework '{framework}'.")


def _build_ag2_llm_config(settings: AgentLensSettings) -> Any:
    try:
        from autogen.llm_config import LLMConfig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "AG2 support requires the 'ag2' package. Install project dependencies again after "
            "updating pyproject.toml."
        ) from exc

    selection = resolve_model_selection(settings.agent_model)
    common: dict[str, Any] = {
        "model": selection.model_name,
        "temperature": 0.0,
        "max_tokens": settings.agent_max_tokens,
    }

    if selection.provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for AG2 Gemini runs.")
        return LLMConfig({
            **common,
            "api_type": "google",
            "api_key": settings.google_api_key,
        })

    if selection.provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for AG2 DeepSeek runs.")
        return LLMConfig({
            **common,
            "api_type": "deepseek",
            "api_key": settings.deepseek_api_key,
            "base_url": _normalize_deepseek_api_base(settings.deepseek_api_base),
        })

    if selection.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for AG2 OpenRouter runs.")
        config = {
            **common,
            "api_type": "openai",
            "api_key": settings.openrouter_api_key,
            "base_url": _normalize_openrouter_api_base(settings.openrouter_api_base),
        }
        default_headers = _openrouter_headers(settings)
        if default_headers:
            config["default_headers"] = default_headers
        return LLMConfig(config)

    if selection.provider == "zhipu":
        if not settings.zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY is required for AG2 Zhipu runs.")
        config = {
            **common,
            "api_type": "openai",
            "api_key": settings.zhipu_api_key,
            "base_url": _normalize_zhipu_api_base(settings.zhipu_api_base),
        }
        price = _zhipu_price_for_model(selection.model_name)
        if price is not None:
            config["price"] = list(price)
        return LLMConfig(config)

    raise ValueError(f"Unsupported AG2 model provider '{selection.provider}'.")


def _openrouter_headers(settings: AgentLensSettings) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_x_title:
        headers["X-Title"] = settings.openrouter_x_title
    return headers


def _normalize_deepseek_api_base(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _zhipu_price_for_model(model_name: str) -> tuple[float, float] | None:
    normalized = model_name.strip().casefold()

    direct_prices_per_million = {
        "glm-4-plus": 5.0,
        "glm-4-air-250414": 0.5,
        "glm-4-airx": 10.0,
        "glm-4-flashx-250414": 0.1,
        "glm-4-flash-250414": 0.0,
    }
    if normalized in direct_prices_per_million:
        per_million = direct_prices_per_million[normalized]
        per_thousand = per_million / 1000.0
        return per_thousand, per_thousand

    # Alias inference based on Zhipu's GLM-4 docs:
    # - GLM-4-FlashX-250414 is the paid enhanced version of free GLM-4-Flash.
    # - GLM-4-Air-250414 and GLM-4-AirX are the currently documented Air variants.
    if normalized == "glm-4-flash" or normalized.startswith("glm-4-flash-"):
        return 0.0, 0.0
    if normalized == "glm-4-flashx" or normalized.startswith("glm-4-flashx-"):
        return 0.1 / 1000.0, 0.1 / 1000.0
    if normalized == "glm-4-air" or normalized.startswith("glm-4-air-"):
        return 0.5 / 1000.0, 0.5 / 1000.0
    if normalized.startswith("glm-4-airx-"):
        return 10.0 / 1000.0, 10.0 / 1000.0

    return None


def _normalize_output_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _clean_terminate_marker(text: str) -> str:
    return _TERMINATE_RE.sub("", text).strip()


def _extract_output_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            if role == "assistant":
                content = message.get("content")
                if content:
                    return _clean_terminate_marker(_normalize_output_content(content))
    return ""


def _normalize_ag2_attributes(span_type: Any, attrs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(attrs)
    span_type_text = str(span_type)

    if span_type_text == "tool":
        normalized["openinference.span.kind"] = "TOOL"
        tool_name = attrs.get("gen_ai.tool.name")
        if tool_name:
            normalized["tool.name"] = str(tool_name)
        tool_args = attrs.get("gen_ai.tool.call.arguments")
        if tool_args is not None:
            normalized["tool.params"] = str(tool_args)
            normalized["input.value"] = str(tool_args)
        tool_result = attrs.get("gen_ai.tool.call.result")
        if tool_result is not None:
            normalized["tool.output"] = str(tool_result)
        return normalized

    if span_type_text == "agent":
        normalized["openinference.span.kind"] = "AGENT"
        output_messages = attrs.get("gen_ai.output.messages")
        output_text = _extract_text_from_otel_messages(output_messages)
        if output_text:
            normalized["output.value"] = output_text
        return normalized

    if span_type_text == "llm":
        normalized["openinference.span.kind"] = "LLM"
        model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
        if model:
            normalized["llm.model_name"] = str(model)
        prompt_tokens = attrs.get("gen_ai.usage.input_tokens")
        if prompt_tokens is not None:
            normalized["llm.token_count.prompt"] = int(prompt_tokens)
        completion_tokens = attrs.get("gen_ai.usage.output_tokens")
        if completion_tokens is not None:
            normalized["llm.token_count.completion"] = int(completion_tokens)
        return normalized

    return normalized


def _extract_text_from_otel_messages(raw_messages: Any) -> str:
    if not raw_messages:
        return ""
    try:
        messages = json.loads(raw_messages) if isinstance(raw_messages, str) else raw_messages
    except json.JSONDecodeError:
        return ""

    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("type") == "text":
                content = part.get("content")
                if content:
                    parts.append(str(content))
    return _clean_terminate_marker("\n".join(parts).strip())


def _should_terminate_ag2_executor_turn(message: dict[str, Any]) -> bool:
    if message.get("tool_calls"):
        return False

    content = message.get("content")
    if content is None:
        return False

    text = _normalize_output_content(content).strip()
    if not text:
        return False
    return True
