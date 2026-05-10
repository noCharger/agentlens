from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from agentlens.agents.tool_registry import (
    build_ag2_tools,
    build_langgraph_tools,
    get_tool_names_for_preset,
)
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
        system_prompt: str | None = None,
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
        kwargs = {}
        if system_prompt:
            kwargs["prompt"] = system_prompt
        self.agent = create_react_agent(llm, tools, **kwargs)

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
        system_prompt: str | None = None,
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
            system_message=system_prompt if system_prompt else _AG2_SYSTEM_MESSAGE,
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


class _SubprocessCLIRuntime:
    framework = ""
    executable_name = ""
    display_name = ""
    executable_env_var = ""
    executable_fallback_paths: tuple[str, ...] = ()

    def __init__(
        self,
        settings: AgentLensSettings,
        *,
        preset: str,
        scenario=None,
        system_prompt: str | None = None,
        timeout_seconds: float = 600.0,
    ):
        self.settings = settings
        self.preset = preset
        self.scenario = scenario
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds
        self.agent = self
        self._last_spans: list[SpanView] = []
        self._tool_names = get_tool_names_for_preset(preset)
        self._allowed_dirs = _cli_allowed_dirs(scenario)
        self._executable = self._resolve_executable()
        if not self._executable:
            raise RuntimeError(
                f"{self.display_name} CLI executable "
                f"'{self.executable_name}' was not found on PATH"
                f"{self._fallback_hint()}."
            )

    def instrument(self, tracer_provider: TracerProvider) -> RuntimeInstrumentor:
        from agentlens.observability.instrument import instrument_runtime

        return instrument_runtime(self.framework, tracer_provider, target=self)

    def invoke(self, query: str, *, max_steps: int) -> AgentInvocationResult:
        cmd = self._build_command(max_steps=max_steps)
        with tempfile.TemporaryDirectory(prefix=f"agentlens-{self.framework}-") as work_dir:
            completed = subprocess.run(
                cmd,
                input=query,
                text=True,
                capture_output=True,
                cwd=work_dir,
                timeout=self.timeout_seconds,
            )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"{self.display_name} CLI failed ({completed.returncode}){suffix}")

        output_text, spans = self._parse_output(completed.stdout or "")
        if not output_text and completed.stdout:
            output_text = completed.stdout.strip()
            if output_text:
                spans.append(_make_output_span(output_text, len(spans)))
        self._last_spans = spans
        return AgentInvocationResult(raw_result=completed, output_text=output_text)

    def normalize_spans(self, spans: list[ReadableSpan]) -> list[ReadableSpan | SpanView]:
        return [*spans, *self._last_spans]

    def _build_command(self, *, max_steps: int) -> list[str]:
        raise NotImplementedError

    def _parse_output(self, stdout: str) -> tuple[str, list[SpanView]]:
        raise NotImplementedError

    def _raw_model(self) -> str:
        return str(getattr(self.settings, "agent_model", "") or "").strip()

    def _configured_executable(self) -> str:
        attr_name = f"{self.framework.replace('-', '_')}_cli_path"
        configured = str(getattr(self.settings, attr_name, "") or "").strip()
        if configured:
            return configured
        if self.executable_env_var:
            return str(os.environ.get(self.executable_env_var, "") or "").strip()
        return ""

    def _resolve_executable(self) -> str | None:
        configured = self._configured_executable()
        if configured:
            path = Path(configured).expanduser()
            if path.is_file():
                return str(path)
            resolved = shutil.which(configured)
            if resolved:
                return resolved
            raise RuntimeError(
                f"{self.display_name} CLI configured path was not found: {configured}"
            )

        resolved = shutil.which(self.executable_name)
        if resolved:
            return resolved

        for fallback in self.executable_fallback_paths:
            path = Path(fallback).expanduser()
            if path.is_file():
                return str(path)
        return None

    def _fallback_hint(self) -> str:
        hints = []
        if self.executable_env_var:
            hints.append(f"set {self.executable_env_var}")
        if self.executable_fallback_paths:
            joined = ", ".join(self.executable_fallback_paths)
            hints.append(f"checked fallbacks: {joined}")
        return "; " + "; ".join(hints) if hints else ""


class ClaudeCodeRuntime(_SubprocessCLIRuntime):
    framework = "claude-code"
    executable_name = "claude"
    display_name = "Claude Code"
    executable_env_var = "CLAUDE_CODE_CLI_PATH"

    def _build_command(self, *, max_steps: int) -> list[str]:
        del max_steps
        cmd = [
            self._executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "text",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            ",".join(_claude_allowed_tools(self._tool_names)),
        ]
        for allowed_dir in self._allowed_dirs:
            cmd.extend(["--add-dir", allowed_dir])
        model = self._raw_model()
        if model:
            cmd.extend(["--model", model])
        if self.system_prompt:
            cmd.extend(["--append-system-prompt", self.system_prompt])
        return cmd

    def _parse_output(self, stdout: str) -> tuple[str, list[SpanView]]:
        return _parse_claude_code_events(stdout)


class CodexRuntime(_SubprocessCLIRuntime):
    framework = "codex"
    executable_name = "codex"
    display_name = "Codex"
    executable_env_var = "CODEX_CLI_PATH"
    executable_fallback_paths = ("/Applications/Codex.app/Contents/Resources/codex",)

    def _build_command(self, *, max_steps: int) -> list[str]:
        del max_steps
        cmd = [
            self._executable,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "-c",
            'approval_policy="never"',
            "--sandbox",
            "workspace-write",
        ]
        for allowed_dir in self._allowed_dirs:
            cmd.extend(["--add-dir", allowed_dir])
        model = self._raw_model()
        if model:
            cmd.extend(["--model", model])
        return cmd

    def _parse_output(self, stdout: str) -> tuple[str, list[SpanView]]:
        return _parse_codex_events(stdout)


def create_agent_runtime(
    settings: AgentLensSettings,
    preset: str = "full",
    *,
    scenario=None,
    system_prompt: str | None = None,
) -> AgentRuntime:
    framework = getattr(settings, "agent_framework", "langgraph")
    if framework == "langgraph":
        return LangGraphRuntime(
            settings, preset=preset, scenario=scenario, system_prompt=system_prompt
        )
    if framework == "ag2":
        return AG2Runtime(
            settings, preset=preset, scenario=scenario, system_prompt=system_prompt
        )
    if framework == "claude-code":
        return ClaudeCodeRuntime(
            settings, preset=preset, scenario=scenario, system_prompt=system_prompt
        )
    if framework == "codex":
        return CodexRuntime(
            settings, preset=preset, scenario=scenario, system_prompt=system_prompt
        )
    raise ValueError(f"Unsupported agent framework '{framework}'.")


def _cli_allowed_dirs(scenario) -> list[str]:
    dirs = ["/tmp"]
    if scenario is None:
        return dirs

    metadata = getattr(scenario, "metadata", {}) or {}
    for key in ("resolved_reference_files", "resolved_deliverable_files"):
        raw_paths = metadata.get(key) or []
        if not isinstance(raw_paths, list):
            continue
        for raw_path in raw_paths:
            value = str(raw_path).strip()
            if not value:
                continue
            dirs.append(str(Path(value).expanduser().resolve(strict=False).parent))

    deduped: list[str] = []
    seen: set[str] = set()
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _claude_allowed_tools(tool_names: list[str]) -> list[str]:
    allowed: list[str] = []
    for name in tool_names:
        if name == "read_file":
            allowed.append("Read")
        elif name == "write_file":
            allowed.extend(["Write", "Edit", "MultiEdit"])
        elif name == "shell":
            allowed.append("Bash")
        elif name == "duckduckgo_search":
            allowed.extend(["WebSearch", "WebFetch"])
    return _dedupe_strings(allowed) or ["Read"]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _parse_json_lines(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _span_time(index: int) -> tuple[int, int]:
    start = 1_000_000_000 + index * 1_000_000
    return start, start + 500_000


def _make_span(name: str, attributes: dict[str, Any], index: int) -> SpanView:
    start, end = _span_time(index)
    return SpanView(name=name, attributes=attributes, start_time=start, end_time=end)


def _make_step_span(step_index: int, action: str, index: int) -> SpanView:
    return _make_span(
        "agent.step",
        {
            "step.index": step_index,
            "step.action": action,
            "step.thought": f"CLI requested {action}",
        },
        index,
    )


def _make_tool_span(
    tool_name: str,
    params: Any,
    index: int,
    *,
    output: Any | None = None,
) -> SpanView:
    params_text = _stringify_tool_params(params)
    attributes: dict[str, Any] = {
        "openinference.span.kind": "TOOL",
        "tool.name": tool_name,
        "tool.params": params_text,
        "input.value": params_text,
    }
    if output is not None:
        attributes["tool.output"] = _normalize_output_content(output)
    return _make_span(f"Tool: {tool_name}", attributes, index)


def _make_output_span(output_text: str, index: int) -> SpanView:
    return _make_span(
        "agent.output",
        {
            "agent.output": output_text,
            "output.value": output_text,
        },
        index,
    )


def _make_llm_span(
    usage: Any,
    model: Any,
    index: int,
) -> SpanView | None:
    if not isinstance(usage, dict):
        return None
    prompt_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("input")
        or usage.get("prompt")
    )
    completion_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("output")
        or usage.get("completion")
    )
    if prompt_tokens is None and completion_tokens is None:
        return None

    attrs: dict[str, Any] = {"openinference.span.kind": "LLM"}
    if model:
        attrs["llm.model_name"] = str(model)
    if prompt_tokens is not None:
        attrs["llm.token_count.prompt"] = int(prompt_tokens)
    if completion_tokens is not None:
        attrs["llm.token_count.completion"] = int(completion_tokens)
    return _make_span("llm.call", attrs, index)


def _stringify_tool_params(params: Any) -> str:
    if params is None:
        return ""
    if isinstance(params, str):
        return params
    try:
        return json.dumps(params, ensure_ascii=False)
    except TypeError:
        return str(params)


def _normalize_cli_tool_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    lowered = name.replace("-", "_").replace(" ", "_").casefold()

    if lowered in {"read", "readfile", "read_file", "ls", "glob", "grep"}:
        return "read_file"
    if lowered in {"write", "edit", "multiedit", "notebookedit", "write_file"}:
        return "write_file"
    if lowered in {
        "bash",
        "shell",
        "terminal",
        "exec",
        "exec_command",
        "shell_command",
        "command",
    }:
        return "terminal"
    if lowered in {"websearch", "web_search", "webfetch", "web_fetch", "search"}:
        return "duckduckgo_search"
    return lowered or "unknown"


def _message_content_blocks(message: Any) -> list[Any]:
    if not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if isinstance(content, list):
        return content
    if content:
        return [content]
    return []


def _extract_text_from_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if isinstance(text, list):
                parts.append(_extract_text_from_blocks(text))
            elif text:
                parts.append(str(text))
    return "\n".join(part for part in parts if part).strip()


def _event_model(event: dict[str, Any]) -> Any:
    message = event.get("message")
    if isinstance(message, dict):
        return event.get("model") or message.get("model")
    return event.get("model")


def _event_usage(event: dict[str, Any]) -> Any:
    message = event.get("message")
    if isinstance(message, dict):
        return event.get("usage") or message.get("usage")
    return event.get("usage")


def _parse_claude_code_events(stdout: str) -> tuple[str, list[SpanView]]:
    spans: list[SpanView] = []
    output_text = ""
    step_count = 0
    tool_spans_by_id: dict[str, SpanView] = {}

    for event in _parse_json_lines(stdout):
        event_type = str(event.get("type", ""))
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        blocks = _message_content_blocks(message)

        if event_type == "assistant":
            text = _extract_text_from_blocks(blocks)
            if text:
                output_text = text

            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_name = _normalize_cli_tool_name(block.get("name"))
                spans.append(_make_step_span(step_count, tool_name, len(spans)))
                tool_span = _make_tool_span(tool_name, block.get("input", {}), len(spans))
                spans.append(tool_span)
                tool_id = str(block.get("id") or "")
                if tool_id:
                    tool_spans_by_id[tool_id] = tool_span
                step_count += 1

        elif event_type == "user":
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = str(block.get("tool_use_id") or "")
                tool_span = tool_spans_by_id.get(tool_id)
                if tool_span is not None:
                    tool_span.attributes["tool.output"] = _normalize_output_content(
                        block.get("content", "")
                    )

        if event_type == "result":
            result = event.get("result") or event.get("output") or event.get("message")
            if result:
                output_text = _normalize_output_content(result)

        usage_span = _make_llm_span(_event_usage(event), _event_model(event), len(spans))
        if usage_span is not None:
            spans.append(usage_span)

    if output_text:
        spans.append(_make_output_span(output_text, len(spans)))
    return output_text, spans


def _codex_payload(event: dict[str, Any]) -> dict[str, Any]:
    msg = event.get("msg")
    if isinstance(msg, dict):
        merged = dict(msg)
        merged.setdefault("type", event.get("type"))
        return merged
    return event


def _codex_item(event: dict[str, Any]) -> dict[str, Any]:
    item = event.get("item") or event.get("data")
    return item if isinstance(item, dict) else event


def _codex_tool_call_fields(event: dict[str, Any]) -> tuple[str, str, Any, Any]:
    item = _codex_item(event)
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    tool_id = str(item.get("id") or item.get("call_id") or event.get("id") or "")
    command = item.get("command")
    raw_tool_name = item.get("name") or item.get("tool_name") or function.get("name")
    tool_name = (
        _codex_tool_name_from_command(str(command))
        if item.get("type") == "command_execution" and command
        else _normalize_cli_tool_name(raw_tool_name)
    )
    params = (
        item.get("arguments")
        or item.get("args")
        or item.get("input")
        or item.get("params")
        or function.get("arguments")
        or ({"command": command} if command else None)
    )
    output = (
        item.get("output")
        or item.get("aggregated_output")
        or item.get("result")
        or item.get("content")
    )
    return tool_id, tool_name, params, output


def _codex_tool_name_from_command(command: str) -> str:
    normalized = command.strip()
    if not normalized:
        return "terminal"

    lowered = normalized.casefold()
    if " -lc " in lowered:
        try:
            tokens = shlex.split(normalized)
            if len(tokens) >= 3 and Path(tokens[0]).name in {"bash", "sh", "zsh"}:
                lowered = tokens[2].casefold()
        except ValueError:
            pass

    read_prefixes = ("cat ", "head ", "tail ", "less ", "more ", "grep ", "sed ", "awk ")
    if lowered.startswith(read_prefixes) or " cat /" in lowered:
        return "read_file"

    if ">" in lowered or lowered.startswith(("tee ", "touch ")):
        return "write_file"

    return "terminal"


def _codex_event_is_tool(event: dict[str, Any]) -> bool:
    item = _codex_item(event)
    event_type = str(event.get("type", ""))
    item_type = str(item.get("type", ""))
    return (
        item_type in {"tool_call", "function_call", "command_execution"}
        or "tool_call" in event_type
        or "function_call" in event_type
        or bool(item.get("tool_name"))
    )


def _codex_final_text(event: dict[str, Any]) -> str:
    for key in ("last_message", "final_message", "output", "text"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = event.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if isinstance(message, dict):
        text = _extract_text_from_blocks(_message_content_blocks(message))
        if text:
            return text

    item = _codex_item(event)
    if str(item.get("type", "")) in {"message", "assistant_message", "agent_message"}:
        content = item.get("content") or item.get("text")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            return _extract_text_from_blocks(content)
    return ""


def _parse_codex_events(stdout: str) -> tuple[str, list[SpanView]]:
    spans: list[SpanView] = []
    output_text = ""
    step_count = 0
    tool_spans_by_id: dict[str, SpanView] = {}

    for raw_event in _parse_json_lines(stdout):
        event = _codex_payload(raw_event)
        if _codex_event_is_tool(event):
            tool_id, tool_name, params, output = _codex_tool_call_fields(event)
            tool_span = tool_spans_by_id.get(tool_id) if tool_id else None
            if tool_span is None:
                spans.append(_make_step_span(step_count, tool_name, len(spans)))
                tool_span = _make_tool_span(tool_name, params or {}, len(spans), output=output)
                spans.append(tool_span)
                if tool_id:
                    tool_spans_by_id[tool_id] = tool_span
                step_count += 1
            elif output is not None:
                tool_span.attributes["tool.output"] = _normalize_output_content(output)

        final_text = _codex_final_text(event)
        if final_text:
            output_text = final_text

        usage_span = _make_llm_span(_event_usage(event), _event_model(event), len(spans))
        if usage_span is not None:
            spans.append(usage_span)

    if output_text:
        spans.append(_make_output_span(output_text, len(spans)))
    return output_text, spans


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
