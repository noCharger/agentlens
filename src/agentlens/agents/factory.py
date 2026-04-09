from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from agentlens.agents.runtime import create_agent_runtime
from agentlens.agents.tool_registry import TOOL_PRESETS, build_langgraph_tools, get_tool_names_for_preset
from agentlens.config import AgentLensSettings
from agentlens.llms import create_chat_llm
from agentlens.sandbox import ShellSandboxPolicy, build_shell_sandbox_policy


def _build_tools(
    tool_names: list[str],
    *,
    shell_policy: ShellSandboxPolicy | None = None,
):
    return build_langgraph_tools(tool_names, shell_policy=shell_policy)


def create_agent(
    settings: AgentLensSettings,
    preset: str = "full",
    *,
    scenario=None,
):
    framework = getattr(settings, "agent_framework", "langgraph")
    if framework == "langgraph":
        tool_names = get_tool_names_for_preset(preset)
        llm = create_chat_llm(
            settings,
            settings.agent_model,
            max_tokens=settings.agent_max_tokens,
        )
        shell_policy = build_shell_sandbox_policy(scenario) if scenario is not None else None
        tools = _build_tools(tool_names, shell_policy=shell_policy)
        return create_react_agent(llm, tools)

    runtime = create_agent_runtime(settings, preset=preset, scenario=scenario)
    return runtime.agent


__all__ = [
    "TOOL_PRESETS",
    "_build_tools",
    "create_agent",
    "create_agent_runtime",
    "get_tool_names_for_preset",
    "build_shell_sandbox_policy",
]
