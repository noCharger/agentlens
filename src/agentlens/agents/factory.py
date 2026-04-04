from langgraph.prebuilt import create_react_agent

from agentlens.config import AgentLensSettings
from agentlens.llms import create_chat_llm
from agentlens.sandbox import GuardedShellTool, ShellSandboxPolicy, build_shell_sandbox_policy


TOOL_PRESETS: dict[str, list[str]] = {
    "file_ops": ["read_file", "write_file"],
    "shell": ["shell"],
    "shell_file": ["shell", "read_file", "write_file"],
    "search": ["duckduckgo_search"],
    "full": ["read_file", "write_file", "shell", "duckduckgo_search"],
}


def _build_tools(
    tool_names: list[str],
    *,
    shell_policy: ShellSandboxPolicy | None = None,
):
    tools = []
    policy = shell_policy or ShellSandboxPolicy.disabled()
    for name in tool_names:
        if name == "read_file":
            from langchain_community.tools import ReadFileTool

            tools.append(ReadFileTool())
        elif name == "write_file":
            from langchain_community.tools import WriteFileTool

            tools.append(WriteFileTool())
        elif name == "shell":
            tools.append(GuardedShellTool(shell_policy=policy))
        elif name == "duckduckgo_search":
            from langchain_community.tools import DuckDuckGoSearchRun

            tools.append(DuckDuckGoSearchRun())
        else:
            raise ValueError(f"Unknown tool: {name}")
    return tools


def create_agent(
    settings: AgentLensSettings,
    preset: str = "full",
    *,
    scenario=None,
):
    tool_names = TOOL_PRESETS.get(preset)
    if tool_names is None:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(TOOL_PRESETS.keys())}")

    llm = create_chat_llm(
        settings,
        settings.agent_model,
        max_tokens=settings.agent_max_tokens,
    )
    shell_policy = build_shell_sandbox_policy(scenario) if scenario is not None else None
    tools = _build_tools(tool_names, shell_policy=shell_policy)
    return create_react_agent(llm, tools)


def get_tool_names_for_preset(preset: str) -> list[str]:
    if preset not in TOOL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(TOOL_PRESETS.keys())}")
    return list(TOOL_PRESETS[preset])
