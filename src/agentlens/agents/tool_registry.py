from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agentlens.sandbox import GuardedShellTool, ShellSandboxPolicy


TOOL_PRESETS: dict[str, list[str]] = {
    "file_ops": ["read_file", "write_file"],
    "shell": ["shell"],
    "shell_file": ["shell", "read_file", "write_file"],
    "search": ["duckduckgo_search"],
    "full": ["read_file", "write_file", "shell", "duckduckgo_search"],
}


@dataclass(frozen=True)
class ToolDefinition:
    logical_name: str
    public_name: str
    description: str
    langgraph_factory: Callable[[ShellSandboxPolicy], Any]
    ag2_factory: Callable[[ShellSandboxPolicy], Any]


def _read_file(file_path: str) -> str:
    """Read a UTF-8 text file from disk and return its contents."""
    return Path(file_path).read_text(encoding="utf-8")


def _write_file(file_path: str, text: str, append: bool = False) -> str:
    """Write UTF-8 text to a file, creating parent directories when needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(text)
    action = "Appended" if append else "Wrote"
    return f"{action} {len(text)} characters to {file_path}"


def _make_terminal_callable(policy: ShellSandboxPolicy) -> Callable[[str], str]:
    shell_tool = GuardedShellTool(shell_policy=policy)

    def terminal(commands: str) -> str:
        """Execute a shell command in the benchmark sandbox and return stdout/stderr."""
        return shell_tool._run(commands)

    terminal.__name__ = "terminal"
    return terminal


def _make_duckduckgo_search_callable() -> Callable[[str], str]:
    from langchain_community.tools import DuckDuckGoSearchRun

    search_tool = DuckDuckGoSearchRun()

    def duckduckgo_search(query: str) -> str:
        """Search the public web with DuckDuckGo and return a short text result."""
        if hasattr(search_tool, "invoke"):
            return str(search_tool.invoke(query))
        return str(search_tool.run(query))

    duckduckgo_search.__name__ = "duckduckgo_search"
    return duckduckgo_search


_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "read_file": ToolDefinition(
        logical_name="read_file",
        public_name="read_file",
        description="Read a UTF-8 text file from disk.",
        langgraph_factory=lambda _policy: __import__(
            "langchain_community.tools", fromlist=["ReadFileTool"]
        ).ReadFileTool(),
        ag2_factory=lambda _policy: _make_ag2_tool(
            name="read_file",
            description="Read a UTF-8 text file from disk.",
            func=_read_file,
        ),
    ),
    "write_file": ToolDefinition(
        logical_name="write_file",
        public_name="write_file",
        description="Write UTF-8 text to a file.",
        langgraph_factory=lambda _policy: __import__(
            "langchain_community.tools", fromlist=["WriteFileTool"]
        ).WriteFileTool(),
        ag2_factory=lambda _policy: _make_ag2_tool(
            name="write_file",
            description="Write UTF-8 text to a file.",
            func=_write_file,
        ),
    ),
    "shell": ToolDefinition(
        logical_name="shell",
        public_name="terminal",
        description="Execute a shell command in the benchmark sandbox.",
        langgraph_factory=lambda policy: GuardedShellTool(shell_policy=policy),
        ag2_factory=lambda policy: _make_ag2_tool(
            name="terminal",
            description="Execute a shell command in the benchmark sandbox.",
            func=_make_terminal_callable(policy),
        ),
    ),
    "duckduckgo_search": ToolDefinition(
        logical_name="duckduckgo_search",
        public_name="duckduckgo_search",
        description="Search the public web with DuckDuckGo.",
        langgraph_factory=lambda _policy: __import__(
            "langchain_community.tools", fromlist=["DuckDuckGoSearchRun"]
        ).DuckDuckGoSearchRun(),
        ag2_factory=lambda _policy: _make_ag2_tool(
            name="duckduckgo_search",
            description="Search the public web with DuckDuckGo.",
            func=_make_duckduckgo_search_callable(),
        ),
    ),
}


def _make_ag2_tool(*, name: str, description: str, func: Callable[..., Any]) -> Any:
    try:
        from autogen.tools import Tool
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "AG2 support requires the 'ag2' package. Install project dependencies again after "
            "updating pyproject.toml."
        ) from exc
    return Tool(name=name, description=description, func_or_tool=func)


def get_tool_definition(name: str) -> ToolDefinition:
    try:
        return _TOOL_DEFINITIONS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown tool: {name}") from exc


def get_tool_names_for_preset(preset: str) -> list[str]:
    if preset not in TOOL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(TOOL_PRESETS.keys())}")
    return list(TOOL_PRESETS[preset])


def build_langgraph_tools(
    tool_names: list[str],
    *,
    shell_policy: ShellSandboxPolicy | None = None,
) -> list[Any]:
    policy = shell_policy or ShellSandboxPolicy.disabled()
    return [get_tool_definition(name).langgraph_factory(policy) for name in tool_names]


def build_ag2_tools(
    tool_names: list[str],
    *,
    shell_policy: ShellSandboxPolicy | None = None,
) -> list[Any]:
    policy = shell_policy or ShellSandboxPolicy.disabled()
    return [get_tool_definition(name).ag2_factory(policy) for name in tool_names]
