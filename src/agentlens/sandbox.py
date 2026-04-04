from __future__ import annotations

import importlib.util
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_community.tools import ShellTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import Field

from agentlens.eval.scenarios import Scenario


class SandboxViolationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchmarkSandboxProfile:
    allowed_commands: tuple[str, ...]
    blocked_commands: tuple[str, ...]
    required_python_modules: tuple[str, ...] = ()
    extra_allowed_roots: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ShellSandboxPolicy:
    enabled: bool
    benchmark: str
    allowed_commands: frozenset[str]
    blocked_commands: frozenset[str]
    allowed_roots: tuple[Path, ...]

    @classmethod
    def disabled(cls) -> ShellSandboxPolicy:
        return cls(
            enabled=False,
            benchmark="",
            allowed_commands=frozenset(),
            blocked_commands=frozenset(),
            allowed_roots=tuple(),
        )


DEFAULT_BENCHMARK_PROFILE = BenchmarkSandboxProfile(
    allowed_commands=(
        "python",
        "python3",
        "ls",
        "cp",
        "mv",
        "mkdir",
        "rm",
        "find",
        "head",
        "tail",
        "wc",
        "grep",
        "sed",
        "awk",
        "cat",
        "file",
        "pwd",
        "echo",
        "test",
        "stat",
        "cut",
        "sort",
        "uniq",
    ),
    blocked_commands=(
        "pip",
        "pip3",
        "curl",
        "wget",
        "open",
        "xdg-open",
        "brew",
        "apt",
        "apt-get",
        "yum",
    ),
)

BENCHMARK_SANDBOX_PROFILES: dict[str, BenchmarkSandboxProfile] = {
    "gdpval-aa": BenchmarkSandboxProfile(
        allowed_commands=DEFAULT_BENCHMARK_PROFILE.allowed_commands,
        blocked_commands=DEFAULT_BENCHMARK_PROFILE.blocked_commands,
        required_python_modules=("openpyxl", "pandas", "numpy"),
    ),
}


def _iter_metadata_paths(scenario: Scenario, key: str) -> Iterable[Path]:
    raw = scenario.metadata.get(key)
    if isinstance(raw, list):
        for item in raw:
            value = str(item).strip()
            if value:
                yield Path(value).expanduser()


def _load_profile_override(benchmark: str, workspace_root: Path | None) -> dict | None:
    if workspace_root is None:
        return None
    profile_path = (
        workspace_root
        / "data"
        / "benchmarks"
        / benchmark
        / "sandbox_profile.json"
    )
    if not profile_path.exists():
        return None

    try:
        payload = json.loads(profile_path.read_text())
    except Exception as exc:
        raise SandboxViolationError(
            f"Invalid sandbox profile file: {profile_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SandboxViolationError(
            f"Sandbox profile must be a JSON object: {profile_path}"
        )
    return payload


def _profile_for_benchmark(
    benchmark: str,
    *,
    workspace_root: Path | None = None,
) -> BenchmarkSandboxProfile:
    base = BENCHMARK_SANDBOX_PROFILES.get(benchmark, DEFAULT_BENCHMARK_PROFILE)
    override = _load_profile_override(benchmark, workspace_root)
    if override is None:
        return base

    def _tuple_str(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = override.get(key)
        if value is None:
            return default
        if not isinstance(value, list):
            raise SandboxViolationError(
                f"sandbox_profile.json field '{key}' must be a list of strings."
            )
        return tuple(str(item) for item in value if str(item).strip())

    return BenchmarkSandboxProfile(
        allowed_commands=_tuple_str("allowed_commands", base.allowed_commands),
        blocked_commands=_tuple_str("blocked_commands", base.blocked_commands),
        required_python_modules=_tuple_str(
            "required_python_modules", base.required_python_modules
        ),
        extra_allowed_roots=_tuple_str("extra_allowed_roots", base.extra_allowed_roots),
    )


def prepare_benchmark_environment(
    scenario: Scenario,
    *,
    workspace_root: Path | None = None,
) -> str | None:
    if not scenario.benchmark:
        return None

    root = (workspace_root or Path.cwd()).resolve()
    profile = _profile_for_benchmark(scenario.benchmark, workspace_root=root)
    missing_modules = [
        name for name in profile.required_python_modules if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        joined = ", ".join(sorted(missing_modules))
        return (
            f"Sandbox prepare failed: missing Python modules ({joined}). "
            "Install benchmark dependencies first (for example: pip install -e \".[benchmarks]\")."
        )

    missing_reference_files = [
        path for path in _iter_metadata_paths(scenario, "resolved_reference_files") if not path.exists()
    ]
    if missing_reference_files:
        sample = ", ".join(str(path) for path in missing_reference_files[:3])
        return (
            "Sandbox prepare failed: missing benchmark reference files. "
            f"Examples: {sample}"
        )

    for output_path in _iter_metadata_paths(scenario, "resolved_deliverable_files"):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    return None


def build_shell_sandbox_policy(
    scenario: Scenario,
    *,
    workspace_root: Path | None = None,
) -> ShellSandboxPolicy:
    if not scenario.benchmark:
        return ShellSandboxPolicy.disabled()

    root = (workspace_root or Path.cwd()).resolve()
    benchmark_root = (root / "data" / "benchmarks" / scenario.benchmark).resolve()
    profile = _profile_for_benchmark(scenario.benchmark, workspace_root=root)

    roots: list[Path] = [root, benchmark_root, Path("/tmp")]
    roots.extend(path.parent.resolve() for path in _iter_metadata_paths(scenario, "resolved_reference_files"))
    roots.extend(path.parent.resolve() for path in _iter_metadata_paths(scenario, "resolved_deliverable_files"))
    roots.extend((benchmark_root / relative).resolve() for relative in profile.extra_allowed_roots)

    deduped_roots: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped_roots.append(p)

    return ShellSandboxPolicy(
        enabled=True,
        benchmark=scenario.benchmark,
        allowed_commands=frozenset(profile.allowed_commands),
        blocked_commands=frozenset(profile.blocked_commands),
        allowed_roots=tuple(deduped_roots),
    )


def _split_command_segments(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    length = len(command)

    while i < length:
        ch = command[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if i + 1 < length and command[i: i + 2] in {"&&", "||"}:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                i += 2
                continue
            if ch in {";", "|"}:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments


def _extract_executable(tokens: list[str]) -> tuple[str | None, int]:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if "=" in token and not token.startswith("/") and token.index("=") > 0:
            idx += 1
            continue
        command = Path(token).name
        return command, idx
    return None, -1


def _ensure_path_allowed(path_str: str, policy: ShellSandboxPolicy) -> None:
    path = Path(path_str).expanduser().resolve(strict=False)
    for allowed_root in policy.allowed_roots:
        try:
            path.relative_to(allowed_root)
            return
        except ValueError:
            continue
    raise SandboxViolationError(
        f"Command references path outside sandbox roots: {path}"
    )


def validate_shell_commands(commands: str | list[str], policy: ShellSandboxPolicy) -> None:
    if not policy.enabled:
        return

    command_list = [commands] if isinstance(commands, str) else list(commands)
    for command in command_list:
        for segment in _split_command_segments(command):
            try:
                tokens = shlex.split(segment)
            except ValueError as exc:
                raise SandboxViolationError(f"Unable to parse shell segment: {segment}") from exc

            if not tokens:
                continue

            executable, idx = _extract_executable(tokens)
            if executable is None:
                continue

            executable_lower = executable.casefold()

            if executable_lower in policy.blocked_commands:
                raise SandboxViolationError(
                    f"Command '{executable}' is blocked in benchmark sandbox mode. "
                    f"If this benchmark requires it, declare an override in "
                    f"data/benchmarks/{policy.benchmark}/sandbox_profile.json."
                )

            if (
                executable_lower in {"python", "python3"}
                and len(tokens) > idx + 2
                and tokens[idx + 1] == "-m"
                and tokens[idx + 2].casefold() in {"pip", "pip3"}
            ):
                raise SandboxViolationError(
                    "Python module installer commands are blocked in benchmark sandbox mode."
                )

            if policy.allowed_commands and executable_lower not in policy.allowed_commands:
                raise SandboxViolationError(
                    f"Command '{executable}' is not allowed for benchmark '{policy.benchmark}'."
                )

            for token in tokens:
                if token.startswith("/"):
                    _ensure_path_allowed(token, policy)


class GuardedShellTool(ShellTool):
    shell_policy: ShellSandboxPolicy = Field(default_factory=ShellSandboxPolicy.disabled)

    def _run(
        self,
        commands,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            validate_shell_commands(commands, self.shell_policy)
        except SandboxViolationError as exc:
            # Return a normal tool response so the agent can recover with an
            # alternative command instead of aborting the whole scenario.
            return f"SandboxViolation: {exc}"
        return super()._run(commands=commands, run_manager=run_manager)
