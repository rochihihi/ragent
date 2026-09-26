"""Task-aware command semantics shared by the agent and approval callback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from veripatch.studio_tools import is_detached_launch


def is_verification_command(command: list[str]) -> bool:
    if not command:
        return False
    lowered = [part.casefold() for part in command]
    executable = Path(lowered[0]).name
    if "pytest" in lowered or executable in {"pytest", "pytest.exe", "ruff", "mypy"}:
        return True
    if executable in {"npm", "npm.cmd", "pnpm", "yarn"}:
        return any(part in {"test", "build", "lint", "check", "typecheck"} for part in lowered[1:])
    if executable == "go":
        return any(part in {"test", "build", "vet"} for part in lowered[1:])
    if executable in {"cargo", "dotnet"}:
        return any(part in {"test", "build", "check", "clippy"} for part in lowered[1:])
    if executable in {"python", "python.exe", "py", "py.exe"}:
        if len(lowered) >= 2 and lowered[1].endswith(".py"):
            return True
        if any(part in {"compileall", "py_compile", "unittest"} for part in lowered):
            return True
        if "-c" in lowered:
            index = lowered.index("-c") + 1
            script = lowered[index] if index < len(lowered) else ""
            return any(
                marker in script
                for marker in ("assert ", "compile(", "py_compile", "compileall")
            )
    return False


def changed_launch_target(root: Path, changed_files: list[str], command: list[str]) -> str | None:
    changed = {path.casefold(): path for path in changed_files}
    root = root.resolve()
    for argument in reversed(command):
        if not Path(argument).suffix:
            continue
        try:
            path = Path(argument)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        if relative.casefold() in changed:
            return changed[relative.casefold()]
    return None


def launch_window_confirmed(payload: dict[str, object]) -> bool:
    """Read structured launch evidence, with support for older saved observations."""
    if payload.get("launch_state") is not None:
        return payload.get("window_confirmed") is True
    return "window_confirmed=true" in str(payload.get("stdout", "")).casefold()


def launch_effect_satisfied(payload: dict[str, object]) -> bool:
    """A window is observed, or Windows accepted an associated-document open."""
    if payload.get("exit_code") not in (None, 0):
        return False
    return (
        payload.get("window_confirmed") is True
        or launch_window_confirmed(payload)
        or (payload.get("launch_state") == "dispatched" and payload.get("exit_code") == 0)
    )


@dataclass(frozen=True)
class CommandExecution:
    command: list[str]
    role: str
    target: str | None


def classify_command(
    command: list[str], *, root: Path, changed_files: list[str], launch_required: bool
) -> CommandExecution:
    target = changed_launch_target(root, changed_files, command)
    normalized = list(command)
    if (
        launch_required
        and target is not None
        and len(command) == 2
        and Path(command[0]).name.casefold() in {"python", "python.exe", "pythonw", "pythonw.exe"}
        and command[1].casefold().endswith(".py")
    ):
        normalized = ["cmd", "/c", "start", "", *command]
    role = "launch" if is_detached_launch(normalized) else (
        "verification" if is_verification_command(normalized) else "command"
    )
    return CommandExecution(command=normalized, role=role, target=target)
