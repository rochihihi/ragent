"""Task-aware command semantics shared by the agent and approval callback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from veripatch.studio_tools import is_detached_launch


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
    # Recognize explicit detached-launch syntax only; never infer verification
    # from executable names or rewrite a command based on task intent.
    target = changed_launch_target(root, changed_files, command)
    role = "launch" if is_detached_launch(command) else "command"
    return CommandExecution(command=list(command), role=role, target=target)
