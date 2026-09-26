"""Controller-owned completion policy shared by all execution routes."""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from veripatch.studio_domain import StudioSession, TaskVerificationPolicy, VerificationMode
from veripatch.studio_harness import TaskEvidence, ToolOutcome

MUTATIONS = {"edit", "create", "patch", "move", "copy", "delete", "git_restore"}
DOCUMENTS = {".txt", ".md", ".rst"}


def is_bulk_delete(message: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:请)?(?:删除|清空|移除)(?:当前工作区|项目中|项目里|这里的)?"
            r"(?:全部|所有)文件[。.!！\s]*",
            message,
        )
    )


def snapshot(root: Path, *, artifacts: bool = True) -> dict[str, str]:
    """Observe workspace files without following links or reading protected metadata."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", ".github", ".codex"}
            and not (Path(directory) / d).is_symlink()
            and (artifacts or d not in {"__pycache__", ".pytest_cache"})
        ]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                result[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return result


def effects(session: StudioSession):
    result = {}
    for item in session.observations[session.turn_observation_start :]:
        if item.kind not in MUTATIONS:
            continue
        for path in [
            item.payload.get("path"),
            item.payload.get("source"),
            item.payload.get("destination"),
            *item.payload.get("paths", []),
        ]:
            if path in session.turn_changed_files:
                result[path] = item
    return result


def file_effects_verified(session: StudioSession) -> bool:
    latest = effects(session)
    if not session.turn_changed_files:
        return False
    for name in session.turn_changed_files:
        item = latest.get(name)
        if item is None:
            return False
        path = Path(session.repo_root) / name
        if item.kind == "delete" or (item.kind == "move" and name == item.payload.get("source")):
            if os.path.lexists(path):
                return False
            continue
        try:
            if not path.is_file():
                return False
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        expected = item.payload.get("content_hashes", {}).get(name) or item.payload.get(
            "content_sha256"
        )
        if actual != expected:
            return False
        if item.kind not in {"move", "copy"} and path.suffix.lower() not in DOCUMENTS:
            return False
    return True


def requires_commands(session: StudioSession) -> bool:
    policy = session.task_state.verification_policy if session.task_state else None
    if policy is TaskVerificationPolicy.SKIPPED_BY_USER:
        return False
    if (
        policy is TaskVerificationPolicy.REQUIRED_BY_USER
        or session.verification_mode is VerificationMode.STRICT
    ):
        return True
    return bool(session.turn_changed_files) and not file_effects_verified(session)


def unresolved_verification_failure(session: StudioSession) -> bool:
    """A different successful check must not erase a failed test result."""
    failed: set[tuple[str, ...]] = set()
    for item in session.observations[session.turn_observation_start :]:
        if item.kind not in {"test", "command"}:
            continue
        payload = item.payload
        command = tuple(str(part) for part in payload.get("command", []))
        text = " ".join(command).casefold()
        verification_command = item.kind == "test" or any(
            marker in text
            for marker in ("pytest", "py_compile", "compileall", "ruff", "mypy", "pyright")
        )
        if not verification_command or not command:
            continue
        if not ToolOutcome.succeeded(item):
            failed.add(command)
        else:
            failed.discard(command)
    return bool(failed)


@dataclass(frozen=True)
class CompletionCheck:
    state: str
    reasons: tuple[str, ...] = ()


def assess(session: StudioSession, unmet: list[str]) -> CompletionCheck:
    if session.pending_permission:
        return CompletionCheck("waiting_permission")
    if unmet:
        return CompletionCheck("blocked", tuple(unmet))
    if unresolved_verification_failure(session):
        return CompletionCheck("needs_verification", ("关联测试曾失败，尚未有同一测试通过证据",))
    if session.verification_mode is VerificationMode.STRICT and not session.turn_changed_files:
        return CompletionCheck("needs_verification", ("严格模式尚未产生修改",))
    if (
        session.verification_mode is not VerificationMode.QUICK
        and requires_commands(session)
        and not session.verification_passed
    ):
        return CompletionCheck("needs_verification", ("尚无满足任务要求的验证证据",))
    return CompletionCheck("ready")


def assess_response(session: StudioSession, unmet: list[str]) -> CompletionCheck:
    """Allow honest blocker reports, but not a premature answer for unfinished work."""
    pending_tools = TaskEvidence.pending_tools(session)
    if pending_tools and any(
        not TaskEvidence.failed_attempt(session, tool) for tool in pending_tools
    ):
        return CompletionCheck(
            "blocked",
            tuple(f"本轮尚未取得 {tool} 的工具结果" for tool in pending_tools),
        )
    contract = session.task_contract
    if contract is None or contract.intent in {"answer", "analysis"}:
        return CompletionCheck("ready")
    check = assess(session, unmet)
    if check.state == "ready" or check.state == "waiting_permission":
        return check
    latest_mutation = max(
        (
            index for index in range(session.turn_observation_start, len(session.observations))
            if session.observations[index].kind in MUTATIONS
        ),
        default=session.turn_observation_start - 1,
    )
    if any(
        item.kind in {"tool_error", "capability_guard"}
        or (item.kind in {"command", "test"} and not ToolOutcome.succeeded(item))
        for item in session.observations[latest_mutation + 1 :]
    ):
        return CompletionCheck("ready")
    return check
