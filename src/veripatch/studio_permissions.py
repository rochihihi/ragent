"""Execution approvals, independent of task intent and filesystem boundaries."""

from pathlib import PurePosixPath

from veripatch.studio_domain import PermissionMode, StudioDecision, StudioSession
from veripatch.studio_tools import ALLOWED_COMMANDS, UnsafeStudioCommand, validate_studio_command

READ_ONLY = {
    "list_files",
    "search",
    "read",
    "git_status",
    "git_diff",
    "git_log",
    "poll_terminal",
    "inspect_processes",
    "respond",
    "finish",
    "fail",
    "request_permission",
    "batch",
}
IMPORTANT = {
    "delete_path",
    "git_restore",
    "git_commit",
    "git_branch",
    "write_terminal",
    "stop_terminal",
    "start_terminal",
}


def fingerprint(decision: StudioDecision) -> str:
    return decision.model_dump_json(exclude={"rationale", "message"}, exclude_none=True)


def _command_rule(command: list[str]) -> str | None:
    """Return the narrowest safe rule shared by equivalent verification commands."""
    if not command:
        return None
    try:
        validate_studio_command(command)
    except UnsafeStudioCommand:
        return None
    executable = command[0].casefold()
    prefixes = ALLOWED_COMMANDS.get(executable)
    if prefixes is None:
        return None
    arguments = tuple(part.casefold() for part in command[1:])
    matching = [prefix for prefix in prefixes if arguments[: len(prefix)] == prefix]
    if not matching:
        return None
    prefix = max(matching, key=len)
    return "rule:command:" + " ".join((executable, *prefix))


def session_rule(decision: StudioDecision) -> str | None:
    """Build a scoped session grant; destructive actions intentionally stay exact."""
    action = decision.action.value
    if action in {"run_command", "run_tests", "start_terminal"}:
        return _command_rule(decision.command)
    if action in {"create", "edit", "move_file", "copy_file"} and decision.path:
        parent = str(PurePosixPath(decision.path).parent)
        return f"rule:file:{action}:{parent.casefold()}"
    if action in {"git_status", "git_diff", "git_log"}:
        return f"rule:action:{action}"
    return None


def session_grant_matches(session: StudioSession, decision: StudioDecision) -> bool:
    key = fingerprint(decision)
    if key in session.action_grants:
        return True  # backwards-compatible grants from older sessions
    rule = session_rule(decision)
    return rule is not None and rule in session.action_grants


def requires_approval(session: StudioSession, decision: StudioDecision) -> bool:
    if decision.action.value == "batch":
        if (
            fingerprint(decision) in session.once_grants
            or session_grant_matches(session, decision)
        ):
            return False
        return any(requires_approval(session, item) for item in decision.actions)
    if decision.action.value == "git_branch" and not decision.branch:
        return False
    if decision.action.value in READ_ONLY:
        return False
    key = fingerprint(decision)
    if key in session.once_grants or session_grant_matches(session, decision):
        return False
    if (
        decision.action.value in {"run_command", "run_tests", "start_terminal"}
        and decision.command in session.approved_commands
    ):
        return False
    if session.permission_mode == PermissionMode.FULL:
        return False
    if session.permission_mode == PermissionMode.ASK:
        return True
    if decision.action.value in IMPORTANT:
        return True
    if decision.action.value in {"run_command", "run_tests"}:
        try:
            validate_studio_command(decision.command or session.test_command)
        except UnsafeStudioCommand:
            return True
    return False
