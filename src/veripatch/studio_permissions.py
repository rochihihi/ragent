"""Execution approvals, independent of task intent and filesystem boundaries."""

import json
from pathlib import Path

from veripatch.mcp_client import external_config_fingerprint
from veripatch.studio_domain import PermissionMode, StudioDecision, StudioSession
from veripatch.studio_sandbox import policy_fingerprint
from veripatch.studio_sandbox import settings as sandbox_settings

READ_ONLY = {
    "update_plan",
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
    "mcp_call",
}


def uses_host_execution(decision: StudioDecision) -> bool:
    return decision.execution_mode == "host" or any(
        uses_host_execution(child) for child in decision.actions
    )


def fingerprint(decision: StudioDecision) -> str:
    def without_runtime_ids(item: StudioDecision) -> StudioDecision:
        return item.model_copy(
            update={
                "call_id": None,
                "actions": [without_runtime_ids(child) for child in item.actions],
            }
        )

    operation = without_runtime_ids(decision).model_dump_json(
        exclude={"rationale", "message", "call_id"}, exclude_none=True
    )

    def bindings(item):
        result = []
        if item.execution_mode == "host":
            result.append("host-execution:v1:")
        if item.action.value in {
            "mcp_call",
            "run_command",
            "run_tests",
            "start_terminal",
            "write_terminal",
            "git_commit",
            "git_branch",
            "git_restore",
        }:
            result.append(policy_fingerprint())
        if item.action.value == "mcp_call":
            digest = external_config_fingerprint(item.mcp_tool or "")
            if digest:
                result.append(digest)
        for child in item.actions:
            result.extend(bindings(child))
        return result

    return operation + "".join(bindings(decision))


def session_rule(decision: StudioDecision, repo_root: str) -> str:
    """Remember this exact operation and workspace, never widen its scope."""
    return "approval:v2:" + json.dumps(
        {"workspace": str(Path(repo_root).resolve()), "operation": fingerprint(decision)},
        ensure_ascii=False,
        sort_keys=True,
    )


def session_grant_matches(session: StudioSession, decision: StudioDecision) -> bool:
    # Old directory/argv-only grants have no reliable workspace/scope binding.
    # Keep saved records, but require explicit reapproval under the v2 key.
    return session_rule(decision, session.repo_root) in session.action_grants


def requires_approval(session: StudioSession, decision: StudioDecision) -> bool:
    if decision.action.value == "run_tests" and session.verification_mode == "strict":
        decision = decision.model_copy(update={"command": session.test_command})
    # Host execution crosses a separate boundary. FULL, historical argv grants,
    # and saved session rules are not substitutes for approval of this call.
    if uses_host_execution(decision):
        if fingerprint(decision) in session.once_grants:
            return False
        if decision.action.value != "batch":
            return True
        return any(requires_approval(session, child) for child in decision.actions)
    if (
        decision.action.value == "mcp_call"
        and decision.mcp_tool == "list_servers"
        and session.permission_mode != PermissionMode.ASK
    ):
        return False
    if decision.action.value == "batch":
        if fingerprint(decision) in session.once_grants or session_grant_matches(session, decision):
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
        and sandbox_settings().mode == "off"
    ):
        return False
    if session.permission_mode == PermissionMode.FULL:
        return False
    if session.permission_mode == PermissionMode.ASK:
        return True
    if decision.action.value in IMPORTANT:
        return True
    # A development-tool prefix cannot prove what project code will do.
    return decision.action.value in {"run_command", "run_tests"}
