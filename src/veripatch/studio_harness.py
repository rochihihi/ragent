"""Runtime tool registry and routing for the Studio harness.

The model chooses the tool surface. This module catalogs tools and normalizes
execution evidence; it does not infer tool preferences from user text or rewrite calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from veripatch.studio_domain import (
    StudioAction, StudioDecision, StudioObservation, StudioSession,
    StudioToolResult, ToolResultStatus,
)


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    surface: str
    read_only: bool = False


class ToolRegistry:
    """Single catalog of logical tools and their execution surfaces."""

    _tools = (
        ToolDescriptor("list_files", "workspace", True),
        ToolDescriptor("read", "workspace", True),
        ToolDescriptor("search", "workspace", True),
        ToolDescriptor("mcp_call", "mcp", False),
        ToolDescriptor("edit", "workspace"),
        ToolDescriptor("apply_patch", "workspace"),
        ToolDescriptor("create", "workspace"),
        ToolDescriptor("move_file", "workspace"),
        ToolDescriptor("copy_file", "workspace"),
        ToolDescriptor("delete_path", "workspace"),
        ToolDescriptor("run_command", "workspace"),
        ToolDescriptor("run_tests", "workspace"),
        ToolDescriptor("start_terminal", "process"),
        ToolDescriptor("poll_terminal", "process", True),
        ToolDescriptor("write_terminal", "process"),
        ToolDescriptor("stop_terminal", "process"),
        ToolDescriptor("inspect_processes", "process", True),
        ToolDescriptor("git_status", "git", True),
        ToolDescriptor("git_diff", "git", True),
        ToolDescriptor("git_commit", "git"),
        ToolDescriptor("git_restore", "git"),
        ToolDescriptor("request_permission", "control"),
        ToolDescriptor("respond", "control"),
        ToolDescriptor("finish", "control"),
        ToolDescriptor("fail", "control"),
    )

    @classmethod
    def get(cls, name: str) -> ToolDescriptor | None:
        return next((tool for tool in cls._tools if tool.name == name), None)

    @classmethod
    def for_action(cls, action: StudioAction) -> ToolDescriptor:
        """Return one logical descriptor for every action, including control actions."""
        return cls.get(action.value) or ToolDescriptor(action.value, "workspace")


class ToolOutcome:
    """One status protocol for workspace, command, process and MCP observations."""

    @staticmethod
    def normalize(action: StudioAction, observation: StudioObservation) -> StudioToolResult:
        if observation.tool_result is not None:
            return observation.tool_result
        payload = observation.payload
        raw_evidence = payload.get("evidence")
        evidence = (
            [str(item) for item in raw_evidence[:40] if item]
            if isinstance(raw_evidence, list) else []
        ) or [observation.summary]
        changed = payload.get("changed_files")
        changed_files = [str(item) for item in changed] if isinstance(changed, list) else []
        if observation.kind == "capability_guard":
            return StudioToolResult(
                status=ToolResultStatus.BLOCKED, evidence=evidence,
                failure_category="policy", retryable=False,
                next_strategy=str(payload.get("required_next_step") or "选择已授权动作"),
                changed_files=changed_files,
            )
        if observation.kind == "tool_error" or (
            observation.kind in {"test", "command", "static_web_check"}
            and (
                (payload.get("exit_code", 1) is not None
                 and str(payload.get("exit_code", 1)) != "0")
                or payload.get("timed_out") is True
            )
        ):
            return StudioToolResult(
                status=ToolResultStatus.FAILED, evidence=evidence,
                failure_category=str(payload.get("failure_category") or "tool_failure"),
                retryable=bool(payload.get("retryable", True)),
                next_strategy=str(payload.get("next_strategy") or "检查失败证据并选择不同诊断动作。"),
                changed_files=changed_files,
            )
        if (
            observation.kind in {"test", "command", "static_web_check"}
            and "exit_code" in payload
            and payload["exit_code"] is None
        ):
            # No exit status yet is neither failure nor completed success.
            return StudioToolResult(
                status=ToolResultStatus.PENDING, evidence=evidence,
                changed_files=changed_files,
            )
        status = (
            ToolResultStatus.NOT_APPLICABLE
            if action in {StudioAction.RESPOND, StudioAction.FINISH}
            else ToolResultStatus.SUCCEEDED
        )
        return StudioToolResult(status=status, evidence=evidence, changed_files=changed_files)

    @staticmethod
    def succeeded(observation: StudioObservation) -> bool:
        if observation.tool_result is not None:
            return observation.tool_result.status is ToolResultStatus.SUCCEEDED
        # Saved sessions from before the result protocol still have payloads.
        if observation.kind in {"tool_error", "capability_guard"}:
            return False
        if observation.kind in {"test", "command", "static_web_check"}:
            return (
                str(observation.payload.get("exit_code", 1)) == "0"
                and not observation.payload.get("timed_out")
            )
        return True


class ActionLedger:
    """Match proposed work to real, current-turn outcomes across tool surfaces."""

    # These inspect live processes. Identical arguments do not imply identical
    # results, unlike rereading an unchanged workspace file.
    _refreshable = frozenset({StudioAction.POLL_TERMINAL, StudioAction.INSPECT_PROCESSES})

    _observations = {
        StudioAction.LIST_FILES: "files",
        StudioAction.READ: "read",
        StudioAction.SEARCH: "search",
        StudioAction.MCP_CALL: "mcp_tool",
        StudioAction.RUN_COMMAND: "command",
        StudioAction.RUN_TESTS: "test",
        StudioAction.START_TERMINAL: "terminal",
    }

    @classmethod
    def refreshable(cls, action: StudioAction) -> bool:
        return action in cls._refreshable

    @staticmethod
    def current_outcomes(session: StudioSession) -> list[dict[str, Any]]:
        """Give the planner a compact, current-turn record even after raw history is trimmed."""
        relevant = set(ActionLedger._observations.values()) | {
            "edit", "patch", "create", "move", "copy", "delete", "tool_error",
            "duplicate_action", "git_status", "git_diff", "processes",
        }
        outcomes: list[dict[str, Any]] = []
        for index in range(session.turn_observation_start, len(session.observations)):
            item = session.observations[index]
            if item.kind not in relevant:
                continue
            payload = item.payload
            status = item.tool_result.status.value if item.tool_result else "observed"
            if item.kind in {"command", "test"} and payload.get("exit_code") not in (None, 0):
                status = "failed"
            outcomes.append({
                "observation_id": index,
                "kind": item.kind,
                "status": status,
                "target": payload.get("path") or payload.get("tool") or payload.get("query")
                or payload.get("command") or payload.get("terminal_id"),
                "changed_files": item.tool_result.changed_files if item.tool_result else [],
            })
        return outcomes[-40:]

    @staticmethod
    def fingerprint(decision: StudioDecision, workspace_epoch: int) -> str:
        payload = decision.model_dump(
            mode="json",
            exclude={"rationale", "message", "memory_update"},
            exclude_defaults=True,
            exclude_none=True,
        )
        payload["workspace_epoch"] = workspace_epoch
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def successful_observation(
        cls, session: StudioSession, decision: StudioDecision
    ) -> tuple[int, StudioObservation] | None:
        """Only reuse evidence produced in this turn, after the last workspace change."""
        kind = cls._observations.get(decision.action)
        if kind is None:
            return None
        for index in range(len(session.observations) - 1, session.turn_observation_start - 1, -1):
            item = session.observations[index]
            if item.kind in {"edit", "patch", "create", "move", "copy", "delete", "git_restore"}:
                return None
            if item.kind != kind or not ToolOutcome.succeeded(item):
                continue
            payload: dict[str, Any] = item.payload
            if decision.action is StudioAction.READ and payload.get("path") != decision.path:
                continue
            if decision.action is StudioAction.SEARCH and payload.get("query") != decision.query:
                continue
            if decision.action is StudioAction.MCP_CALL and (
                payload.get("tool") != decision.mcp_tool
                or payload.get("arguments") != decision.mcp_arguments
            ):
                continue
            if decision.action is StudioAction.RUN_COMMAND and (
                payload.get("command") != decision.command or payload.get("exit_code") != 0
            ):
                continue
            if decision.action is StudioAction.START_TERMINAL and (
                payload.get("command") != decision.command or not payload.get("terminal_id")
            ):
                continue
            if decision.action is StudioAction.RUN_TESTS and not payload.get("passed"):
                continue
            return index, item
        return None


class TaskEvidence:
    """Check structured tool requirements against current-turn execution evidence."""

    _kinds = {
        "mcp_call": "mcp_tool",
        "list_files": "files",
        "run_tests": "test",
        "run_command": "command",
    }

    @classmethod
    def evidence(cls, session: StudioSession, tool: str) -> int | None:
        kind = cls._kinds.get(tool)
        if kind is None:
            return None
        start = session.turn_observation_start
        for index in range(start, len(session.observations)):
            if session.observations[index].kind in {
                "edit", "patch", "create", "move", "copy", "delete", "git_restore",
            }:
                start = index + 1
        for index in range(len(session.observations) - 1, start - 1, -1):
            item = session.observations[index]
            if item.kind != kind or not ToolOutcome.succeeded(item):
                continue
            if tool == "mcp_call" and item.payload.get("tool") == "list_tools":
                continue
            if tool in {"run_tests", "run_command"} and item.payload.get("exit_code") != 0:
                continue
            return index
        return None

    @classmethod
    def pending_tools(cls, session: StudioSession) -> list[str]:
        return [
            item.expected
            for item in session.task_contract.requirements
            if item.key == "tool_use" and item.expected and cls.evidence(session, item.expected) is None
        ] if session.task_contract else []

    @classmethod
    def failed_attempt(cls, session: StudioSession, tool: str) -> bool:
        kind = cls._kinds.get(tool)
        for item in session.observations[session.turn_observation_start :]:
            action = item.payload.get("action")
            action_name = getattr(action, "value", action)
            if item.kind == "tool_error" and action_name == tool:
                return True
            if item.kind == kind and tool in {"run_tests", "run_command"} and item.payload.get("exit_code") not in (None, 0):
                return True
        return False
