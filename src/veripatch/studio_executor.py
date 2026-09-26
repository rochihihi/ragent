"""Concrete action execution boundary for Studio sessions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from veripatch.domain import FileEdit
from veripatch.studio_domain import StudioAction, StudioDecision, StudioObservation, StudioSession
from veripatch.mcp_client import call_project_tool
from veripatch.studio_tools import SafeStudioCommandRunner
from veripatch.studio_git import StudioGit
from veripatch.studio_execution import changed_launch_target
from veripatch.domain import TestOutcome
from veripatch.studio_tools import TERMINALS, inspect_visible_processes
from veripatch import studio_completion as completion
from veripatch.workspace import SafeWorkspace


class StudioActionExecutor:
    """Execute workspace read/mutation actions without owning agent policy."""

    def __init__(
        self,
        *,
        file_tree: Callable[[Path], list[str]],
        record_changed_paths: Callable[[StudioSession, list[str]], None],
        record_command_effects: Callable[[StudioSession, SafeWorkspace, dict[str, str]], list[str]],
        run_verification: Callable[..., TestOutcome],
    ) -> None:
        self.file_tree = file_tree
        self.record_changed_paths = record_changed_paths
        self.record_command_effects = record_command_effects
        self.run_verification = run_verification

    def read(self, workspace: SafeWorkspace, decision: StudioDecision) -> StudioObservation | None:
        if decision.action is StudioAction.LIST_FILES:
            files = self.file_tree(workspace.root)
            return StudioObservation(
                kind="files", summary=f"发现 {len(files)} 个文件。",
                payload={"files": files, "truncated": len(files) >= 240},
            )
        if decision.action is StudioAction.SEARCH:
            assert decision.query is not None
            results = workspace.search(decision.query)
            return StudioObservation(
                kind="search", summary=f"找到 {len(results)} 个匹配项。",
                payload={"query": decision.query, "results": results},
            )
        if decision.action is StudioAction.READ:
            assert decision.path is not None
            return StudioObservation(
                kind="read", summary=f"已读取 {decision.path}。",
                payload=workspace.read(decision.path),
            )
        if decision.action is StudioAction.MCP_CALL:
            assert decision.mcp_tool is not None
            result = call_project_tool(
                workspace.root, decision.mcp_tool, decision.mcp_arguments
            )
            return StudioObservation(
                kind="mcp_tool", summary=f"MCP 工具 {decision.mcp_tool} 已调用。",
                payload={
                    "tool": decision.mcp_tool,
                    "arguments": decision.mcp_arguments,
                    "result": result,
                },
            )
        return None

    def mutate(
        self,
        session: StudioSession,
        workspace: SafeWorkspace,
        decision: StudioDecision,
    ) -> StudioObservation | None:
        action = decision.action
        if action is StudioAction.EDIT:
            assert decision.path is not None
            assert decision.old_text is not None and decision.new_text is not None
            path = workspace.resolve(decision.path)
            before = path.read_text(encoding="utf-8")
            changed = workspace.apply_edits(
                [FileEdit(path=decision.path, old_text=decision.old_text, new_text=decision.new_text)],
                protect_tests=False,
            )
            self.record_changed_paths(session, changed)
            return StudioObservation(
                kind="edit", summary=f"已修改 {decision.path}。",
                payload={
                    "path": decision.path, "intent": decision.rationale,
                    "before": before, "after": path.read_text(encoding="utf-8"),
                    "diff": workspace.diff(),
                },
            )
        if action is StudioAction.APPLY_PATCH:
            assert decision.patch is not None
            before = workspace.diff()
            changed = workspace.apply_patch(decision.patch, protect_tests=False)
            self.record_changed_paths(session, changed)
            return StudioObservation(
                kind="patch", summary=f"已应用补丁，修改 {len(changed)} 个文件。",
                payload={
                    "paths": changed, "intent": decision.rationale,
                    "patch": decision.patch, "previous_diff": before,
                    "diff": workspace.diff(),
                },
            )
        if action is StudioAction.CREATE:
            assert decision.path is not None and decision.content is not None
            created = workspace.create_file(decision.path, decision.content)
            self.record_changed_paths(session, [created])
            return StudioObservation(
                kind="create", summary=f"已创建 {created}。",
                payload={
                    "path": created, "intent": decision.rationale, "before": "",
                    "after": workspace.resolve(created).read_text(encoding="utf-8"),
                    "diff": workspace.diff(),
                },
            )
        if action in {StudioAction.MOVE_FILE, StudioAction.COPY_FILE}:
            assert decision.path is not None and decision.destination is not None
            payload: dict[str, Any] = (
                workspace.move_file(decision.path, decision.destination)
                if action is StudioAction.MOVE_FILE
                else workspace.copy_file(decision.path, decision.destination)
            )
            touched = [payload["destination"]]
            if action is StudioAction.MOVE_FILE:
                touched.insert(0, payload["source"])
            self.record_changed_paths(session, touched)
            payload["diff"] = workspace.diff()
            return StudioObservation(
                kind="move" if action is StudioAction.MOVE_FILE else "copy",
                summary=(
                    f"已移动 {payload['source']} 到 {payload['destination']}。"
                    if action is StudioAction.MOVE_FILE
                    else f"已复制 {payload['source']} 到 {payload['destination']}。"
                ), payload=payload,
            )
        if action is StudioAction.DELETE_PATH:
            assert decision.path is not None
            payload = workspace.delete_path(decision.path)
            self.record_changed_paths(session, [payload["path"]])
            payload["diff"] = workspace.diff()
            return StudioObservation(
                kind="delete", summary=f"已删除 {payload['path']}。", payload=payload
            )
        return None

    def command(
        self,
        session: StudioSession,
        workspace: SafeWorkspace,
        decision: StudioDecision,
        execution: Any,
        command_grants: list[list[str]],
        action_approved: bool,
        approved_capabilities: list[str],
    ) -> StudioObservation | None:
        """Execute verification and command actions outside the orchestrator."""
        if decision.action is StudioAction.RUN_TESTS:
            command = (
                session.test_command
                if session.verification_mode.value == "strict"
                else decision.command
            )
            outcome = self.run_verification(
                workspace.root,
                command,
                approved_commands=(
                    [command]
                    if action_approved or session.permission_mode.value == "full"
                    else None
                ),
            )
            session.verification_passed = outcome.passed
            return StudioObservation(
                kind="test",
                summary="测试通过。" if outcome.passed else "测试未通过。",
                payload=outcome.model_dump(mode="json"),
            )
        if decision.action is not StudioAction.RUN_COMMAND:
            return None
        if execution is None:
            raise ValueError("命令尚未完成语义分类")
        before = completion.snapshot(workspace.root, artifacts=False)
        runner = SafeStudioCommandRunner(
            workspace.root,
            approved_commands=command_grants,
            approved_capabilities=approved_capabilities,
        )
        outcome = (
            runner.launch(decision.command)
            if execution.role == "launch" else runner.run(decision.command)
        )
        changed = self.record_command_effects(session, workspace, before)
        if (
            session.verification_mode.value == "auto"
            and session.turn_changed_files
            and execution.role == "verification"
        ):
            session.verification_passed = outcome.passed
        return StudioObservation(
            kind="command",
            summary=(
                "进程已启动，尚未确认窗口；可轮询现有进程。"
                if outcome.launch_state == "running_unconfirmed"
                else "系统已接受打开文件的请求。"
                if outcome.launch_state == "dispatched"
                else "命令执行成功。" if outcome.exit_code == 0
                else "命令执行未通过。"
            ),
            payload={
                **outcome.model_dump(mode="json"),
                "execution_role": execution.role,
                "launch_target": execution.target if execution.role == "launch" else None,
                "changed_files": changed,
            },
        )

    def git(
        self,
        session: StudioSession,
        workspace: SafeWorkspace,
        decision: StudioDecision,
    ) -> StudioObservation | None:
        """Execute Git actions outside the orchestrator."""
        action = decision.action
        if action not in {
            StudioAction.GIT_STATUS, StudioAction.GIT_DIFF, StudioAction.GIT_LOG,
            StudioAction.GIT_BRANCH, StudioAction.GIT_COMMIT, StudioAction.GIT_RESTORE,
        }:
            return None
        git = StudioGit(workspace.root)
        if action is StudioAction.GIT_STATUS:
            payload = git.status()
            return StudioObservation(
                kind="git_status",
                summary=f"Git 状态：{len(payload['changes'])} 项工作区变化。",
                payload=payload,
            )
        if action is StudioAction.GIT_DIFF:
            payload = git.diff(decision.path, decision.revision)
            return StudioObservation(
                kind="git_diff",
                summary=f"已读取 Git Diff，共 {len(payload['stdout'].splitlines())} 行。",
                payload=payload,
            )
        if action is StudioAction.GIT_LOG:
            payload = git.log()
            return StudioObservation(
                kind="git_log",
                summary=f"已读取最近 {len(payload['commits'])} 条 Git 提交。",
                payload=payload,
            )
        if action is StudioAction.GIT_BRANCH:
            payload = git.branches(decision.branch)
            return StudioObservation(
                kind="git_branch",
                summary=(
                    f"已创建并切换到分支 {payload['created']}。"
                    if payload["created"]
                    else f"当前分支 {payload['current']}，共 {len(payload['branches'])} 个本地分支。"
                ),
                payload=payload,
            )
        if action is StudioAction.GIT_COMMIT:
            assert decision.commit_message is not None
            payload = git.commit(decision.commit_message, session.turn_changed_files)
            return StudioObservation(
                kind="git_commit",
                summary=f"已创建 Git 提交 {payload['commit'][:12]}。",
                payload=payload,
            )
        assert decision.path is not None
        normalized = workspace.resolve(decision.path).relative_to(workspace.root).as_posix()
        if normalized not in session.turn_changed_files:
            raise ValueError("Git restore only permits files changed by RAgent in this turn")
        payload = git.restore(normalized, decision.revision or "HEAD")
        self.record_changed_paths(session, [normalized])
        return StudioObservation(
            kind="git_restore",
            summary=f"已从 {payload['revision']} 恢复 {normalized}。",
            payload=payload,
        )

    def process(
        self,
        session: StudioSession,
        workspace: SafeWorkspace,
        decision: StudioDecision,
        command_grants: list[list[str]],
    ) -> StudioObservation | None:
        """Execute terminal lifecycle and live-process inspection actions."""
        action = decision.action
        if action is StudioAction.START_TERMINAL:
            launch_target = changed_launch_target(
                workspace.root, session.turn_changed_files, decision.command
            )
            expect_window = bool(
                launch_target
                and session.task_contract
                and any(
                    item.key == "launch_after_change"
                    for item in session.task_contract.requirements
                )
            )
            payload = TERMINALS.start(
                workspace.root,
                decision.command,
                approved_commands=command_grants,
                approved_capabilities=session.approved_capabilities,
                expect_window=expect_window,
            )
            payload["command"] = decision.command
            payload["launch_target"] = launch_target
            return StudioObservation(
                kind="terminal",
                summary=f"已启动持续终端 {payload['terminal_id']}。",
                payload=payload,
            )
        if action is StudioAction.POLL_TERMINAL:
            assert decision.terminal_id is not None
            payload = TERMINALS.poll(workspace.root, decision.terminal_id)
            launch_target = next(
                (
                    item.payload.get("launch_target")
                    for item in reversed(session.observations)
                    if item.payload.get("terminal_id") == decision.terminal_id
                    and item.payload.get("launch_target")
                ),
                None,
            )
            if launch_target:
                payload["launch_target"] = launch_target
            return StudioObservation(
                kind="terminal",
                summary="终端仍在运行。" if payload["running"] else "终端已结束。",
                payload=payload,
            )
        if action is StudioAction.WRITE_TERMINAL:
            assert decision.terminal_id is not None and decision.input is not None
            payload = TERMINALS.write(workspace.root, decision.terminal_id, decision.input)
            return StudioObservation(
                kind="terminal",
                summary=f"已向终端 {decision.terminal_id} 发送输入。",
                payload=payload,
            )
        if action is StudioAction.STOP_TERMINAL:
            assert decision.terminal_id is not None
            payload = TERMINALS.stop(workspace.root, decision.terminal_id)
            return StudioObservation(
                kind="terminal",
                summary=f"已停止终端 {decision.terminal_id}。",
                payload=payload,
            )
        if action is StudioAction.INSPECT_PROCESSES:
            query = decision.query or ""
            windows = inspect_visible_processes(query)
            return StudioObservation(
                kind="processes",
                summary=f"实时检查到 {len(windows)} 个匹配的可见窗口。",
                payload={"query": query, "windows": windows, "live": True},
            )
        return None
