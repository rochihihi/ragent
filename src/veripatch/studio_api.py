"""FastAPI routes for the interactive RAgent workspace."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from veripatch import studio_completion
from veripatch.config import Settings
from veripatch.studio_agent import StudioAgent, _file_tree, context_limit_for_model
from veripatch.studio_domain import (
    PermissionMode,
    ResponseStyle,
    StudioAction,
    StudioDecision,
    StudioMessage,
    StudioObservation,
    StudioSession,
    VerificationMode,
)
from veripatch.studio_execution import classify_command
from veripatch.studio_git import GitToolError, StudioGit
from veripatch.studio_model import StudioProviderModel
from veripatch.studio_store import StudioStore
from veripatch.studio_tools import (
    TERMINALS,
    SafeStudioCommandRunner,
    UnsafeStudioCommand,
    command_capability,
    detect_project,
    validate_studio_command,
)
from veripatch.studio_ui import frontend_root, studio_html
from veripatch.workspace import SafeWorkspace


class CreateStudioSession(BaseModel):
    permission_mode: PermissionMode = PermissionMode.IMPORTANT
    repo_root: str
    provider: str
    model: str
    reasoning_effort: str = "high"
    response_style: ResponseStyle | None = None
    verification_mode: VerificationMode = VerificationMode.AUTO
    test_command: list[str] | str = Field(default_factory=list)


class SkillImport(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class SkillSelection(BaseModel):
    names: list[str] = Field(max_length=10)


class StudioUserMessage(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class UpdateStudioSession(BaseModel):
    permission_mode: PermissionMode | None = None
    provider: str
    model: str
    reasoning_effort: str
    response_style: ResponseStyle = ResponseStyle.CONCISE
    verification_mode: VerificationMode
    test_command: list[str] | str = Field(default_factory=list)


class PermissionDecision(BaseModel):
    scope: str = Field(default="once", pattern="^(once|session)$")
    approved: bool
    instruction: str | None = Field(default=None, max_length=2_000)


class StudioGitAction(BaseModel):
    action: str
    path: str | None = None
    branch: str | None = None
    message: str | None = Field(default=None, max_length=500)
    paths: list[str] = Field(default_factory=list)


class CreateProjectEntry(BaseModel):
    kind: str = Field(pattern="^(file|folder)$")
    path: str = Field(min_length=1, max_length=500)


UPGRADE_CHECKS = {
    "task_contract": ("任务契约", "识别目标文件、保护范围、功能要求和兼容性约束"),
    "failure_strategy": ("失败恢复", "区分超时、缺少工具、认证及上游服务异常"),
    "dynamic_budget": ("动态预算", "按任务复杂度分配步骤，并为有效进展保留续跑空间"),
    "verification_gate": ("验证门禁", "只把真实测试、构建或静态检查视为验证证据"),
    "path_permission": ("路径与权限", "兼容相对/绝对路径，并保持工作区边界"),
}


def _run_upgrade_check(check_id: str) -> dict[str, Any]:
    """Exercise one production capability without requiring the source test suite."""
    if check_id not in UPGRADE_CHECKS:
        raise KeyError(check_id)
    title, description = UPGRADE_CHECKS[check_id]
    started = time.perf_counter()
    evidence: list[str] = []
    try:
        if check_id == "task_contract":
            session = StudioSession(
                session_id="upgrade-check",
                repo_root=".",
                provider="openai",
                model="check",
                reasoning_effort="low",
            )
            contract = StudioAgent._build_task_contract(
                session,
                "修改 app.py，增加复制按钮，不要修改 tests/test_app.py，并保留现有功能。",
            )
            keys = {item.key for item in contract.requirements}
            required = {
                "workspace_change",
                "target_file",
                "protected_path",
                "preserve_behavior",
            }
            assert required <= keys, f"缺少契约项：{sorted(required - keys)}"
            assert "feature" not in keys
            assert "增加复制按钮" in contract.objective
            evidence.append("功能需求保留在任务目标中；文件和验证约束独立检查")
        elif check_id == "failure_strategy":
            messages = (
                "request timed out",
                "HTTP 401 invalid api key",
                "HTTP 502 bad gateway",
                "command not found: go executable",
            )
            actual = {StudioAgent._classify_tool_failure("command", item)[0] for item in messages}
            required = {"timeout", "authentication", "upstream_unavailable", "missing_command"}
            assert required <= actual, f"分类不完整：{sorted(required - actual)}"
            evidence.append("超时、认证、上游异常和缺少工具均得到不同恢复策略")
        elif check_id == "dynamic_budget":
            agent = object.__new__(StudioAgent)
            agent.max_steps = 60
            simple = agent._dynamic_budget("解释这个项目", 8)
            complex_budget = agent._dynamic_budget(
                "1. 重构跨文件状态恢复\n2. 修复并发问题\n3. 运行完整测试", 180
            )
            assert 0 < simple < complex_budget <= 60
            evidence.append(f"简单任务 {simple} 步，复杂任务 {complex_budget} 步，上限 60 步")
        elif check_id == "verification_gate":
            accepted = [
                ["python", "-m", "py_compile", "app.py"],
                ["python", "-m", "pytest", "-q"],
                ["go", "test", "./..."],
                ["npm", "run", "build"],
            ]
            rejected = [
                ["python", "-c", "print('not a verification')"],
                ["cmd", "/c", "start", "app.py"],
            ]
            assert all(StudioAgent.is_verification_command(item) for item in accepted)
            assert not any(StudioAgent.is_verification_command(item) for item in rejected)
            evidence.append("语法、测试和构建可作为证据；启动程序不会被误判为验证")
        elif check_id == "path_permission":
            with tempfile.TemporaryDirectory(prefix="ragent-check-") as directory:
                root = Path(directory)
                (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
                workspace = SafeWorkspace(root)
                relative = StudioDecision(
                    action=StudioAction.READ, rationale="检查文件", path="app.py"
                )
                absolute = StudioDecision(
                    action=StudioAction.READ, rationale="检查文件", path=str(root / "app.py")
                )
                StudioAgent._normalize_workspace_decision_path(workspace, relative)
                StudioAgent._normalize_workspace_decision_path(workspace, absolute)
                assert relative.path == absolute.path == "app.py"
                try:
                    workspace.read("../outside.txt")
                except Exception:
                    pass
                else:
                    raise AssertionError("路径逃逸未被阻止")
            evidence.append("相对与绝对路径统一为工作区路径，目录逃逸已阻止")
        return {
            "id": check_id,
            "title": title,
            "description": description,
            "status": "passed",
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "evidence": evidence,
            "error": None,
        }
    except Exception as exc:
        return {
            "id": check_id,
            "title": title,
            "description": description,
            "status": "failed",
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "evidence": evidence,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _valid_provider_model(provider: str, model: str) -> bool:
    """Validate provider model names while allowing proxy-discovered OpenAI aliases."""
    if provider == "deepseek":
        return model in {"deepseek-v4-flash", "deepseek-v4-pro"}
    if provider in {"openai", "openai_official"}:
        return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", model))
    return False


def _go_executable() -> str | None:
    discovered = shutil.which("go")
    if discovered:
        return discovered
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    installed = program_files / "Go" / "bin" / "go.exe"
    return str(installed) if installed.is_file() else None


def _is_process_inspection(command: list[str]) -> bool:
    lowered = [part.casefold() for part in command]
    return "tasklist" in lowered or any("get-process" in part for part in lowered)


def _latest_task_message(session: StudioSession) -> str:
    """Return the real user task, excluding permission-dialog instructions."""
    prefixes = ("调整执行方案：", "关于当前待批准操作，请调整方案：")
    return next(
        (
            message.content
            for message in reversed(session.messages)
            if message.role == "user" and not message.content.startswith(prefixes)
        ),
        "继续当前任务",
    )


def _is_pure_bulk_delete_request(message: str) -> bool:
    """Identify an explicit delete-everything task with no requested follow-up work."""
    normalized = re.sub(r"\s+", "", message).casefold()
    bulk_delete = bool(
        re.search(r"(?:删除|清空|移除).*(?:全部|所有|所有东西|全部东西|全部文件)", normalized)
        or re.search(r"(?:delete|remove|clear).*(?:all|everything|entire)", normalized)
    )
    follow_up = bool(
        re.search(
            r"(?:然后|之后|再|接着|并且|创建|新建|安装|启动|打开|修改|写入|then|after|andcreate|install|launch|open)",
            normalized,
        )
    )
    return bulk_delete and not follow_up


def _restore_before_from_unified_diff(current: str, patch: str, path: str) -> str | None:
    """Reverse one file's diff for sessions created before snapshots existed."""
    lines = patch.splitlines(keepends=True)
    marker = f"+++ b/{path}"
    try:
        file_start = next(index for index, line in enumerate(lines) if line.rstrip() == marker)
    except StopIteration:
        return None
    file_end = next(
        (index for index in range(file_start + 1, len(lines)) if lines[index].startswith("--- a/")),
        len(lines),
    )
    hunks = lines[file_start + 1 : file_end]
    current_lines = current.splitlines(keepends=True)
    restored: list[str] = []
    current_index = 0
    saw_hunk = False
    for index, line in enumerate(hunks):
        if not line.startswith("@@"):
            continue
        saw_hunk = True
        try:
            new_range = line.split("@@", 2)[1].strip().split()[1][1:]
            new_start = int(new_range.split(",", 1)[0])
        except (IndexError, ValueError):
            return None
        restored.extend(current_lines[current_index : new_start - 1])
        current_index = new_start - 1
        cursor = index + 1
        while cursor < len(hunks) and not hunks[cursor].startswith("@@"):
            change = hunks[cursor]
            if change.startswith("+"):
                current_index += 1
            elif change.startswith("-"):
                restored.append(change[1:])
            elif change.startswith(" "):
                if current_index >= len(current_lines):
                    return None
                restored.append(current_lines[current_index])
                current_index += 1
            cursor += 1
    if not saw_hunk:
        return None
    restored.extend(current_lines[current_index:])
    return "".join(restored)


def create_studio_router(settings: Settings) -> APIRouter:
    from veripatch import studio_skills

    router = APIRouter()
    store = StudioStore(settings.database_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    steer_queues: dict[str, deque[str]] = {}

    # A process exit destroys in-memory tasks but persisted sessions used to remain
    # `running` forever. Recover those sessions as resumable instead of making the UI
    # poll a task that no longer exists.
    for interrupted in store.list_sessions():
        if interrupted.status == "running":
            interrupted.status = "idle"
            interrupted.activity = "idle"
            interrupted.failure_reason = None
            store.save(
                interrupted,
                "interrupted",
                {"summary": "上次执行因应用关闭或重启而中断，可继续发送消息。"},
            )
        elif interrupted.status == "waiting_permission" and interrupted.pending_permission is None:
            # Older builds could persist the approval before the approved command
            # was started. That left a session waiting for a permission request
            # which no longer existed, so every subsequent click returned 409.
            interrupted.status = "idle"
            interrupted.activity = "idle"
            interrupted.failure_reason = None
            store.save(
                interrupted,
                "permission_recovered",
                {"summary": "上次权限操作未完整结束，已恢复为可继续状态。"},
            )

    def load_session(session_id: str) -> StudioSession:
        session = store.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Studio session not found")
        internal_prefixes = ("用户已批准并已执行命令 ", "用户已批准只读访问 ")
        visible_messages = [
            message
            for message in session.messages
            if not (message.role == "user" and message.content.startswith(internal_prefixes))
        ]
        if len(visible_messages) != len(session.messages):
            session.messages = visible_messages
            store.save(
                session,
                "message_history_repaired",
                {"summary": "已从聊天记录中移除旧版内部权限续跑消息。"},
            )
        return session

    def session_payload(session: StudioSession) -> dict[str, Any]:
        return {
            **session.model_dump(mode="json"),
            "context_limit_tokens": context_limit_for_model(session.provider, session.model),
        }

    @router.get("/studio-api/sessions/{session_id}/skills")
    def list_skills(session_id: str):
        session = load_session(session_id)
        try:
            return {
                "items": studio_skills.discover(session.repo_root),
                "enabled": session.enabled_skills,
            }
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/studio-api/sessions/{session_id}/skills", status_code=201)
    def import_skill(session_id: str, request: SkillImport):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            return studio_skills.install(session.repo_root, request.content)
        except FileExistsError as exc:
            raise HTTPException(409, "同名技能已存在，不会覆盖") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.put("/studio-api/sessions/{session_id}/skills")
    def choose_skills(session_id: str, request: SkillSelection):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            studio_skills.selected(session.repo_root, request.names)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session.enabled_skills = list(dict.fromkeys(request.names))
        store.save(session, "skills_updated", {"names": session.enabled_skills})
        return {"enabled": session.enabled_skills}

    @router.get("/studio", response_class=HTMLResponse)
    async def studio_page() -> str:
        return studio_html()

    @router.get("/assets/{asset_path:path}", response_class=FileResponse)
    async def studio_asset(asset_path: str) -> Path:
        assets = (frontend_root() / "assets").resolve()
        target = (assets / asset_path).resolve()
        if target.parent != assets or not target.is_file():
            raise HTTPException(status_code=404, detail="Frontend asset not found")
        return target

    @router.get("/studio-brand", response_class=FileResponse)
    async def studio_brand() -> Path:
        root = (
            Path(sys.__dict__["_MEIPASS"])
            if getattr(sys, "frozen", False)
            else Path(__file__).parents[2]
        )
        return root / "assets" / "ragent-logo-transparent.png"

    @router.get("/studio-api/upgrade-checks")
    async def list_upgrade_checks() -> dict[str, Any]:
        return {
            "version": "3.0.0",
            "checks": [
                {
                    "id": key,
                    "title": value[0],
                    "description": value[1],
                    "status": "untested",
                }
                for key, value in UPGRADE_CHECKS.items()
            ],
        }

    @router.post("/studio-api/upgrade-checks/{check_id}")
    async def run_upgrade_check(check_id: str) -> dict[str, Any]:
        if check_id == "all":
            results = [await asyncio.to_thread(_run_upgrade_check, key) for key in UPGRADE_CHECKS]
            return {"version": "3.0.0", "results": results}
        if check_id not in UPGRADE_CHECKS:
            raise HTTPException(status_code=404, detail="Unknown upgrade check")
        return await asyncio.to_thread(_run_upgrade_check, check_id)

    @router.post("/studio-api/sessions", status_code=201)
    async def create_session(request: CreateStudioSession) -> dict[str, str]:
        root = Path(request.repo_root).resolve()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail="Repository does not exist")
        efforts = {
            "deepseek": {"low", "high", "max"},
            "openai": {"low", "medium", "high", "xhigh", "max"},
            "openai_official": {"low", "medium", "high", "xhigh", "max"},
        }
        if not _valid_provider_model(request.provider, request.model):
            raise HTTPException(status_code=400, detail="Unsupported provider model")
        if request.reasoning_effort not in efforts[request.provider]:
            raise HTTPException(status_code=400, detail="Unsupported reasoning effort")
        test_command = (
            shlex.split(request.test_command, posix=False)
            if isinstance(request.test_command, str)
            else request.test_command
        )
        if request.verification_mode is VerificationMode.STRICT and not test_command:
            raise HTTPException(status_code=400, detail="Strict verification requires a command")
        if request.verification_mode is VerificationMode.STRICT:
            try:
                validate_studio_command(test_command)
            except UnsafeStudioCommand as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        session = StudioSession(
            session_id=uuid4().hex,
            repo_root=str(root),
            provider=request.provider,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            response_style=(
                request.response_style
                or store.project_response_style(str(root))
                or ResponseStyle.CONCISE
            ),
            verification_mode=request.verification_mode,
            permission_mode=request.permission_mode,
            test_command=test_command,
        )
        store.save(session, "created", {"repo_root": str(root), "provider": request.provider})
        return {"session_id": session.session_id}

    @router.get("/studio-api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return [session_payload(session) for session in store.list_sessions()]

    @router.get("/studio-api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        return session_payload(load_session(session_id))

    @router.patch("/studio-api/sessions/{session_id}/settings")
    async def update_session_settings(
        session_id: str, request: UpdateStudioSession
    ) -> dict[str, Any]:
        session = load_session(session_id)
        if session_id in tasks or session.status == "running":
            raise HTTPException(status_code=409, detail="运行中的会话不能修改配置")
        efforts = {
            "deepseek": {"low", "high", "max"},
            "openai": {"low", "medium", "high", "xhigh", "max"},
            "openai_official": {"low", "medium", "high", "xhigh", "max"},
        }
        if not _valid_provider_model(request.provider, request.model):
            raise HTTPException(status_code=400, detail="Unsupported provider model")
        if request.reasoning_effort not in efforts[request.provider]:
            raise HTTPException(status_code=400, detail="Unsupported reasoning effort")
        command = (
            shlex.split(request.test_command, posix=False)
            if isinstance(request.test_command, str)
            else request.test_command
        )
        switching_to_strict = (
            request.verification_mode is VerificationMode.STRICT
            and session.verification_mode is not VerificationMode.STRICT
        )
        if switching_to_strict and session.changed_files:
            raise HTTPException(
                status_code=409,
                detail="已有代码修改的会话不能切换为严格验证，请新建严格验证对话",
            )
        command_changed = command != session.test_command
        if (
            session.verification_mode is VerificationMode.STRICT
            and session.baseline_completed
            and command_changed
        ):
            raise HTTPException(status_code=409, detail="基线执行后不能更改固定验证命令")
        if request.verification_mode is VerificationMode.STRICT:
            try:
                validate_studio_command(command)
            except UnsafeStudioCommand as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        model_changed = (session.provider, session.model) != (request.provider, request.model)
        session.provider = request.provider
        session.model = request.model
        if model_changed:
            session.context_estimated_tokens = 0
            session.context_actual_input_tokens = None
            session.context_trimmed_items = []
        session.reasoning_effort = request.reasoning_effort
        session.response_style = request.response_style
        store.save_project_response_style(session.repo_root, session.response_style.value)
        session.verification_mode = request.verification_mode
        if (
            request.permission_mode is not None
            and request.permission_mode != session.permission_mode
        ):
            session.permission_mode = request.permission_mode
            session.action_grants.clear()
            session.once_grants.clear()
            session.approved_commands.clear()
            session.approved_capabilities.clear()
        session.test_command = command
        if switching_to_strict:
            session.baseline_completed = False
            session.baseline_reproduced = False
            session.verification_passed = False
        store.save(
            session,
            "settings_updated",
            {
                "provider": session.provider,
                "model": session.model,
                "model_changed": model_changed,
                "reasoning_effort": session.reasoning_effort,
                "response_style": session.response_style,
                "verification_mode": session.verification_mode,
                "test_command": session.test_command,
            },
        )
        return session_payload(session)

    @router.delete("/studio-api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        session = load_session(session_id)
        if session_id in tasks or session.status == "running":
            raise HTTPException(status_code=409, detail="运行中的会话不能删除")
        store.delete(session_id)

    @router.get("/studio-api/sessions/{session_id}/events")
    async def session_events(
        session_id: str, after: int = Query(default=0, ge=0)
    ) -> list[dict[str, Any]]:
        load_session(session_id)
        return store.events(session_id, after)

    @router.get("/studio-api/sessions/{session_id}/files")
    async def session_files(session_id: str) -> dict[str, list[str]]:
        session = load_session(session_id)
        root = Path(session.repo_root)
        ignored = {".git", ".venv", "venv", "node_modules", "runs", "build", "dist"}
        directories = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
            and not any(part in ignored for part in path.relative_to(root).parts)
        ][:240]
        return {"files": _file_tree(root), "directories": directories}

    @router.post("/studio-api/sessions/{session_id}/files", status_code=201)
    async def create_project_entry(
        session_id: str, request: CreateProjectEntry
    ) -> dict[str, str]:
        session = load_session(session_id)
        if session_id in tasks or session.status == "running":
            raise HTTPException(status_code=409, detail="Agent 运行中不能新建项目文件")
        workspace = SafeWorkspace(Path(session.repo_root))
        try:
            target = workspace.resolve(request.path)
            relative = target.relative_to(workspace.root)
            if any(part in {".git", ".github", ".codex"} for part in relative.parts):
                raise ValueError("不能在受保护的项目元数据中创建内容")
            if target.exists():
                raise FileExistsError(f"路径已存在：{relative.as_posix()}")
            if request.kind == "file":
                created = workspace.create_file(relative.as_posix(), "")
                if created not in session.changed_files:
                    session.changed_files.append(created)
            else:
                target.mkdir(parents=True)
                created = relative.as_posix()
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.save(
            session,
            "workspace_entry_created",
            {"kind": request.kind, "path": created, "summary": f"已新建{('文件' if request.kind == 'file' else '文件夹')} {created}"},
        )
        return {"kind": request.kind, "path": created}

    @router.get("/studio-api/sessions/{session_id}/project")
    async def session_project(session_id: str) -> dict[str, object]:
        session = load_session(session_id)
        return detect_project(Path(session.repo_root))

    @router.get("/studio-api/sessions/{session_id}/git")
    async def session_git(session_id: str, path: str | None = None) -> dict[str, Any]:
        session = load_session(session_id)
        root = Path(session.repo_root)
        if not (root / ".git").exists():
            return {"initialized": False, "repo_root": str(root)}
        try:
            git = StudioGit(root)
            commits = git.log(20)["commits"]
            return {
                "initialized": True,
                "status": git.status(),
                "branches": git.branches(),
                "commits": commits,
                "has_commits": bool(commits),
                "diff": git.diff(path)["stdout"] if path else "",
                "commit_eligible": session.changed_files,
            }
        except GitToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/studio-api/sessions/{session_id}/git")
    async def session_git_action(session_id: str, request: StudioGitAction) -> dict[str, Any]:
        session = load_session(session_id)
        if session_id in tasks or session.status == "running":
            raise HTTPException(status_code=409, detail="Agent 运行中不能执行 Git 操作")
        try:
            root = Path(session.repo_root)
            if request.action == "initialize":
                return StudioGit.initialize(root, request.branch or "main")
            git = StudioGit(root)
            if request.action == "create_branch" and request.branch:
                return git.branches(request.branch)
            if request.action == "switch_branch" and request.branch:
                return git.switch(request.branch)
            if request.action == "commit" and request.message:
                eligible = set(session.changed_files)
                paths = [path for path in request.paths if path in eligible]
                if paths != request.paths:
                    raise GitToolError("只能提交 RAgent 在当前会话修改的文件")
                return git.commit(request.message, paths)
            raise HTTPException(status_code=400, detail="不支持的 Git 操作")
        except GitToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/studio-api/sessions/{session_id}/file")
    async def session_file(session_id: str, path: str = Query(min_length=1)) -> dict[str, Any]:
        session = load_session(session_id)
        try:
            return SafeWorkspace(Path(session.repo_root)).read(path, 1, 400)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/studio-api/sessions/{session_id}/file-change")
    async def session_file_change(
        session_id: str, path: str = Query(min_length=1)
    ) -> dict[str, Any]:
        session = load_session(session_id)
        workspace = SafeWorkspace(Path(session.repo_root))
        try:
            current_path = workspace.resolve(path)
            current = current_path.read_text(encoding="utf-8")
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        changes = [
            event["payload"].get("payload", {})
            for event in store.events(session_id)
            if event["event_type"] == "observation"
            and event["payload"].get("kind") in {"edit", "create"}
            and event["payload"].get("payload", {}).get("path") == path
        ]
        latest = changes[-1] if changes else {}
        before = latest.get("before")
        if before is None and latest.get("diff"):
            before = _restore_before_from_unified_diff(current, latest["diff"], path)
        return {
            "path": path,
            "changed": bool(changes),
            "before": before,
            "before_available": before is not None,
            "after": latest.get("after", current),
            "current": current,
            "diff": latest.get("diff", ""),
        }

    async def execute(
        session: StudioSession,
        content: str,
        *,
        continuation: bool = False,
        record_user_message: bool = True,
        resume_after_permission: bool = False,
    ) -> None:
        model_settings = replace(
            settings,
            model=(
                session.model
                if session.provider in {"openai", "openai_official"}
                else settings.model
            ),
            deepseek_model=(
                session.model if session.provider == "deepseek" else settings.deepseek_model
            ),
            reasoning_effort=session.reasoning_effort,
        )
        try:
            model = StudioProviderModel(session.provider, model_settings)
            if resume_after_permission and session.pending_model_call:
                model.restore_tool_continuation(
                    session.pending_model_call,
                    session_id=session.session_id,
                    turn_observation_start=session.turn_observation_start,
                )
            queue = steer_queues.setdefault(session.session_id, deque())
            await StudioAgent(
                model,
                store,
                max_steps=max(settings.max_steps, 60),
                max_context_tokens=context_limit_for_model(session.provider, session.model),
                consume_steer=lambda: queue.popleft() if queue else None,
            ).handle(
                session,
                content,
                continuation=continuation,
                record_user_message=record_user_message,
                resume_after_permission=resume_after_permission,
            )
        except Exception as exc:
            session.status = "failed"
            session.failure_reason = f"RAgent 后台任务失败：{type(exc).__name__}: {exc}"
            store.save(session, "failed", {"reason": session.failure_reason})
        finally:
            tasks.pop(session.session_id, None)
            if not steer_queues.get(session.session_id):
                steer_queues.pop(session.session_id, None)

    @router.post("/studio-api/sessions/{session_id}/permissions/{request_id}")
    async def decide_permission(
        session_id: str, request_id: str, request: PermissionDecision
    ) -> dict[str, str]:
        session = load_session(session_id)
        pending = session.pending_permission
        if pending is None or pending.request_id != request_id:
            raise HTTPException(status_code=409, detail="权限请求已失效")
        if session_id in tasks or session.status == "running":
            raise HTTPException(status_code=409, detail="会话仍在运行")
        instruction = (request.instruction or "").strip()
        if instruction:
            session.remaining_actions.clear()
            content = f"调整执行方案：{instruction}"
            session.pending_permission = None
            session.status = "running"
            session.activity = "preparing_context"
            # This is attached to the permission decision, not a second chat
            # message. Keep it out of the visible conversation.
            store.save(
                session,
                "permission_revised",
                {
                    "summary": "用户要求调整待批准的执行方案。",
                    "instruction": instruction,
                    "replaced_command": pending.command,
                },
            )
            store.save(session, "permission_instruction", {"content": content})
            task = asyncio.create_task(
                execute(
                    session,
                    content,
                    continuation=True,
                    record_user_message=False,
                    resume_after_permission=True,
                ),
                name=f"studio-{session_id}-permission-revision",
            )
            tasks[session_id] = task
            return {"status": "revising"}
        if request.approved:
            if pending.decision is not None:
                from veripatch.studio_permissions import fingerprint, session_rule

                decision = StudioDecision.model_validate(pending.decision)
                action_name = decision.action.value
                if (
                    session.task_contract is not None
                    and action_name not in session.task_contract.allowed_actions
                ):
                    session.task_contract.allowed_actions.append(action_name)
                if (
                    session.task_state is not None
                    and action_name not in session.task_state.allowed_actions
                ):
                    session.task_state.allowed_actions.append(action_name)
                grants = (
                    session.action_grants if request.scope == "session" else session.once_grants
                )
                # Session approval stores a scoped rule (command family or
                # same-directory file action); one-time approval remains exact.
                grants.append(
                    (session_rule(decision) or fingerprint(decision))
                    if request.scope == "session"
                    else fingerprint(decision)
                )
                session.resume_decision = None if pending.operation == "baseline" else decision
                session.pending_permission = None
                session.status = "running"
                session.activity = "preparing_context"
                store.save(
                    session,
                    "permission_approved",
                    {"request_id": request_id, "scope": request.scope},
                )
                tasks[session_id] = asyncio.create_task(
                    execute(
                        session,
                        _latest_task_message(session),
                        continuation=True,
                        record_user_message=False,
                        resume_after_permission=True,
                    ),
                    name=f"studio-{session_id}-permission",
                )
                return {"status": "approved"}
            if pending.access == "execute" and pending.command:
                execution_commands = list(session.approved_commands)
                execution_capabilities = list(session.approved_capabilities)
                if pending.command not in execution_commands:
                    execution_commands.append(pending.command)
                    execution_commands = execution_commands[-40:]
                capability = pending.capability or command_capability(
                    pending.command, Path(session.repo_root)
                )
                if capability and capability not in execution_capabilities:
                    execution_capabilities.append(capability)
                    execution_capabilities = execution_capabilities[-40:]
                if request.scope == "session":
                    session.approved_commands = execution_commands
                    session.approved_capabilities = execution_capabilities
                # Claim and persist the request before awaiting command execution.
                # A double click, frontend retry, or duplicated HTTP request must
                # never execute an approved launch/install command twice.
                session.pending_permission = None
                session.status = "running"
                session.activity = "executing_tool"
                store.save(
                    session,
                    "permission_approved",
                    {
                        "command": pending.command,
                        "access": "execute",
                        "request_id": pending.request_id,
                        "claimed": True,
                    },
                )
                # Execute the exact approved command immediately. Asking the model to
                # repeat it added latency and could leave the UI spinning without any
                # tool event when the provider stalled.
                try:
                    workspace = SafeWorkspace(Path(session.repo_root))
                    before_files = studio_completion.snapshot(workspace.root, artifacts=False)
                    runner = SafeStudioCommandRunner(
                        Path(session.repo_root),
                        approved_commands=execution_commands,
                        approved_capabilities=execution_capabilities,
                    )
                    command_execution = classify_command(
                        pending.command,
                        root=workspace.root,
                        changed_files=session.turn_changed_files,
                        launch_required=bool(
                            session.task_contract
                            and any(
                                item.key == "launch_after_change"
                                for item in session.task_contract.requirements
                            )
                        ),
                    )
                    if (
                        pending.operation != StudioAction.START_TERMINAL.value
                        and command_execution.command != pending.command
                    ):
                        raise UnsafeStudioCommand(
                            "旧审批请求的命令语义已变化；请重新提交任务以获取新的启动审批"
                        )
                    if pending.operation == StudioAction.START_TERMINAL.value:
                        expect_window = bool(
                            command_execution.target
                            and session.task_contract
                            and any(
                                item.key == "launch_after_change"
                                for item in session.task_contract.requirements
                            )
                        )
                        terminal_payload = await asyncio.to_thread(
                            TERMINALS.start,
                            Path(session.repo_root),
                            pending.command,
                            approved_commands=execution_commands,
                            approved_capabilities=execution_capabilities,
                            expect_window=expect_window,
                        )
                        terminal_payload["command"] = pending.command
                        terminal_payload["launch_target"] = (
                            command_execution.target if expect_window else None
                        )
                        outcome = None
                    else:
                        outcome = await asyncio.to_thread(
                            runner.launch if command_execution.role == "launch" else runner.run,
                            pending.command,
                        )
                        StudioAgent._record_command_effects(session, workspace, before_files, store)
                except Exception as exc:
                    session.pending_permission = None
                    summary = f"命令无法启动：{type(exc).__name__}: {exc}"
                    observation = StudioObservation(
                        kind="tool_error",
                        summary=summary,
                        payload={
                            "command": pending.command,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "exit_code": 127,
                        },
                    )
                    session.observations.append(observation)
                    session.status = "running"
                    session.activity = "preparing_context"
                    store.save(
                        session,
                        "permission_execution_failed",
                        {"command": pending.command, "summary": summary, "resuming": True},
                    )
                    store.save(session, "observation", observation.model_dump(mode="json"))
                    task = asyncio.create_task(
                        execute(
                            session,
                            _latest_task_message(session),
                            continuation=True,
                            record_user_message=False,
                            resume_after_permission=True,
                        ),
                        name=f"studio-{session_id}-permission-recovery",
                    )
                    tasks[session_id] = task
                    return {"status": "recovering"}
                if outcome is None:
                    observation = StudioObservation(
                        kind="terminal",
                        summary=f"已启动持续终端 {terminal_payload['terminal_id']}。",
                        payload=terminal_payload,
                    )
                else:
                    observation = StudioObservation(
                        kind="command",
                        summary=(
                            "进程已启动，尚未确认窗口；可轮询现有进程。"
                            if outcome.launch_state == "running_unconfirmed"
                            else "系统已接受打开文件的请求。"
                            if outcome.launch_state == "dispatched"
                            else "已执行允许的命令。" if outcome.passed
                            else "允许的命令执行未通过。"
                        ),
                        payload={
                            **outcome.model_dump(mode="json"),
                            "execution_role": command_execution.role,
                            "launch_target": (
                                command_execution.target
                                if command_execution.role == "launch" else None
                            ),
                        },
                    )
                session.observations.append(observation)
                if outcome is None:
                    store.save(session, "observation", observation.model_dump(mode="json"))
                    session.status = "running"
                    session.activity = "preparing_context"
                    task = asyncio.create_task(
                        execute(
                            session,
                            _latest_task_message(session),
                            continuation=True,
                            record_user_message=False,
                            resume_after_permission=True,
                        ),
                        name=f"studio-{session_id}-terminal",
                    )
                    tasks[session_id] = task
                    return {"status": "approved"}
                if (
                    session.verification_mode is VerificationMode.AUTO
                    and session.turn_changed_files
                    and outcome.passed
                    and command_execution.role == "verification"
                ):
                    session.verification_passed = True
                store.save(session, "observation", observation.model_dump(mode="json"))
                if (
                    pending.destructive
                    and outcome.passed
                    and _is_pure_bulk_delete_request(_latest_task_message(session))
                    and not studio_completion.snapshot(workspace.root)
                ):
                    # A bulk-delete command is intentionally terminal once the exact
                    # command has been approved and succeeded. Sending it back to the
                    # model caused follow-up probes and slightly different delete
                    # commands to open an endless series of approval dialogs.
                    session.status = "completed"
                    session.activity = "completed"
                    session.failure_reason = None
                    session.pause_reason = None
                    message = "已按你的授权完成清空操作；不会继续生成或执行其他删除命令。"
                    session.messages.append(StudioMessage(role="assistant", content=message))
                    store.save(
                        session,
                        "completed",
                        {
                            "summary": message,
                            "command": pending.command,
                            "destructive": True,
                            "single_approval": True,
                        },
                    )
                    return {"status": "approved"}
                install_launch = pending.capability == "install:go"
                if install_launch and pending.follow_up_command:
                    artifact_target = pending.follow_up_command[-1]
                    go_executable = _go_executable() if outcome.passed else None
                    if go_executable is None:
                        session.status = "idle"
                        session.activity = "idle"
                        message = "Go 安装未完成，程序尚未启动。请查看右侧工具结果中的安装错误。"
                        session.messages.append(StudioMessage(role="assistant", content=message))
                        store.save(session, "assistant_message", {"content": message})
                        return {"status": "approved"}
                    launch_command = [go_executable, "run", artifact_target]
                    launch_capability = command_capability(launch_command, Path(session.repo_root))
                    execution_commands.append(launch_command)
                    execution_commands = execution_commands[-40:]
                    if launch_capability:
                        execution_capabilities.append(launch_capability)
                        execution_capabilities = execution_capabilities[-40:]
                    launch_runner = SafeStudioCommandRunner(
                        Path(session.repo_root),
                        approved_commands=execution_commands,
                        approved_capabilities=execution_capabilities,
                    )
                    launch_outcome = await asyncio.to_thread(launch_runner.launch, launch_command)
                    launch_observation = StudioObservation(
                        kind="command",
                        summary=(
                            "Go 已安装，程序已启动。"
                            if launch_outcome.passed
                            else "Go 已安装，但程序启动失败。"
                        ),
                        payload=launch_outcome.model_dump(mode="json"),
                    )
                    session.observations.append(launch_observation)
                    store.save(session, "observation", launch_observation.model_dump(mode="json"))
                    session.status = "idle"
                    session.activity = "idle"
                    message = (
                        f"Go 已安装，并已启动 {artifact_target}。"
                        if launch_outcome.passed
                        else f"Go 已安装，但 {artifact_target} 启动失败。"
                    )
                    session.messages.append(StudioMessage(role="assistant", content=message))
                    store.save(session, "assistant_message", {"content": message})
                    return {"status": "approved"}
                if _is_process_inspection(pending.command):
                    session.status = "idle"
                    session.activity = "idle"
                    output = (outcome.stdout or outcome.stderr or "命令没有返回文本输出").strip()
                    message = "已实时检查当前进程，命令结果如下：\n\n" + output[-12_000:]
                    session.messages.append(StudioMessage(role="assistant", content=message))
                    store.save(session, "assistant_message", {"content": message})
                    return {"status": "approved"}
                session.status = "running"
                session.activity = "preparing_context"
                task = asyncio.create_task(
                    execute(
                        session,
                        _latest_task_message(session),
                        continuation=True,
                        record_user_message=False,
                        resume_after_permission=True,
                    ),
                    name=f"studio-{session_id}-permission",
                )
                tasks[session_id] = task
                return {"status": "approved"}
            target = Path(pending.path).resolve()
            approved = str(target)
            grants = (
                session.approved_write_paths
                if pending.access == "write"
                else session.approved_paths
            )
            if approved not in grants:
                grants.append(approved)
                if pending.access == "write":
                    session.approved_write_paths = grants[-40:]
                else:
                    session.approved_paths = grants[-40:]
            if pending.operation in {
                StudioAction.EDIT.value,
                StudioAction.APPLY_PATCH.value,
                StudioAction.CREATE.value,
                StudioAction.MOVE_FILE.value,
                StudioAction.COPY_FILE.value,
                StudioAction.DELETE_PATH.value,
            }:
                if (
                    session.task_contract is not None
                    and pending.operation not in session.task_contract.allowed_actions
                ):
                    session.task_contract.allowed_actions.append(pending.operation)
                if (
                    session.task_state is not None
                    and pending.operation not in session.task_state.allowed_actions
                ):
                    session.task_state.allowed_actions.append(pending.operation)
            session.pending_permission = None
            session.status = "running"
            session.activity = "preparing_context"
            store.save(
                session,
                "permission_approved",
                {"path": approved, "access": pending.access},
            )
            task = asyncio.create_task(
                execute(
                    session,
                    _latest_task_message(session),
                    continuation=True,
                    record_user_message=False,
                    resume_after_permission=True,
                ),
                name=f"studio-{session_id}-permission",
            )
            tasks[session_id] = task
            return {"status": "approved"}
        session.status = "idle"
        session.activity = "idle"
        session.pending_permission = None
        session.remaining_actions.clear()
        target_label = " ".join(pending.command) if pending.command else pending.path
        message = f"用户拒绝了权限请求：{target_label}。请在现有权限内继续或解释限制。"
        session.messages.append(StudioMessage(role="assistant", content=message))
        store.save(
            session,
            "permission_denied",
            {"path": pending.path, "command": pending.command, "access": pending.access},
        )
        return {"status": "denied"}

    @router.post("/studio-api/sessions/{session_id}/messages", status_code=202)
    async def send_message(session_id: str, request: StudioUserMessage) -> dict[str, str]:
        session = load_session(session_id)
        if session_id in tasks or session.status == "running":
            queue = steer_queues.setdefault(session_id, deque())
            queue.append(request.content)
            store.append_event(
                session_id,
                "steer_queued",
                {
                    "summary": "运行中纠正已排队，将在当前安全边界应用。",
                    "content": request.content,
                    "queue_depth": len(queue),
                },
            )
            return {"session_id": session_id, "status": "queued"}
        if session.status == "waiting_permission":
            raise HTTPException(status_code=409, detail="请先在权限对话框中拒绝或调整方案")
        task = asyncio.create_task(execute(session, request.content), name=f"studio-{session_id}")
        tasks[session_id] = task
        # Let the worker persist the user message and running state before the
        # client performs its immediate post-submit refresh.
        await asyncio.sleep(0)
        return {"session_id": session_id, "status": "accepted"}

    return router
