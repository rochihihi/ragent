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
from threading import Event
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, model_validator

from veripatch.config import Settings
from veripatch.studio_agent import StudioAgent, _file_tree, context_limit_for_model
from veripatch.studio_domain import (
    PermissionMode,
    ResponseStyle,
    StudioAction,
    StudioDecision,
    StudioMessage,
    StudioSession,
    VerificationMode,
)
from veripatch.studio_git import GitToolError, StudioGit
from veripatch.studio_model import StudioProviderModel
from veripatch.studio_store import StudioStore
from veripatch.studio_tools import (
    TERMINALS as TERMINALS,  # Compatibility export; execution lives in the Agent.
)
from veripatch.studio_tools import (
    SafeStudioCommandRunner as SafeStudioCommandRunner,  # Compatibility export.
)
from veripatch.studio_tools import (
    UnsafeStudioCommand,
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


class SkillPackageFile(BaseModel):
    path: str = Field(min_length=1, max_length=240)
    data: str = Field(max_length=2_800_000)


class SkillImport(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=12000)
    files: list[SkillPackageFile] | None = Field(default=None, max_length=200)
    archive: str | None = Field(default=None, max_length=11_200_000)
    replace: bool = False
    expected_version: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_source(self):
        if sum(value is not None for value in (self.content, self.files, self.archive)) != 1:
            raise ValueError("请选择一种技能导入来源")
        if self.files and sum(len(f.data) for f in self.files) > 11_200_000:
            raise ValueError("技能包超过 8 MiB")
        return self


class SkillEdit(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    expected_version: str = Field(min_length=64, max_length=64)


class SkillSelection(BaseModel):
    names: list[str] | None = Field(default=None, max_length=10)
    modes: dict[str, Literal["auto", "pinned", "disabled"]] | None = None


class _SkillBodyLimitedRoute(APIRoute):
    """Bound streamed package JSON before FastAPI decodes Base64 or validates models."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request: Request):
            if "/skills" in request.url.path and request.method in {"POST", "PUT", "PATCH"}:
                chunks, length = [], 0
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > 16 * 1024 * 1024:
                        return JSONResponse({"detail": "技能请求超过 16 MiB"}, status_code=413)
                    chunks.append(chunk)
                request._body = b"".join(chunks)
            return await handler(request)

        return bounded


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
    "task_contract": ("任务契约", "保留原始请求与文件范围，不用关键词提前决定任务类型"),
    "failure_strategy": ("失败恢复", "区分超时、缺少工具、认证及上游服务异常"),
    "dynamic_budget": ("执行控制", "默认持续执行，通过暂停和继续保存并恢复任务进度"),
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
                "target_file",
                "protected_path",
            }
            assert required <= keys, f"缺少契约项：{sorted(required - keys)}"
            assert contract.intent == "unresolved"
            assert contract.intent_source == "model_pending"
            assert not contract.requested_actions
            assert "workspace_change" not in keys
            assert "preserve_behavior" not in keys
            assert "feature" not in keys
            assert "增加复制按钮" in contract.objective
            evidence.append("原始需求交给执行模型理解；不再以本地分类生成修改和验证要求")
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
            agent.max_steps = None
            simple = agent._dynamic_budget("解释这个项目", 8)
            complex_budget = agent._dynamic_budget(
                "1. 重构跨文件状态恢复\n2. 修复并发问题\n3. 运行完整测试", 180
            )
            assert simple == complex_budget == agent.max_steps
            evidence.append("交互会话默认无固定步数上限，由用户暂停并保存进度")
        elif check_id == "verification_gate":
            from veripatch.studio_execution import classify_command
            command = ["python", "app.py"]
            execution = classify_command(
                command, root=Path.cwd(), changed_files=["app.py"], launch_required=True,
            )
            assert execution.command == command
            assert execution.role == "command"
            evidence.append("普通命令不猜测为验证，也不按任务目标改写启动形式")
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


def create_studio_router(settings: Settings, *, ui_token: str | None = None) -> APIRouter:
    from veripatch import studio_skills

    router = APIRouter(route_class=_SkillBodyLimitedRoute)
    store = StudioStore(settings.database_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    steer_queues: dict[str, deque[str]] = {}
    pause_signals: dict[str, Event] = {}

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
            session.summary_message_end = sum(
                not (m.role == "user" and m.content.startswith(internal_prefixes))
                for m in session.messages[:session.summary_message_end]
            )
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
            listing = studio_skills.listing(session.repo_root)
            return {
                **listing,
                "enabled": session.enabled_skills,
                "modes": {
                    item["name"]: studio_skills.mode_for(session, item["name"])
                    for item in listing["items"]
                },
                "active": list(session.active_skill_contents),
                "locked": session.status in {"running", "waiting_permission"},
            }
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    def skill_package(request: SkillImport):
        return studio_skills.decode_package(
            content=request.content, archive=request.archive,
            files=[f.model_dump() for f in request.files] if request.files is not None else None,
        )

    @router.post("/studio-api/sessions/{session_id}/skills/preview")
    def preview_skill(session_id: str, request: SkillImport):
        session = load_session(session_id)
        try:
            return studio_skills.preview(session.repo_root, skill_package(request))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/studio-api/sessions/{session_id}/skills/{name}")
    def skill_detail(session_id: str, name: str):
        session = load_session(session_id)
        try:
            return studio_skills.detail(session.repo_root, name)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/studio-api/sessions/{session_id}/skills/{name}/resource")
    def skill_resource(session_id: str, name: str, path: str = Query(max_length=240)):
        session = load_session(session_id)
        try:
            return studio_skills.resource(session.repo_root, name, path)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/studio-api/sessions/{session_id}/skills", status_code=201)
    def import_skill(session_id: str, request: SkillImport):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            result = studio_skills.install_package(
                session.repo_root, skill_package(request), replace=request.replace,
                expected_version=request.expected_version,
            )
            session.skill_modes.setdefault(result["name"], "auto")
            session.active_skill_contents.pop(result["name"], None)
            store.save(session, "skill_imported", {"name": result["name"],
                       "backup": result["backup"], "summary": f"已导入技能 {result['name']}。"})
            return result
        except FileExistsError as exc:
            raise HTTPException(409, "同名技能已存在，不会覆盖") from exc
        except studio_skills.SkillConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/studio-api/sessions/{session_id}/skills/{name}/repair")
    def repair_skill_permissions(
        session_id: str, name: str, version: str = Query(min_length=64, max_length=64)
    ):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            result = studio_skills.repair_permissions(session.repo_root, name, version)
            session.active_skill_contents.pop(name, None)
            store.save(session, "skill_permissions_repaired", {
                "name": name, "backup": result["backup"],
                "summary": f"技能 {name} 已按项目权限重新保存，原包已备份。",
            })
            return result
        except studio_skills.SkillConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.patch("/studio-api/sessions/{session_id}/skills/{name}")
    def edit_skill(session_id: str, name: str, request: SkillEdit):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            result = studio_skills.edit(
                session.repo_root, name, request.content, request.expected_version,
            )
            session.active_skill_contents.pop(name, None)
            store.save(session, "skill_updated", {"name": name, "backup": result["backup"],
                       "summary": f"已更新技能 {name}，原版本已备份。"})
            return result
        except studio_skills.SkillConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.delete("/studio-api/sessions/{session_id}/skills/{name}")
    def delete_skill(session_id: str, name: str, version: str = Query(min_length=64, max_length=64)):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            result = studio_skills.remove(session.repo_root, name, version)
            session.skill_modes.pop(name, None)
            session.enabled_skills = [n for n in session.enabled_skills if n != name]
            session.active_skill_contents.pop(name, None)
            store.save(session, "skill_deleted", {**result, "summary": f"已移除技能 {name}，文件已备份。"})
            return result
        except studio_skills.SkillConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.put("/studio-api/sessions/{session_id}/skills")
    def choose_skills(session_id: str, request: SkillSelection):
        session = load_session(session_id)
        if session.status in {"running", "waiting_permission"}:
            raise HTTPException(409, "请等待任务结束后管理技能")
        try:
            if request.modes is None and request.names is None:
                raise ValueError("需要提供模式或启用项")
            if request.modes is not None:
                available = {i["name"] for i in studio_skills.listing(session.repo_root)["items"]}
                if len(request.modes) > 200 or not set(request.modes).issubset(available):
                    raise ValueError("技能模式包含不存在的技能或超过 200 项")
                modes = {n: request.modes.get(n, "disabled") for n in available}
                names = [n for n, mode in modes.items() if mode == "pinned"]
            else:
                names = list(dict.fromkeys(request.names or []))
                modes = {n: "pinned" for n in names}
            if len(names) > 10:
                raise ValueError("最多固定启用 10 项技能")
            studio_skills.selected(session.repo_root, names)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session.enabled_skills = names
        session.skill_modes = modes
        session.active_skill_contents = {}
        store.save(session, "skills_updated", {"names": session.enabled_skills})
        return {"enabled": session.enabled_skills, "modes": session.skill_modes}

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
        session_id: str, after: int = Query(default=0, ge=0), full: bool = False
    ) -> list[dict[str, Any]]:
        load_session(session_id)
        return store.events(session_id, after, full=full)

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
            if (resume_after_permission or continuation) and session.pending_model_call:
                model.restore_tool_continuation(
                    session.pending_model_call,
                    session_id=session.session_id,
                    turn_observation_start=session.turn_observation_start,
                )
            queue = steer_queues.setdefault(session.session_id, deque())
            signal = pause_signals.setdefault(session.session_id, Event())
            agent = StudioAgent(
                model,
                store,
                max_steps=None,
                max_context_tokens=context_limit_for_model(session.provider, session.model),
                consume_steer=lambda: queue.popleft() if queue else None,
                pause_requested=signal.is_set,
            )
            await asyncio.to_thread(
                lambda: asyncio.run(agent.handle(
                    session, content, continuation=continuation,
                    record_user_message=record_user_message,
                    resume_after_permission=resume_after_permission,
                ))
            )
        except Exception as exc:
            session.status = "failed"
            session.failure_reason = f"RAgent 后台任务失败：{type(exc).__name__}: {exc}"
            store.save(session, "failed", {"reason": session.failure_reason})
        finally:
            tasks.pop(session.session_id, None)
            pause_signals.pop(session.session_id, None)
            if not steer_queues.get(session.session_id):
                steer_queues.pop(session.session_id, None)

    @router.post("/studio-api/sessions/{session_id}/permissions/{request_id}")
    async def decide_permission(
        session_id: str, request_id: str, request: PermissionDecision, http_request: Request,
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
                from veripatch.studio_permissions import (
                    fingerprint,
                    session_rule,
                    uses_host_execution,
                )

                decision = StudioDecision.model_validate(pending.decision)
                if uses_host_execution(decision):
                    from veripatch.sandbox_management import require_desktop_approval

                    require_desktop_approval(
                        ui_token, http_request.headers.get("x-veripatch-ui"),
                        http_request.headers.get("x-ragent-sandbox-key"),
                    )
                if pending.approval_digest is not None:
                    import hashlib

                    try:
                        digest = hashlib.sha256(fingerprint(decision).encode()).hexdigest()
                    except (ValueError, OSError, RuntimeError):
                        raise HTTPException(
                            status_code=409, detail="执行配置已失效，请重新生成审批"
                        ) from None
                    if pending.approval_digest != digest:
                        raise HTTPException(
                            status_code=409,
                            detail="执行策略或 MCP 配置已变化，请拒绝或调整后重新审批",
                        )
                if request.scope == "session" and uses_host_execution(decision):
                    raise HTTPException(
                        status_code=400, detail="沙箱外执行仅允许本次批准，不可保存会话授权"
                    )
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
                # Session approval extends duration, not operation scope.
                grants.append(
                    session_rule(decision, session.repo_root)
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
                # Legacy saved requests also resume through the Agent executor.
                # The HTTP endpoint grants permission; it never runs the command.
                from veripatch.studio_permissions import fingerprint, session_rule
                operation = (
                    StudioAction.START_TERMINAL
                    if pending.operation == StudioAction.START_TERMINAL.value
                    else StudioAction.RUN_COMMAND
                )
                decision = StudioDecision(
                    call_id=pending.request_id, action=operation,
                    rationale=pending.reason, command=pending.command,
                )
                grant = (session_rule(decision, session.repo_root)
                         if request.scope == "session" else fingerprint(decision))
                grants = session.action_grants if request.scope == "session" else session.once_grants
                grants.append(grant)
                session.resume_decision = decision
                if pending.follow_up_command:
                    follow_up = StudioDecision(
                        call_id=f"{pending.request_id}:follow_up",
                        action=StudioAction.RUN_COMMAND,
                        rationale=pending.reason, command=pending.follow_up_command,
                    )
                    session.remaining_actions = [follow_up]
                    session.once_grants.append(fingerprint(follow_up))
                session.pending_permission = None
                session.status = "running"
                session.activity = "preparing_context"
                store.save(session, "permission_approved",
                           {"request_id": request_id, "scope": request.scope})
                tasks[session_id] = asyncio.create_task(
                    execute(session, _latest_task_message(session), continuation=True,
                            record_user_message=False, resume_after_permission=True),
                    name=f"studio-{session_id}-permission",
                )
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

    @router.post("/studio-api/sessions/{session_id}/pause", status_code=202)
    async def pause_session(session_id: str) -> dict[str, str]:
        session = load_session(session_id)
        if session_id not in tasks:
            if session.status == "paused":
                return {"status": "paused"}
            raise HTTPException(status_code=409, detail="会话当前没有运行中的任务")
        pause_signals.setdefault(session_id, Event()).set()
        store.append_event(session_id, "pause_requested", {
            "summary": "已请求暂停；当前工具完成并保存结果后停止。",
        })
        return {"status": "pausing"}

    @router.post("/studio-api/sessions/{session_id}/resume", status_code=202)
    async def resume_session(session_id: str) -> dict[str, str]:
        session = load_session(session_id)
        if session_id in tasks or session.status != "paused":
            raise HTTPException(status_code=409, detail="仅可继续已暂停的任务")
        content = _latest_task_message(session)
        session.status = "running"
        store.save(session, "resume_requested", {"summary": "从已保存的任务和证据继续。"})
        tasks[session_id] = asyncio.create_task(execute(
            session, content, continuation=True, record_user_message=False,
        ), name=f"studio-{session_id}")
        await asyncio.sleep(0)
        return {"status": "accepted"}

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
        store.save(session.model_copy(update={
            "status": "running", "activity": "preparing_context",
        }), "run_requested", {"summary": "任务已提交后台执行。"})
        task = asyncio.create_task(execute(session, request.content), name=f"studio-{session_id}")
        tasks[session_id] = task
        # Let the worker persist the user message and running state before the
        # client performs its immediate post-submit refresh.
        await asyncio.sleep(0)
        return {"session_id": session_id, "status": "accepted"}

    return router
