"""Local background-run API and graphical trajectory viewer."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Coroutine
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, SecretStr

from veripatch.agent import VeriPatchAgent
from veripatch.config import Settings
from veripatch.credentials import credential_status, delete_api_key, load_api_key, save_api_key
from veripatch.domain import AgentRunState, IssueSpec, RunnerKind, RunPhase
from veripatch.models.base import AgentModel
from veripatch.providers import create_model
from veripatch.quota import get_provider_quota, refresh_pixel_tokens
from veripatch.store import SQLiteRunStore
from veripatch.studio_api import create_studio_router
from veripatch.studio_model import parse_studio_decision
from veripatch.testing import InProcessDemoPytestRunner


class CreateRunRequest(BaseModel):
    repo_root: str
    issue: IssueSpec
    test_command: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    provider: str = "openai"
    model: str | None = None
    reasoning_effort: str | None = None
    runner: RunnerKind = RunnerKind.DOCKER


class RunAccepted(BaseModel):
    run_id: str
    phase: RunPhase


class ProviderCredentialRequest(BaseModel):
    api_key: SecretStr = Field(min_length=1, max_length=10_000)


class ProviderModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=100)


class OpenAIConnectionRequest(BaseModel):
    base_url: str = Field(min_length=8, max_length=2_000)
    model: str = Field(min_length=1, max_length=100)
    quota_access_token: SecretStr | None = Field(default=None, max_length=10_000)
    quota_refresh_token: SecretStr | None = Field(default=None, max_length=10_000)
    quota_user_id: str | None = Field(default=None, max_length=100)


class OpenAIModelDiscoveryRequest(BaseModel):
    base_url: str = Field(min_length=8, max_length=2_000)
    api_key: SecretStr | None = Field(default=None, max_length=10_000)


class ProviderConnectionTestRequest(BaseModel):
    model: str = Field(min_length=1, max_length=100)
    base_url: str | None = Field(default=None, min_length=8, max_length=2_000)
    api_key: SecretStr | None = Field(default=None, max_length=10_000)


async def _test_provider_connection(
    *, provider: str, base_url: str, model: str, api_key: str
) -> dict[str, Any]:
    """Make one tiny, non-streaming request without exposing credentials."""
    started = time.perf_counter()
    official = provider == "openai_official"
    endpoint = base_url.rstrip("/") + ("/responses" if official else "/chat/completions")
    payload = (
        {"model": model, "input": "Reply with OK.", "max_output_tokens": 256}
        if official
        else {
            "model": model,
            "messages": [{
                "role": "user",
                "content": (
                    "Return only this JSON object, without Markdown: "
                    '{"action":"respond","rationale":"connection test",'
                    '"message":"OK","command":[]}'
                ),
            }],
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": 96,
        }
    )
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(20, connect=8, write=10, pool=8),
            follow_redirects=False,
        ) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "RAgent/connection-test",
                    "X-Title": "RAgent",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "status_code": None,
            "message": "连接超时；请检查网络、代理或中转站状态。",
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "status_code": None,
            "message": f"网络连接失败：{type(exc).__name__}。",
        }
    latency_ms = round((time.perf_counter() - started) * 1000)
    messages = {
        400: "请求格式或模型参数不被中转站支持。",
        401: "API Key 无效或已过期。",
        402: "账户余额不足。",
        403: "密钥没有该模型的访问权限。",
        404: "接口路径或模型不存在。",
        429: "请求过于频繁或供应商容量已满。",
    }
    protocol_ok = False
    if response.is_success:
        try:
            result = response.json()
            if official:
                protocol_ok = any(
                    item.get("type") == "output_text" and bool(item.get("text", "").strip())
                    for output in result.get("output", [])
                    for item in output.get("content", [])
                )
            else:
                content = result["choices"][0]["message"]["content"]
                decision = parse_studio_decision(content)
                protocol_ok = decision.action.value == "respond" and decision.message == "OK"
        except (KeyError, IndexError, TypeError, ValueError):
            protocol_ok = False
    ok = response.is_success and protocol_ok
    return {
        "ok": ok,
        "transport_ok": response.is_success,
        "protocol_ok": protocol_ok,
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "status_code": response.status_code,
        "message": (
            "连接成功，且模型能够遵循 RAgent 动作协议。"
            if ok
            else (
                "基础连接成功，但模型没有返回有效的 Agent JSON 动作。"
                if response.is_success
                else messages.get(response.status_code, f"服务返回 HTTP {response.status_code}。")
            )
        ),
    }


class CreateDirectoryRequest(BaseModel):
    parent: str = Field(min_length=1, max_length=2_000)
    name: str = Field(min_length=1, max_length=255)


async def _discover_openai_models(base_url: str, api_key: str) -> list[str]:
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("Models response does not contain a data array")
    models = sorted(
        {
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        },
        key=str.casefold,
    )
    if not models:
        raise ValueError("Models response did not contain model IDs")
    return models[:500]


def _model(provider: str, settings: Settings) -> AgentModel:
    return create_model(provider, settings)


def _public_state(state: AgentRunState) -> dict[str, Any]:
    return state.model_dump(
        mode="json",
        exclude={
            "original_files",
            "working_file_hashes",
            "action_fingerprints",
            "pending_edit",
            "request_ids",
        },
    )


_PRIVATE_EVENT_KEYS = {
    "api_key",
    "original_files",
    "working_file_hashes",
    "action_fingerprints",
    "pending_edit",
    "request_id",
    "request_ids",
    "transaction_id",
}
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|ds)-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
)


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", redacted
        )
    return redacted


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_value(item)
            for key, item in value.items()
            if key.casefold() not in _PRIVATE_EVENT_KEYS
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _public_value(event))


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    store = SQLiteRunStore(resolved_settings.database_path)
    app = FastAPI(
        title="RAgent API",
        version="3.0.0",
        description="Verifiable repository-level software repair agent.",
    )
    app.include_router(create_studio_router(resolved_settings))

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {key: error[key] for key in ("type", "loc", "msg") if key in error}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    app.state.store = store
    app.state.tasks = {}
    model_config_path = resolved_settings.database_path.resolve().parent / "provider_models.json"
    connection_config_path = (
        resolved_settings.database_path.resolve().parent / "provider_connections.json"
    )
    model_choices = {
        "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
        "openai_official": ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"],
    }
    model_preferences = {
        "deepseek": resolved_settings.deepseek_model,
        "openai": resolved_settings.model,
        "openai_official": "gpt-6-astra",
    }
    openai_base_url = resolved_settings.openai_base_url
    openai_quota_user_id: str | None = None
    try:
        saved_models = json.loads(model_config_path.read_text(encoding="utf-8"))
        for provider, choices in model_choices.items():
            saved_model = saved_models.get(provider)
            if isinstance(saved_model, dict):
                choices.extend(model for model in saved_model.get("choices", []) if isinstance(model, str) and model not in choices)
                saved_model = saved_model.get("selected")
            if isinstance(saved_model, str) and (provider in {"openai", "openai_official"} or saved_model in choices):
                model_preferences[provider] = saved_model
                if saved_model not in choices:
                    choices.append(saved_model)
    except (OSError, ValueError, TypeError):
        pass

    def persist_models() -> None:
        model_config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({provider: {"selected": selected, "choices": model_choices[provider]}
                        for provider, selected in model_preferences.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(model_config_path)
    try:
        saved_connections = json.loads(connection_config_path.read_text(encoding="utf-8"))
        candidate_url = saved_connections.get("openai", {}).get("base_url")
        if isinstance(candidate_url, str) and candidate_url.startswith(("http://", "https://")):
            openai_base_url = candidate_url.rstrip("/")
        candidate_user_id = saved_connections.get("openai", {}).get("quota_user_id")
        if isinstance(candidate_user_id, str) and candidate_user_id.strip():
            openai_quota_user_id = candidate_user_id.strip()
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    def provider_settings(
        provider: str,
        requested_model: str | None = None,
        requested_effort: str | None = None,
    ) -> Settings:
        selected_model = requested_model or model_preferences.get(provider)
        if requested_model is not None and requested_model not in model_choices.get(provider, []):
            raise ValueError("Unsupported model for provider")
        allowed_efforts = {
            "openai": {"low", "medium", "high", "xhigh", "max"},
            "openai_official": {"low", "medium", "high", "xhigh", "max"},
            "deepseek": {"low", "high", "max"},
            "scripted-demo": {"none"},
        }
        effort = requested_effort or (
            "none" if provider == "scripted-demo" else resolved_settings.reasoning_effort
        )
        if effort not in allowed_efforts.get(provider, set()):
            raise ValueError("Unsupported reasoning effort for provider")
        if provider == "deepseek":
            return replace(
                resolved_settings,
                deepseek_model=selected_model or "deepseek-v4-flash",
                reasoning_effort=effort,
            )
        if provider in {"openai", "openai_official"}:
            return replace(
                resolved_settings,
                model=selected_model or resolved_settings.model,
                openai_base_url=(
                    "https://api.openai.com/v1"
                    if provider == "openai_official" else openai_base_url
                ),
                reasoning_effort=effort,
            )
        return replace(resolved_settings, reasoning_effort=effort)

    def schedule(run_id: str, coroutine: Coroutine[Any, Any, None], *, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        app.state.tasks[run_id] = task

        def remove_completed(completed_task: asyncio.Task[None]) -> None:
            if app.state.tasks.get(run_id) is completed_task:
                app.state.tasks.pop(run_id, None)

        task.add_done_callback(remove_completed)

    async def execute_new(
        run_id: str,
        request: CreateRunRequest,
        model: AgentModel,
    ) -> None:
        runner_factory = None
        if request.provider == "scripted-demo":

            def demo_runner_factory(root: Path, _kind: RunnerKind) -> InProcessDemoPytestRunner:
                return InProcessDemoPytestRunner(root)

            runner_factory = demo_runner_factory
        agent = VeriPatchAgent(
            model,
            settings=resolved_settings,
            store=store,
            runner_factory=runner_factory,
        )
        try:
            await agent.run(
                repo_root=Path(request.repo_root),
                issue=request.issue,
                test_command=request.test_command,
                run_id=run_id,
                runner_kind=request.runner,
                provider=request.provider,
            )
        except Exception as exc:
            state = store.load(run_id)
            if state is not None and not state.terminal:
                state.phase = RunPhase.FAILED
                state.failure_reason = f"Background run failed: {type(exc).__name__}: {exc}"
                store.append_event(run_id, "failed", {"reason": state.failure_reason})
                store.checkpoint(state)

    async def execute_resume(run_id: str, model: AgentModel) -> None:
        agent = VeriPatchAgent(model, settings=resolved_settings, store=store)
        try:
            await agent.resume(run_id)
        except Exception as exc:
            state = store.load(run_id)
            if state is not None and not state.terminal:
                state.phase = RunPhase.FAILED
                state.failure_reason = f"Background resume failed: {type(exc).__name__}: {exc}"
                store.append_event(run_id, "failed", {"reason": state.failure_reason})
                store.checkpoint(state)

    def queue(request: CreateRunRequest, *, run_id: str | None = None) -> RunAccepted:
        repository = Path(request.repo_root).resolve()
        if not repository.is_dir():
            raise HTTPException(status_code=400, detail="Repository does not exist")
        try:
            run_settings = provider_settings(
                request.provider, request.model, request.reasoning_effort
            )
            model = _model(request.provider, run_settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        accepted_run_id = run_id or uuid4().hex
        queued = AgentRunState(
            run_id=accepted_run_id,
            repo_root=str(repository),
            issue=request.issue,
            test_command=request.test_command,
            provider=request.provider,
            model=(
                request.model
                or model_preferences.get(request.provider)
                or ("scripted-demo" if request.provider == "scripted-demo" else None)
            ),
            reasoning_effort=run_settings.reasoning_effort,
            runner=request.runner,
            phase=RunPhase.QUEUED,
        )
        store.checkpoint(queued)
        store.append_event(
            accepted_run_id,
            "queued",
            {
                "provider": request.provider,
                "model": queued.model,
                "reasoning_effort": queued.reasoning_effort,
            },
        )
        schedule(
            accepted_run_id,
            execute_new(accepted_run_id, request, model),
            name=f"veripatch-{accepted_run_id}",
        )
        return RunAccepted(run_id=accepted_run_id, phase=RunPhase.QUEUED)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/studio")

    @app.get("/providers")
    async def providers() -> dict[str, dict[str, str | bool]]:
        statuses = credential_status()
        for provider in ("deepseek", "openai", "openai_official"):
            statuses[provider]["model"] = model_preferences[provider]
        statuses["openai"]["base_url"] = openai_base_url
        statuses["openai_official"]["base_url"] = "https://api.openai.com/v1"
        statuses["openai"]["quota_configured"] = bool(load_api_key("openai_quota"))
        statuses["openai"]["quota_refresh_configured"] = bool(load_api_key("openai_quota_refresh"))
        statuses["openai"]["quota_user_id"] = openai_quota_user_id or ""
        statuses["scripted-demo"] = {"configured": True, "source": "built-in"}
        return statuses

    @app.get("/provider-models")
    async def provider_models() -> dict[str, dict[str, Any]]:
        return {
            provider: {"selected": model_preferences[provider], "choices": choices}
            for provider, choices in model_choices.items()
        }

    @app.put("/providers/{provider}/model")
    async def configure_provider_model(
        provider: str, request: ProviderModelRequest
    ) -> dict[str, Any]:
        if provider not in model_choices:
            raise HTTPException(status_code=404, detail="Unknown model provider")
        if provider not in {"openai", "openai_official"} and request.model not in model_choices[provider]:
            raise HTTPException(status_code=400, detail="Unsupported model for provider")
        if provider in {"openai", "openai_official"} and request.model not in model_choices[provider]:
            model_choices[provider].append(request.model)
        model_preferences[provider] = request.model
        persist_models()
        return {"provider": provider, "model": request.model}

    @app.put("/providers/openai/connection")
    async def configure_openai_connection(
        request: OpenAIConnectionRequest,
    ) -> dict[str, str]:
        nonlocal openai_base_url, openai_quota_user_id
        base_url = request.base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, detail="Base URL must start with http:// or https://"
            )
        openai_base_url = base_url
        if request.quota_user_id is not None:
            openai_quota_user_id = request.quota_user_id.strip() or None
        if request.quota_access_token and request.quota_access_token.get_secret_value().strip():
            save_api_key("openai_quota", request.quota_access_token.get_secret_value())
        if request.quota_refresh_token and request.quota_refresh_token.get_secret_value().strip():
            save_api_key("openai_quota_refresh", request.quota_refresh_token.get_secret_value())
        model_preferences["openai"] = request.model.strip()
        if model_preferences["openai"] not in model_choices["openai"]:
            model_choices["openai"].append(model_preferences["openai"])
        connection_config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = connection_config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "openai": {
                        "base_url": openai_base_url,
                        "quota_user_id": openai_quota_user_id,
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(connection_config_path)
        persist_models()
        return {
            "provider": "openai",
            "base_url": openai_base_url,
            "model": model_preferences["openai"],
        }

    @app.post("/providers/{provider}/models/discover")
    async def discover_openai_models(
        provider: str,
        request: OpenAIModelDiscoveryRequest,
    ) -> dict[str, list[str]]:
        if provider not in {"openai", "openai_official"}:
            raise HTTPException(status_code=404, detail="Unknown model provider")
        base_url = (
            "https://api.openai.com/v1"
            if provider == "openai_official" else request.base_url.strip().rstrip("/")
        )
        if not base_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, detail="Base URL must start with http:// or https://"
            )
        supplied_key = request.api_key.get_secret_value().strip() if request.api_key else ""
        api_key = supplied_key or load_api_key(provider)
        if not api_key:
            raise HTTPException(status_code=400, detail="请先输入或保存 OpenAI API Key")
        try:
            models = await _discover_openai_models(base_url, api_key)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"模型列表请求失败（HTTP {exc.response.status_code}）",
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"无法读取模型列表：{exc}") from exc
        model_choices[provider] = list(dict.fromkeys([*model_choices[provider], *models]))
        persist_models()
        return {"models": models}

    @app.post("/providers/{provider}/connection-test")
    async def test_provider_connection(
        provider: str, request: ProviderConnectionTestRequest
    ) -> dict[str, Any]:
        if provider not in {"openai", "openai_official", "deepseek"}:
            raise HTTPException(status_code=404, detail="Unknown model provider")
        supplied_key = request.api_key.get_secret_value().strip() if request.api_key else ""
        api_key = supplied_key or load_api_key(provider)
        if not api_key:
            raise HTTPException(status_code=400, detail="请先输入或保存 API Key")
        base_url = (
            request.base_url.strip().rstrip("/")
            if provider == "openai" and request.base_url
            else (
                "https://api.openai.com/v1" if provider == "openai_official" else openai_base_url
                if provider == "openai"
                else resolved_settings.deepseek_base_url.rstrip("/")
            )
        )
        if not base_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Base URL 必须以 http:// 或 https:// 开头")
        return await _test_provider_connection(
            provider=provider,
            base_url=base_url,
            model=request.model.strip(),
            api_key=api_key,
        )

    @app.get("/system/directories")
    async def list_directories(
        path: str | None = Query(default=None),
        x_veripatch_ui: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if x_veripatch_ui != "1":
            raise HTTPException(status_code=403, detail="Local UI request header is required")
        if path is None:
            if os.name == "nt":
                roots = [
                    f"{letter}:\\"
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    if Path(f"{letter}:\\").is_dir()
                ]
            else:
                roots = ["/"]
            directories: list[dict[str, str]] = []
            desktop = Path.home() / "Desktop"
            if desktop.is_dir():
                directories.append({"name": "桌面", "path": str(desktop.resolve())})
            directories.extend({"name": root, "path": root} for root in roots)
            return {
                "current": None,
                "parent": None,
                "directories": directories,
            }
        current = Path(path).expanduser().resolve()
        if not current.is_dir():
            raise HTTPException(status_code=404, detail="Folder does not exist")
        try:
            directories = sorted(
                (
                    {"name": child.name, "path": str(child)}
                    for child in current.iterdir()
                    if child.is_dir()
                ),
                key=lambda item: item["name"].casefold(),
            )[:500]
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="没有权限读取这个文件夹") from exc
        parent = None if current.parent == current else str(current.parent)
        return {"current": str(current), "parent": parent, "directories": directories}

    @app.post("/system/directories", status_code=201)
    async def create_directory(
        request: CreateDirectoryRequest,
        x_veripatch_ui: str | None = Header(default=None),
    ) -> dict[str, str]:
        if x_veripatch_ui != "1":
            raise HTTPException(status_code=403, detail="Local UI request header is required")
        parent = Path(request.parent).expanduser().resolve()
        if not parent.is_dir():
            raise HTTPException(status_code=404, detail="父文件夹不存在")
        name = request.name.strip()
        if (
            not name
            or name in {".", ".."}
            or any(separator in name for separator in ("/", "\\"))
            or any(character in name for character in '<>:"|?*')
        ):
            raise HTTPException(status_code=400, detail="文件夹名称无效")
        target = (parent / name).resolve()
        if target.parent != parent:
            raise HTTPException(status_code=400, detail="文件夹路径超出当前目录")
        try:
            target.mkdir()
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail="同名文件夹已经存在") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="没有权限在这里新建文件夹") from exc
        return {"name": target.name, "path": str(target)}

    @app.post("/providers/{provider}/credentials")
    async def configure_provider(
        provider: str, request: ProviderCredentialRequest
    ) -> dict[str, Any]:
        if provider not in {"openai", "openai_official", "deepseek"}:
            raise HTTPException(status_code=404, detail="Unknown credential provider")
        try:
            save_api_key(provider, request.api_key.get_secret_value())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Credential store unavailable") from exc
        status = credential_status()[provider]
        if not status["configured"]:
            raise HTTPException(status_code=503, detail="Credential could not be persisted")
        return {"provider": provider, **status}

    @app.get("/providers/{provider}/credentials/reveal")
    async def reveal_provider_credential(
        provider: str,
        x_veripatch_ui: str | None = Header(default=None),
    ) -> JSONResponse:
        if x_veripatch_ui != "1":
            raise HTTPException(status_code=403, detail="Local UI request header is required")
        if provider not in {"openai", "openai_official", "deepseek"}:
            raise HTTPException(status_code=404, detail="Unknown credential provider")
        api_key = load_api_key(provider)
        if not api_key:
            raise HTTPException(status_code=404, detail="Credential is not configured")
        return JSONResponse(
            {"provider": provider, "api_key": api_key},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.delete("/providers/{provider}/credentials")
    async def remove_provider(provider: str) -> dict[str, Any]:
        if provider not in {"openai", "openai_official", "deepseek"}:
            raise HTTPException(status_code=404, detail="Unknown credential provider")
        current = credential_status()[provider]
        if current["source"] == "environment":
            raise HTTPException(
                status_code=409,
                detail="Credential is managed by an environment variable and cannot be removed here",
            )
        removed = delete_api_key(provider)
        status = credential_status()[provider]
        return {"provider": provider, "removed": removed, **status}

    @app.get("/providers/{provider}/quota")
    async def provider_quota(provider: str) -> dict[str, Any]:
        try:
            api_key = load_api_key(provider)
            quota_access_token = load_api_key("openai_quota") if provider == "openai" else None
            quota_refresh_token = (
                load_api_key("openai_quota_refresh") if provider == "openai" else None
            )
            snapshot = await asyncio.to_thread(
                get_provider_quota,
                provider,
                replace(resolved_settings, openai_base_url=openai_base_url),
                api_key=api_key,
                quota_access_token=quota_access_token,
                quota_user_id=openai_quota_user_id,
            )
            if (
                provider == "openai"
                and quota_refresh_token
                and snapshot.error
                and "鉴权失败" in snapshot.error
            ):
                pair = await asyncio.to_thread(
                    refresh_pixel_tokens, openai_base_url, quota_refresh_token
                )
                save_api_key("openai_quota", pair.access_token)
                save_api_key("openai_quota_refresh", pair.refresh_token)
                snapshot = await asyncio.to_thread(
                    get_provider_quota,
                    provider,
                    replace(resolved_settings, openai_base_url=openai_base_url),
                    api_key=api_key,
                    quota_access_token=pair.access_token,
                    quota_user_id=openai_quota_user_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return snapshot.model_dump(mode="json")

    @app.get("/runs")
    def list_runs(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return [_public_state(state) for state in store.list_runs(limit=limit, offset=offset)]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        state = store.load(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _public_state(state)

    @app.get("/runs/{run_id}/events")
    def get_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        if store.load(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [
            _public_event(event)
            for event in store.events(run_id, after_sequence=after, limit=limit)
        ]

    @app.get("/runs/{run_id}/result")
    def get_result(run_id: str) -> dict[str, Any]:
        state = store.load(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if not state.terminal:
            raise HTTPException(status_code=409, detail="Run is not complete")
        return {"state": _public_state(state), "diff": state.final_diff}

    @app.post("/runs", response_model=RunAccepted, status_code=202)
    async def create_run(request: CreateRunRequest) -> RunAccepted:
        return queue(request)

    @app.post("/demo", response_model=RunAccepted, status_code=202)
    async def create_demo() -> RunAccepted:
        source = _project_root() / "examples" / "discount_bug"
        if not source.is_dir():
            raise HTTPException(status_code=500, detail="Demo repository is missing")
        run_id = uuid4().hex
        workspace = resolved_settings.database_path.resolve().parent / "demo_workspaces" / run_id
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, workspace)
        request = CreateRunRequest(
            repo_root=str(workspace),
            issue=IssueSpec(
                issue_id="discount-percentage",
                title="折扣百分比被当成了整数",
                description=(
                    "calculate_discount(100, 10) 应返回 90。请复现现有失败测试，"
                    "定位百分比换算错误并完成可验证修复。"
                ),
            ),
            test_command=["python", "-m", "pytest", "-q"],
            provider="scripted-demo",
            runner=RunnerKind.LOCAL,
        )
        return queue(request, run_id=run_id)

    @app.post("/runs/{run_id}/resume", response_model=RunAccepted, status_code=202)
    async def resume_run(run_id: str) -> RunAccepted:
        state = store.load(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if state.terminal:
            raise HTTPException(status_code=409, detail="Terminal run cannot be resumed")
        if run_id in app.state.tasks:
            raise HTTPException(status_code=409, detail="Run is already active")
        try:
            model = _model(
                state.provider,
                provider_settings(state.provider, state.model, state.reasoning_effort),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        schedule(
            run_id,
            execute_resume(run_id, model),
            name=f"veripatch-resume-{run_id}",
        )
        return RunAccepted(run_id=run_id, phase=state.phase)

    return app


app = create_app()
