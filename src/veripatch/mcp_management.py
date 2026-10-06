"""Local UI management of user-owned MCP configuration."""

import asyncio
import json
import os
import tempfile
from threading import Lock
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from veripatch.mcp_client import (
    _exchange, configuration_path, configured_servers, validate_servers,
)

_LOCK = Lock()


class ServerUpdate(BaseModel):
    config: dict


class TestConnection(BaseModel):
    confirmed: bool = False
    repo_root: str = Field(min_length=1, max_length=4096)


def _save(servers):
    validate_servers(servers, include_disabled=True)
    path = configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing["mcpServers"] = servers
    descriptor, temp = tempfile.mkstemp(dir=path.parent, prefix=".mcp-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(existing, stream, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def create_mcp_management_router():
    router = APIRouter(prefix="/mcp-servers")

    def check_origin(request):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "不允许跨站管理 MCP 服务")

    @router.get("")
    def list_servers():
        try:
            return {"path": str(configuration_path()), "servers": [
                {"name": name, "config": {key: value for key, value in config.items()
                    if key in {"transport", "command", "args", "cwd", "url", "headers_env", "enabled"}},
                 "has_env": bool(config.get("env"))}
                for name, config in configured_servers(include_disabled=True).items()
            ]}
        except (ValueError, OSError):
            raise HTTPException(400, "MCP 配置无法读取，请检查配置文件") from None

    @router.put("/{name}")
    def update_server(name: str, body: ServerUpdate, request: Request):
        check_origin(request)
        try:
            with _LOCK:
                servers = configured_servers(include_disabled=True)
                config = dict(body.config)
                if "env" not in config and name in servers and "env" in servers[name]:
                    config["env"] = servers[name]["env"]
                servers[name] = config
                _save(servers)
            return {"status": "saved"}
        except (ValueError, OSError):
            raise HTTPException(400, "配置无效或保存失败，请检查命令、参数、地址及环境变量映射") from None

    @router.delete("/{name}")
    def delete_server(name: str, request: Request):
        check_origin(request)
        try:
            with _LOCK:
                servers = configured_servers(include_disabled=True)
                if name not in servers:
                    raise HTTPException(404, "找不到服务")
                del servers[name]
                _save(servers)
            return {"status": "deleted"}
        except (ValueError, OSError):
            raise HTTPException(400, "删除配置失败") from None

    @router.post("/{name}/test")
    async def test_server(name: str, body: TestConnection, request: Request):
        check_origin(request)
        if not body.confirmed:
            raise HTTPException(403, "测试将启动程序或连接网络，需要明确确认")
        from pathlib import Path
        root = Path(body.repo_root).resolve()
        if not root.is_dir():
            raise HTTPException(400, "测试工作目录不存在")
        try:
            if name not in configured_servers():
                raise HTTPException(409, "服务不存在或已禁用")
            tools = await _exchange(root, f"{name}::list_tools", {})
            return {"ok": True, "tools": tools}
        except HTTPException:
            raise
        except (Exception, asyncio.CancelledError) as exc:
            # External errors may echo credentials; expose category only.
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(400, f"连接失败（{type(exc).__name__}），请检查程序、地址或认证配置") from None

    return router
