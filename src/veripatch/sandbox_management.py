"""User-facing sandbox settings; deliberately not a model-callable tool."""

from __future__ import annotations

import asyncio
import hmac
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from veripatch import studio_sandbox


class SandboxUpdate(BaseModel):
    settings: studio_sandbox.SandboxSettings
    confirm_unrestricted: bool = False


class SandboxInstall(BaseModel):
    confirm_system_changes: Literal[True]


def require_desktop_approval(ui_token: str | None, header: str | None, token: str | None) -> None:
    """A native-window capability, never delivered through HTTP or model context."""
    if header != "1" or not ui_token or not token or not hmac.compare_digest(token, ui_token):
        raise HTTPException(403, "请从 RAgent 桌面窗口确认沙箱设置或本次沙箱外操作。")


def create_sandbox_router(ui_token: str) -> APIRouter:
    router = APIRouter(prefix="/sandbox-api", tags=["sandbox"])

    def require_ui(header: str | None, token: str | None) -> None:
        # Capability is provided only through the native desktop JS bridge,
        # never in served HTML, tool context, config, or GET responses.
        require_desktop_approval(ui_token, header, token)

    @router.get("/settings")
    async def get_settings() -> dict[str, Any]:
        try:
            return {
                "settings": studio_sandbox.settings().model_dump(),
                "policy": studio_sandbox.description(),
                "configuration_path": str(studio_sandbox.configuration_path()),
            }
        except studio_sandbox.SandboxError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.put("/settings")
    async def update_settings(
        request: SandboxUpdate,
        x_veripatch_ui: str | None = Header(default=None),
        x_ragent_sandbox_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_ui(x_veripatch_ui, x_ragent_sandbox_key)
        if request.settings.mode == "off" and not request.confirm_unrestricted:
            raise HTTPException(409, "关闭沙箱需确认：批准的程序将拥有当前用户权限。")
        studio_sandbox.save_settings(request.settings)
        return await get_settings()

    @router.post("/probe")
    async def probe(
        x_veripatch_ui: str | None = Header(default=None),
        x_ragent_sandbox_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_ui(x_veripatch_ui, x_ragent_sandbox_key)
        try:
            return await asyncio.to_thread(studio_sandbox.management, "probe")
        except studio_sandbox.SandboxError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/install")
    async def install(
        request: SandboxInstall,
        x_veripatch_ui: str | None = Header(default=None),
        x_ragent_sandbox_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_ui(x_veripatch_ui, x_ragent_sandbox_key)
        try:
            return await asyncio.to_thread(studio_sandbox.management, "install")
        except studio_sandbox.SandboxError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
