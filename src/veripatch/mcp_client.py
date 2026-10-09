"""Client for user-configured external MCP servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from veripatch import studio_sandbox


def configuration_path() -> Path:
    return Path(
        os.environ.get("RAGENT_MCP_CONFIG")
        or str(Path(os.environ.get("APPDATA", str(Path.home()))) / "RAgent" / "mcp_servers.json")
    )


def configured_servers(*, include_disabled: bool = False) -> dict[str, dict[str, Any]]:
    """Load user-owned configuration, never configuration from the workspace."""
    path = configuration_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ValueError("MCP configuration requires an mcpServers object")
    return validate_servers(servers, include_disabled=include_disabled)


def validate_servers(servers, *, include_disabled=False):
    for name, config in servers.items():
        if not isinstance(name, str) or not name or "::" in name or name == "builtin":
            raise ValueError("Invalid MCP server name")
        if not isinstance(config, dict):
            raise ValueError("Invalid MCP server configuration")
        if not isinstance(config.get("enabled", True), bool):
            raise ValueError("MCP enabled must be boolean")
        if config.get("cwd") is not None and not isinstance(config["cwd"], str):
            raise ValueError("MCP cwd must be a string")
        transport = config.get("transport", "stdio")
        if transport == "stdio":
            if not isinstance(config.get("command"), str) or not config["command"]:
                raise ValueError("stdio MCP requires command")
            args = config.get("args", [])
            if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                raise ValueError("MCP args must be strings")
        elif transport == "streamable_http":
            from urllib.parse import urlparse

            url = urlparse(config.get("url", ""))
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
            ):
                raise ValueError("Invalid MCP HTTP URL")
            if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Remote MCP requires HTTPS")
        else:
            raise ValueError("Unsupported MCP transport")
        for field in ("env", "headers_env"):
            mapping = config.get(field, {})
            if not isinstance(mapping, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
            ):
                raise ValueError(f"MCP {field} must map strings to strings")
    return {
        name: config
        for name, config in servers.items()
        if include_disabled or config.get("enabled", True)
    }


def external_config_fingerprint(tool: str) -> str | None:
    if tool == "list_servers":
        return None
    if "::" not in tool:
        return None
    name = tool.split("::", 1)[0]
    config = configured_servers().get(name)
    if config is None:
        raise ValueError("Unknown configured MCP server")
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


async def _request(streams, tool: str, arguments: dict[str, Any]) -> Any:
    async with ClientSession(*streams[:2], read_timeout_seconds=timedelta(seconds=20)) as client:
        await client.initialize()
        tools = []
        cursor = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
        if tool == "list_tools":
            return [item.model_dump(mode="json", exclude_none=True) for item in tools]
        if tool not in {item.name for item in tools}:
            raise ValueError(f"Unknown MCP tool: {tool}")
        result = await client.call_tool(tool, arguments)
        # Preserve structured and non-text content; remote errors are tool evidence.
        payload = result.model_dump(mode="json", exclude_none=True)
        if result.isError:
            raise ValueError(json.dumps(payload, ensure_ascii=False))
        return payload


async def _exchange(
    root: Path,
    tool: str,
    arguments: dict[str, Any],
    *,
    read_paths: list[Path] | None = None,
    write_paths: list[Path] | None = None,
    host_execution_approved: bool = False,
) -> Any:
    if host_execution_approved:
        studio_sandbox.ensure_host_execution_allowed()
    if tool == "list_servers":
        return [
            {"name": name, "transport": config.get("transport", "stdio")}
            for name, config in configured_servers().items()
        ]
    server, separator, name = tool.partition("::")
    if not separator or not name:
        raise ValueError("MCP tools require server::tool; use native tools for project files")
    if separator:
        config = configured_servers().get(server)
        if config is None:
            raise ValueError("Unknown configured MCP server")
        async with asyncio.timeout(30):
            if config.get("transport", "stdio") == "streamable_http":
                headers = {}
                for header, variable in config.get("headers_env", {}).items():
                    if variable not in os.environ:
                        raise ValueError("Missing MCP authentication environment variable")
                    headers[header] = os.environ[variable]
                async with streamablehttp_client(config["url"], headers=headers) as streams:
                    return await _request(streams, name, arguments)
            cwd = Path(config.get("cwd") or root).resolve()
            # A configured cwd is not automatically a filesystem write grant.
            if (
                not host_execution_approved
                and studio_sandbox.settings().mode == "required"
                and not any(
                    cwd == p.resolve() or cwd.is_relative_to(p.resolve())
                    for p in [root, *(read_paths or []), *(write_paths or [])]
                )
            ):
                raise studio_sandbox.SandboxError("SandboxError: MCP 工作目录在授权路径之外。")
            with studio_sandbox.prepare(
                root,
                [config["command"], *config.get("args", [])],
                {**os.environ, **config.get("env", {})},
                read_paths=read_paths,
                write_paths=write_paths,
                explicit_env=config.get("env"),
                interactive=True,
                host_execution_approved=host_execution_approved,
            ) as launch:
                if launch.enabled and cwd != root.resolve():
                    assert launch.request is not None
                    request = json.loads(launch.request.read_text(encoding="utf-8"))
                    request["cwd"] = str(cwd)
                    launch.request.write_text(json.dumps(request), encoding="utf-8")
                params = StdioServerParameters(
                    command=launch.argv[0],
                    args=launch.argv[1:],
                    env=launch.env,
                    cwd=studio_sandbox.runtime_root() if launch.enabled else cwd,
                )
                try:
                    async with stdio_client(params) as streams:
                        return await _request(streams, name, arguments)
                except BaseException:
                    launch.check_error()
                    raise


def call_project_tool(
    root: Path,
    tool: str,
    arguments: dict[str, Any],
    *,
    read_paths: list[Path] | None = None,
    write_paths: list[Path] | None = None,
    host_execution_approved: bool = False,
) -> Any:
    # Studio tools are synchronous, including when called inside its async loop.
    # Keep the SDK session and subprocess cleanup on the same worker event loop.
    with ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(
            lambda: asyncio.run(
                _exchange(
                    root, tool, arguments, read_paths=read_paths, write_paths=write_paths,
                    host_execution_approved=host_execution_approved,
                )
            )
        ).result()
