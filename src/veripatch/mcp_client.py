"""Client for the bundled, workspace-scoped read-only MCP subprocess."""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _exchange(root: Path, tool: str, arguments: dict[str, Any]) -> Any:
    if getattr(sys, "frozen", False):
        command = str(Path(sys.executable).with_name("RAgent-MCP.exe"))
        args = ["--repo", str(root.resolve())]
    else:
        command = sys.executable
        args = ["-m", "veripatch.mcp_server", "--repo", str(root.resolve())]
    async with asyncio.timeout(30):
        params = StdioServerParameters(command=command, args=args)
        async with stdio_client(params) as streams, ClientSession(
            *streams, read_timeout_seconds=timedelta(seconds=20)
        ) as client:
            await client.initialize()
            tools = (await client.list_tools()).tools
            if tool == "list_tools":
                return [item.model_dump(mode="json", exclude_none=True) for item in tools]
            if tool not in {item.name for item in tools}:
                raise ValueError(f"Unknown MCP tool: {tool}")
            result = await client.call_tool(tool, arguments)
            text = "\n".join(item.text for item in result.content if item.type == "text")
            if result.isError:
                raise ValueError(text)
            return json.loads(text)


def call_project_tool(root: Path, tool: str, arguments: dict[str, Any]) -> Any:
    # Studio tools are synchronous, including when called inside its async loop.
    # Keep the SDK session and subprocess cleanup on the same worker event loop.
    with ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(lambda: asyncio.run(_exchange(root, tool, arguments))).result()
