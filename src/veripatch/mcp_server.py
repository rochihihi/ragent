"""Minimal read-only MCP server for exposing a local project to an Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import validate

from veripatch.indexing import IGNORED_DIRECTORIES
from veripatch.workspace import SafeWorkspace

PROTOCOL_VERSION = "2024-11-05"


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_project_files",
            "description": "List UTF-8 project files while skipping generated directories.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "read_project_file",
            "description": "Read a UTF-8 text file in the selected project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_project",
            "description": "Search text in UTF-8 project files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]


class MCPServer:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.workspace = SafeWorkspace(self.root)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = next((tool for tool in _tools() if tool["name"] == name), None)
        if tool is None:
            raise ValueError(f"Unknown MCP tool: {name}")
        validate(arguments, tool["inputSchema"])
        if name == "list_project_files":
            return [
                path.relative_to(self.root).as_posix()
                for path in sorted(self.root.rglob("*"))
                if path.is_file()
                and not any(
                    part in IGNORED_DIRECTORIES for part in path.relative_to(self.root).parts
                )
            ][:500]
        if name == "read_project_file":
            return self.workspace.read(
                str(arguments.get("path", "")),
                int(arguments.get("start_line", 1)),
                int(arguments.get("end_line", 240)),
            )
        if name == "search_project":
            return self.workspace.search(
                str(arguments.get("query", "")),
                limit=min(int(arguments.get("limit", 30)), 100),
            )
        raise ValueError(f"Unknown MCP tool: {name}")

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if "id" not in request:
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "ragent-project", "version": "3.0.0"},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tools()}}
        if method == "tools/call":
            params = request.get("params") or {}
            try:
                value = self.call(str(params.get("name", "")), params.get("arguments") or {})
                content = [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]
                return {"jsonrpc": "2.0", "id": request_id, "result": {"content": content}}
            except (OSError, ValueError, TypeError) as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": str(exc)},
                }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def serve(root: Path) -> None:
    server = MCPServer(root)
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        request = json.loads(line)
        response = server.dispatch(request)
        if response is None:
            continue
        payload = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sys.stdout.buffer.write(payload + b"\n")
        sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragent-mcp")
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    serve(args.repo)


if __name__ == "__main__":
    main()
