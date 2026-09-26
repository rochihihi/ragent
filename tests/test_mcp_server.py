from pathlib import Path

from veripatch.mcp_server import MCPServer


def test_mcp_evidence_survives_model_context_projection() -> None:
    from veripatch.studio_agent import StudioAgent
    from veripatch.studio_domain import StudioObservation

    payload = {"tool": "list_tools", "arguments": {}, "result": [
        {"name": "read_project_file", "inputSchema": {"required": ["path"]}}
    ]}
    projected = StudioAgent._compact_observation(
        StudioObservation(kind="mcp_tool", summary="discovered tools", payload=payload)
    )
    assert projected["payload"] == payload


def test_mcp_server_discovers_and_calls_read_only_project_tools(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    server = MCPServer(tmp_path)

    initialized = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert initialized and initialized["result"]["serverInfo"]["name"] == "ragent-project"

    listed = server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed and {tool["name"] for tool in listed["result"]["tools"]} == {
        "list_project_files",
        "read_project_file",
        "search_project",
    }

    called = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "read_project_file", "arguments": {"path": "app.py"}},
        }
    )
    assert called and "return a + b" in called["result"]["content"][0]["text"]


def test_mcp_server_rejects_workspace_escape(tmp_path: Path) -> None:
    server = MCPServer(tmp_path)
    result = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_project_file", "arguments": {"path": "../secret.txt"}},
        }
    )
    assert result and result["error"]["code"] == -32602
