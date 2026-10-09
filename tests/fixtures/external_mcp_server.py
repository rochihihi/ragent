"""Independent SDK echo server used only by external MCP integration tests."""
from mcp.server.fastmcp import FastMCP

server = FastMCP("test-external")


@server.tool()
def echo(message: str) -> str:
    return message


if __name__ == "__main__":
    server.run(transport="stdio")
