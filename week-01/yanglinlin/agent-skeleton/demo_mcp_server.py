"""一个极简的 stdio MCP server，用来验证 MCP 接入链路（不联网、不依赖外部包）。

启动方式（骨架会自动按 mcp_servers.json 里的配置启动它）：

    python demo_mcp_server.py

它只暴露两个工具：add_numbers 和 server_info。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """返回两个数字之和。"""
    return a + b


@mcp.tool()
def server_info() -> str:
    """返回这个 MCP server 的自我介绍。"""
    return "demo MCP server，提供 add_numbers / server_info 两个工具。"


if __name__ == "__main__":
    mcp.run(transport="stdio")
