from .base import (
    FunctionTool,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    tool,
    truncate,
)
from .read_url import ReadUrlTool
from .web_search import WebSearchTool

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "FunctionTool",
    "tool",
    "truncate",
    "WebSearchTool",
    "ReadUrlTool",
    "build_default_registry",
]


def build_default_registry(
    *,
    enable_web_search: bool = True,
    enable_read_url: bool = True,
    search_backend: str = "auto",
) -> ToolRegistry:
    """装好内置工具的工具箱。"""
    registry = ToolRegistry()
    if enable_web_search:
        registry.register(WebSearchTool(backend=search_backend))
    if enable_read_url:
        registry.register(ReadUrlTool())
    return registry
