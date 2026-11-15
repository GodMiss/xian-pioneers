"""MCP（Model Context Protocol）接入。

作用：把外部 MCP server 暴露的工具，自动注册成本地 Tool，模型调用时无感知。

支持三种传输方式（写在配置文件里）：
  - stdio：本地进程，如 npx 启动的 filesystem server
  - sse：远程 HTTP SSE 端点
  - http：远端 streamable-http 端点

依赖官方 SDK：pip install mcp
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..tools.base import Tool, ToolError, ToolRegistry

log = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


def _sanitize(name: str) -> str:
    """OpenAI 要求 function name 匹配 [a-zA-Z0-9_-]，长度 <= 64。"""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return cleaned[:64]


class MCPTool(Tool):
    """把一个远端 MCP 工具包装成本地 Tool。"""

    def __init__(
        self,
        *,
        server: str,
        remote_name: str,
        local_name: str,
        description: str,
        input_schema: dict[str, Any],
        session: Any,
    ) -> None:
        self.server = server
        self.remote_name = remote_name
        self.name = local_name
        self.description = description or f"[MCP:{server}] {remote_name}"
        self.parameters = input_schema or {"type": "object", "properties": {}}
        self._session = session

    async def run(self, **kwargs: Any) -> str:
        if self._session is None:
            raise ToolError(f"MCP server '{self.server}' 未连接")
        try:
            result = await self._session.call_tool(self.remote_name, arguments=kwargs)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"调用 MCP 工具失败：{type(e).__name__}: {e}") from e
        return _format_result(result, self.server)

    def __repr__(self) -> str:
        return f"<MCPTool {self.server}.{self.remote_name} as {self.name}>"


def _format_result(result: Any, server: str) -> str:
    """MCP 返回的是 content 块列表，这里拍平成文本。"""
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
            continue
        btype = getattr(block, "type", "")
        if btype == "image":
            parts.append("[图片内容已省略：MCP 返回了图片，当前模型不支持图片输入]")
        elif btype == "resource":
            res = getattr(block, "resource", None)
            body = getattr(res, "text", None)
            parts.append(body if isinstance(body, str) else f"[资源 {getattr(res, 'uri', '')}]")
        else:
            parts.append(f"[{btype or '未知类型'} 内容]")

    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if not parts and structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, indent=2))

    text = "\n".join(parts).strip() or "（工具执行成功，无返回内容）"
    if is_error:
        text = "错误：" + text
    return text


@dataclass
class MCPStats:
    servers_connected: list[str] = field(default_factory=list)
    tools_registered: int = 0
    errors: dict[str, str] = field(default_factory=dict)


class MCPManager:
    """管理多个 MCP server 的连接生命周期，并把工具注入 ToolRegistry。"""

    def __init__(
        self,
        servers: dict[str, dict[str, Any]],
        registry: ToolRegistry,
        *,
        prefix_with_server: bool = True,
        read_timeout: float = 30.0,
    ) -> None:
        self.servers = servers or {}
        self.registry = registry
        self.prefix = prefix_with_server
        self.read_timeout = read_timeout
        self.stats = MCPStats()
        self._stacks: list[AsyncExitStack] = []
        self._closed = False

    # ------------------------------------------------------------------ #

    async def connect_all(self) -> MCPStats:
        if not _MCP_AVAILABLE:
            raise RuntimeError("未安装 MCP SDK，请先执行：pip install mcp")
        if not self.servers:
            return self.stats

        for name, cfg in self.servers.items():
            try:
                tools = await self._connect_one(name, cfg)
                self.stats.servers_connected.append(name)
                self.stats.tools_registered += tools
                log.info("MCP server '%s' 已连接，注册 %d 个工具", name, tools)
            except Exception as e:  # noqa: BLE001 - 单个 server 失败不该拖垮整体
                self.stats.errors[name] = f"{type(e).__name__}: {e}"
                log.warning("MCP server '%s' 连接失败：%s", name, e)
        return self.stats

    async def _connect_one(self, name: str, cfg: dict[str, Any]) -> int:
        stack = AsyncExitStack()
        try:
            read, write = await self._open_transport(stack, name, cfg)
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.read_timeout))
            )
            await session.initialize()
            listed = await session.list_tools()
            tools = getattr(listed, "tools", []) or []

            for remote in tools:
                remote_name = getattr(remote, "name", "")
                if not remote_name:
                    continue
                local_name = _sanitize(f"{name}_{remote_name}" if self.prefix else remote_name)
                if self.registry.get(local_name) is not None:
                    log.warning("MCP 工具名冲突，跳过：%s", local_name)
                    continue
                self.registry.register(
                    MCPTool(
                        server=name,
                        remote_name=remote_name,
                        local_name=local_name,
                        description=getattr(remote, "description", "") or "",
                        input_schema=getattr(remote, "inputSchema", None)
                        or getattr(remote, "input_schema", None)
                        or {},
                        session=session,
                    )
                )
        except BaseException:
            await _safe_close(stack)
            raise
        self._stacks.append(stack)
        return len(tools)

    async def _open_transport(self, stack: AsyncExitStack, name: str, cfg: dict[str, Any]):
        if "url" in cfg and cfg["url"]:
            transport = (cfg.get("transport") or "http").lower()
            url = cfg["url"]
            headers = cfg.get("headers") or {}
            if transport == "sse":
                from mcp.client.sse import sse_client

                read, write = await stack.enter_async_context(sse_client(url, headers=headers))
                return read, write
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(url, headers=headers)
            )
            return read, write

        command = cfg.get("command")
        if not command:
            raise ValueError(f"MCP server '{name}' 缺少 command 或 url")
        env = {**os.environ, **(cfg.get("env") or {})}
        params = StdioServerParameters(
            command=command,
            args=cfg.get("args") or [],
            env=env,
            cwd=cfg.get("cwd") or None,
        )
        from mcp.client.stdio import stdio_client

        read, write = await stack.enter_async_context(stdio_client(params))
        return read, write

    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stack in self._stacks:
            await _safe_close(stack)
        self._stacks.clear()


async def _safe_close(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except Exception as e:  # noqa: BLE001 - 关闭阶段的报错不影响主流程
        log.debug("关闭 MCP 连接时出错（可忽略）：%s", e)


# ---------------------------------------------------------------------- #
# 配置读取
# ---------------------------------------------------------------------- #

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    """把配置里的 ${ENV_VAR} 展开成真实值。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_servers(path: str) -> dict[str, dict[str, Any]]:
    """读取 MCP 配置，兼容 Claude Desktop 的 {"mcpServers": {...}} 格式。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 MCP 配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise RuntimeError("读取 YAML 配置需要安装 pyyaml：pip install pyyaml") from e
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    servers = data.get("mcpServers") or data.get("servers") or data
    if not isinstance(servers, dict):
        raise ValueError("MCP 配置格式错误：应为 {server_name: {...}} 或 {mcpServers: {...}}")
    return {k: _expand(v) for k, v in servers.items()}
