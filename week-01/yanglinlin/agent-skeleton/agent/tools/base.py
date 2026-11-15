"""工具抽象层。

每个工具 = 名字 + 描述 + JSON Schema + 一个 async 执行函数。
Agent 把 schema 交给模型，模型决定调用后，由 ToolRegistry 统一分发执行。

新增一个自定义工具，最省事的方式是：

    from agent.tools import tool

    @tool(name="get_weather", description="查询某城市天气",
          parameters={"type": "object",
                      "properties": {"city": {"type": "string", "description": "城市名"}},
                      "required": ["city"]})
    async def get_weather(city: str) -> str:
        return f"{city} 今天晴，25℃"

装饰器会把普通函数包装成 Tool 对象，直接 register 即可。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8000


class ToolError(Exception):
    """工具执行失败。会以错误文本回传给模型，让它自己决定要不要换个方式。"""


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[结果已截断，原始长度 {len(text)} 字符]"


class Tool(ABC):
    """工具基类。子类实现 name / description / parameters / run。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def definition(self) -> dict[str, Any]:
        """转成 OpenAI function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """执行工具，返回字符串（给模型看）。"""

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class FunctionTool(Tool):
    """把普通（或 async）函数包装成 Tool。"""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").split("\n")[0]
        self.parameters = parameters or {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        try:
            result = self.func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as e:
            raise ToolError(f"参数不匹配：{e}") from e
        if result is None:
            return "（执行成功，无输出）"
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """装饰器：@tool(...) -> FunctionTool"""

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, parameters=parameters)

    return decorator


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    elapsed: float = 0.0


class ToolRegistry:
    """工具注册表 + 执行器。"""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    # ---------------- 注册 ----------------

    def register(self, t: Tool | Callable[..., Any], *, override: bool = False) -> Tool:
        if not isinstance(t, Tool):
            t = FunctionTool(t)
        if not t.name:
            raise ValueError("工具必须有 name")
        if t.name in self._tools and not override:
            raise ValueError(f"工具名 '{t.name}' 已存在")
        self._tools[t.name] = t
        return t

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        return [t.definition() for t in self._tools.values()]

    # ---------------- 生命周期 ----------------

    async def aclose(self) -> None:
        """关闭工具持久的连接（例如内置 HTTP 客户端）。"""
        for t in self._tools.values():
            closer = getattr(t, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001
                log.debug("关闭工具 %s 时出错", t.name, exc_info=True)

    # ---------------- 执行 ----------------

    async def execute(self, call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行一次工具调用。任何异常都转成错误文本，不让它打断 agent 循环。"""
        t = self._tools.get(name)
        start = time.perf_counter()
        if t is None:
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"错误：不存在名为 '{name}' 的工具。可用工具：{', '.join(self.names())}",
                is_error=True,
            )
        try:
            output = await t.run(**arguments)
        except ToolError as e:
            output, is_error = f"错误：{e}", True
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - 工具错误不应中断主循环
            log.exception("工具 %s 执行异常", name)
            output, is_error = f"错误：{type(e).__name__}: {e}", True
        else:
            is_error = False
        return ToolResult(
            call_id=call_id,
            name=name,
            content=truncate(output),
            is_error=is_error,
            elapsed=time.perf_counter() - start,
        )
