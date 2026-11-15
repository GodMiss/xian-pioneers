"""LLM 抽象层。

这里只做三件事：
1. 定义一套与厂商无关的消息结构（Message / ToolCall / LLMResponse）；
2. 定义统一的 async 调用接口 BaseLLM；
3. 提供 OpenAI 兼容格式的序列化/反序列化，方便切换 GLM、DeepSeek 等提供方。

设计原则：上层 Agent 只依赖这一层，换模型时不用改 Agent 代码。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(Exception):
    """模型调用失败（网络、鉴权、限流、返回格式异常等）。"""

    def __init__(self, message: str, *, provider: str = "", status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:  # pragma: no cover - 仅用于展示
        head = f"[{self.provider}] " if self.provider else ""
        return f"{head}{super().__str__()}"


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # 保留模型返回的原始 JSON 字符串：回传时原样带回，避免二次序列化导致格式漂移
    raw_arguments: str = ""

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class Message:
    """与厂商无关的一条消息。"""

    role: Role
    content: str = ""
    # 仅 assistant 使用
    tool_calls: list[ToolCall] = field(default_factory=list)
    # 仅 tool 使用：对应哪个 tool_call
    tool_call_id: str | None = None
    name: str | None = None

    # ---------- 构造快捷方法 ----------

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Sequence[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    # ---------- 序列化 ----------

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI chat/completions 的 message 结构（GLM、DeepSeek 均兼容）。"""
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id or "", "content": self.content}
        if self.role == "assistant":
            payload: dict[str, Any] = {"role": "assistant", "content": self.content or None}
            if self.tool_calls:
                payload["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
            return payload
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "Message":
        raw_calls = data.get("tool_calls") or []
        return cls(
            role=data.get("role", "assistant"),  # type: ignore[arg-type]
            content=data.get("content") or "",
            tool_calls=[parse_tool_call(c) for c in raw_calls],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


def parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    """把 OpenAI 格式的 tool_call 解析成 ToolCall，容忍 arguments 不是合法 JSON。"""
    fn = raw.get("function") or {}
    args_str = fn.get("arguments") or ""
    if isinstance(args_str, dict):  # 少数实现直接给 dict
        arguments, args_str = args_str, json.dumps(args_str, ensure_ascii=False)
    else:
        try:
            arguments = json.loads(args_str) if args_str.strip() else {}
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
        except json.JSONDecodeError:
            arguments = {}
    return ToolCall(
        id=raw.get("id") or f"call_{abs(hash(fn.get('name', ''))) % 10**8}",
        name=fn.get("name") or "",
        arguments=arguments,
        raw_arguments=args_str if isinstance(args_str, str) else "",
    )


@dataclass
class LLMResponse:
    """一次模型调用的结果。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class BaseLLM(ABC):
    """所有模型客户端的统一接口。"""

    provider: str = "unknown"
    model: str = ""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发起一次对话补全。tools 为 OpenAI function schema 列表。"""

    async def aclose(self) -> None:  # pragma: no cover - 默认无事可做
        """释放连接资源。"""
        return None

    async def __aenter__(self) -> "BaseLLM":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()
