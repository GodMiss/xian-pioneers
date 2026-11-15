"""Agent 主循环：ReAct 风格的「模型思考 -> 调用工具 -> 观察结果 -> 继续」。

一次 run() 内部的时序：

    user ─► LLM ─┬─ 有 tool_calls ─► 并发执行工具 ─► 结果回填 ─┐
                 │                                          │
                 └─ 无 tool_calls ─► 最终答案 ◄───────────────┘

要点：
- 工具失败不会中断循环，错误文本会回给模型让它自己调整；
- 单轮多个工具调用并发执行（asyncio.gather）；
- 有最大步数保护，避免模型陷入无限调用；
- 所有过程可通过 on_event 回调暴露给 UI / 日志。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import BaseLLM, LLMError, Message, ToolCall
from .tools import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

DEFAULT_SYSTEM_PROMPT = """你是一个可靠、严谨的通用助手，可以调用工具来完成任务。

工作原则：
1. 需要实时信息（新闻、价格、天气、最新数据、你不确定的事实）时，先调用 web_search，不要凭记忆编造。
2. 搜索结果不足时，用 read_url 打开具体链接读正文；必要时换关键词再搜一次。
3. 工具调用要克制，一次只解决当前子问题，不要为了凑步骤而调用。
4. 回答基于工具返回的真实内容；引用网上信息时给出链接。信息不足就明确说明不足并给出现在能确认的部分。
5. 最终回答直接给结论，不要复述思考过程或工具调用细节。
"""


@dataclass
class AgentResult:
    answer: str
    messages: list[Message] = field(default_factory=list)
    steps: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "completed"  # completed | max_steps | error


class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_history_messages: int = 40,
        on_event: EventCallback | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_messages = max_history_messages
        self.on_event = on_event
        self._messages: list[Message] = [Message.system(system_prompt)] if system_prompt else []
        self._usage: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 会话状态
    # ------------------------------------------------------------------ #

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def reset(self) -> None:
        self._messages = [Message.system(self.system_prompt)] if self.system_prompt else []
        self._usage = {}

    # ------------------------------------------------------------------ #

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001 - UI 回调出错不影响主流程
                log.debug("on_event 回调异常", exc_info=True)

    async def run(self, user_input: str) -> AgentResult:
        if not user_input.strip():
            raise ValueError("user_input 不能为空")

        self._emit("user", content=user_input)
        self._messages.append(Message.user(user_input))
        self._trim_history()

        used_calls: list[ToolCall] = []
        usage_total: dict[str, Any] = {}
        last_content = ""

        for step in range(1, self.max_steps + 1):
            self._emit("step", index=step)
            try:
                resp = await self.llm.chat(
                    self._messages,
                    tools=self.tools.definitions() or None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except LLMError as e:
                self._emit("error", message=str(e), step=step)
                return AgentResult(
                    answer=f"模型调用失败：{e}",
                    messages=self._messages,
                    steps=step,
                    usage=usage_total,
                    tool_calls=used_calls,
                    stop_reason="error",
                )

            _merge_usage(usage_total, resp.usage)
            if resp.content:
                last_content = resp.content
            self._emit(
                "assistant",
                content=resp.content,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
                step=step,
            )

            # 没有工具调用 -> 这就是最终答案
            if not resp.tool_calls:
                answer = (resp.content or "").strip() or "（模型没有返回内容）"
                self._messages.append(Message.assistant(resp.content or ""))
                self._emit("final", content=answer, steps=step)
                return AgentResult(
                    answer=answer,
                    messages=self._messages,
                    steps=step,
                    usage=usage_total,
                    tool_calls=used_calls,
                    stop_reason="completed",
                )

            # 有工具调用 -> 记录 assistant 消息，并发执行工具
            self._messages.append(Message.assistant(resp.content or "", resp.tool_calls))
            used_calls.extend(resp.tool_calls)

            for call in resp.tool_calls:
                self._emit("tool_call", name=call.name, arguments=call.arguments, call_id=call.id)

            results: list[ToolResult] = await asyncio.gather(
                *(
                    self.tools.execute(call.id, call.name, call.arguments)
                    for call in resp.tool_calls
                )
            )

            for call, result in zip(resp.tool_calls, results):
                self._emit(
                    "tool_result",
                    name=result.name,
                    content=result.content,
                    is_error=result.is_error,
                    elapsed=round(result.elapsed, 2),
                    call_id=result.call_id,
                )
                self._messages.append(
                    Message.tool(result.call_id, result.content, name=result.name)
                )

        # 步数用尽
        self._emit("max_steps", steps=self.max_steps)
        note = (
            f"（已达到最大步数 {self.max_steps}，任务可能未完成。"
            f"最后一次模型输出如下）\n\n{last_content.strip() or '（无输出）'}"
        )
        return AgentResult(
            answer=note,
            messages=self._messages,
            steps=self.max_steps,
            usage=usage_total,
            tool_calls=used_calls,
            stop_reason="max_steps",
        )

    # ------------------------------------------------------------------ #

    def _trim_history(self) -> None:
        """防止多轮对话无限增长；只在 user 消息处切断，保证 tool 消息不成孤儿。"""
        if len(self._messages) <= self.max_history_messages:
            return
        head = self._messages[:1] if self._messages and self._messages[0].role == "system" else []
        tail_budget = self.max_history_messages - len(head)
        start = len(self._messages) - tail_budget
        while start < len(self._messages) and self._messages[start].role != "user":
            start += 1
        if start >= len(self._messages):
            self._messages = self._messages[: len(head)] + [Message.user("…")]
            return
        log.debug("裁剪历史：丢弃前 %d 条消息", start - len(head))
        self._messages = head + self._messages[start:]


def _merge_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value
