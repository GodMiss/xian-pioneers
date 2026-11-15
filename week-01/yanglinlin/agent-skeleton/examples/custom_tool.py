"""自定义工具的三种写法，以及如何接进 Agent。

运行（需要先配好某个模型的 API Key）：
    python examples/custom_tool.py "北京今天天气怎么样？"
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, Tool, ToolRegistry, create_llm, tool  # noqa: E402
from agent.tools.base import FunctionTool  # noqa: E402


# --------------------------------------------------------------------- #
# 写法一：装饰器（最省事，推荐）
# --------------------------------------------------------------------- #
@tool(
    name="get_weather",
    description="查询指定城市当天的天气。数据来自本地演示桩，金额类/实时类需求请接真实 API。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，例如 北京"},
        },
        "required": ["city"],
    },
)
async def get_weather(city: str) -> str:
    # 真实项目里这里换成 httpx 调用天气 API
    return f"{city}：晴，24℃，东南风 2 级（演示数据）"


# --------------------------------------------------------------------- #
# 写法二：FunctionTool 包装一个已有函数（不改动原函数）
# --------------------------------------------------------------------- #
def word_count(text: str) -> str:
    """统计字符数和中文字数，纯函数，便于复用。"""
    return f"字符数 {len(text)}，其中汉字 {sum(1 for c in text if '\u4e00' <= c <= '\u9fff')} 个"


word_count_tool = FunctionTool(
    word_count,
    name="word_count",
    description="统计一段文本的字符数",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "待统计的文本"}},
        "required": ["text"],
    },
)


# --------------------------------------------------------------------- #
# 写法三：继承 Tool（适合带状态、需要连接池/生命周期的工具）
# --------------------------------------------------------------------- #
class DatabaseQueryTool(Tool):
    name = "query_orders"
    description = "按用户 ID 查询最近订单数量（演示：返回固定值）"
    parameters = {
        "type": "object",
        "properties": {"user_id": {"type": "string", "description": "用户 ID"}},
        "required": ["user_id"],
    }

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn  # 真实项目里在这里建连接池

    async def run(self, user_id: str, **_: Any) -> str:
        return f"用户 {user_id} 最近 30 天有 3 笔订单（dsn={self.dsn}，演示数据）"

    async def aclose(self) -> None:
        pass  # 关闭连接池


async def main() -> None:
    registry = ToolRegistry()
    registry.register(get_weather)
    registry.register(word_count_tool)
    registry.register(DatabaseQueryTool("sqlite://demo.db"))

    llm = create_llm(provider=os.getenv("AGENT_PROVIDER", "deepseek"))
    agent = Agent(
        llm,
        registry,
        system_prompt="你是一个简洁的助手。能用工具确认的信息就去确认，不要编造。",
        max_steps=4,
        on_event=lambda kind, data: print(f"[{kind}] {data}") if kind.startswith("tool") else None,
    )

    question = " ".join(sys.argv[1:]) or "北京今天天气怎么样？"
    result = await agent.run(question)
    print("\n回答：", result.answer)

    await registry.aclose()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
