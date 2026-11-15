"""离线冒烟测试：不需要 API Key、不联网，验证 agent 主循环与 MCP 接入。

运行：
    python scripts/smoke_test.py            # 只测主循环
    python scripts/smoke_test.py --mcp      # 额外测 MCP（会拉起 demo_mcp_server.py）

思路：用一个脚本化的假模型 FakeLLM 替代真实模型，
按预设剧本让它在第 1 轮调用工具、第 2 轮给出答案，
从而在没有任何外部依赖的情况下验证「模型 -> 工具 -> 模型」闭环。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, MCPManager, ToolRegistry, create_llm, tool  # noqa: E402
from agent.llm.base import BaseLLM, LLMResponse, ToolCall  # noqa: E402


class FakeLLM(BaseLLM):
    """按剧本返回的假模型：第一次要求调用工具，第二次直接给答案。"""

    provider = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="我需要先算一下。",
                tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 17, "b": 25})],
                finish_reason="tool_calls",
                usage={"total_tokens": 20},
            )
        # 从历史里把工具结果捞出来，模拟"模型读到了工具返回"
        tool_output = ""
        for m in messages:
            if m.role == "tool":
                tool_output = m.content
        return LLMResponse(
            content=f"17 + 25 = {tool_output}（来自工具）",
            finish_reason="stop",
            usage={"total_tokens": 15},
        )


@tool(
    name="add",
    description="把两个数字相加",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "第一个数"},
            "b": {"type": "number", "description": "第二个数"},
        },
        "required": ["a", "b"],
    },
)
async def add(a: float, b: float) -> str:
    return str(a + b)


async def test_loop() -> bool:
    registry = ToolRegistry([add])
    events: list[str] = []
    agent = Agent(FakeLLM(), registry, system_prompt="测试用 system", max_steps=3,
                  on_event=lambda kind, data: events.append(kind))

    result = await agent.run("17 + 25 等于多少？")
    print("  回答：", result.answer)
    print("  事件：", " -> ".join(events))
    print("  步数：", result.steps, " 停止原因：", result.stop_reason, " 用量：", result.usage)

    assert result.stop_reason == "completed", result.stop_reason
    assert "42" in result.answer, result.answer
    assert result.steps == 2, result.steps
    assert result.usage.get("total_tokens") == 35, result.usage
    assert "tool_call" in events and "tool_result" in events
    # 历史结构检查：assistant(tool_calls) 后面必须紧跟 tool 消息
    idx = next(i for i, m in enumerate(result.messages) if m.tool_calls)
    assert result.messages[idx + 1].role == "tool"
    return True


async def test_tool_error_does_not_break_loop() -> bool:
    registry = ToolRegistry([add])

    class ErrorLLM(BaseLLM):
        provider = "fake"
        model = "fake"
        calls = 0

        async def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # 故意用错误参数调用，验证异常被转成文本而不是崩掉
                return LLMResponse(tool_calls=[ToolCall(id="c1", name="add", arguments={"a": "x"})])
            # 第二次应能拿到错误文本并继续
            tool_msg = next(m for m in messages if m.role == "tool")
            return LLMResponse(content=f"工具报错了：{tool_msg.content[:40]}")

    result = await Agent(ErrorLLM(), registry, max_steps=3).run("算一下")
    print("  回答：", result.answer[:80])
    assert result.stop_reason == "completed"
    assert "错误" in result.answer
    return True


async def test_mcp() -> bool:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    servers = {
        "demo": {
            "command": sys.executable,
            "args": [os.path.join(here, "demo_mcp_server.py")],
        }
    }
    registry = ToolRegistry()
    manager = MCPManager(servers, registry)
    stats = await manager.connect_all()
    print("  已连接 server：", stats.servers_connected, " 工具数：", stats.tools_registered, " 错误：", stats.errors)
    try:
        assert stats.servers_connected == ["demo"], stats.errors
        assert "demo_add_numbers" in registry.names(), registry.names()

        result = await registry.execute("c1", "demo_add_numbers", {"a": 2, "b": 39})
        print("  MCP 工具返回：", result.content)
        assert not result.is_error, result.content
        assert "41" in result.content, result.content

        # 再走一遍完整 agent 循环，验证「模型 -> MCP 工具 -> 模型」也通
        class MCPLLM(BaseLLM):
            provider = "fake"
            model = "fake"
            calls = 0

            async def chat(self, messages, tools=None, **kwargs):
                self.calls += 1
                assert tools, "MCP 工具的 schema 没有传给模型"
                assert any(t["function"]["name"] == "demo_add_numbers" for t in tools), tools
                if self.calls == 1:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(id="m1", name="demo_add_numbers", arguments={"a": 100, "b": 23})
                        ]
                    )
                tool_msg = next(m for m in messages if m.role == "tool")
                return LLMResponse(content=f"MCP 算了：{tool_msg.content}")

        out = await Agent(MCPLLM(), registry, max_steps=3).run("用 MCP 工具算 100+23")
        print("  agent 走 MCP 的回答：", out.answer)
        assert "123" in out.answer, out.answer
    finally:
        await manager.aclose()
    return True


async def test_provider_factory() -> bool:
    # 只验证工厂能正确解析配置，不发真实请求
    try:
        create_llm("deepseek", api_key="sk-test")
    except Exception as e:  # pragma: no cover
        print("  deepseek 构造失败：", e)
        return False
    llm_glm = create_llm("glm", api_key="test-key")
    assert llm_glm.base_url.endswith("/api/paas/v4")
    assert llm_glm.model == "glm-4.6"
    assert create_llm("ollama").model == "qwen3:8b"
    print("  提供方工厂：GLM ->", llm_glm.base_url, "| model:", llm_glm.model)
    return True


async def amain(with_mcp: bool) -> int:
    cases = [
        ("agent 主循环", test_loop()),
        ("工具报错不中断", test_tool_error_does_not_break_loop()),
        ("模型工厂", test_provider_factory()),
    ]
    if with_mcp:
        cases.append(("MCP 接入", test_mcp()))

    failed = 0
    for name, coro in cases:
        print(f"[测试] {name}")
        try:
            ok = await coro
        except Exception as e:  # noqa: BLE001
            ok = False
            print("  失败：", type(e).__name__, e)
        print("  =>", "通过" if ok else "失败")
        failed += 0 if ok else 1

    print(f"\n共 {len(cases)} 项，失败 {failed} 项")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp", action="store_true", help="同时测试 MCP 接入")
    argv = parser.parse_args()
    raise SystemExit(asyncio.run(amain(argv.mcp)))
