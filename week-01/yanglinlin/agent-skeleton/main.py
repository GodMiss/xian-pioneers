#!/usr/bin/env python3
"""命令行入口。

示例：
    python main.py --list-providers
    python main.py -p deepseek -q "帮我搜一下 2026 年 GLM-5 的消息并总结"
    python main.py -p glm --mcp mcp_servers.json        # 进入多轮对话
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

from agent import Agent, MCPManager, build_default_registry, create_llm, list_providers
from agent.llm import LLMError

try:  # .env 是可选依赖
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


BANNER = """────────────────────────────────────────
 Agent 骨架 · 支持 GLM / DeepSeek / OpenAI 兼容模型
 内置 web_search / read_url · 可接入 MCP
 输入 /help 查看命令，/exit 退出
────────────────────────────────────────"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent",
        description="一个支持多模型、web_search 与 MCP 的 Agent 骨架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-p", "--provider", default=os.getenv("AGENT_PROVIDER", "deepseek"),
                   help="模型提供方：glm / deepseek / moonshot / qwen / openai / ollama")
    p.add_argument("-m", "--model", default=None, help="模型名，缺省用提供方默认模型")
    p.add_argument("--api-key", default=None, help="覆盖环境变量里的密钥")
    p.add_argument("--base-url", default=None, help="自定义接口地址（如自建代理）")
    p.add_argument("-q", "--query", default=None, help="一次性执行该问题后退出；不填进入交互模式")
    p.add_argument("--mcp", default=None, help="MCP 配置文件路径（JSON/YAML）")
    p.add_argument("--no-mcp", action="store_true", help="不加载 MCP")
    p.add_argument("--no-web-search", action="store_true", help="不挂载内置 web_search")
    p.add_argument("--search-backend", default="auto",
                   choices=["auto", "tavily", "bing", "searxng", "duckduckgo"], help="搜索后端")
    p.add_argument("--max-steps", type=int, default=8, help="最大工具调用轮次，默认 8")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--system", default=None, help="自定义 system prompt")
    p.add_argument("--list-providers", action="store_true", help="列出内置提供方后退出")
    p.add_argument("-v", "--verbose", action="store_true", help="打印调试日志")
    return p


def make_printer() -> Any:
    """把 Agent 事件打印成人能看懂的过程日志。"""

    def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "step":
            print(f"\n[第 {data['index']} 步]")
        elif kind == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in (data.get("arguments") or {}).items())
            print(f"  → 调用 {data['name']}({args})")
        elif kind == "tool_result":
            flag = "✗" if data.get("is_error") else "✓"
            preview = (data.get("content") or "").replace("\n", " ")[:120]
            print(f"  {flag} {data['name']} 用时 {data.get('elapsed')}s：{preview}...")
        elif kind == "error":
            print(f"  ! {data.get('message')}")
        elif kind == "max_steps":
            print("  ! 已达最大步数")

    return on_event


async def create_agent(args: argparse.Namespace) -> tuple[Agent, MCPManager | None]:
    llm = create_llm(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    registry = build_default_registry(
        enable_web_search=not args.no_web_search,
        search_backend=args.search_backend,
    )

    mcp: MCPManager | None = None
    if not args.no_mcp:
        mcp_path = args.mcp or ("mcp_servers.json" if os.path.exists("mcp_servers.json") else None)
        if mcp_path:
            mcp = await _connect_mcp(mcp_path, registry)
        else:
            print("（未发现 MCP 配置，跳过。可用 --mcp 指定配置文件）")

    agent_kwargs: dict[str, Any] = {
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "on_event": make_printer(),
    }
    if args.system:
        agent_kwargs["system_prompt"] = args.system

    agent = Agent(llm, registry, **agent_kwargs)
    return agent, mcp


async def _connect_mcp(path: str, registry) -> MCPManager:
    from agent.mcp import load_servers

    servers = load_servers(path)
    manager = MCPManager(servers, registry)
    print(f"正在连接 {len(servers)} 个 MCP server：{', '.join(servers) or '（空）'}")
    stats = await manager.connect_all()
    if stats.servers_connected:
        print(f"  MCP 已连接 {len(stats.servers_connected)} 个，注册工具 {stats.tools_registered} 个")
    for name, err in stats.errors.items():
        print(f"  MCP '{name}' 连接失败：{err}")
    return manager


async def main_async(args: argparse.Namespace) -> int:
    if args.list_providers:
        print("内置提供方：\n" + list_providers())
        return 0

    try:
        agent, mcp = await create_agent(args)
    except ValueError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2

    try:
        print(f"模型：{agent.llm.provider} / {agent.llm.model}")
        print(f"可用工具：{', '.join(agent.tools.names())}")

        if args.query:
            result = await agent.run(args.query)
            print("\n" + "═" * 60)
            print(result.answer)
            print("═" * 60)
            if result.usage:
                print(f"[用量] {result.usage}")
            return 0 if result.stop_reason == "completed" else 1

        print(BANNER)
        while True:
            try:
                user = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                break
            if not user:
                continue
            if user in ("/exit", "/quit", "exit", "quit"):
                break
            if user == "/help":
                print("/help 帮助 | /tools 查看工具 | /reset 清空上下文 | /exit 退出")
                continue
            if user == "/tools":
                for name in agent.tools.names():
                    print(f"  - {name}")
                continue
            if user == "/reset":
                agent.reset()
                print("  上下文已清空")
                continue

            result = await agent.run(user)
            print("\n助手 >", result.answer)
        return 0
    finally:
        await agent.tools.aclose()
        await agent.llm.aclose()
        if mcp:
            await mcp.aclose()


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(main_async(args))
    except LLMError as e:
        print(f"模型调用失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
