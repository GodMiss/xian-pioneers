"""最小可用的 Agent 骨架。

用法速览：

    from agent import Agent, create_llm, build_default_registry

    llm = create_llm("deepseek")                 # 或 "glm"
    agent = Agent(llm, build_default_registry()) # 已内置 web_search / read_url
    result = await agent.run("今天上海天气如何？")
"""

from .agent import Agent, AgentResult, DEFAULT_SYSTEM_PROMPT
from .llm import PROVIDERS, BaseLLM, LLMError, Message, ToolCall, create_llm, list_providers
from .mcp import MCPManager, load_servers
from .tools import Tool, ToolRegistry, build_default_registry, tool

__all__ = [
    "Agent",
    "AgentResult",
    "DEFAULT_SYSTEM_PROMPT",
    "BaseLLM",
    "LLMError",
    "Message",
    "ToolCall",
    "PROVIDERS",
    "create_llm",
    "list_providers",
    "MCPManager",
    "load_servers",
    "Tool",
    "ToolRegistry",
    "build_default_registry",
    "tool",
]
