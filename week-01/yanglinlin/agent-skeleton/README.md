# Agent 骨架（Python）

一个能直接跑起来、也方便往里填业务逻辑的 Agent 骨架。

- **多模型**：GLM（智谱）、DeepSeek 开箱可用，另内置 Moonshot、通义千问、OpenAI、本地 Ollama；换厂商只改一个参数
- **内置 web_search**：Tavily / Bing / SearXNG / DuckDuckGo 四后端，自动降级，**没有 key 也能用**
- **接入 MCP**：stdio / SSE / streamable-http 三种传输，MCP 工具自动注册成模型可调用的工具
- **透明的执行过程**：`on_event` 回调可拿到每一步的思考、工具调用、耗时，方便接 CLI / Web UI
- **无框架依赖**：不依赖 LangChain 之类重型库，核心代码约 800 行，改起来没有黑盒

---

## 目录结构

```
agent-skeleton/
├── main.py                     # 命令行入口（单轮问答 + 多轮对话）
├── demo_mcp_server.py          # 自带的示例 MCP server，用于验证接入链路
├── requirements.txt
├── .env.example                # 复制为 .env 填密钥
├── mcp_servers.example.json    # 复制为 mcp_servers.json 后自动加载
├── agent/
│   ├── agent.py                # ★ 主循环：模型 -> 工具 -> 模型
│   ├── llm/
│   │   ├── base.py             # Message / ToolCall / LLMResponse / BaseLLM
│   │   ├── openai_compat.py    # OpenAI 兼容协议实现（GLM、DeepSeek 共用）
│   │   └── registry.py         # ★ 提供方注册表，加厂商改这里
│   ├── tools/
│   │   ├── base.py             # Tool / FunctionTool / ToolRegistry / @tool
│   │   ├── web_search.py       # ★ 内置搜索，四后端自动降级
│   │   └── read_url.py         # 抓网页正文（带 SSRF 防护）
│   └── mcp/
│       └── client.py           # ★ MCP 连接管理 + 工具适配
├── examples/
│   └── custom_tool.py          # 自定义工具的三种写法
└── scripts/
    └── smoke_test.py           # 离线冒烟测试（不花钱、不联网）
```

三个 ★ 是最常改的文件：`agent.py` 改循环逻辑，`llm/registry.py` 加模型，`tools/` 加能力。

---

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 DEEPSEEK_API_KEY 或 GLM_API_KEY

python main.py -p deepseek -q "搜一下 DeepSeek 最新的模型，并总结要点"
python main.py -p glm                       # 不带 -q 进入多轮对话
python main.py --list-providers            # 看支持哪些厂商
```

代码里用：

```python
import asyncio
from agent import Agent, build_default_registry, create_llm

async def main():
    llm = create_llm("glm", model="glm-4.6")     # 或 "deepseek"
    agent = Agent(llm, build_default_registry(), max_steps=6)
    result = await agent.run("今天杭州适合骑行吗？")
    print(result.answer)
    await agent.tools.aclose()
    await llm.aclose()

asyncio.run(main())
```

---

## 多模型支持

所有提供方都是 OpenAI 兼容协议（`/chat/completions` + `tools` 字段），因此共用同一个客户端实现，差异只在 `agent/llm/registry.py` 的一张表：

| provider | 默认模型 | 密钥环境变量 | Base URL |
|---|---|---|---|
| `glm` | `glm-4.6` | `GLM_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| `moonshot` | `kimi-k2-0905-preview` | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` |
| `qwen` | `qwen-plus` | `DASHSCOPE_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `ollama` | `qwen3:8b` | 可留空 | `http://localhost:11434/v1` |

**加一个新厂商**：在 `PROVIDERS` 里加一条 `ProviderSpec` 即可（若接口不是 OpenAI 兼容，比如 Anthropic Messages API，就在 `agent/llm/` 下再写一个 `BaseLLM` 子类）。

**模型选型提示**：
- 需要稳定工具调用 → `deepseek-chat`、`glm-4.6`、`glm-4.5-air`
- `deepseek-reasoner` 偏长篇推理，工具调用支持不稳定，骨架里保留了 `no_tool_models` 开关可关掉工具
- 结构化输出要求高时把 `temperature` 调到 0.1～0.2

---

## 工具系统

内置两个：

| 工具 | 作用 | 说明 |
|---|---|---|
| `web_search` | 联网搜索 | 返回标题 / 链接 / 摘要；后端自动降级 |
| `read_url` | 抓网页正文 | 去标签，最多 6000 字符；带内网地址防护 |

**搜索后端选择**（`WebSearchTool(backend="auto")` 的规则）：

```
TAVILY_API_KEY        → Tavily（推荐：有 answer 字段，摘要质量最好）
BING_SEARCH_API_KEY   → Bing Web Search v7
SEARXNG_URL           → 自建 SearXNG
以上都没有            → DuckDuckGo（免费，可能被限流）
某个后端失败          → 自动换下一个，最终全失败才报错给模型
```

**加自己的工具**，三种写法见 `examples/custom_tool.py`：

```python
from agent import tool

@tool(name="get_weather", description="查询城市天气",
      parameters={"type": "object",
                  "properties": {"city": {"type": "string"}},
                  "required": ["city"]})
async def get_weather(city: str) -> str:
    return f"{city}：晴，24℃"
```

要点：工具返回**字符串**（模型能读的文本）；抛 `ToolError` 会被转成错误文本回给模型，让它自己换策略，不会打断循环；工具结果超长会自动截断（默认 8000 字符）。

---

## 接入 MCP

MCP server 的工具会在启动时被拉取并注册进来，模型调用它们时和本地工具没有区别（名字默认带 server 前缀，如 `demo_add_numbers`，避免多 server 重名）。

### 1. 装 SDK

```bash
pip install mcp
```

### 2. 写配置（复制 `mcp_servers.example.json` 为 `mcp_servers.json`）

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "remote": {
      "url": "https://your-mcp-server.example.com/mcp",
      "transport": "http",
      "headers": {"Authorization": "Bearer ${MCP_REMOTE_TOKEN}"}
    }
  }
}
```

- `command` + `args` → **stdio**（本地进程）
- `url` + `transport: "sse"` → **SSE**（老式远程）
- `url` + `transport: "http"` → **streamable-http**（新式远程，默认）
- 配置里的 `${VAR}` 会从环境变量展开，密钥不用写进文件

### 3. 跑

```bash
python main.py -p glm                   # 自动加载当前目录的 mcp_servers.json
python main.py -p glm --mcp ./conf/x.json
python main.py -p glm --no-mcp          # 临时关掉
```

代码里手动控制：

```python
from agent import MCPManager, ToolRegistry, load_servers

registry = ToolRegistry()
manager = MCPManager(load_servers("mcp_servers.json"), registry, prefix_with_server=True)
stats = await manager.connect_all()
print(stats.servers_connected, stats.tools_registered, stats.errors)
# ...用完
await manager.aclose()
```

### 注意

- 单个 server 连接失败只会在 `stats.errors` 里记录，不会让整个 Agent 起不来
- MCP session 与事件循环绑定，`connect_all()` 和 `aclose()` 要在同一个 asyncio 任务里调用
- MCP 返回的图片内容当前会被降级成占位文本（模型侧若支持图片可自行处理 `content` 块）

---

## 执行流程

```
        ┌──────────────────────────────────────────┐
user ──▶│  messages + tools schema → LLM           │
        └──────────────────────────────────────────┘
                       │
        有 tool_calls  │  无 tool_calls
                       ▼                 ▼
        ┌───────────────────────┐   ┌──────────┐
        │ 并发执行（gather）     │   │ 最终答案 │
        │ 结果转文本回填 messages│   └──────────┘
        └───────────────────────┘
                       │
                       └──── 回到第一步（最多 max_steps 轮）
```

`on_event` 会依次抛出 `user / step / assistant / tool_call / tool_result / final`（异常时还有 `error / max_steps`），`main.py` 里的 `make_printer()` 就是一个最小实现，接 Web UI 时换成往 SSE 推即可。

---

## 方便继续扩展的地方

| 想做的事 | 改哪里 |
|---|---|
| 流式输出 | `OpenAICompatLLM.chat` 加 `stream=True` 分支，`Agent.run` 里改成增量 yield |
| 短期记忆 / 摘要压缩 | `Agent._trim_history`，现在是按 `max_history_messages` 截断，可换成 LLM 摘要 |
| 长期记忆 / RAG | 加一个 `Tool` 做检索，或加在 `Agent.run` 前做 retrieve 注入 |
| 多 Agent / 子任务 | 新建一个 `Agent` 实例并包成 `Tool` 注册进去 |
| 结构化输出 | 给 `chat()` 传 `response_format`，或在最终答案上加 JSON 校验重试 |
| 成本控制 | `AgentResult.usage` 已累加 token，可在循环里加预算上限 |

---

## 已知限制与安全提醒

骨架把「能跑通」放在第一位，上生产前建议补齐：

1. **Prompt 注入**：`web_search` / `read_url` / MCP 返回的都是不可信内容，会把外部文本喂给模型。现在靠 system prompt 约束，正式场景应加一层「工具结果不视为指令」的清洗或隔离。
2. **工具权限**：MCP filesystem 这类工具能读写真实文件，务必限制目录、用容器或只读挂载；本骨架不做沙箱。
3. **密钥管理**：只从环境变量 / `.env` 读，不要写进代码；日志里不要打印请求体（当前只打印错误摘要）。
4. **SSRF**：`read_url` 已拦 file://、内网/回环地址，但如果走公司代理，代理侧也要有出口白名单。
5. **重试与限流**：模型侧对 429/5xx 有 3 次指数退避重试；搜索侧没有，高频调用建议自己加限流。
6. **历史长度**：多轮对话按消息条数截断（默认 40 条），长文本场景建议改成按 token 估算。

---

## 测试

```bash
python scripts/smoke_test.py          # 主循环 + 工具报错 + 模型工厂（不联网、不花钱）
python scripts/smoke_test.py --mcp    # 额外拉起 demo_mcp_server.py 验证 MCP 全链路
```

smoke test 用一个脚本化的 `FakeLLM` 替代真实模型，所以不需要任何 API Key。
