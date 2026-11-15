"""提供方注册表：一行配置就能接一个新模型厂商。

目前内置 GLM（智谱）、DeepSeek、Moonshot、通义千问兼容模式、OpenAI、Ollama。
它们都是 OpenAI 兼容协议，所以共用 OpenAICompatLLM。
如果需要接入非兼容协议（例如 Anthropic Messages API、Gemini），
照着 BaseLLM 再写一个实现，然后在 create_llm 里加一个分支即可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import BaseLLM
from .openai_compat import OpenAICompatLLM


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    base_url: str
    api_key_env: str
    default_model: str
    models: tuple[str, ...] = ()
    # 某些模型（如早期的 DeepSeek reasoner）不支持 function calling，需要关掉工具
    no_tool_models: tuple[str, ...] = ()
    requires_key: bool = True
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "glm": ProviderSpec(
        name="glm",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        default_model="glm-4.6",
        models=("glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4-plus", "glm-4-flash"),
        notes="国内直连、支持 function calling；文档 https://open.bigmodel.cn/dev/api",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        notes="deepseek-chat 支持工具调用；deepseek-reasoner 偏推理、工具调用支持不稳定",
    ),
    "moonshot": ProviderSpec(
        name="moonshot",
        label="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-k2-0905-preview",
        models=("kimi-k2-0905-preview", "moonshot-v1-8k"),
    ),
    "qwen": ProviderSpec(
        name="qwen",
        label="阿里通义千问（兼容模式）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
        models=("qwen-plus", "qwen-max", "qwen-turbo"),
    ),
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"),
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="本地 Ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        default_model="qwen3:8b",
        models=("qwen3:8b", "llama3.1:8b", "deepseek-r1:7b"),
        requires_key=False,
        notes="本地跑，无需联网；模型名用 ollama list 里的实际名字",
    ),
}


def list_providers() -> str:
    lines = []
    for spec in PROVIDERS.values():
        key_hint = spec.api_key_env if spec.requires_key else f"{spec.api_key_env}（可留空）"
        lines.append(
            f"- {spec.name:<9} {spec.label:<16} 默认模型 {spec.default_model:<22} 密钥环境变量 {key_hint}"
        )
    return "\n".join(lines)


def create_llm(
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseLLM:
    """按提供方名字创建模型客户端。

    api_key 解析顺序：显式传入 > 环境变量。
    """
    key = provider.lower().strip()
    if key not in PROVIDERS:
        raise ValueError(f"未知提供方 '{provider}'，可选：{', '.join(PROVIDERS)}")

    spec = PROVIDERS[key]
    model = model or spec.default_model
    base_url = base_url or os.getenv(f"{key.upper()}_BASE_URL") or spec.base_url

    resolved_key = api_key or os.getenv(spec.api_key_env, "").strip()
    if spec.requires_key and not resolved_key:
        raise ValueError(
            f"缺少 {spec.label} 的 API Key：请设置环境变量 {spec.api_key_env}，"
            f"或用 --api-key 传入。"
        )

    supports_tools = model not in spec.no_tool_models

    return OpenAICompatLLM(
        provider=spec.name,
        model=model,
        api_key=resolved_key,
        base_url=base_url,
        supports_tools=supports_tools,
        **kwargs,
    )
