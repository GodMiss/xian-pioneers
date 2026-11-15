"""OpenAI 兼容协议的模型客户端。

智谱 GLM、DeepSeek、Moonshot、通义千问（兼容模式）、OpenAI、本地 Ollama
都提供 /v1/chat/completions 接口，且 function calling 结构一致，
因此用同一个实现即可覆盖，差异收敛到 registry.py 里的 ProviderSpec。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence

import httpx

from .base import BaseLLM, LLMError, Message, LLMResponse, parse_tool_call

log = logging.getLogger(__name__)

# 这些状态码值得重试：限流 / 网关抖动 / 上游超时
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_temperature: float | None = 0.3,
        default_max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        supports_tools: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError(f"[{provider}] base_url 不能为空")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.supports_tools = supports_tools
        self.extra_headers = extra_headers or {}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Ollama 等本地服务不校验 key，但也允许为空
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "Bearer not-needed",
        }
        headers.update(self.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "stream": False,
        }
        temp = self.default_temperature if temperature is None else temperature
        if temp is not None:
            payload["temperature"] = temp
        tokens = self.default_max_tokens if max_tokens is None else max_tokens
        if tokens is not None:
            payload["max_tokens"] = tokens
        if tools and self.supports_tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        payload.update(extra)
        return payload

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, tools, temperature, max_tokens, kwargs)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.post(url, headers=self._headers(), json=payload)
            except httpx.HTTPError as e:  # 网络层异常
                last_error = e
                if attempt == self.max_retries:
                    raise LLMError(f"请求失败：{e!r}", provider=self.provider) from e
                await self._backoff(attempt)
                continue

            if resp.status_code == 200:
                return self._parse(resp)

            body = resp.text[:2000]
            # 鉴权/参数错误重试没有意义，直接抛出
            if resp.status_code not in RETRYABLE_STATUS:
                raise LLMError(
                    f"HTTP {resp.status_code}：{_extract_error(body)}",
                    provider=self.provider,
                    status_code=resp.status_code,
                    body=body,
                )
            last_error = LLMError(
                f"HTTP {resp.status_code}：{_extract_error(body)}",
                provider=self.provider,
                status_code=resp.status_code,
                body=body,
            )
            if attempt == self.max_retries:
                break
            log.warning("%s 返回 %s，第 %s 次重试", self.provider, resp.status_code, attempt)
            await self._backoff(attempt)

        raise last_error or LLMError("未知错误", provider=self.provider)

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(2 ** (attempt - 1), 8))

    # ------------------------------------------------------------------ #

    def _parse(self, resp: httpx.Response) -> LLMResponse:
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError("返回体不是合法 JSON", provider=self.provider, body=resp.text[:2000]) from e

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"返回体缺少 choices：{resp.text[:500]}", provider=self.provider)

        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # 部分实现返回 content 数组
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        tool_calls = [parse_tool_call(tc) for tc in (msg.get("tool_calls") or [])]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            model=data.get("model") or self.model,
            raw=data,
        )

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()


def _extract_error(body: str) -> str:
    """尽量从错误响应里捞出人话。"""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:300] or "空响应"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if isinstance(err, str):
            return err
        for key in ("message", "msg", "detail"):
            if data.get(key):
                return str(data[key])
    return json.dumps(data, ensure_ascii=False)[:300]
