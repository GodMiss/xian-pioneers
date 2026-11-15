"""内置 web_search 工具（v3）。

按要求只保留两个搜索源：

  bing   —— 走 cn.bing.com 的 RSS 出口，不需要 key，国内网络可直连。默认后端。
  baidu  —— 走 m.baidu.com 网页端解析，不需要 key；百度反爬较严，可能返回「安全验证」，
            被拦时会明确报错而不是静默失败。需要时显式指定 backend=baidu。

降级顺序：bing -> baidu。

其余后端（Tavily / Bing API / SearXNG / DuckDuckGo）已全部移除。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

from .base import Tool, ToolError

log = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class BackendProbe:
    backend: str
    ok: bool
    elapsed: float
    count: int = 0
    error: str = ""


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "联网搜索互联网信息（Bing / 百度）。当你需要最新的事实、新闻、数据、天气、价格，"
        "或对不确定的信息做核实时应调用。返回结果的标题、链接和摘要；"
        "如果摘要不足以回答，可再用 read_url 打开其中某条链接读全文。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词。用自然语言描述即可，越具体越好；建议带上时间、地点等限定词。",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果条数，1-10，默认 5。",
                "minimum": 1,
                "maximum": 10,
            },
            "backend": {
                "type": "string",
                "description": "搜索源，可选 bing / baidu；不填则 bing 优先、失败自动降级到 baidu。",
                "enum": ["bing", "baidu"],
            },
        },
        "required": ["query"],
    }

    BACKEND_LABELS = {
        "bing": "Bing（cn.bing.com RSS 出口，无需 key）",
        "baidu": "百度（网页端解析，无需 key，可能触发人机验证）",
    }

    def __init__(
        self,
        backend: str = "auto",
        *,
        bing_host: str = "cn.bing.com",
        timeout: float = 15.0,
        max_results: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.backend = backend
        self.bing_host = bing_host
        self.timeout = timeout
        self.max_results = max_results
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        )
        self._owns_client = client is None
        self._baidu_warmed = False

    # ------------------------------------------------------------------ #

    def backend_chain(self) -> list[str]:
        if self.backend != "auto":
            return [self.backend]
        return ["bing", "baidu"]

    async def run(
        self,
        query: str,
        max_results: int | None = None,
        backend: str | None = None,
        freshness: str | None = None,
        **_: Any,
    ) -> str:
        if not query or not query.strip():
            raise ToolError("query 不能为空")
        n = max(1, min(int(max_results or self.max_results), 10))
        freshness = (freshness or "").lower() or None

        chain = [backend] if backend else self.backend_chain()
        unknown = [b for b in chain if b not in ("bing", "baidu")]
        if unknown:
            raise ToolError(f"不支持的搜索源 {unknown}，只支持 bing / baidu")

        errors: list[str] = []
        for name in chain:
            handler = getattr(self, f"_search_{name}", None)
            start = time.perf_counter()
            try:
                results, answer = await asyncio.wait_for(
                    handler(query.strip(), n, freshness), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                errors.append(f"{name}：超时（>{self.timeout}s，可能被网络屏蔽）")
                continue
            except Exception as e:  # noqa: BLE001 - 换下一个源
                msg = str(e).strip() or type(e).__name__
                errors.append(f"{name}：{_short(msg)}")
                log.warning(
                    "web_search 后端 %s 失败：%s（耗时 %.1fs）", name, msg, time.perf_counter() - start
                )
                continue
            if results:
                return self._format(query, name, results, answer)
            errors.append(f"{name}：返回 0 条结果")

        raise ToolError(
            "web_search 全部搜索源失败：\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n建议：默认的 bing 源通常可用；若 bing 被限流，可指定 backend=baidu 或稍后重试。"
            "排查用 `python scripts/check_search.py`。"
        )

    # ------------------------------------------------------------------ #
    # Bing：RSS 出口
    # ------------------------------------------------------------------ #

    async def _search_bing(
        self, query: str, n: int, freshness: str | None
    ) -> tuple[list[SearchResult], str]:
        """Bing RSS：https://cn.bing.com/search?q=...&format=rss

        count 不保证被遵守，返回后自行截断；qft 用于时效过滤。
        """
        params: dict[str, Any] = {"q": query, "format": "rss", "count": n}
        age = {"day": "24h", "week": "168h", "month": "720h", "year": "8760h"}.get(freshness or "")
        if age:
            params["qft"] = f"+filterui:age-lt{age}"

        resp = await self._client.get(f"https://{self.bing_host}/search", params=params)
        if resp.status_code >= 400:
            raise ToolError(f"HTTP {resp.status_code}")
        if not resp.text.strip():
            raise ToolError("返回空响应（可能被限流）")

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            raise ToolError(f"RSS 解析失败：{e}；前 120 字：{resp.text[:120]}") from e

        results: list[SearchResult] = []
        for item in root.findall(".//item")[:n]:
            title = _clean(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            desc = _clean(item.findtext("description") or "")
            if title or link:
                results.append(SearchResult(title=title or link, url=link, snippet=desc))
        if not results:
            raise ToolError("RSS 里没有结果项（可能被限流，或该查询词无结果）")
        return results, ""

    # ------------------------------------------------------------------ #
    # 百度：网页端解析
    # ------------------------------------------------------------------ #

    async def _search_baidu(
        self, query: str, n: int, freshness: str | None
    ) -> tuple[list[SearchResult], str]:
        """百度没有公开的免费搜索 API，这里解析 m.baidu.com 的 HTML。

        百度反爬较严：同一出口 IP 连续请求会跳「安全验证」页。
        所以先访问 baidu.com 拿 BAIDUID 等 cookie，命中验证页时直接报错、不重试
        （重试只会加深封锁）。
        """
        if not self._baidu_warmed:
            try:
                await self._client.get("https://www.baidu.com/", timeout=10)
                self._baidu_warmed = True
            except Exception:  # noqa: BLE001 - 预热失败也继续尝试
                pass

        params: dict[str, Any] = {"word": query, "ie": "utf-8", "rn": str(n)}
        resp = await self._client.get("https://m.baidu.com/s", params=params)

        if "wappass" in str(resp.url) or "安全验证" in resp.text[:3000]:
            raise ToolError(
                "百度触发人机验证（wappass 验证页）。百度网页端对同一出口 IP 连续访问会临时封禁，"
                "建议改用 bing，或过一段时间再试 baidu"
            )
        if resp.status_code >= 400:
            raise ToolError(f"HTTP {resp.status_code}")

        results = _parse_baidu(resp.text, n)
        if not results:
            raise ToolError("百度页面未解析出结果（页面结构可能已变，或返回了验证页/空结果页）")
        return results, ""

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #

    async def diagnose(self, query: str = "Python 3.13 新特性") -> list[BackendProbe]:
        probes: list[BackendProbe] = []
        for name in self.backend_chain():
            start = time.perf_counter()
            handler = getattr(self, f"_search_{name}", None)
            if handler is None:
                probes.append(BackendProbe(name, False, 0.0, error="未知后端"))
                continue
            try:
                results, _ = await asyncio.wait_for(handler(query, 3, None), timeout=self.timeout)
                probes.append(
                    BackendProbe(name, True, time.perf_counter() - start, count=len(results))
                )
            except asyncio.TimeoutError:
                probes.append(
                    BackendProbe(name, False, time.perf_counter() - start, error=f"超时 >{self.timeout}s")
                )
            except Exception as e:  # noqa: BLE001
                probes.append(
                    BackendProbe(
                        name,
                        False,
                        time.perf_counter() - start,
                        error=f"{type(e).__name__}: {_short(str(e))}",
                    )
                )
        return probes

    # ------------------------------------------------------------------ #

    @staticmethod
    def _format(query: str, backend: str, results: list[SearchResult], answer: str) -> str:
        parts = [f'搜索「{query}」的结果（来源：{backend}）：']
        if answer:
            parts.append(f"\n【摘要答案】{answer}")
        for i, r in enumerate(results, 1):
            snippet = (r.snippet or "").replace("\n", " ").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            parts.append(f"\n[{i}] {r.title}\n    URL: {r.url}\n    摘要: {snippet or '（无摘要）'}")
        parts.append(
            "\n\n提示：以上信息来自实时网页，引用时请注明链接；如摘要不足以判断，可用 read_url 打开链接获取正文。"
        )
        return "\n".join(parts)

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------- #
# 解析与去噪
# ---------------------------------------------------------------------- #

_HREF_RE = re.compile(r'<a[^>]+href="([^"]+)"', re.I)
_H3_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S | re.I)


def _parse_baidu(html: str, n: int) -> list[SearchResult]:
    """百度结果页解析。

    页面结构随版本变化，这里不依赖完整 DOM：
    以 class 含 c-container 的位置切块，从每块取 h3 标题、第一个链接、剩余正文当摘要；
    找不到 c-container 时退化为「所有 h3 + 其后片段」。
    """
    anchors = [m.start() for m in re.finditer(r'class="[^"]*c-container', html)]
    results: list[SearchResult] = []
    seen: set[str] = set()

    def push(title: str, url: str, snippet: str) -> None:
        title = _clean(title)
        if not title or title in seen:
            return
        seen.add(title)
        results.append(SearchResult(title=title, url=url, snippet=_clean(snippet)))

    if anchors:
        for idx, pos in enumerate(anchors):
            if len(results) >= n:
                break
            # 切块要从标签的 ">" 之后开始，否则会把 class="..." 残片当正文
            tag_end = html.find(">", pos)
            block_start = tag_end + 1 if tag_end != -1 else pos
            end = anchors[idx + 1] if idx + 1 < len(anchors) else min(len(html), pos + 4000)
            chunk = html[block_start:end]
            h3 = _H3_RE.search(chunk)
            if not h3:
                continue
            href = _HREF_RE.search(chunk)
            title_text = _clean(re.sub(r"<[^>]+>", "", h3.group(1)))
            body = _clean(re.sub(r"<[^>]+>", " ", chunk))
            snippet = body.replace(title_text, "", 1).strip(" |·-—")
            push(h3.group(1), href.group(1) if href else "", snippet[:200])
    else:
        for m in re.finditer(r"(?s)<h3[^>]*>(.*?)</h3>(.{0,600})", html):
            if len(results) >= n:
                break
            href = _HREF_RE.search(m.group(2))
            push(m.group(1), href.group(1) if href else "", re.sub(r"<[^>]+>", " ", m.group(2)))

    return results[:n]


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"<[^>]*$", "", text)  # 切块末尾可能留下半个标签
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return re.sub(r"\s+", " ", text).strip()


def _short(msg: str, limit: int = 120) -> str:
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg[:limit] + ("..." if len(msg) > limit else "")
#（注：内容由AI生成）
