"""内置 read_url 工具：把网页正文抓成纯文本。

配合 web_search 使用：先搜出链接，再打开最相关的一两条读全文。
只做轻量解析（去 script/style/标签），不做 JS 渲染；需要渲染时请在 MCP 侧接
Playwright MCP server。
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from .base import Tool, ToolError, truncate

DEFAULT_MAX_CHARS = 6000
MAX_BYTES = 2 * 1024 * 1024


class _TextExtractor(HTMLParser):
    """把 HTML 压成可读文本，尽量保留段落结构。"""

    SKIP = {"script", "style", "noscript", "svg", "head", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":  # 标题在 head 里，要先于 SKIP 判断
            self._in_title = True
        elif tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip() + " "
            return
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t\u3000]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _is_public_url(url: str) -> bool:
    """挡掉 file://、内网地址等，避免被模型拿来做 SSRF。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


class ReadUrlTool(Tool):
    name = "read_url"
    description = (
        "打开一个网页并返回其正文纯文本（已去除 HTML 标签）。"
        "用于在 web_search 得到链接后获取详细内容，或读取用户给出的网页链接。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要读取的完整网页地址，必须以 http:// 或 https:// 开头"},
            "max_chars": {
                "type": "integer",
                "description": "返回正文的最大字符数，默认 6000，最大 20000",
            },
        },
        "required": ["url"],
    }

    def __init__(self, timeout: float = 20.0, max_chars: int = DEFAULT_MAX_CHARS,
                 client: httpx.AsyncClient | None = None) -> None:
        self.timeout = timeout
        self.max_chars = max_chars
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AgentSkeleton/1.0)",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        self._owns_client = client is None

    async def run(self, url: str, max_chars: int | None = None, **_: Any) -> str:
        if not _is_public_url(url):
            raise ToolError(f"拒绝访问：{url} 不是可公开访问的 http(s) 地址")
        limit = max(500, min(int(max_chars or self.max_chars), 20000))

        try:
            resp = await self._client.get(url)
        except httpx.HTTPError as e:
            raise ToolError(f"打开失败：{e!r}") from e
        if resp.status_code >= 400:
            raise ToolError(f"打开失败：HTTP {resp.status_code}")

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or resp.text.lstrip().startswith("<"):
            parser = _TextExtractor()
            parser.feed(resp.text[:MAX_BYTES])
            body = parser.text()
            head = f"标题：{parser.title.strip()}\nURL：{resp.url}\n\n"
        else:
            body = resp.text
            head = f"URL：{resp.url}\n\n"

        if not body.strip():
            raise ToolError("页面没有可提取的正文，可能需要 JS 渲染或登录")
        return head + truncate(body, limit)

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
