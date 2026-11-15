"""搜索源自检脚本：逐项打印连通性、耗时和失败原因。

用法（在项目根目录）：
    python scripts/check_search.py
    python scripts/check_search.py "今天西安天气"
    python scripts/check_search.py --backend baidu

输出示例：
    [ok  ] bing      0.31s  3 条
    [fail] baidu    0.55s  百度触发人机验证（wappass 验证页）

据此判断：是网络把域名屏蔽了、还是百度反爬拦了。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import WebSearchTool  # noqa: E402


async def amain(query: str, backend: str) -> int:
    tool = WebSearchTool(backend=backend)
    print(f"搜索源：{' -> '.join(tool.backend_chain())}\n")

    exit_code = 0
    try:
        for probe in await tool.diagnose(query):
            if probe.ok:
                print(f"  [ok  ] {probe.backend:<8} {probe.elapsed:6.2f}s  {probe.count} 条")
            else:
                label = tool.BACKEND_LABELS.get(probe.backend, probe.backend)
                print(f"  [fail] {probe.backend:<8} {probe.elapsed:6.2f}s  {probe.error}")
                print(f"         ({label})")

        print("\n实际调用（run 按顺序降级，返回第一个成功的结果）：")
        try:
            out = await tool.run(query, max_results=3)
            print("  成功，返回前 200 字：")
            print("  " + out[:200].replace("\n", "\n  "))
        except Exception as e:  # noqa: BLE001
            print(f"  失败：{e}")
            exit_code = 1
    finally:
        await tool.aclose()
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Python 3.13 新特性")
    parser.add_argument("--backend", default="auto", choices=["auto", "bing", "baidu"])
    args = parser.parse_args()
    raise SystemExit(asyncio.run(amain(args.query, args.backend)))
#（注：内容由AI生成）
