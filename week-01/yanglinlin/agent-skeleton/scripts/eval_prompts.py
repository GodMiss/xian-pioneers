#!/usr/bin/env python3
"""提示策略对比评测：Zero-Shot / Few-Shot / CoT 在同一测试集上的准确率。

用法：
    # 真实模型
    python scripts/eval_prompts.py --data data/sample_testset.jsonl \
        --examples data/fewshot_examples.jsonl --k 3 -p deepseek

    # 不花钱跑通流程（假模型，结果无意义，只验证管道）
    python scripts/eval_prompts.py --data data/sample_testset.jsonl --mock

公平性控制（脚本层保证）：
  1. 三个策略用同一个模型、同一 temperature（默认 0）、同一测试集；
  2. Few-Shot 的示例只从一个独立文件读，若示例与测试集 id/题干重复会直接报警；
  3. 判分逻辑与提示策略无关，统一走 agent/prompts.py 的 parse_answer + score。

输出：
  results/eval_<时间戳>.json   全量明细
  results/eval_<时间戳>.csv    逐题结果（方便 Excel 透视）
  results/eval_<时间戳>.md     对比表 + 差异题 + 结论
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import BaseLLM, LLMError, LLMResponse, create_llm, list_providers
from agent.prompts import (
    STRATEGY_LABELS,
    STRATEGY_NAMES,
    ExamItem,
    answer_display,
    build_messages,
    parse_answer,
    parse_item,
    score,
)


# --------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------- #

@dataclass
class Trial:
    strategy: str
    item_id: str
    repeat: int
    pred: str
    gold: str
    correct: bool
    parsed: bool
    latency: float
    tokens: int = 0
    error: str = ""
    raw: str = field(default="", repr=False)


@dataclass
class StrategySummary:
    strategy: str
    total: int
    correct: int
    parse_failed: int
    errors: int
    accuracy: float
    avg_latency: float
    avg_tokens: float
    trials: list[Trial] = field(default_factory=list)


# --------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------- #

def load_items(path: str) -> list[ExamItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到测试集：{path}")
    ext = os.path.splitext(path)[1].lower()

    rows: list[dict[str, Any]] = []
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("items", [])
    else:
        raise ValueError(f"不支持的测试集格式：{ext}（请用 .jsonl / .json / .csv）")

    if not rows:
        raise ValueError(f"测试集为空：{path}")
    return [parse_item(r, i) for i, r in enumerate(rows)]


def check_leakage(test_items: Sequence[ExamItem], examples: Sequence[ExamItem]) -> list[str]:
    """Few-Shot 示例不能来自测试集，否则准确率会被示例直接泄题抬高。"""
    problems: list[str] = []
    test_ids = {i.id for i in test_items}
    test_qs = {i.question.strip() for i in test_items}
    for ex in examples:
        if ex.id in test_ids:
            problems.append(f"示例 id={ex.id} 与测试集重复")
        elif ex.question.strip() in test_qs:
            problems.append(f"示例题干与测试集重复：{ex.question[:30]}")
    return problems


# --------------------------------------------------------------------- #
# Mock 模型（离线验证流程用）
# --------------------------------------------------------------------- #

class MockLLM(BaseLLM):
    """假模型：按题面哈希决定答对与否，答案是测试集里的真实标准答案。

    用途只有一个：在没有 API Key 的情况下验证「提示词构造 -> 调用 -> 解析 -> 判分 -> 汇总」
    整条链路是通的。准确率数字是编的（zero_shot 1/3、few_shot 2/3、cot 3/3），
    千万不要当成真实结论。
    """

    provider = "mock"
    model = "mock-model"

    def __init__(self, items: Sequence[ExamItem]) -> None:
        self._by_prompt = {item.render(): item for item in items}

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None, **kwargs):
        system = messages[0].content if messages else ""
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content
                break

        if "答案：<最终答案>" in system:            # CoT：模拟"先推理后作答"
            correct_rate = 3
            prefix = "分析：这道题考的是基础知识。\n推理：先回忆定义，再对照选项。\n"
        elif len(messages) > 2:                     # Few-Shot
            correct_rate = 2
            prefix = ""
        else:                                       # Zero-Shot
            correct_rate = 1
            prefix = ""

        item = self._by_prompt.get(user_text)
        seed = abs(hash(user_text)) % 3
        if item is not None and seed < correct_rate:
            content = prefix + "答案：" + _mock_gold(item)
        else:
            content = prefix + "答案：不确定"
        await asyncio.sleep(0.01)
        return LLMResponse(
            content=content,
            usage={"total_tokens": 60, "prompt_tokens": 40, "completion_tokens": 20},
        )


def _mock_gold(item: ExamItem) -> str:
    """把标准答案转成模型该输出的形式（选择题给字母，简答题给文本）。"""
    gold = item.gold_list[0] if item.gold_list else ""
    if item.kind == "choice" and item.options:
        if len(gold) == 1 and gold.upper() in "ABCD":
            return gold.upper()
        for i, opt in enumerate(item.options):
            if opt.strip() == gold.strip():
                return chr(ord("A") + i)
    return gold


# --------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------- #

async def run_trial(
    llm: BaseLLM,
    strategy: str,
    item: ExamItem,
    examples: Sequence[ExamItem],
    k: int,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    progress: dict[str, int],
) -> Trial:
    messages = build_messages(strategy, item, examples, k)
    start = time.perf_counter()
    error = ""
    content = ""
    tokens = 0
    finish_reason = ""

    async with semaphore:
        try:
            resp = await asyncio.wait_for(
                llm.chat(
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
                timeout=args.timeout,
            )
            content = resp.content or ""
            finish_reason = resp.finish_reason or ""
            tokens = int((resp.usage or {}).get("total_tokens") or 0)
        except asyncio.TimeoutError:
            error = f"超时 >{args.timeout}s"
        except LLMError as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"

    latency = time.perf_counter() - start
    pred, parsed = parse_answer(content, item) if not error else ("", False)
    if not error and not parsed and finish_reason == "length":
        error = "输出被截断（finish_reason=length），请调大 --max-tokens 或改用更短的推理要求"
    correct = (not error) and parsed and score(pred, item, args.match)
    progress["done"] += 1
    print(
        f"\r  [{progress['done']}/{progress['total']}] {strategy} {item.id} "
        f"{'✓' if correct else '✗'}",
        end="",
        file=sys.stderr,
    )

    return Trial(
        strategy=strategy,
        item_id=item.id,
        repeat=progress["repeat"],
        pred=pred,
        gold=answer_display(item),
        correct=correct,
        parsed=parsed,
        latency=latency,
        tokens=tokens,
        error=error,
        raw=content[:500],
    )


async def eval_strategy(
    llm: BaseLLM,
    strategy: str,
    items: Sequence[ExamItem],
    examples: Sequence[ExamItem],
    k: int,
    args: argparse.Namespace,
    progress: dict[str, int],
) -> StrategySummary:
    semaphore = asyncio.Semaphore(args.concurrency)
    trials: list[Trial] = []

    for rep in range(args.repeat):
        progress["repeat"] = rep
        tasks = [
            run_trial(llm, strategy, item, examples, k, args, semaphore, progress)
            for item in items
        ]
        trials.extend(await asyncio.gather(*tasks))
    print("", file=sys.stderr)

    total = len(trials)
    correct = sum(1 for t in trials if t.correct)
    parse_failed = sum(1 for t in trials if not t.parsed and not t.error)
    errors = sum(1 for t in trials if t.error)
    valid = total - errors
    return StrategySummary(
        strategy=strategy,
        total=total,
        correct=correct,
        parse_failed=parse_failed,
        errors=errors,
        accuracy=(correct / valid) if valid else 0.0,
        avg_latency=statistics.mean(t.latency for t in trials) if trials else 0.0,
        avg_tokens=statistics.mean(t.tokens for t in trials) if trials else 0.0,
        trials=trials,
    )


# --------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------- #

def build_report(summaries: list[StrategySummary], items: Sequence[ExamItem], args) -> str:
    lines = [
        "# 提示策略准确率对比",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：{args.provider} / {args.model or '默认'}",
        f"- 测试集：{args.data}（{len(items)} 题）"
        + (f"，每题重复 {args.repeat} 次" if args.repeat > 1 else ""),
        f"- Few-Shot：k={args.k}，示例来自 {args.examples or '未提供'}",
        f"- 解码参数：temperature={args.temperature}，max_tokens={args.max_tokens}",
        "",
        "## 汇总",
        "",
        "| 策略 | 准确率 | 正确/有效 | 解析失败 | 调用异常 | 平均耗时(s) | 平均 tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        valid = s.total - s.errors
        lines.append(
            f"| {STRATEGY_LABELS.get(s.strategy, s.strategy)} | **{s.accuracy:.1%}** | "
            f"{s.correct}/{valid} | {s.parse_failed} | {s.errors} | "
            f"{s.avg_latency:.2f} | {s.avg_tokens:.0f} |"
        )

    # 每题横向对比，只有策略间结果不一致的题才列出来（差异题）
    by_item: dict[str, dict[str, bool]] = {}
    for s in summaries:
        # 多次重复时按多数票归档，避免"偶尔蒙对"被记成正确
        for t in s.trials:
            by_item.setdefault(t.item_id, {}).setdefault(s.strategy, [])
            by_item[t.item_id][s.strategy].append(t.correct)
        for iid, row in by_item.items():
            if s.strategy in row and isinstance(row[s.strategy], list):
                votes = row[s.strategy]
                row[s.strategy] = sum(votes) * 2 > len(votes)

    diff_items = [iid for iid, m in by_item.items() if len(set(m.values())) > 1]
    if diff_items:
        lines += [
            "",
            "## 策略间结果不一致的题目",
            "",
            "| 题目 | 标准答案 | " + " | ".join(STRATEGY_LABELS[s.strategy] for s in summaries) + " |",
            "|---|---|" + "---|" * len(summaries),
        ]
        item_map = {i.id: i for i in items}
        for iid in diff_items:
            item = item_map.get(iid)
            q = (item.question[:40] + "…") if item and len(item.question) > 40 else (item.question if item else "")
            marks = " | ".join("✓" if by_item[iid].get(s.strategy) else "✗" for s in summaries)
            lines.append(f"| {iid} {q} | {answer_display(item) if item else ''} | {marks} |")

    # 失败样本（解析失败或调用异常）
    bad = [(s.strategy, t) for s in summaries for t in s.trials if t.error or not t.parsed]
    if bad:
        lines += ["", "## 需要人工抽查的样本（解析失败 / 调用异常）", ""]
        for strategy, t in bad[:12]:
            label = STRATEGY_LABELS.get(strategy, strategy)
            reason = t.error or "未解析出答案"
            lines.append(f"- `{label}` / `{t.item_id}`：{reason}；模型原始输出：{t.raw[:160]!r}")
    return "\n".join(lines) + "\n"


def write_outputs(summaries: list[StrategySummary], items, args, out_dir: str) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"eval_{stamp}")

    detail = {
        "meta": {
            "time": datetime.now().isoformat(timespec="seconds"),
            "provider": args.provider,
            "model": args.model,
            "data": args.data,
            "examples": args.examples,
            "k": args.k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "match": args.match,
            "mock": args.mock,
            "items": len(items),
        },
        "summary": [
            {k: v for k, v in asdict(s).items() if k != "trials"} for s in summaries
        ],
        "summary_mock_note": "mock=true 时数据为虚构，仅用于验证流程" if args.mock else "",
        "trials": [asdict(t) for s in summaries for t in s.trials],
    }
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    csv_path = base + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["strategy", "item_id", "repeat", "correct", "parsed", "pred", "gold",
             "latency_s", "total_tokens", "error"]
        )
        for s in summaries:
            for t in s.trials:
                writer.writerow(
                    [t.strategy, t.item_id, t.repeat, int(t.correct), int(t.parsed),
                     t.pred, t.gold, f"{t.latency:.2f}", t.tokens, t.error]
                )

    md_path = base + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(summaries, items, args))

    return json_path, csv_path, md_path


# --------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="对比 Zero-Shot / Few-Shot / CoT 在同一测试集上的准确率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", default="data/sample_testset.jsonl", help="测试集文件（jsonl/json/csv）")
    p.add_argument("--examples", default=None, help="Few-Shot 示例文件（必须与测试集不重叠）")
    p.add_argument("--k", type=int, default=3, help="Few-Shot 示例条数，默认 3")
    p.add_argument("--strategies", default=",".join(STRATEGY_NAMES), help="要跑的策略，逗号分隔")
    p.add_argument("-p", "--provider", default=os.getenv("AGENT_PROVIDER", "deepseek"))
    p.add_argument("-m", "--model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--temperature", type=float, default=0.0, help="评测建议固定 0")
    p.add_argument("--max-tokens", type=int, default=1024, help="CoT 需要更大值，过小会被截断")
    p.add_argument("--timeout", type=float, default=90.0, help="单题超时（秒）")
    p.add_argument("--concurrency", type=int, default=4, help="并发请求数")
    p.add_argument("--repeat", type=int, default=1, help="每题重复次数（>1 可看稳定性）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    p.add_argument("--match", default="auto", choices=["auto", "exact", "contains", "choice"],
                   help="判分方式；auto 时选择题比字母、简答题精确匹配")
    p.add_argument("--out", default="results", help="结果输出目录")
    p.add_argument("--mock", action="store_true", help="用假模型跑通流程（结果无意义）")
    p.add_argument("--list-providers", action="store_true")
    return p


async def amain(args: argparse.Namespace) -> int:
    if args.list_providers:
        print(list_providers())
        return 0

    items = load_items(args.data)
    if args.limit:
        items = items[: args.limit]

    examples: list[ExamItem] = []
    if args.examples:
        examples = load_items(args.examples)
        problems = check_leakage(items, examples)
        if problems:
            print("检测到 Few-Shot 示例与测试集重叠，这会让结果失真：", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            if not args.mock:
                return 2
    elif "few_shot" in args.strategies:
        print(
            "警告：没有提供 --examples，Few-Shot 将退化成 Zero-Shot（准确率会与 Zero-Shot 接近）。",
            file=sys.stderr,
        )

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGY_NAMES]
    if unknown:
        print(f"未知策略 {unknown}，可选：{', '.join(STRATEGY_NAMES)}", file=sys.stderr)
        return 2

    if args.mock:
        print("!! MOCK 模式：模型输出是假的，只有流程和判分逻辑是真的。\n", file=sys.stderr)
        llm: BaseLLM = MockLLM(items)
        args.provider, args.model = llm.provider, llm.model
    else:
        try:
            llm = create_llm(
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        except ValueError as e:
            print(f"配置错误：{e}", file=sys.stderr)
            return 2
        args.model = llm.model  # 报告里记实际模型名，便于复现
        print(f"模型：{args.provider} / {llm.model}\n", file=sys.stderr)

    print(
        f"测试集 {len(items)} 题，策略 {len(strategies)} 个，共 "
        f"{len(items) * len(strategies) * args.repeat} 次调用\n",
        file=sys.stderr,
    )

    progress = {"done": 0, "total": len(items) * len(strategies) * args.repeat, "repeat": 0}
    summaries: list[StrategySummary] = []
    started = time.perf_counter()
    try:
        for strategy in strategies:
            print(f"→ {STRATEGY_LABELS.get(strategy, strategy)}", file=sys.stderr)
            summaries.append(
                await eval_strategy(llm, strategy, items, examples, args.k, args, progress)
            )
    finally:
        await llm.aclose()

    elapsed = time.perf_counter() - started
    # 控制台汇总表
    print(f"\n{'策略':<12}{'准确率':>9}{'正确/有效':>12}{'解析失败':>10}{'耗时(s)':>10}")
    print("-" * 55)
    for s in summaries:
        valid = s.total - s.errors
        print(
            f"{STRATEGY_LABELS.get(s.strategy, s.strategy):<12}"
            f"{s.accuracy:>8.1%}{f'{s.correct}/{valid}':>12}{s.parse_failed:>10}{s.avg_latency:>10.2f}"
        )

    json_path, csv_path, md_path = write_outputs(summaries, items, args, args.out)
    print(f"\n结果已写入：\n  {json_path}\n  {csv_path}\n  {md_path}\n总耗时 {elapsed:.0f}s")
    if args.mock:
        print("\n提醒：这是 MOCK 结果，只能用来确认评测流程是否正常，不要当作真实结论。")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
