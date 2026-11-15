#!/usr/bin/env python3
"""诊断评测结果：三种策略准确率为什么一样？

用法：
    python scripts/analyze_eval.py results/eval_20261115_143000.json

它只读结果文件，不发请求、不花钱。回答四个问题：
  1. 三种策略是不是真的产生了不同的输出？（排除脚本 bug）
  2. 哪些题三种策略都对、哪些都错、哪些只有部分对？
  3. CoT 到底有没有在推理？（看输出里有没有推理痕迹）
  4. 准确率相同，是题集造成的，还是代码造成的？
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

STRATEGY_LABELS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "cot": "CoT",
}
REASON_MARKERS = ("分析", "推理", "步骤", "因为", "所以", "先", "然后", "计算")


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="评测结果诊断")
    ap.add_argument("result", help="eval_*.json 路径")
    args = ap.parse_args()

    data = load(args.result)
    trials = data.get("trials") or []
    if not trials:
        print("结果文件里没有 trials，无法分析。", file=sys.stderr)
        return 2

    meta = data.get("meta") or {}
    strategies = [s for s in STRATEGY_LABELS if any(t["strategy"] == s for t in trials)]
    items: dict[str, dict[str, dict]] = defaultdict(dict)
    for t in trials:
        if t.get("repeat", 0) != 0:  # repeat>1 时只看第一轮，避免混淆
            continue
        items[t["item_id"]][t["strategy"]] = t

    print(f"结果文件：{args.result}")
    print(f"模型：{meta.get('provider')} / {meta.get('model')}    "
          f"温度：{meta.get('temperature')}    题数：{len(items)}    "
          f"max_tokens：{meta.get('max_tokens')}")
    if meta.get("mock"):
        print("⚠ 这是 MOCK 数据，下面的分析只用于演示，不代表真实结论。")
    print()

    # ---------------- 1. 输出是否真的不同 ----------------
    identical = 0
    compared = 0
    for iid, per in items.items():
        raws = [per[s]["raw"] for s in strategies if s in per and per[s].get("raw")]
        if len(raws) >= 2:
            compared += 1
            if len(set(raws)) == 1:
                identical += 1
    print("【1】三种策略的输出是否真的不同")
    if compared and identical == compared:
        print(f"  ✗ 全部 {compared} 题在三策略下输出完全相同 → 大概率没真正切换提示词/命中了缓存")
    else:
        print(f"  ✓ {compared - identical}/{compared} 题的输出在三策略间存在差异（正常）")

    # ---------------- 2. 对错分布 ----------------
    all_correct, all_wrong, sensitive = [], [], []
    for iid, per in sorted(items.items(), key=lambda kv: (len(kv[0]), kv[0])):
        flags = {s: bool(per[s].get("correct")) for s in strategies if s in per}
        vals = set(flags.values())
        if vals == {True}:
            all_correct.append(iid)
        elif vals == {False}:
            all_wrong.append(iid)
        else:
            sensitive.append((iid, flags))

    print("\n【2】逐题对错分布")
    header = f"  {'题目':<6}" + "".join(f"{STRATEGY_LABELS[s]:>12}" for s in strategies)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for iid, per in sorted(items.items(), key=lambda kv: (len(kv[0]), kv[0])):
        row = f"  {iid:<6}"
        for s in strategies:
            t = per.get(s)
            if t is None:
                row += f"{'-':>12}"
            elif t.get("error"):
                row += f"{'异常':>12}"
            else:
                row += f"{'✓' if t['correct'] else '✗':>12}"
        print(row)

    print(f"\n  三策略全对：{len(all_correct)} 题  ({', '.join(all_correct) or '无'})")
    print(f"  三策略全错：{len(all_wrong)} 题  ({', '.join(all_wrong) or '无'})")
    print(f"  策略敏感（对错不一致）：{len(sensitive)} 题")

    # ---------------- 3. 全错题的共性 ----------------
    if all_wrong:
        print("\n【3】三策略都错的题——这是天花板所在，换提示词救不回来")
        for iid in all_wrong:
            per = items[iid]
            any_t = next(iter(per.values()))
            print(f"\n  {iid}  标准答案：{any_t.get('gold')}")
            for s in strategies:
                t = per.get(s)
                if not t:
                    continue
                raw = (t.get("raw") or "").replace("\n", " ")[:110]
                verdict = "解析失败" if not t.get("parsed") else "答错"
                print(f"    {STRATEGY_LABELS[s]:<10} {verdict}：{raw}")

    # ---------------- 4. CoT 有没有真推理 ----------------
    print("\n【4】各策略输出特征")
    for s in strategies:
        rows = [t for t in trials if t["strategy"] == s and not t.get("error")]
        if not rows:
            continue
        lens = [len(t.get("raw") or "") for t in rows]
        tokens = [t.get("tokens") or 0 for t in rows]
        with_reason = sum(1 for t in rows if any(m in (t.get("raw") or "") for m in REASON_MARKERS))
        print(f"  {STRATEGY_LABELS[s]:<10} 输出字符中位数 {statistics.median(lens):>5.0f}   "
              f"平均 tokens {statistics.mean(tokens):>6.0f}   "
              f"含推理痕迹 {with_reason}/{len(rows)}")
    if "cot" in strategies:
        cot_rows = [t for t in trials if t["strategy"] == "cot" and not t.get("error")]
        if cot_rows and all(not any(m in (t.get("raw") or "") for m in REASON_MARKERS) for t in cot_rows):
            print("  ⚠ CoT 输出里没有任何推理痕迹，说明模型忽略了 CoT 指令，或 max_tokens 太小被截断")

    # ---------------- 5. 结论 ----------------
    s = next((x for x in data.get("summary") or []), None)
    accs = {x["strategy"]: x["accuracy"] for x in (data.get("summary") or [])}
    print("\n【5】结论")
    if accs and len(set(round(a, 4) for a in accs.values())) == 1:
        if not sensitive:
            print(f"  三策略准确率完全相同（{list(accs.values())[0]:.1%}），且没有任何一题对策略敏感。")
            print(f"  → 这批 {len(items)} 题里，{len(all_correct)} 题三策略都对、{len(all_wrong)} 题三策略都错。")
            print("     说明题目考的是「知不知道」而不是「会不会推」，提示词改变不了答案。")
            print("     这是题集性质导致的，不是脚本 bug。要么换推理型题目，要么扩大题量。")
        else:
            print(f"  三策略准确率相同（{list(accs.values())[0]:.1%}），但有 {len(sensitive)} 题对策略敏感。")
            print("  → 各策略在多题上互有胜负，正好抵消。样本太小时很常见（12 题差 1 题 = 8.3%）。")
            print("     建议 --repeat 3 看稳定性，并把题量提到 100 以上再比较。")
    elif accs:
        best = max(accs, key=accs.get)
        print(f"  最优策略：{STRATEGY_LABELS.get(best, best)}（{accs[best]:.1%}），"
              f"最差 {min(accs.values()):.1%}，差 {(max(accs.values()) - min(accs.values())) * 100:.1f} 个百分点。")
        if len(items) < 50:
            print(f"  ⚠ 只有 {len(items)} 题，差 1 题就是 {100 / len(items):.1f} 个百分点，别据此下结论。")
    else:
        print("  结果文件里没有 summary。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#（注：内容由AI生成）
