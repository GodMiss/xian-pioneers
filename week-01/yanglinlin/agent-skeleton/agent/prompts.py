"""提示策略与判分。

三种策略只改「提示词怎么组织」，其他变量（模型、温度、测试集、max_tokens）保持一致，
这样准确率差异才能归因到提示策略本身。

    zero_shot : 只给题目 + "直接给答案"
    few_shot  : 题目 + k 个同类示例（示例来自独立文件，绝不从测试集里取）
    cot       : 要求"先逐步推理，最后一行以「答案：」给出结论"

判分：
    choice 题目 -> 抽取选项字母，与标准答案比对
    short  题目 -> 抽取答案文本，归一化后精确匹配（可用 contains 放宽）
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .llm import Message

STRATEGY_NAMES = ("zero_shot", "few_shot", "cot")

STRATEGY_LABELS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "cot": "CoT",
}


# --------------------------------------------------------------------- #
# 题目
# --------------------------------------------------------------------- #

@dataclass
class ExamItem:
    id: str
    question: str
    answer: Any
    options: list[str] = field(default_factory=list)
    kind: Literal["choice", "short"] = "short"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_list(self) -> list[str]:
        """标准答案统一成列表（支持多参考答案）。"""
        if isinstance(self.answer, (list, tuple)):
            return [str(a) for a in self.answer if str(a).strip()]
        if isinstance(self.answer, dict):  # 形如 {"value": ..., "aliases": [...]}
            values = [str(self.answer.get("value", ""))]
            values += [str(a) for a in self.answer.get("aliases", [])]
            return [v for v in values if v.strip()]
        return [str(self.answer)]

    def render(self) -> str:
        """拼成模型看到的题面文本。"""
        text = self.question.strip()
        if self.kind == "choice" and self.options:
            opts = "\n".join(_fmt_option(i, o) for i, o in enumerate(self.options))
            text = f"{text}\n{opts}"
        return text


def _fmt_option(index: int, option: str) -> str:
    letter = chr(ord("A") + index)
    opt = str(option).strip()
    # 选项本身可能已经带 "A. xxx"，避免重复加前缀
    if re.match(rf"^{letter}\s*[.、:：]", opt):
        return opt
    return f"{letter}. {opt}"


def parse_item(raw: dict[str, Any], index: int) -> ExamItem:
    """把一条原始数据（JSONL 一行 / CSV 一行）转成 ExamItem。

    字段别名：
      question: question / prompt / query / 题目
      answer:   answer / answers / gold / label / 答案
      options:  options / choices / 选项（list 或 "A. x|B. y" 字符串）
    """
    question = _first(raw, ("question", "prompt", "query", "题目", "input"))
    if not question:
        raise ValueError(f"第 {index + 1} 条数据缺少 question 字段：{raw}")

    answer = _first(raw, ("answer", "answers", "gold", "label", "答案", "target"))
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        raise ValueError(f"第 {index + 1} 条数据缺少 answer 字段：{raw}")

    options = _first(raw, ("options", "choices", "选项"))
    if isinstance(options, str):
        options = [o for o in re.split(r"[|\n]", options) if o.strip()]
    options = list(options or [])

    declared = str(_first(raw, ("type", "kind", "题型")) or "").lower()
    if declared in ("choice", "choice_question", "选择题"):
        kind = "choice"
    elif declared in ("short", "short_answer", "qa", "问答", "简答"):
        kind = "short"
    else:
        # 自动判断：有选项，或答案是单个字母，就当选择题
        kind = "choice" if (options or _looks_like_letter(answer)) else "short"

    return ExamItem(
        id=str(_first(raw, ("id", "uid", "编号")) or f"q{index + 1}"),
        question=str(question),
        answer=answer,
        options=[str(o) for o in options],
        kind=kind,  # type: ignore[arg-type]
        meta={k: v for k, v in raw.items() if k not in {"question", "answer", "options", "id"}},
    )


def _first(raw: dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in raw and raw[k] not in (None, ""):
            return raw[k]
    return None


def _looks_like_letter(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_looks_like_letter(v) for v in value) and bool(value)
    return bool(re.fullmatch(r"[A-Da-d]", str(value).strip()))


# --------------------------------------------------------------------- #
# 提示词构造
# --------------------------------------------------------------------- #

_BASE_RULES = """你是一个严谨的答题助手。请严格遵守以下要求：
1. 只依据题目本身作答，不要假设题目信息之外的内容。
2. 不要输出与答案无关的寒暄、免责声明或题目复述。"""

_CHOICE_FORMAT = "3. 这是选择题：最终只输出一个选项字母（A / B / C / D），不要输出选项内容。"
_SHORT_FORMAT = "3. 这是简答题：最终只输出答案本身，尽量简短，不要整句解释。"

_COT_RULES = """你是一个严谨的答题助手。请按以下步骤作答：
1. 先分析题目考查的知识点，再一步步推理（推理过程简洁，不超过 8 行）。
2. 检查推理是否有漏洞，如有则纠正。
3. 最后单独一行，严格按 `答案：<最终答案>` 的格式给出结论。"""

_COT_CHOICE_FORMAT = "4. 这是选择题：`答案：` 后面只写一个选项字母（A / B / C / D）。"
_COT_SHORT_FORMAT = "4. 这是简答题：`答案：` 后面只写答案本身，不要解释。"


def system_prompt(strategy: str, kind: str) -> str:
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"未知策略 '{strategy}'，可选：{', '.join(STRATEGY_NAMES)}")

    if strategy == "cot":
        fmt = _COT_CHOICE_FORMAT if kind == "choice" else _COT_SHORT_FORMAT
        return f"{_COT_RULES}\n{fmt}"

    rules = _BASE_RULES
    fmt = _CHOICE_FORMAT if kind == "choice" else _SHORT_FORMAT
    if strategy == "few_shot":
        rules += "\n4. 你会看到若干示例，请严格模仿示例的回答格式与粒度。"
    return f"{rules}\n{fmt}"


def build_messages(
    strategy: str,
    item: ExamItem,
    examples: Sequence[ExamItem] = (),
    k: int = 0,
) -> list[Message]:
    """构造一次完整的 messages（不含历史）。"""
    messages = [Message.system(system_prompt(strategy, item.kind))]

    if strategy == "few_shot":
        for ex in list(examples)[:k]:
            messages.append(Message.user(ex.render()))
            messages.append(Message.assistant(_example_answer(ex)))
        messages.append(Message.user(item.render()))
    elif strategy == "cot":
        messages.append(Message.user(item.render()))
    else:
        messages.append(Message.user(item.render()))
    return messages


def _example_answer(ex: ExamItem) -> str:
    """示例的标准答案展示格式：选择题给字母，简答题给文本。"""
    gold = ex.gold_list[0] if ex.gold_list else ""
    if ex.kind == "choice":
        letter = _extract_letter(gold, ex.options)
        return letter or gold
    return gold


# --------------------------------------------------------------------- #
# 答案解析
# --------------------------------------------------------------------- #

_FW_MAP = {ord(c): ord(c) - 0xFEE0 for c in "０１２３４５６７８９ＡＢＣＤａｂｃｄ：（）"}

_ANSWER_LINE_RE = re.compile(r"答案\s*[:：]\s*(.+)")
_NOTE_LINE_RE = re.compile(
    r"^[（(][^（）()]{0,30}[)）]$"                      # 只有括号备注的一行
    r"|^(?:注|备注|说明|提示|简要回答|回答完毕|解析)\s*[:：]?.*$"
)
_LETTER_PATTERNS = (
    re.compile(r"答案\s*[:：]\s*[（(]?\s*([A-Da-d])\b"),
    re.compile(r"^\s*[（(]?\s*([A-Da-d])\s*[)）.、:：]"),
    re.compile(r"(?:选择|选|应选|正确答案是|答案为)\s*[:：]?\s*[（(]?\s*([A-Da-d])\b"),
    re.compile(r"[（(]\s*([A-Da-d])\s*[)）]"),
)
_TRAILING_PUNCT = "。．.、，,；;！!？?：: \t\"'“”‘’`（）()[]【】<>《》"


def parse_answer(text: str, item: ExamItem) -> tuple[str, bool]:
    """从模型输出里抽取答案。返回 (答案文本, 是否成功解析)。"""
    if not text or not text.strip():
        return "", False
    text = unicodedata.normalize("NFKC", text)

    if item.kind == "choice":
        letter = _extract_letter(text, item.options)
        return (letter, True) if letter else ("", False)

    # 简答题：优先取「答案：」之后；否则去掉备注行后取第一行（回答通常就是这个）
    candidate = ""
    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        candidate = matches[-1]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        kept = [line for line in lines if not _NOTE_LINE_RE.match(line)]
        candidate = kept[0] if kept else (lines[0] if lines else "")
    candidate = _strip_wrapper(candidate)
    return (candidate, bool(candidate))


def _extract_letter(text: str, options: Sequence[str] = ()) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()

    # 1) 各种显式格式
    for pattern in _LETTER_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return m.group(1).upper()

    # 2) 整体就是一个字母
    if re.fullmatch(r"[A-Da-d]", normalized):
        return normalized.upper()

    # 3) 模型直接输出了选项内容 -> 反查字母
    if options:
        for i, opt in enumerate(options):
            opt_text = _strip_option_prefix(str(opt))
            if opt_text and _normalize(opt_text) and _normalize(opt_text) in _normalize(normalized):
                return chr(ord("A") + i)

    # 4) 最后一行里的孤立字母
    last_line = next((l for l in reversed(normalized.splitlines()) if l.strip()), "")
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", last_line)
    if m:
        return m.group(1).upper()
    return ""


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-Da-d]\s*[.、:：)]\s*", "", option.strip())


def _strip_wrapper(text: str) -> str:
    text = text.strip()
    # 去掉 markdown 强调、引号、句末标点
    text = re.sub(r"^\*+(.*?)\*+$", r"\1", text)
    text = text.strip("\"'“”‘’`")
    text = text.strip(_TRAILING_PUNCT)
    # 去掉"最终答案是X"这类前缀
    text = re.sub(r"^(最终)?答案(是|为)?\s*[:：]?\s*", "", text)
    return text.strip()


# --------------------------------------------------------------------- #
# 归一化与判分
# --------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[\s\u3000。．.、，,；;！!？?：:\"'“”‘’`（）()\[\]【】<>《》\-—_/\\|~·*#]+")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _PUNCT_RE.sub("", text)
    for filler in ("的", "是", "了"):
        pass  # 中文虚词不做删除，避免误伤（如"是"本身可能是答案的一部分）
    return text


def _as_number(text: str) -> float | None:
    cleaned = str(text).replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def score(pred: str, item: ExamItem, mode: str = "auto") -> bool:
    """判分。mode: auto / exact / contains / choice"""
    if not pred:
        return False

    golds = item.gold_list
    if item.kind == "choice" or mode == "choice":
        return pred.strip().upper() in {g.strip().upper() for g in golds}

    if mode == "auto":
        mode = "exact"

    pred_n = _normalize(pred)
    for gold in golds:
        gold_n = _normalize(gold)
        if not gold_n:
            continue
        if pred_n == gold_n:
            return True
        # 数值等价：1/1.0、85/85.0、百分之
        p_num, g_num = _as_number(pred), _as_number(gold)
        if p_num is not None and g_num is not None and abs(p_num - g_num) < 1e-9:
            return True
        if mode == "contains" and (gold_n in pred_n or pred_n in gold_n):
            return True
    return False


def answer_display(item: ExamItem) -> str:
    """标准答案在报告里的展示形式。"""
    if item.kind == "choice":
        gold = item.gold_list[0] if item.gold_list else ""
        letter = _extract_letter(gold, item.options) if gold else ""
        if letter:
            idx = ord(letter) - ord("A")
            opt = _strip_option_prefix(item.options[idx]) if 0 <= idx < len(item.options) else ""
            return f"{letter}{('（' + opt + '）') if opt else ''}"
    return " / ".join(item.gold_list)


def iter_items_from_dicts(rows: Iterable[dict[str, Any]]) -> list[ExamItem]:
    return [parse_item(row, i) for i, row in enumerate(rows)]
