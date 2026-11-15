# 提示策略准确率评测（Zero-Shot / Few-Shot / CoT）

这是给 Agent 骨架加的**增量模块**，只依赖你已有的 `agent/llm`，不需要 web_search / MCP。
它把「提示词策略」当成唯一自变量，同一测试集、同一模型、同一温度下跑三组，输出准确率对比。

## 文件

```
agent/prompts.py            三种策略的提示词构造 + 答案解析 + 判分（可单独复用）
scripts/eval_prompts.py     评测入口：跑批、统计、出报告
data/sample_testset.jsonl   12 题示例测试集（选择题 + 简答题混排）
data/fewshot_examples.jsonl 3 条 Few-Shot 示例（与测试集不重叠）
```

## 三种策略到底差在哪

| 策略 | 提示词差异 | 期望 | 主要风险 |
|---|---|---|---|
| Zero-Shot | 只给题面 + "直接给答案"，无示例、无推理要求 | 快、省 token | 格式不稳定；复杂题靠一步直觉 |
| Few-Shot | 题面 + k 条同类示例（用户/助手成对），要求模仿格式与粒度 | 格式最稳，准确率常高于 zero-shot | 示例选择偏差；示例泄题会让结果虚高 |
| CoT | 要求"先分析→推理→自检→最后一行 `答案：X`" | 多步推理题提升明显 | token 与延迟成本高；输出被截断会直接判错 |

## 公平性控制（脚本已内置）

1. 三个策略共用同一个 LLM 实例、同一 `temperature`（默认 **0**）、同一 `max_tokens`、同一测试集
2. Few-Shot 示例只从 `--examples` 指定的独立文件读；若示例 id 或题干与测试集重复，脚本会报警并中止
3. 判分逻辑与策略无关，统一走 `parse_answer` + `score`
4. 每题可 `--repeat N` 重复，多轮按**多数票**归档，避免"偶尔蒙对"
5. CoT 若因 `max_tokens` 不足被截断（`finish_reason=length`），会单独标记为「输出被截断」，而不是悄悄算错

## 用法

```bash
# 真实模型（推荐先小样本试跑，确认解析没问题）
python scripts/eval_prompts.py \
  --data data/sample_testset.jsonl \
  --examples data/fewshot_examples.jsonl --k 3 \
  -p deepseek --limit 5

# 全量跑，每题重复 3 次看稳定性
python scripts/eval_prompts.py \
  --data data/my_testset.jsonl --examples data/my_fewshot.jsonl \
  -p glm -m glm-4.6 --repeat 3 --concurrency 4 --out results/glm46

# 不花钱验证流程（假模型，数字是编的）
python scripts/eval_prompts.py --mock
```

输出三个文件到 `--out`：

- `eval_<时间戳>.json`：全量明细 + meta（模型、温度、k、测试集路径），可复现
- `eval_<时间戳>.csv`：逐题结果，直接丢进 Excel 做透视
- `eval_<时间戳>.md`：汇总表 + 策略间不一致的题目 + 需要人工抽查的样本

## 测试集格式

JSONL，一行一题，字段名支持中英文别名：

```json
{"id":"q1","question":"中国的首都是哪座城市？","type":"choice","options":["上海","北京","广州","深圳"],"answer":"B"}
{"id":"q2","question":"12 × 7 等于多少？","type":"short","answer":"84"}
{"id":"q3","question":"水的化学式？","answer":["H2O","H₂O"]}
```

- `type` 可省略：有 `options` 或答案是单个字母 → 自动判为选择题
- `answer` 支持多参考答案（列表），任一命中即算对
- 也支持 `.csv`（表头用 `question,answer,options,type`，`options` 用 `|` 分隔）

## 判分方式

| 题型 | 默认（`--match auto`） | 说明 |
|---|---|---|
| choice | 抽取选项字母比对 | 兼容 `答案：B`、`(B)`、`选 B`、仅输出 `B`、输出选项原文（反查字母） |
| short | 归一化后精确匹配 | 去标点/空格/全半角，数值按 float 比较（`84` = `84.0`） |

觉得判分太严可以 `--match contains`（标准答案出现在输出里即算对），但报告里务必注明，否则与精确匹配的结果不可比。

## 怎么读结果（别只看一个百分比）

1. **样本量**：12 题差 1 题就是 8.3 个百分点，别下结论。要下结论建议 ≥ 200 题，且同一批题跑 3 次看方差。
2. **解析失败要单列**：`解析失败` 不等于 `答错`。解析失败多，说明提示词格式约束不够或输出被截断，应先改提示词/调大 `max_tokens`，否则准确率是被判分器拖低的。
3. **成本一起看**：CoT 的 tokens 常是 Zero-Shot 的 2~5 倍，延迟也更高。准确率涨 3 个点但成本涨 4 倍，要按业务场景权衡。
4. **同一模型内比较**：不同模型（GLM vs DeepSeek）的绝对准确率不能直接横向比较谁"更懂提示词"。
5. **差异题最有信息量**：报告里列出了策略间结果不一致的题，逐题看能看出 Few-Shot 是在"教格式"还是真在"教知识"。
6. **想要统计显著性**：同一批题的策略对比应用配对检验（McNemar），脚本目前没内置，样本上量后建议补上。

## 常见坑

- **Few-Shot 示例与测试集重叠**：最常见的作弊，脚本会检测并中止
- **temperature 不为 0**：结果不可复现，比较失真
- **CoT + 小 max_tokens**：答案被截断，全判错
- **示例全是一种题型**：比如示例全是选择题，简答题表现会掉——示例要覆盖题型分布
- **k 太大**：示例占满上下文，反而压准确率；一般 k=2~5 就够
