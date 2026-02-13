---
title: "LLM 评测体系"
date: 2026-02-13
tags:
  - ai/llm/evaluation
  - ai/llm/benchmark
  - type/reference
  - interview/hot
status: active
---

# LLM 评测体系

> MMLU、HumanEval、Chatbot Arena——主流评测方法论、Benchmark 解读与局限性分析

## 1. 评测维度总览

```
LLM 评测
├── 知识与理解: MMLU, MMLU-Pro, ARC, HellaSwag
├── 推理: GSM8K, MATH, BBH, ARC-Challenge
├── 代码: HumanEval, MBPP, SWE-bench, LiveCodeBench
├── 对话: Chatbot Arena, MT-Bench, AlpacaEval
├── 安全: TruthfulQA, ToxiGen, SafetyBench
├── 长文本: RULER, LongBench, InfiniteBench
├── 多模态: MMMU, MathVista, RealWorldQA
└── 前沿: HLE (Humanity's Last Exam), FrontierMath
```

## 2. 核心 Benchmark 详解

### 2.1 MMLU (Massive Multitask Language Understanding)

**设计**：Hendrycks et al. (2020)，57 个学科（STEM/人文/社科/其他），14,042 道四选一题。

**评测方式**：Few-shot（5-shot），取各学科准确率平均。

```python
# MMLU 评测示例
prompt = """The following is a multiple choice question about abstract algebra.

Q: Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.
(A) 0 (B) 4 (C) 2 (D) 6

Answer:"""

# 评测: 取 logprobs 最高的选项 (A/B/C/D)
```

**现状 (2026)**：
- GPT-4o: 88.7%, Claude Opus 4: 89.1%, Gemini Ultra: 90.0%
- 已接近饱和 → MMLU-Pro 应运而生（10 选一 + 推理题）

**MMLU-Pro**：Wang et al. (2024)，更难的版本：
- 10 个选项（vs 原版 4 个）
- 加入更多推理题
- GPT-4o: 72.6%，区分度更好

### 2.2 HumanEval

**设计**：Chen et al. (2021, OpenAI)，164 道 Python 编程题，每题包含函数签名、docstring 和测试用例。

**评测指标**：**pass@k** — 生成 k 个候选，至少一个通过所有测试的概率。

```python
# HumanEval 示例
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers 
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
    # 模型需要生成正确实现

# pass@1 计算（无偏估计）
import numpy as np
def pass_at_k(n, c, k):
    """n: 总生成数, c: 通过数, k: 取 k 个"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

**现状 (2026)**：
- Claude Opus 4: 92.1%, GPT-4o: 90.2%, DeepSeek-V3: 89.4%
- 已趋于饱和 → SWE-bench 成为新标准

### 2.3 SWE-bench

**设计**：Jimenez et al. (2024)，真实 GitHub issue 修复。模型需要在完整代码库中定位问题、理解上下文并提交 patch。

**难度级别**：
- SWE-bench Lite: 300 个精选问题
- SWE-bench Full: 2,294 个问题
- SWE-bench Verified: 500 个人工验证问题

**现状 (2026)**：Claude Opus 4 + 脚手架: 80.9%（SWE-bench Verified）

### 2.4 Chatbot Arena (LMSYS)

**设计**：人类盲评对战。用户提交 prompt，两个匿名模型同时回答，用户投票选更好的。

**排名方法**：Bradley-Terry 模型计算 **Elo 评分**。

```
用户 → 提交问题 → Model A & Model B 匿名回答 → 用户投票 A>B / B>A / Tie
                                                        ↓
                                        Bradley-Terry 模型更新 Elo
```

**优势**：
- 真实用户偏好（非合成数据）
- 开放域（不限特定任务）
- 动态更新（持续收集新数据）

**现状 (2026)**：
- 累计 2M+ 投票
- 细分类别：Coding、Math、Hard Prompts、Style Control
- 分离 "Style" 和 "Substance"（避免偏好长回答的偏见）

### 2.5 其他重要 Benchmark

| Benchmark | 评测维度 | 关键特点 |
|-----------|----------|----------|
| **GSM8K** | 小学数学 | 8.5K 题，CoT 评测标配 |
| **MATH** | 竞赛数学 | 12.5K 题，难度 1-5 级 |
| **BBH** (BIG-Bench Hard) | 综合推理 | 23 个难任务子集 |
| **MT-Bench** | 多轮对话 | 80 题，GPT-4 打分 |
| **AlpacaEval 2.0** | 指令遵循 | 805 题，自动评测 |
| **TruthfulQA** | 真实性 | 817 题，测试模型是否复制人类常见错误 |
| **HLE** | 前沿知识 | 3K+ 专家题，当前最强模型 <10% |

## 3. 评测方法论

### 3.1 静态 Benchmark vs 动态评测

| 维度 | 静态 (MMLU/HumanEval) | 动态 (Arena/LiveBench) |
|------|----------------------|----------------------|
| 数据 | 固定题库 | 持续更新 |
| 污染风险 | 高（可能泄入训练集） | 低（新题/人类实时） |
| 可复现 | 完全可复现 | 需要足够投票量 |
| 成本 | 低（自动评测） | 高（人类标注） |
| 覆盖面 | 特定能力 | 开放域 |

### 3.2 自动评测 vs 人类评测

**自动评测**：
- 精确匹配 (Exact Match)：MMLU、GSM8K
- 代码执行 (Execute & Test)：HumanEval、SWE-bench
- LLM-as-Judge：MT-Bench（GPT-4 打分）、AlpacaEval

**LLM-as-Judge 的问题**：
```python
# 常见偏见
# 1. 位置偏见：倾向于选第一个回答
# 2. 冗长偏见：倾向于选更长的回答
# 3. 自我偏好：GPT-4 做 judge 倾向选 GPT-4 的回答

# 缓解方案
def debiased_judge(response_a, response_b):
    # 交换顺序评两次
    score_ab = judge(response_a, response_b)  # A 在前
    score_ba = judge(response_b, response_a)  # B 在前
    # 取平均，消除位置偏见
    return (score_ab + (1 - score_ba)) / 2
```

### 3.3 评测流水线

```python
# 标准评测流程（以 lm-evaluation-harness 为例）
from lm_eval import evaluator, tasks

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.1-70B-Instruct",
    tasks=["mmlu", "gsm8k", "humaneval"],
    num_fewshot=5,  # MMLU 标准 5-shot
    batch_size=8,
)

# 结果
# mmlu: 82.3% (acc)
# gsm8k: 87.5% (acc, strict match)
# humaneval: 78.0% (pass@1)
```

## 4. 局限性与批判

### 4.1 数据污染 (Contamination)

最严重的问题：模型训练数据可能包含评测题目。

```
问题: MMLU 题目出现在 Common Crawl → 被用于训练
影响: 模型"背答案"而非真正理解
检测: n-gram overlap 检测、canary string
      但难以完全排除间接污染（如讨论题目的网页）
```

**缓解方案**：
- 动态 Benchmark（LiveBench、Arena）
- Contamination 检测报告
- 定期更换题库

### 4.2 Benchmark 饱和

```
MMLU: 2023 GPT-4 86.4% → 2026 多模型 >90%
HumanEval: 2023 GPT-4 67% → 2026 多模型 >90%
GSM8K: 几乎所有前沿模型 >95%

结果: 无法区分前沿模型差异
应对: MMLU-Pro、HLE、GPQA、FrontierMath
```

### 4.3 评测与真实能力的 Gap

- **Cherry-picking**：厂商只报最好的 benchmark
- **Prompt 敏感性**：同一模型不同 prompt 格式差异可达 10%+
- **任务覆盖不全**：大多 benchmark 测的是知识和推理，缺少创造力、情感理解、长期规划
- **过拟合评测**：模型可能针对热门 benchmark 优化（类似应试教育）

### 4.4 多维度评估的缺失

```
当前主流评测:
✅ 知识 (MMLU)
✅ 代码 (HumanEval)  
✅ 数学 (GSM8K/MATH)
✅ 对话偏好 (Arena)

缺失维度:
❌ 长期记忆和学习能力
❌ 真实世界 Agent 任务完成率
❌ 成本效率 (quality per dollar)
❌ 安全性的系统评测
❌ 多语言能力（大多 benchmark 英文为主）
```

## 5. 2025-2026 趋势

### 新一代 Benchmark

- **HLE (Humanity's Last Exam)**：3K+ 博士级专家出题，当前最强模型 <10%
- **FrontierMath**：数学研究级难度
- **SWE-bench**：从代码生成到工程能力
- **GPQA**：博士级科学推理

### Arena 的演进

- **Hard Prompts** 子排名：区分简单聊天 vs 复杂任务
- **Style Control**：分离表达风格与实质内容
- **Category Arena**：Coding、Math、Creative 分开排名
- **Cost-adjusted**：单位成本的性能比较

### 评测框架成熟

- **lm-evaluation-harness**（EleutherAI）：标准自动评测
- **OpenCompass**：中文生态评测平台
- **HELM**（Stanford）：全面的 holistic 评测

## 6. 面试常见问题

1. **Q: MMLU 分数高就一定好吗？**
   A: 不一定。MMLU 有数据污染风险，且已趋于饱和（多模型 >90%）。需要结合 Arena 等动态评测、和领域特定 benchmark 综合判断。MMLU-Pro 和 HLE 是更好的区分指标。

2. **Q: Chatbot Arena 的 Elo 评分怎么理解？**
   A: 基于 Bradley-Terry 模型，类似国际象棋 Elo。每次对战根据结果更新分数，差 100 分 ≈ 64% 胜率。优点是反映真实偏好，缺点是受用户群体偏见影响（如偏好长回答）。

3. **Q: 怎么检测 benchmark 数据污染？**
   A: 常见方法：n-gram overlap 检测（训练数据与题目的文本重叠度）、canary string（在数据中埋标记词）、membership inference attack、对比旧题/新题的得分差异。

4. **Q: LLM-as-Judge 靠谱吗？**
   A: 有一定参考价值但有已知偏见：位置偏见（倾向选第一个）、冗长偏见（选更长的）、自我偏好。缓解方法包括交换顺序双评、使用多个 judge 模型、结合人类评估。

5. **Q: 怎么评测自己训练的模型？**
   A: 推荐流程：(1) lm-evaluation-harness 跑标准 benchmark（MMLU/GSM8K/HumanEval）；(2) 领域特定评测集（自建或开源）；(3) 人工 A/B 测试（vs baseline）；(4) 关注 [[PPL 困惑度]]  作为训练质量的快速指标。

## 相关笔记

- [[PPL 困惑度]] — 困惑度计算
- [[PPL 计算 交叉熵损失与 ignore_index]] — PPL 实现细节
- [[模型评估]] — 机器学习评估方法
- [[SFT 原理]] — 有监督微调
- [[RLHF 全链路]] — RLHF 与偏好学习
- [[DeepSeek-R1]] — DeepSeek R1 评测表现
