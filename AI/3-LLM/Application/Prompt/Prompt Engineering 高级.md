---
brief: "Prompt Engineering 高级技巧——CoT/ToT/Self-Consistency/ReAct/Least-to-Most 等高阶提示策略；包含面试场景的典型问题和示范回答；interview/hot 标注，直接可用的面试武器。"
title: "Prompt Engineering 高级技巧"
date: 2026-02-13
tags:
  - ai/llm/prompting
  - ai/llm/application
  - type/technique
  - interview/hot
status: active
---

# Prompt Engineering 高级技巧

> Chain of Thought、Tree of Thought、Self-Consistency、Constitutional AI、Few-shot 策略——面试高频进阶话题

## 1. 技术全景

```
Prompting 策略演进
├── Zero-Shot → Few-Shot → Chain of Thought (2022)
├── CoT → Self-Consistency (2022) → Tree of Thought (2023)
├── Constitutional AI (2022) → RLHF 替代方案
└── 2025: Agentic Prompting、Structured Output、Multi-modal CoT
```

## 2. Chain of Thought (CoT)

### 核心思想

由 Wei et al. (2022) 提出：在 prompt 中加入中间推理步骤，让 LLM "展示思考过程"，显著提升复杂推理能力。

### 三种变体

**Few-shot CoT**：提供带推理过程的示例

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?
A: Roger started with 5 balls. 2 cans × 3 balls = 6. Total = 5 + 6 = 11.

Q: The cafeteria had 23 apples. They used 20 and bought 6 more. How many?
A: Started with 23. Used 20, so 23 - 20 = 3. Bought 6, so 3 + 6 = 9. Answer: 9.
```

**Zero-shot CoT**：只需附加一句 "Let's think step by step"

```python
prompt = f"""
{question}

Let's think step by step.
"""
# GSM8K 准确率：Zero-shot 17.7% → Zero-shot CoT 40.7% → Few-shot CoT 58.1%
```

**Auto-CoT** (Zhang et al. 2022)：自动从训练集中聚类选取多样化示例，消除手工编写 few-shot 的成本。

### 何时有效

- ✅ 数学推理、逻辑推理、多步问题
- ✅ 模型越大效果越好（涌现能力 emergent ability）
- ❌ 简单事实查询无需 CoT（反而增加 token 成本）
- ❌ 小模型（<10B）效果不稳定

## 3. Tree of Thought (ToT)

### 核心思想

Yao et al. (2023) 提出：将 CoT 从线性链扩展为树状搜索。每一步生成多个候选思路，通过评估函数选择最优路径，支持回溯（backtracking）。

```
                    [Problem]
                   /    |    \
              [思路A] [思路B] [思路C]
              /  \      |
         [A1] [A2]  [B1]  ← 评估后剪枝C
          |
        [A1-1]  ← 最终选择
```

### 实现方式

```python
import openai

def tree_of_thought(problem, breadth=3, depth=3):
    thoughts = [problem]
    
    for step in range(depth):
        candidates = []
        for thought in thoughts:
            # 生成多个候选
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""Given the problem and current reasoning:
{thought}

Generate {breadth} different next reasoning steps.
Format: [Step 1] ... [Step 2] ... [Step 3] ..."""
                }],
                n=1
            )
            new_thoughts = parse_steps(response.choices[0].message.content)
            candidates.extend([(thought + "\n" + t, t) for t in new_thoughts])
        
        # 评估并选择 top-k
        scores = evaluate_thoughts(candidates)
        thoughts = [c[0] for c, s in sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )][:breadth]
    
    return thoughts[0]

def evaluate_thoughts(candidates):
    """让 LLM 评估每条思路的可行性 (1-10)"""
    scores = []
    for full_thought, step in candidates:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Rate this reasoning step (1-10):\n{step}\nScore:"
            }]
        )
        scores.append(float(resp.choices[0].message.content.strip()))
    return scores
```

### 搜索策略

- **BFS**：每层保留 top-k 个节点，适合需要全面探索的问题
- **DFS**：深度优先 + 回溯，适合有明确终止条件的问题
- **Beam Search**：BFS 变体，固定 beam width

### 性能数据

- Game of 24：CoT 4% → ToT 74% (GPT-4)
- Creative Writing：人类评估 ToT 显著优于 CoT
- 代价：API 调用量 ×10~50

## 4. Self-Consistency

### 核心思想

Wang et al. (2022)：对同一问题多次采样（temperature > 0），取多数投票（majority voting）作为最终答案。基于直觉：正确推理路径更容易被多次采样到。

```python
import collections

def self_consistency(question, n_samples=5, temperature=0.7):
    answers = []
    for _ in range(n_samples):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"{question}\nLet's think step by step."
            }],
            temperature=temperature
        )
        answer = extract_final_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # 多数投票
    counter = collections.Counter(answers)
    return counter.most_common(1)[0][0]

# GSM8K: CoT 58.1% → Self-Consistency (40 paths) 74.4%
```

### 关键参数

- **采样数 n**：5~40，边际收益递减
- **temperature**：0.5~0.8 平衡多样性与质量
- **成本**：线性增长，n=10 则成本 ×10

## 5. Constitutional AI (CAI)

### 核心思想

Anthropic (Bai et al. 2022)：用一组 **原则（constitution）** 来指导模型自我修正，替代大量人类标注的 RLHF。两阶段流程：

1. **Critique + Revision**：模型生成 → 模型根据原则自我批评 → 模型修正回答
2. **RLAIF**：用修正后的数据做偏好学习（AI 反馈替代人类反馈）

```python
# Stage 1: Self-critique and revision
constitution = [
    "Choose the response that is least harmful or toxic.",
    "Choose the response that is most helpful while being safe.",
    "Choose the response that best acknowledges uncertainty."
]

def constitutional_revision(question, initial_response):
    critique_prompt = f"""
Question: {question}
Response: {initial_response}

Principles:
{chr(10).join(f'- {p}' for p in constitution)}

Critique this response based on the above principles:
"""
    critique = llm(critique_prompt)
    
    revision_prompt = f"""
Based on the critique: {critique}
Please revise the response to better align with the principles.
"""
    revised = llm(revision_prompt)
    return revised
```

### 与 RLHF 的关系

| 对比 | RLHF | CAI (RLAIF) |
|------|------|-------------|
| 反馈来源 | 人类标注员 | AI 自身 + 原则 |
| 成本 | 高（需大量人类标注） | 低（自动化） |
| 可扩展性 | 受限于标注产能 | 容易扩展 |
| 一致性 | 标注员间差异大 | 原则固定，一致性高 |
| 风险 | 标注偏见 | AI 盲点放大 |

参见 [[AI/3-LLM/RL/实践/RLHF-工程全栈|RLHF 全链路]] 和 [[AI/3-LLM/RL/算法/PPO 原理|PPO 原理]]。

## 6. Few-shot 高级策略

### 示例选择

```python
# 1. 随机选择（baseline）
examples = random.sample(example_pool, k=5)

# 2. 相似度选择（推荐）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
query_emb = model.encode(query)
example_embs = model.encode([e["question"] for e in example_pool])
similarities = cosine_similarity([query_emb], example_embs)[0]
top_k_indices = similarities.argsort()[-5:][::-1]
examples = [example_pool[i] for i in top_k_indices]

# 3. 多样性选择（覆盖不同模式）
# 先聚类，每个 cluster 选一个最相似的
```

### 示例排列

- **递增难度**：从简单到复杂，让模型"热身"
- **相关性排序**：最相关的放最后（recency bias 利用）
- **格式一致性**：所有示例保持相同格式，减少模型困惑

### 示例数量

- 一般 3~8 个最优（受限于 context window）
- 超过 8 个边际收益快速下降
- 长 context 模型（128k+）可以用更多示例

## 7. 2025 前沿趋势

### Agentic Prompting

将 CoT 与 Agent 能力结合：模型在推理中主动调用工具、查询数据库、验证中间结果。

```python
system_prompt = """You are a research assistant. When reasoning:
1. If you need current data, use search_web(query)
2. If you need to verify a fact, use fact_check(claim)
3. If you need to calculate, use calculator(expression)
Think step by step, calling tools as needed."""
```

### Structured Output Prompting

利用 JSON Schema 约束输出格式，减少解析错误：

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                    "answer": {"type": "string"}
                }
            }
        }
    },
    messages=[{"role": "user", "content": question}]
)
```

## 8. 面试常见问题

1. **Q: CoT 为什么有效？**
   A: 主流假说是 CoT 将复杂问题分解为子问题，降低每步推理难度，且中间步骤提供了"工作记忆"。另一种解释是训练数据中存在大量推理文本，CoT 触发了这些能力。

2. **Q: ToT 和 CoT 的本质区别？**
   A: CoT 是贪心搜索（一条路走到底），ToT 是树搜索（探索多条路径 + 评估 + 回溯）。ToT 适合搜索空间大、需要试错的问题，但成本高一个数量级。

3. **Q: Self-Consistency 为什么能提升准确率？**
   A: 类似集成学习（ensemble）的思想。错误的推理路径通常更随机，正确路径更集中。多次采样后多数投票可以过滤掉随机错误。

4. **Q: 实际项目中怎么选 prompting 策略？**
   A: 按成本-效果排序：Zero-shot → Few-shot → CoT → Self-Consistency → ToT。先用最简单的，效果不够再升级。生产环境中 Self-Consistency（n=3~5）是性价比最高的。

5. **Q: Constitutional AI 解决了 RLHF 的什么问题？**
   A: 三个核心问题：标注成本高、标注者间不一致、难以规模化。CAI 用可编辑的原则列表替代模糊的人类偏好，更透明可控。

## 相关笔记

- [[AI/3-LLM/Application/Prompt/Prompt-Engineering-基础|Prompt Engineering 基础]] — 基础 Prompt Engineering
- [[AI/3-LLM/Application/Prompt/Prompt-Engineering-概述|Prompt Engineering 概述]] — PE 概述
- [[AI/3-LLM/RL/实践/RLHF-工程全栈|RLHF 全链路]] — RLHF 完整流程
- [[AI/3-LLM/RL/算法/PPO 原理|PPO 原理]] — PPO 算法
- [[AI/3-LLM/Application/RAG/RAG 工程实践]] — 检索增强生成
- Transformer — Transformer 架构基础
