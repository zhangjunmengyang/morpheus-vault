---
title: "Test-Time Compute (TTC) — 推理时扩展综述"
type: survey
domain: ai/llm/inference
tags:
  - ai/llm/inference
  - concept/ttc
  - concept/scaling
  - type/survey
created: 2026-02-19
---

# Test-Time Compute (TTC) — 推理时扩展综述

> 关键词：inference-time scaling, test-time compute, chain-of-thought, self-verification, budget forcing

## 一句话定义

**在推理阶段分配更多算力（而不是训练更大的模型）来提升任务准确率。**

这是对 training-time scaling law（Chinchilla）的正交补充维度。

---

## 背景与动机

### Chinchilla Scaling 的瓶颈

传统 scaling 路线：更多数据 + 更多参数 + 更多训练算力 → 性能提升。

但这条路越来越贵：GPT-4 训练成本 ~$100M，GPT-5 数量级更高。

### TTC 的核心洞察

**人类面对难题时会多思考，而不是替换自己的大脑。**

LLM 可以做同样的事：在 inference 时多分配算力，而不是训练更大的模型。

关键 empirical 发现（Snell et al., 2024；Google DeepMind, 2024）：
- **对难题**：小模型 + 大量 TTC > 大模型 + 少量 TTC
- **compute-optimal 点随任务难度移动**：难题值得分配更多推理算力

---

## 核心技术路径

### 1. Chain-of-Thought（CoT）

最早期的 TTC 形式。让模型显式输出推理步骤，而不是直接输出答案。

```
[直接回答] "The answer is 42."
[CoT]      "Step 1: ... Step 2: ... Therefore 42."
```

CoT 把 latent reasoning 变成 token generation，模型可以在中间步骤上做 conditional generation。

### 2. Process Reward Model（PRM）

在生成过程中，对每一步推理做评分，而不是只评最终答案。

- **Outcome Reward Model (ORM)**：只看结果对不对
- **Process Reward Model (PRM)**：每步都打分
- PRM 能更早发现错误路径，引导模型 self-correct

### 3. Best-of-N / Self-Consistency

- 生成 N 个候选答案，用 majority vote 或 reward model 选最优
- 简单有效，compute 线性扩展
- 限制：N 个 sample 彼此独立，无法互相纠错

### 4. Beam Search / Tree-of-Thought

- 并行维护多条推理路径（beam）
- 中间剪枝不靠谱的路径，扩展有潜力的
- Tree-of-Thought 是泛化：树形搜索 + 显式状态评估

### 5. Budget Forcing

Stanford s1 论文提出（2025）：

**强制模型分配固定 thinking budget（token 数），不允许提前停止。**

即使模型想输出 "答案是 X"，也 force 它继续 "think longer"。

实验结论：同一小模型，给更多 thinking budget → 性能大幅提升，接近 GPT-4 水平。

### 6. Extended Thinking / Internal Verification（前沿）

Gemini 3 Deep Think（2026）的路线：

- 推理步骤不一定输出，在 internal state 里完成
- 模型内部做 **self-verification**：生成答案 → 内部检查逻辑一致性 → 发现错误 → 回溯修正
- 这层验证对用户透明，不占输出 token

---

## 工业落地里程碑

| 时间 | 模型 | 技术路线 | 关键 Benchmark |
|------|------|---------|---------------|
| 2024-09 | OpenAI o1 | Long CoT + PRM | AIME 2024: 83% |
| 2025-01 | DeepSeek-R1 | GRPO + RL-trained long CoT | AIME 2025: 79.8% |
| 2025-01 | Stanford s1 | Budget forcing on 1B model | 超越 o1 on AIME |
| 2026-02 | Gemini 3 Deep Think | TTC + internal verification | ARC-AGI-2: 84.6% (超人类均值) |

> 参考：[[Gemini-3-Deep-Think]] [[ICLR-2026-趋势分析]]

---

## 核心 Trade-off

```
更多 TTC → 更高准确率
         → 更高 latency
         → 更高 cost
```

**Compute-optimal 分配**（Snell et al., 2024）：

- 给定总算力预算 B
- 如何在 "训练更大模型" vs "推理用更多算力" 之间分配？
- 答：**任务难度 > threshold 时，分配给 TTC 更划算**

---

## TTC vs Training-time Scaling

| 维度 | Training-time Scaling | Test-time Compute |
|------|----------------------|-------------------|
| 投入时机 | 训练阶段 | 推理阶段 |
| 成本 | 一次性（amortized） | 每次推理都付 |
| 灵活性 | 固定 | 可动态调整 |
| 适合场景 | 通用能力提升 | 特定难题攻坚 |
| 典型代表 | Chinchilla, GPT-4 | o1, R1, s1 |

两者 **orthogonal，可以组合**：大模型 + 大 TTC = 最强推理能力，但最贵。

---

## 对算法工程师的启示

### 面试高频考点（ICLR 2026 TTC 论文 257 篇）

1. **TTC 为什么 work？**
   - 等效于搜索：在 solution space 中搜索更久
   - 难题需要更长推理链，CoT 提供中间状态
   - PRM 提供 guidance，避免盲目搜索

2. **与 RLVR 的关系**
   - RLVR（[[RLVR-Edge-of-Competence]]）训练出能做长 CoT 的模型
   - TTC 是这些模型在推理时的 deployment 策略
   - 两者协同：RL 训练赋予能力，TTC 在推理时激发

3. **Budget forcing 的直觉**
   - 类比：考试不到最后一分钟不允许交卷
   - 模型被迫 "再想想"，有时候能纠正第一印象的错误

### 工程实践注意

- 实时应用慎用大 TTC（latency 问题）
- 离线任务（代码 review、论文分析）可以给大 budget
- Best-of-N 是最简单的 TTC 实现，低成本验证有效性

---

## 知识缺口

- [ ] Inference-time scaling law 数学推导（Snell et al. 2024 原论文）
- [ ] PRM 的训练方式（哪些数据，怎么打 step label）
- [ ] o3 TTC 的具体机制（未公开）

---

## 关联笔记

- [[Gemini-3-Deep-Think]] — TTC 前沿落地案例
- [[推理优化]] — 工程层面的 inference 优化
- [[Speculative Decoding]] — 另一种推理加速方向（减少 latency，与 TTC 目标相反）
- [[ICLR-2026-趋势分析]] — TTC 是 ICLR 2026 最大热点（257篇）
- [[RLVR-Edge-of-Competence]] — RL 训练如何赋予模型 TTC 能力
- [[采样策略]] — Best-of-N、Beam Search 实现细节

---
*Created: 2026-02-19 by Librarian heartbeat — 补全知识缺口 TTC*
