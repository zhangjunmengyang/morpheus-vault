---
title: Test-Time Compute (TTC) — 推理时扩展综述
brief: TTC 是对 training-time scaling 的正交补充——在推理阶段分配更多算力（CoT/Best-of-N/PRM/Budget Forcing）来提升准确率，而非训练更大模型；Snell et al. (arXiv:2408.03314) 证明对难题小模型+大TTC可超越大模型+小TTC；理解 TTC 是读懂 o1/R1/s1 推理范式的关键
type: survey
domain: ai/llm/inference
created: 2026-02-19
updated: 2026-02-22
tags:
  - ai/llm/inference
  - concept/ttc
  - concept/scaling
  - type/survey
  - interview/hot
status: complete
sources:
  - Scaling LLM Test-Time Compute Optimally arXiv:2408.03314 (Snell et al., Google DeepMind, 2024)
  - Let's Verify Step by Step (PRM) arXiv:2305.20050 (Lightman et al., OpenAI, 2023)
  - Self-Consistency Improves Chain of Thought Reasoning arXiv:2203.11171 (Wang et al., 2022)
  - Chain-of-Thought Prompting arXiv:2201.11903 (Wei et al., Google, 2022)
  - Deep-Thinking Ratio arXiv:2602.13517 (2026)
  - "s1: Simple Test-Time Scaling (Stanford, 2025)"
related:
  - "[[推理优化|推理优化]]"
  - "[[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO]]"
  - "[[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]"
  - "[[Gemini-3-Deep-Think|Gemini-3-Deep-Think]]"
  - "[[AI/3-LLM/Evaluation/ICLR-2026-趋势分析|ICLR-2026-趋势分析]]"
  - "[[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]]"
  - "[[Deep-Thinking-Ratio-DTR|Deep-Thinking-Ratio-DTR]]"
  - "[[采样策略|采样策略]]"
---

# Test-Time Compute (TTC) — 推理时扩展综述

> **Brief**：TTC 在推理阶段分配更多算力（CoT/Best-of-N/PRM/Budget Forcing）来提升准确率，是对 training-time scaling 的正交补充。Snell et al. (arXiv:2408.03314) 的关键发现：对难题，小模型+大TTC 可超越大模型+小TTC。
>
> 来源：Snell et al. arXiv:2408.03314; Let's Verify Step by Step arXiv:2305.20050; Self-Consistency arXiv:2203.11171

> 关键词：inference-time scaling, test-time compute, chain-of-thought, self-verification, budget forcing

## 一句话定义

**在推理阶段分配更多算力（而不是训练更大的模型）来提升任务准确率。**

这是对 training-time scaling law（Chinchilla, arXiv:2203.15556）的正交补充维度。

---

## 背景与动机

### Chinchilla Scaling 的瓶颈

传统 scaling 路线：更多数据 + 更多参数 + 更多训练算力 → 性能提升。

但这条路越来越贵：GPT-4 训练成本 ~$100M，GPT-5 数量级更高。

### TTC 的核心洞察

**人类面对难题时会多思考，而不是替换自己的大脑。**

LLM 可以做同样的事：在 inference 时多分配算力，而不是训练更大的模型。

关键 empirical 发现：
- **对难题**：小模型 + 大量 TTC > 大模型 + 少量 TTC
- **compute-optimal 点随任务难度移动**：难题值得分配更多推理算力

> 来源：Snell et al. arXiv:2408.03314 "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters", Sec. 4

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

> 来源：Lightman et al. arXiv:2305.20050 "Let's Verify Step by Step" (OpenAI, 2023)

- **Outcome Reward Model (ORM)**：只看结果对不对
- **Process Reward Model (PRM)**：每步都打分
- PRM 能更早发现错误路径，引导模型 self-correct

### 3. Best-of-N / Self-Consistency

> 来源：Wang et al. arXiv:2203.11171 "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

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

> 参考：[[Gemini-3-Deep-Think|Gemini-3-Deep-Think]] [[AI/3-LLM/Evaluation/ICLR-2026-趋势分析|ICLR-2026-趋势分析]]

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
   - RLVR（[[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]]）训练出能做长 CoT 的模型
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

## 2026-02 新进展：质量 vs 数量的重新定义

### Deep-Thinking Ratio (DTR)（2602.13517）

一个颠覆"长度 = 质量"假设的重要发现：

**关键数字**：
- Token count 与准确率 Pearson r = **-0.544**（负相关！）
- DTR 与准确率 Pearson r = **+0.828**（强正相关）

**DTR 定义**：序列中"深层才收敛"的 token 比例。使用 logit lens 技术，计算每层 hidden state 投影的分布与最终层的 JSD 散度，收敛层深的 token = 深度思考 token。

**Think@n**：基于 DTR 的 test-time scaling 策略，用 DTR 早期筛选 + 拒绝低质量生成，在匹配 SC@n 准确率的同时降低约 50% 计算成本。

详见：[[Deep-Thinking-Ratio-DTR|Deep-Thinking-Ratio-DTR]]

**对 TTC 的启示**：TTC 的目标不应该是"更多 token"，而是"更多深层推理 token"。Budget forcing 只约束数量，未来的改进方向可能是约束 DTR，鼓励模型用有限 token 进行更深度的思考。

---

## 知识缺口

- [ ] Inference-time scaling law 数学推导（Snell et al. 2024 原论文）
- [ ] PRM 的训练方式（哪些数据，怎么打 step label）
- [ ] o3 TTC 的具体机制（未公开）
- [ ] DTR 在 MoE 模型上的适用性
- [ ] DTR 作为 RL reward 信号的可行性

---

## 关联笔记

- [[Gemini-3-Deep-Think|Gemini-3-Deep-Think]] — TTC 前沿落地案例
- [[推理优化|推理优化]] — 工程层面的 inference 优化
- [[Speculative Decoding|Speculative Decoding]] — 另一种推理加速方向（减少 latency，与 TTC 目标相反）
- [[AI/3-LLM/Evaluation/ICLR-2026-趋势分析|ICLR-2026-趋势分析]] — TTC 是 ICLR 2026 最大热点（257篇）
- [[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]] — RL 训练如何赋予模型 TTC 能力
- [[采样策略|采样策略]] — Best-of-N、Beam Search 实现细节
- [[Deep-Thinking-Ratio-DTR|Deep-Thinking-Ratio-DTR]] — 推理质量新指标，超越 token 长度

---

## 🔧 落地应用

### 什么时候用 TTC？

| 场景 | TTC 策略 | 效果 | 成本 |
|------|---------|------|------|
| 数学/代码竞赛 | Best-of-N + PRM | 极好 | 高（N 次推理） |
| 复杂推理任务 | Long CoT + Budget Forcing | 很好 | 中高（长输出） |
| 实时对话 | 不适合大 TTC | — | 延迟不可接受 |
| 离线批处理（代码 review、论文分析） | 大 Budget + PRM | 最佳 | 可接受 |
| 简单 QA | 不需要 TTC | — | 浪费算力 |

### 工程实现要点

- **Best-of-N 是最简单的 TTC 实现**：N=8-16 通常就能显著提升准确率，低成本验证有效性
- **PRM 训练需要 step-level 标注**：成本高，但一旦训练好可以复用
- **Budget Forcing 的实现**：在生成时设置最小 token 数，遇到 `<end>` token 替换为 `\n` 继续生成
- **动态预算分配**：简单问题给少量 TTC，难题给大量 TTC → 需要一个难度评估器（可以用 prompt 的 perplexity 或初始 logits 的 entropy 估算）

### 面试高频问法

- **Q: TTC 为什么 work？和搜索有什么关系？**
  A: TTC 本质是在 solution space 中搜索更久。CoT 提供中间状态使搜索有方向，PRM 提供 guidance 避免盲目搜索，Best-of-N 是最简单的并行搜索。

- **Q: 给你固定算力预算，是训练更大模型还是用更多 TTC？**
  A: 取决于任务难度。Snell et al. (arXiv:2408.03314) 证明：难题上 TTC 的 ROI 更高，简单任务用大模型更划算。Compute-optimal 点随难度移动。

- **Q: Budget Forcing 和 Chain-of-Thought 的区别？**
  A: CoT 是"让模型展示推理步骤"，模型自己决定想多久。Budget Forcing 是"强制模型想更久"，即使模型想早停也不允许。后者往往能发现第一印象的错误。

---

## 💡 启发与思考

### So What？

TTC 揭示了一个深刻的范式转变：**推理能力不仅可以通过训练获得，也可以通过推理时的计算分配来增强**。这类似于人类的直觉和深思——快速直觉（System 1）对应标准推理，深入思考（System 2）对应 TTC。

对老板的启示：
- **Agent 设计中应该内置 TTC 策略**：对简单请求快速响应，对复杂任务自动分配更多思考预算
- **成本控制的新维度**：不再只是"用多大的模型"，还要考虑"给多少推理预算"
- **小模型 + 大 TTC 可能是性价比最高的方案**：比如 7B 模型 + Best-of-16 可能比 70B 模型 + 单次推理更好

### 局限与未解问题

- **延迟问题**：TTC 增加推理时间，实时应用难以接受（o1 的"思考中..."就是延迟的体现）
- **成本线性增长**：Best-of-N 的成本随 N 线性增长，没有"免费午餐"
- **PRM 的训练数据获取困难**：step-level 标注成本极高，自动标注方法仍在探索
- **DTR 研究还很初步**：Deep-Thinking Ratio 证明"思考深度比长度重要"，但如何引导模型深度思考仍不清楚
- **TTC 对不同模型架构的效果差异**：MoE vs Dense 模型在 TTC 上的表现是否不同？

### 脑暴拓展

- **自适应 TTC 预算分配器**：训练一个轻量模型预测"这个问题需要多少 TTC"→ 在延迟和准确率间动态 trade-off
- **TTC + MoE 的协同**：不同 expert 处理不同推理深度？浅层 expert 负责 System 1 推理，深层 expert 负责 System 2？
- **DTR 作为 RL reward 信号**：在 RLVR 训练中用 DTR 代替（或辅助）ORM/PRM，鼓励模型生成"深度思考 token"而非"长但浅的 token"

> 🔗 See also:
> - [[推理优化|推理优化]] — 工程层面的 inference 优化（减少 latency）
> - [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO]] — RL 训练赋予模型 TTC 能力
> - [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — GRPO + long CoT 的实践
> - [[Deep-Thinking-Ratio-DTR|Deep-Thinking-Ratio-DTR]] — 推理质量新指标，思考深度 > 思考长度

---

## 📚 推荐阅读

### 原始论文
- [Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) — TTC 领域最重要的理论工作，证明 compute-optimal 分配策略 ⭐⭐⭐⭐⭐
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — PRM 的奠基论文，process reward 比 outcome reward 更有效 ⭐⭐⭐⭐⭐
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Best-of-N + majority vote 的简单但强大的 TTC 方法 ⭐⭐⭐⭐⭐
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — CoT 的开山之作，最早的 TTC 形式 ⭐⭐⭐⭐⭐
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — 树形搜索 + 显式状态评估 ⭐⭐⭐⭐

### 深度解读
- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) — o1 的设计哲学和安全考量 ⭐⭐⭐⭐
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) — GRPO + long CoT 训练的工程实践 ⭐⭐⭐⭐⭐

### 实践资源
- [s1: Simple Test-Time Scaling (GitHub)](https://github.com/simplescaling/s1) — Stanford 的 Budget Forcing 实现，小模型超越 o1 ⭐⭐⭐⭐
- [PRM800K Dataset (OpenAI)](https://github.com/openai/prm800k) — PRM 训练数据集，800K step-level 标注 ⭐⭐⭐⭐

---

*Created: 2026-02-19 by Librarian heartbeat — 补全知识缺口 TTC*
*Updated: 2026-02-22 — 补充 frontmatter/出处/推荐阅读/落地应用/启发思考*
