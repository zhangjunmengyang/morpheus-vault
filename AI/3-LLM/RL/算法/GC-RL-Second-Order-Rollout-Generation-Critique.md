---
title: "GC-RL: Generation-Critique Joint Training via Second-Order Rollout"
tags: [RL, GRPO, CritiqueTraining, DataEfficiency, SelfImprovement, ReasoningRL]
created: 2026-02-27
status: permanent
rating: ★★★☆
arxiv: "2602.22765"
related:
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解]]"
  - "[[AI/2-Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents]]"
  - "[[AI/2-Agent/Agentic-RL/Search-P1-Path-Centric-Reward-Agentic-RAG]]"
---

# GC-RL: 二阶 Rollout——用批判能力提升 RL 数据利用效率

> arXiv:2602.22765 | 2026-02-26 | ★★★☆

---

## 一句话定位

Vanilla GRPO 只训了**生成能力**（对问题采样多个回答）。GC-RL 引入**二阶 Rollout**（对回答再采样多个评判），让同一 policy 同时学会"做题"和"批改"，在相同训练数据量下获得更好的推理性能。

---

## 核心观察：Vanilla RL 的数据浪费

标准 GRPO 的 rollout 流程：

```
问题 q → [response_1, ..., response_k] → GRPO advantage → policy 更新
                                              ↑
                      只有 first-order rollout（问题 → 回答）
```

**被忽略的训练信号**：模型生成了 k 条回答，但从未被要求**评判**这些回答的质量。评判能力（critique capability）在 vanilla RL 中完全没有被训练。

论文的核心主张：**生成能力和评判能力是互相促进的**——会批改题目的模型做题也更好；做题更好的模型，批改时的基准也更准确。但 vanilla RL 的训练信号完全来自 first-order rollout，critique 能力的训练被浪费了。

---

## 方法：GC-RL 框架

### 1. 二阶 Rollout 定义

```
First-order rollout（vanilla RL）：
  q → [r_1, r_2, ..., r_k]           # 对问题采样 k 个 response

Second-order rollout（GC-RL 新增）：
  (q, r_i) → [c_1, c_2, ..., c_m]   # 对某个 response 采样 m 个 critique
```

Critique 的定义：给定问题 q 和某个回答 r_i，模型生成评判文本，判断 r_i 是否正确并给出理由。

### 2. Mixed Rollout Group

GC-RL 的核心设计：把 first-order 和 second-order rollout 的结果**混入同一个 GRPO group**：

```
Mixed Group = {r_1, ..., r_k, c_1, ..., c_m}
                 ↑                ↑
          generation responses   critique responses

所有样本在同一 group 内计算 GRPO advantage：
A_i = (R_i - mu_group) / sigma_group
```

这意味着 responses 和 critiques 相互竞争 advantage——高质量的生成和高质量的评判都会被强化。

### 3. 分离的 Reward 函数

两种 rollout 用不同的 reward：

**Generation reward**（correctness，standard GRPO）：
- R_gen = 1.0 if correct(r_i), else 0.0

**Critique reward**（评判是否准确）：
- R_crit = 1.0 if accurate_critique(c_j, r_i), else 0.0
- "正确识别了 r_i 的对/错"即为准确评判

**关键**：critique reward 是**自动可计算的**（因为 r_i 的 ground truth correctness 已知），不需要额外标注——这是 GC-RL 端到端训练的前提。

### 4. 训练流程

```
迭代 t：
1. 对每个训练问题 q，生成 k 个 responses（first-order rollout）
2. 对部分 responses，再各生成 m 个 critiques（second-order rollout）
3. 计算 R_gen 和 R_crit
4. 混合 group → GRPO advantage 计算
5. Policy 梯度更新（generation + critique 同一 policy）
```

---

## 设计哲学：为什么混合同一 group？

**如果分离训练**（generation 和 critique 各自 GRPO）：
- 两个 group 各自的 baseline 不一样
- 两种能力没有被相互比较，缺少跨能力竞争的压力

**混合一个 group 的效果**：
- 一个好的 critique（准确判断对/错）的 advantage 会高于一个差的 generation
- 这让模型发现：准确评判和准确生成都是有价值的策略
- 形成内部激励：模型会逐渐把批改能力当成和做题能力同等重要的目标

类似 Multi-Task RL 中的共享 advantage space——让不同任务在同一个奖励竞争框架内互相校准。

---

## 与相关工作的关系

### vs. Self-Critique / Self-Refinement

Self-Critique（Constitutional AI, Self-Refine）：用 critique 的结果去**修改**当前答案——critique 是后处理步骤，不是训练目标。

GC-RL：critique 本身作为训练目标，和 generation 同时被 RL 优化——**critique 能力本身被提升**，而不只是被使用。

### vs. Search-P1 (arXiv:2602.22576)

Search-P1 的 Self-Consistency Track（模型执行自己计划的程度）也是"生成-评判"双轨结构。但 Search-P1 的评判是离线参考计划的对齐度（路径级），GC-RL 的评判是对自身 response 的实时质量判断（回答级）。两者场景不同（RAG 搜索 vs 数学推理）。

### vs. AgentPRM (arXiv:2502.10325)

AgentPRM MC rollout：向前模拟（估计未来价值）
GC-RL second-order rollout：向后评判（判断当前质量）

两种从 rollout 提取额外信号的互补方向。

---

## 关键洞察

### 1. "Critique 是免费的训练信号"

Critique reward 自动可计算（已知 ground truth correctness），不需要额外人工标注。同样的训练问题，两层 rollout 挖出比 vanilla RL 更多的梯度信号。

### 2. 数据效率的重新定义

传统：RL 数据效率 = 用多少问题达到多好的性能。
GC-RL：同样的问题集，second-order rollout 在不增加数据量的情况下提升性能——**问题固定，信号密度可以增加**。

这和 Search-P1 的 "soft scoring 不浪费失败样本" 是同一个大方向：挖掘已有训练数据中未被利用的信号。

### 3. 双能力正反馈回路

生成能力↑ → critique 准确性↑（更能理解推理对/错模式）
critique 准确性↑ → 生成能力↑（更能识别并避免自身错误）

理论上比单独训练收敛更快，但需要实验验证。

---

## 局限与批判

**Critique reward 的质量上限**：只能判断"有没有正确识别答案对/错"，不能评判 critique 的**推理质量**（理由是否正确）。模型可能通过记忆"这类问题通常对/错"来 hack critique reward，而不是真正理解。

**Second-order rollout 的成本**：每个 response 再生成 m 个 critiques，总 rollout 量增加 (1+m) 倍。"免费"的说法是针对数据量，不是计算量。

**Mixed group 的量纲问题**：critique 任务通常比 generation 更容易（随机猜测 baseline ~50%），可能导致 advantage 被高 critique reward 主导，generation 的梯度信号被稀释。

**实验范围**：已知实验在数学推理场景（GRPO on MATH）。Critique 能力在 agent 场景（工具调用、多步任务）的效果未知。

---

## 在 RL 数据效率谱系中的位置

RL 训练数据效率提升方向：

- 更好的 reward 信号：MIG（信息论量化贡献）/ Search-P1（soft scoring）/ SORL（稳定 credit 传递）
- 更好的数据选择：Rejection Sampling / DoReMi / D4
- **更丰富的训练信号（同样数据挖出更多梯度）：GC-RL（second-order rollout）**

GC-RL 属于第三类，思路最新颖：不是换更好的数据，而是从同样数据中产生更多维度的训练信号。

---

## See Also

- [[AI/2-Agent/Agentic-RL/SCoRe-Self-Correction-via-Reinforcement-Learning]] — 另一种双阶段结构（Phase 1 初始化 + Phase 2 精炼），与 GC-RL 的 generation-critique 双轮不同但都是两层结构
- [[AI/2-Agent/Agentic-RL/Search-P1-Path-Centric-Reward-Agentic-RAG]] — 同样"挖掘已有数据的未利用信号"，RAG 场景路径级 soft scoring
- [[AI/2-Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents]] — forward MC rollout vs. GC-RL backward critique rollout

> 注：论文实验数字（具体 benchmark 精度、消融结果）在本笔记写作时无法获取（网络限制）。机制分析基于 abstract + snippet 推导，可信度高；具体数字待补充。

