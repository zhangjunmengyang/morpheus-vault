---
title: "iGRPO: 迭代式自反馈 GRPO"
date: 2026-02-15
tags: [RL, GRPO, reasoning, self-feedback, interview]
type: note
---

# iGRPO: Iterative Group Relative Policy Optimization with Self-Feedback

> **论文**: [arXiv:2602.09000](https://arxiv.org/abs/2602.09000) — *iGRPO: Self-Feedback-Driven LLM Reasoning*
> **作者**: Ali Hatamizadeh, Shrimai Prabhumoye, Igor Gitman, Ximing Lu, Seungju Han, Wei Ping, Yejin Choi, Jan Kautz
> **机构**: NVIDIA
> **发表时间**: 2026 年 2 月
> **一句话总结**: 在标准 GRPO 基础上加入两阶段自反馈机制——模型先生成 draft，选最优 draft 作为条件，再生成 refined output 做 GRPO 更新——以相同 rollout 预算实现 AIME24 85.62%、AIME25 79.64% SOTA。

---

## 1. GRPO 回顾：Group Relative Policy Optimization

### 1.1 核心思想

GRPO 是 DeepSeek 团队在 DeepSeek-Math 中提出的策略优化算法（Shao et al., 2024），是 PPO 的一种**无 value function** 变体。核心创新在于：不需要训练一个 critic 模型来估计 advantage，而是通过**组内相对奖励归一化**来计算 advantage。

### 1.2 算法流程

1. 给定 prompt `q`，从当前策略快照 `π_θ_old` 采样一组 G 个候选输出 `{o_1, o_2, ..., o_G}`
2. 用奖励模型对每个输出打分得到 `{R_1, ..., R_G}`
3. 组内归一化计算 advantage：

```
Â_i = (R_i - mean({R_1, ..., R_G})) / std({R_1, ..., R_G})
```

4. 使用 PPO-style 的 clipped surrogate objective 更新策略：

```
J_GRPO(θ) = E[1/G Σ_i 1/|o_i| Σ_t [min(r_i,t(θ)·Â_i, clip(r_i,t(θ), 1-ε, 1+ε)·Â_i) - β·D_KL]]
```

其中 `r_i,t(θ) = π_θ(o_i,t | q, o_i,<t) / π_θ_old(o_i,t | q, o_i,<t)` 是 importance sampling ratio。

### 1.3 关键优势

- **无需 critic 模型**：不需要训练和维护价值网络，极大减少显存和计算开销
- **组内归一化**：天然提供了 baseline 减方差，同一 prompt 内的多个输出互相比较
- **简单高效**：相比 PPO 的 actor-critic 架构，实现更简洁，适合大规模 LLM 微调
- **已验证有效**：DeepSeek-R1 的核心训练算法之一

### 1.4 KL 惩罚

GRPO 使用 Schulman (2020) 提出的非负 per-token KL 估计器：

```
D̂_KL^(i,t) = π_ref(o_i,t | ...) / π_θ(o_i,t | ...) - log[π_ref(o_i,t | ...) / π_θ(o_i,t | ...)] - 1
```

该估计器保证非负，且在 `π_θ` 下采样时是对 `D_KL(π_θ || π_ref)` 的无偏估计。

---

## 2. iGRPO 的动机：标准 GRPO 的局限性

### 2.1 单次生成的天花板

标准 GRPO 的目标函数将每次生成视为**独立事件**——模型对每个 prompt 只有一次尝试的机会。这忽略了模型自身生成过程中的信息：

- **无自反思机制**：模型不能参考自己之前的尝试来改进
- **exploration 不足**：在复杂推理任务上，单次生成命中正确答案的概率较低
- **reward 信号稀疏**：对于困难问题，一组 G 个采样可能全部失败（reward 全为 0），此时 advantage 全为 0，该 prompt 对训练无贡献

### 2.2 人类解题的启示

人类解决复杂问题很少一步到位——通常是：
1. 先写一个初步方案（draft）
2. 回顾和检查这个方案
3. 基于发现的错误和不足进行修正
4. 迭代直到满意

这种 **"draft → reflect → refine"** 的模式在认知心理学中有大量研究支持（Flower & Hayes 1981; Polya 2014）。现有 RL 框架完全没有利用这一机制。

### 2.3 静态 ICL vs 动态自条件化

In-Context Learning (ICL) 通过在 prompt 中附加示例来引导生成：

```
J_ICL(θ) = E[R(o)]  where o ~ π_θ(·|q, e)
```

但 ICL 的示例 `e` 是**静态的**，不随策略进化而变化。当模型能力提升时，固定的示例可能不再是最优的条件信号。

iGRPO 提出 **dynamic self-conditioning**：条件信号由策略自身生成，并随训练共同进化。

### 2.4 GRPO 变体的定位

现有 GRPO 改进（Dr. GRPO、DAPO、GSPO 等）主要集中在**优化目标的稳定性**：
- **Dr. GRPO**：去除序列长度归一化和标准差归一化中的 bias
- **DAPO**：动态采样 + 解耦裁剪 + reward shaping
- **GSPO**：序列级重要性比率和裁剪

而 iGRPO 是 **正交（orthogonal）** 的——它不修改优化目标本身，而是改变**优化器看到的数据分布**。因此 iGRPO 可以与上述所有变体组合使用。

---

## 3. Self-Feedback 机制详解

### 3.1 两阶段架构概览

iGRPO 在每个优化步骤中执行两个紧密耦合的阶段：

```
                    ┌─────────────────────┐
                    │   Stage 1: Draft    │
                    │   Exploration       │
   prompt q ───────►│                     │──► best draft d̂
                    │   N 个候选 draft     │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
   prompt q' =     │   Stage 2: Conditioned│
   Concat(q, d̂) ──►│   Refinement        │──► G 个 refinement
                    │                     │    ──► GRPO 更新
                    └─────────────────────┘
```

**关键约束**：只有 Stage 2 的输出接收梯度更新；Stage 1 是不可微的探索机制。

### 3.2 Stage 1：Exploratory Draft Generation（探索性草稿生成）

**目标**：从当前策略中采样多个候选解，选出最优的作为自反馈信号。

**步骤**：
1. 给定 prompt `q`，从策略快照 `π_θ_old` 采样 N 个 draft：`d_i ~ π_θ_old(·|q), i=1,...,N`
2. 对每个 draft 计算奖励：`R_φ(d_i)`
3. 选择得分最高的 draft：`d̂ = argmax_i R_φ(d_i)`

**直觉**：
- 训练早期，`d̂` 可能是一个较弱的解，但至少是当前策略的"最好尝试"
- 随着训练进行，`d̂` 的质量逐步提升，趋近（但未达到）最优解
- 这实现了一种**隐式课程学习（implicit curriculum）**

**理论保证（Proposition 3.1）**：
对于二值奖励 `R ∈ {0, 1}`，如果策略成功概率为 `p`，则从 N 个 i.i.d. 采样中选出至少一个成功 draft 的概率为：

```
E[R(d̂)] = 1 - (1 - p)^N
```

当 `p` 随训练增大时，`E[R(d̂)]` 单调递增——这就是 **bootstrapping effect（自举效应）**。

### 3.3 Stage 2：Conditioned Refinement（条件化精炼）

**目标**：利用最佳 draft 作为上下文条件，生成比 draft 更好的答案。

**步骤**：
1. 构造增强 prompt：`q' = Concat(q, d̂)`（使用固定的 prompt template 连接）
2. 从 `π_θ_old` 采样 G 个 refinement：`o_j ~ π_θ_old(·|q'), j=1,...,G`
3. 对 refinement 计算奖励并归一化计算 advantage
4. 用标准 GRPO 目标函数更新策略

**核心洞察**：
- 模型不仅仅学习"抄写" draft——它学习的是一个**从 draft 到更好解的映射函数**
- Stage 2 的条件生成在 augmented prompt `q'` 上进行，所有 importance ratio 和 KL 都相应调整
- draft 提供了丰富的上下文信息：解题方向、可能的错误、中间步骤等

### 3.4 iGRPO 完整目标函数

```
J_iGRPO(θ) = E[q ~ P(Q)] E[{d_i} ~ π_θ_old(·|q), d̂ = argmax_i R(d_i),
                              q' = Concat(q, d̂), {o_j} ~ π_θ_old(·|q')]
             × 1/G Σ_j 1/|o_j| Σ_t [min(r_j,t(θ)·Â_j, clip(r_j,t(θ), 1-ε, 1+ε)·Â_j) - β·D̂_KL]
```

与标准 GRPO 的**唯一结构差异**：所有 Stage 2 的生成都以 augmented prompt `q'` 为条件，而非原始 prompt `q`。

### 3.5 奖励函数

采用标准的 rule-based binary reward（可验证推理任务的通用做法）：

```
R_φ(o) = 1[extract(o) = a]
```

即：从模型输出中提取最终答案，与 ground truth 比较，正确为 1，错误为 0。

### 3.6 推理时行为

**重要**：iGRPO 的两阶段机制**仅在训练时使用**。推理时，模型以标准的单次生成方式运行——直接从原始 prompt `q` 生成，无需 draft 生成、conditioning 或选择。

这意味着 iGRPO 训练出的模型在推理时没有额外计算开销。

---

## 4. 迭代训练流程

### 4.1 算法伪代码

```
Algorithm: iGRPO

输入: 预训练模型 M, 训练集 B = {(q, a)}, 奖励函数 R_φ,
      draft 数 N, 组大小 G, clipping ε, KL 系数 β, 迭代数 I, batch size S

初始化: π_θ ← M, π_ref ← M

for iteration = 1 to I:
    π_θ_old ← π_θ              # 快照策略用于采样
    采样 batch {(q^(k), a^(k))} ~ B

    for k = 1 to S:
        # ---- Stage 1: Draft Generation ----
        采样 drafts {d_i^(k)} ~ π_θ_old(·|q^(k)), i=1,...,N
        计算奖励 {R_φ(d_i^(k))}
        选择最佳 draft: d̂^(k) ← argmax_i R_φ(d_i^(k))

        # ---- Stage 2: Conditioned Refinement ----
        构造增强 prompt: q'^(k) ← Concat(q^(k), d̂^(k))
        采样 refinements {o_j^(k)} ~ π_θ_old(·|q'^(k)), j=1,...,G
        计算奖励 {R_φ(o_j^(k))}
        计算 advantage {Â_j^(k)}

    # ---- Policy Update ----
    θ ← θ + η·∇_θ J_iGRPO(θ)     # Eq. 5

return π_θ
```

### 4.2 计算开销分析

**核心结论：在相同 rollout 预算下，iGRPO 与 GRPO 计算开销相当。**

| 项目 | GRPO | iGRPO |
|------|------|-------|
| 每个 prompt 采样数 | G_GRPO | N (Stage 1) + G (Stage 2) |
| 预算约束 | — | N + G = G_GRPO |
| 实际配置示例 | G_GRPO = 16 | N = 8, G = 8 |
| 主要计算成本 | G_GRPO × C_gen | (N + G) × C_gen ≈ G_GRPO × C_gen |

论文保持 `N + G = G_GRPO`，所以 iGRPO **重新分配了相同数量的 rollout**，而非增加总量。额外的开销仅来自 prompt 拼接和 best draft 选择，可忽略不计。

### 4.3 Bootstrapping 动态

训练过程中的正反馈循环：

```
更好的策略 → 更好的 draft → 更好的 conditioning → 更好的 refinement → 更好的策略
    ↑                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
```

这种自举效应是 iGRPO 区别于所有先前方法的核心特性：
- 不需要外部 teacher 模型
- 不需要人工设计 curriculum
- conditioning 信号的质量自然随训练提升

### 4.4 Entropy Collapse 延迟

论文发现 iGRPO 的一个重要副作用：**延迟 entropy collapse**。

标准 GRPO 训练中，策略的 entropy 会较快下降（模型过早收敛到某些固定模式）。iGRPO 由于 draft conditioning 提供了额外的探索信号，使策略在更长时间内保持较高 entropy，从而：
- 保持更好的探索能力
- 避免过早陷入局部最优
- 最终达到更高的最终性能

---

## 5. 实验结果对比

### 5.1 实验设置

- **基础模型**：Nemotron-H-8B-Base-8K、DeepSeek-R1-Distill-Qwen-7B/14B、OpenMath-Nemotron-7B
- **训练数据**：MATH 数据集 (Hendrycks et al., 2021)；大规模实验使用 AceReason-Math
- **评估基准**：AIME24、AIME25、MATH500、AMC23、GSM8K、Minerva Math
- **对比方法**：GRPO、Self-Verification、Critique-GRPO
- **统一条件**：相同 rollout 预算（N + G = G_GRPO）

### 5.2 主要结果（Table 1 摘要）

#### 8B 模型（Nemotron-H-8B-Base-8K）

| 方法 | AIME25 | AIME24 | MATH500 | AMC | GSM8K | Minerva | Avg |
|------|--------|--------|---------|-----|-------|---------|-----|
| Base | 6.20 | 8.65 | 61.23 | 43.21 | 41.02 | 17.60 | 29.65 |
| + GRPO | 7.78 | 9.01 | 73.13 | 45.10 | 81.93 | 29.56 | 41.08 |
| + Self-Verification | 8.50 | 9.25 | 75.60 | 46.50 | 86.20 | 31.10 | 42.86 |
| + Critique-GRPO | 8.42 | 9.15 | 76.05 | 46.80 | 88.40 | 31.50 | 43.39 |
| **+ iGRPO** | **9.17** | **9.56** | **78.80** | **48.75** | **91.26** | **32.72** | **45.04** |

#### 7B 模型（DeepSeek-R1-Distill-Qwen-7B）

| 方法 | AIME25 | AIME24 | MATH500 | AMC | GSM8K | Minerva | Avg |
|------|--------|--------|---------|-----|-------|---------|-----|
| Base | 38.60 | 54.40 | 92.80 | 90.00 | 92.00 | 39.10 | 61.93 |
| + GRPO | 38.90 | 55.00 | 93.25 | 90.00 | 92.12 | 40.44 | 68.29 |
| + Self-Verification | 39.45 | 55.80 | 93.50 | 92.50 | 92.20 | 41.00 | 69.08 |
| + Critique-GRPO | 39.60 | 55.65 | 93.45 | 92.80 | 92.25 | 41.10 | 69.14 |
| **+ iGRPO** | **40.16** | **56.30** | **93.80** | **95.00** | **92.42** | **41.54** | **69.87** |

#### SOTA 结果（OpenReasoning-Nemotron-7B + AceReason-Math）

| Benchmark | 成绩 | 说明 |
|-----------|------|------|
| **AIME24** | **85.62%** | 7B 模型 SOTA |
| **AIME25** | **79.64%** | 7B 模型 SOTA |

### 5.3 关键观察

1. **一致性优势**：iGRPO 在所有测试的基础模型（7B、8B、14B）和所有 benchmark 上均优于标准 GRPO
2. **困难任务上增益更大**：在 AIME（最难的竞赛级数学题）上的提升比 GSM8K（简单数学题）上更显著
3. **超越同类方法**：iGRPO 在相同预算下始终优于 Self-Verification 和 Critique-GRPO
4. **大规模有效**：在大规模数据（AceReason-Math）和强基座（OpenReasoning-Nemotron-7B）上进一步推高 SOTA

### 5.4 Ablation 研究要点

论文通过消融实验验证了以下发现：

1. **Refinement wrapper 的通用性**：iGRPO 的两阶段机制不限于标准 GRPO，可以包裹 DAPO 等其他 GRPO 变体并带来提升
2. **Generative Judge 的价值**：使用生成式判断器（而非纯 rule-based）可进一步提升效果
3. **Entropy 动态变化**：iGRPO 显著延迟了 entropy collapse，使模型在训练后期仍保持探索能力
4. **Stage 1/Stage 2 的采样分配**：N 和 G 的比例影响性能，论文建议均分（如 N=8, G=8）
5. **Draft 选择策略**：选择最高 reward draft（vs 随机 draft）是关键——验证了 self-feedback 信号质量的重要性

---

## 6. 与其他方法的对比

### 6.1 iGRPO vs PPO

| 维度 | PPO | iGRPO |
|------|-----|-------|
| Value Function | 需要 critic 网络 | 不需要 |
| 计算开销 | 高（actor + critic） | 低（仅 policy） |
| Advantage 估计 | GAE（需要 V(s)） | 组内归一化 |
| 自反馈 | ❌ | ✅ 两阶段 self-conditioning |
| 显存占用 | ~2x 模型参数 | ~1x 模型参数 |

### 6.2 iGRPO vs DPO

| 维度 | DPO | iGRPO |
|------|-----|-------|
| 训练方式 | 离线（固定偏好对） | 在线（on-policy 采样） |
| 数据需求 | 需要 preference pair | 只需 verifiable reward |
| 探索能力 | 受限于离线数据 | 在线探索 + self-feedback |
| 迭代改进 | ❌ | ✅ bootstrapping |
| 适用范围 | 对齐任务为主 | 推理任务为主 |

### 6.3 iGRPO vs 标准 GRPO

| 维度 | GRPO | iGRPO |
|------|------|-------|
| 生成模式 | 单次独立生成 | 两阶段：draft → refine |
| 条件信号 | 无 | 模型自生成的 best draft |
| 信息利用 | 仅当前 prompt | prompt + 历史最佳尝试 |
| Entropy 动态 | 较快 collapse | 延迟 collapse |
| 计算开销 | N 次采样 | N 次采样（N_draft + N_refine） |
| 实现复杂度 | 简单 | 略增（需 draft 选择 + prompt 拼接） |

### 6.4 iGRPO vs DAPO

| 维度 | DAPO | iGRPO |
|------|------|-------|
| 创新方向 | 优化目标改进（动态采样、解耦裁剪） | 数据分布改变（self-conditioning） |
| 与 GRPO 关系 | 替代 GRPO 的变体 | GRPO 的扩展（可包裹 GRPO 或 DAPO） |
| 兼容性 | 替换 GRPO | 可叠加在 GRPO/DAPO 之上 |
| 核心机制 | reward shaping + clipping 改进 | 两阶段 draft-refine |

### 6.5 iGRPO vs Critique-GRPO

| 维度 | Critique-GRPO | iGRPO |
|------|---------------|-------|
| 反馈形式 | 自然语言 critique | 完整的 best draft |
| 反馈来源 | 模型生成 critique 文本 | 模型生成解并选最优 |
| 奖励信号 | 模型内部（可能不可靠） | 外部可验证奖励 |
| 分离性 | 奖励提供者和接受者角色模糊 | 生成过程和奖励信号明确分离 |

### 6.6 iGRPO vs Self-Verification

| 维度 | Self-Verification | iGRPO |
|------|-------------------|-------|
| 推理时 | 需要验证+重加权 | 标准单次生成 |
| 训练时 | 统一求解和验证 | 两阶段 draft-refine |
| 额外推理开销 | 有（验证） | 无 |

---

## 7. 对 Reasoning 模型训练的启示

### 7.1 Self-Feedback 是 Reasoning RL 的关键范式

iGRPO 证明了在 RL 训练中引入自反馈机制的价值：
- 模型能力与条件信号质量形成**正反馈循环**
- 不需要外部 teacher 或 human-in-the-loop
- 在可验证推理任务上效果显著

### 7.2 数据分布 > 优化算法

一个重要启示：**改变优化器看到的数据分布**可能比**改进优化目标本身**更有效。iGRPO 不修改 GRPO 的 loss function，但通过 self-conditioning 改变了训练样本的分布，获得了更大的性能提升。

### 7.3 Exploration 的重要性

iGRPO 延迟 entropy collapse 的发现强化了一个观点：在推理任务训练中，**保持探索能力**至关重要。过早收敛到某种固定策略会限制模型在困难问题上的表现。

### 7.4 Compute-Optimal 的 Rollout 分配

iGRPO 展示了一种新的 rollout 预算分配思路：与其把所有采样都用于独立生成，不如把一部分用于"先看一遍"（draft），另一部分用于"改进"（refine）。这种 **exploration-exploitation 分割**在固定计算预算下能获得更好的信号。

### 7.5 训练-推理分离

iGRPO 的设计哲学：**训练时可以复杂，推理时必须简单**。两阶段机制只在训练时使用，训练出的模型在推理时没有任何额外开销。这与一些需要推理时额外计算的方法（如 Self-Verification、best-of-N sampling）形成对比。

### 7.6 Orthogonal 改进的组合空间

iGRPO 与优化目标改进（Dr. GRPO、DAPO、GSPO）正交，与数据改进（更好的训练集）正交，与模型架构改进正交。这意味着它可以与几乎所有其他改进叠加使用，暗示了一个巨大的组合改进空间。

### 7.7 从 Draft 到 Agent 的推理范式

iGRPO 的 "draft → feedback → refine" 范式与 Agent 系统中的 "plan → execute → reflect → replan" 循环高度相似。这暗示了 RL 训练和 Agent 架构之间可能存在更深的统一框架。

### 7.8 Scale 效应

论文在 7B、8B、14B 上验证了一致的提升。一个自然的问题是：iGRPO 在更大规模（70B+）上效果如何？bootstrapping 效应是否会随 scale 增强（因为更大模型生成更好的 draft）？这是值得关注的方向。

---

## 8. 面试题及回答要点

### 面试题 1：请解释 iGRPO 的核心机制，以及它相比标准 GRPO 的关键改进是什么？

**回答要点**：

iGRPO（Iterative GRPO）是 NVIDIA 于 2026 年提出的 GRPO 扩展算法，核心创新是在标准 GRPO 的基础上引入了**两阶段自反馈机制**。

**标准 GRPO 的做法**是：对每个 prompt 采样一组 G 个候选输出，用组内相对奖励归一化计算 advantage，然后做 PPO-style 的策略更新。每次生成是独立的，不利用模型自身的先验尝试。

**iGRPO 的改进**分两个阶段：

- **Stage 1（Exploratory Draft Generation）**：对每个 prompt 先采样 N 个候选 draft，用奖励函数评估后选出得分最高的 `d̂` 作为"最佳草稿"。这一步不参与梯度计算。

- **Stage 2（Conditioned Refinement）**：将 `d̂` 拼接到原始 prompt 后面形成增强 prompt `q' = Concat(q, d̂)`，然后从增强 prompt 采样 G 个 refinement 输出，对这些输出执行标准 GRPO 更新。

**关键改进**体现在三个方面：
1. **Dynamic Self-Conditioning**：条件信号由模型自身生成，随训练共同进化，形成 bootstrapping 正反馈循环
2. **更高效的信号利用**：在相同 rollout 预算下（N + G = G_GRPO），通过重新分配采样获得更强的训练信号
3. **延迟 Entropy Collapse**：draft conditioning 帮助模型保持更长时间的探索能力

在实验中，iGRPO 在所有测试的基础模型和 benchmark 上一致优于标准 GRPO，在 OpenReasoning-Nemotron-7B 上达到了 AIME24 85.62%、AIME25 79.64% 的 SOTA。

---

### 面试题 2：iGRPO 如何在不增加计算预算的情况下超越标准 GRPO？请从理论和实践两个角度分析。

**回答要点**：

**实践角度——计算预算不变**：

iGRPO 通过约束 `N + G = G_GRPO` 来保持相同的 rollout 预算。例如，标准 GRPO 用 G=16 个采样，iGRPO 把这 16 个采样分为 N=8 个 draft 采样和 G=8 个 refinement 采样。主要计算成本 `C = (N + G) × C_gen ≈ G_GRPO × C_gen`，与标准 GRPO 相同。额外开销仅有 prompt 拼接和 argmax 选择，可忽略。

**那为什么性能更好？核心在于信号质量的提升。**

**理论角度——Bootstrapped Policy Improvement**：

1. **更好的条件信号**：Proposition 3.1 证明了对于二值奖励，从 N 个采样中选出至少一个正确 draft 的概率为 `1 - (1-p)^N`，远高于单个采样的成功率 `p`。这意味着模型更可能获得一个包含正确解法信息的 draft 作为条件。

2. **信息增益**：Stage 2 的 refinement 不是"从零开始"，而是基于一个已有的（通常是正确的）解法进行改进。这降低了问题难度——模型不需要独立发现完整解法，只需在已有解法基础上精炼。

3. **Bootstrapping 循环**：`更好的策略 → 更好的 draft → 更好的 conditioning → 更好的 refinement → 更好的策略`。这种正反馈循环使得性能提升可以复利积累。

4. **Entropy 动态**：标准 GRPO 在训练中 entropy 快速下降（策略过早收敛），而 iGRPO 的 draft conditioning 提供了隐式的多样性鼓励，延迟了 entropy collapse，使模型能在更大的策略空间中搜索更优解。

**本质上，iGRPO 用相同数量的采样获得了更高质量的训练信号**——它不是生成更多样本，而是更聪明地使用同样数量的样本。

---

### 面试题 3：如果你要将 iGRPO 应用到代码生成或多模态推理任务上，你会如何修改或调整？可能会遇到什么挑战？

**回答要点**：

**应用到代码生成**：

1. **奖励函数调整**：代码任务天然支持 verifiable reward（单元测试通过 = 1，否则 = 0），这与 iGRPO 使用的 binary reward 完全兼容。可以进一步设计粒度更细的 reward（部分测试通过给部分分）。

2. **Draft 的作用更大**：代码生成中，一个有 bug 但思路正确的 draft 提供的信息价值极高——模型可以识别 bug 并修复。这天然适合 iGRPO 的 "draft → refine" 范式。

3. **Prompt Template 设计**：需要设计代码场景下的连接模板，如 "以下是一个尝试解法，请审查并提供改进版本"。

**应用到多模态推理**：

1. **Draft 形式多样化**：对于视觉推理，draft 可能包含对图像的文字描述、初步推理链等，这些都可以作为 Stage 2 的条件。

2. **跨模态信息传递**：draft 可以将视觉信息"翻译"为文本，为 Stage 2 提供更易处理的推理起点。

**可能的挑战**：

1. **长序列问题**：代码生成的输出通常比数学解答长得多。Draft + refinement 会使 context 变得非常长，可能超出模型上下文窗口。需要考虑 draft 压缩或摘要。

2. **Reward 设计**：数学题有 ground truth 答案可以直接校验，但很多任务的 reward 更难 verifiable（如开放域代码生成、创意写作）。iGRPO 强依赖于 reward 信号的准确性。

3. **Draft 质量的初始冷启动**：如果基座模型在目标任务上能力很弱（如 base model 几乎无法生成有效代码），Stage 1 的所有 draft 可能都很差，导致 conditioning 信号无用甚至误导。可能需要先用 SFT 暖启动。

4. **计算成本的平衡**：对于需要执行测试用例的代码任务，每个 draft 的 reward 评估本身就有较高成本。N + G 的分配需要更仔细地权衡。

5. **Generalize 到非 binary reward**：论文的理论保证（Proposition 3.1）基于 binary reward。对于连续 reward，bootstrapping 效应的理论保证需要重新推导。不过实践中，选择 reward 最高的 draft 仍然是一个合理的启发式。

6. **组合爆炸**：iGRPO 与 DAPO 等其他改进正交，但组合调参空间也随之扩大。需要系统的 ablation 来找到最优配置。

**总体判断**：iGRPO 的核心思想（self-conditioning via best draft）是 task-agnostic 的，迁移到其他领域的主要工程挑战在于 reward 设计和 context length 管理，而非算法本身的局限。

---

## 附录：核心概念速查

| 概念 | 定义 |
|------|------|
| GRPO | Group Relative Policy Optimization，无 critic 的 PPO 变体 |
| iGRPO | Iterative GRPO，加入 draft-refine 两阶段自反馈 |
| Stage 1 | Exploratory Draft Generation：采样 N 个 draft，选最优 |
| Stage 2 | Conditioned Refinement：基于 best draft 生成并更新 |
| Dynamic Self-Conditioning | 条件信号由策略自身生成，随训练共同进化 |
| Bootstrapping Effect | 更好策略 → 更好 draft → 更好 conditioning → 更好策略 |
| Entropy Collapse 延迟 | iGRPO 训练中策略 entropy 下降更慢，保持探索能力 |
| Binary Reward | R(o) ∈ {0, 1}，提取答案与 ground truth 比较 |
| Rollout Budget | N + G = G_GRPO，计算开销与标准 GRPO 相当 |
| Orthogonal | iGRPO 与优化目标改进（DAPO 等）正交，可叠加 |

---

*最后更新: 2026-02-15*

## See Also

- [[AI/LLM/RL/_MOC|RL MOC]] — LLM 强化学习全图谱
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — iGRPO 在七大维度框架中的位置（迭代自反馈维度）
- [[AI/LLM/RL/GRPO/ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO]] — 同方向 GRPO 改进，改 advantage 计算方式
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 同期 GRPO 改进，改 trust region 适应性
- [[AI/LLM/RL/Frameworks/Slime-RL-Framework|Slime-RL]] — GRPO 系列的工程化框架参考
