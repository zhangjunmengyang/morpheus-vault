---
title: "LAD: Learning Advantage Distribution for Reasoning"
brief: 把 GRPO 的'最大化期望 advantage'重构为'匹配优势诱导分布'：Lemma 3.1 证明 trust-region RL 最优策略 ∝ π_old·exp(A/η)，对应两分布相等的分布匹配问题；Lemma 3.2 推导出无需计算配分函数的实用代理 loss（f-divergence 最小化）；自然保留多模式推理轨迹，无需显式熵正则。Qwen2.5-7B AIME2024 +3.31，多样性 dist-4 +0.154，Codeforces 82.5%。UW-Madison，arXiv:2602.20132。
date: 2026-02-25
type: paper-note
rating: ★★★★☆
venue: arXiv (cs.LG)
arxiv: "2602.20132"
authors: Wendi Li, Sharon Li
affiliation: UW-Madison
tags:
  - rl
  - grpo
  - advantage-distribution
  - f-divergence
  - diversity
  - policy-optimization
  - llm-reasoning
sources:
  - arXiv:2602.20132 https://arxiv.org/abs/2602.20132
related:
  - "[[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO改进全景]]"
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO深度理解]]"
  - "[[AI/3-LLM/RL/算法/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]]"
  - "[[AI/3-LLM/RL/算法/REBEL-Regret-Based-RL-LLM-Alignment|REBEL]]"
  - "[[AI/3-LLM/RL/算法/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]]"
---

# LAD: Learning Advantage Distribution for Reasoning

**arXiv**: 2602.20132  
**作者**: Wendi Li, Sharon Li (UW-Madison)  
**提交**: 2026-02-23  
**状态**: cs.LG，开源代码  
**评分**: ★★★★☆

---

## 一句话总结

GRPO 等算法的核心问题是"最大化期望奖励"——这会把策略 collapse 到单一高奖励轨迹。LAD 把问题重构为**分布匹配**：让策略诱导的分布去匹配优势函数诱导的分布，通过 f-divergence 最小化实现，自然地保留多模式推理轨迹，无需额外熵正则。

---

## 问题诊断

### GRPO 的根本限制

GRPO/PPO 的目标函数：

$$J(\pi_\theta) = \mathbb{E}_{x,y}[A(x,y) \log \pi_\theta(y|x)]$$

这个目标有个结构性缺陷：**它对 advantage 的方向敏感，但对已经高概率的轨迹没有任何抑制机制**。后果：
- 高奖励轨迹概率不断提升 → 占据几乎所有概率质量
- 其他同样正确但"非主流"的推理轨迹被压制
- 多样性丧失 → Pass@k 提升有限，Avg@k 停滞

熵正则（EntAdv/KLCov/ClipCov）只是治标：它们加了额外的惩罚项，但核心目标仍是期望奖励最大化。

---

## 核心理论

### Lemma 3.1 — 最优策略的形式

Trust-region 约束 RL（PPO/TRPO 形式）的最优策略满足：

$$\frac{\pi^*_\theta(y|x) / \pi_{old}(y|x)}{Z_\pi(x)} = \frac{e^{A(x,y)/\eta}}{Z_A(x)}$$

其中 $\eta > 0$ 是 Lagrange 乘子，$Z_A(x) = \sum_y e^{A(x,y)/\eta}$ 是归一化常数。

**这个等式说明**：最优策略更新 $\propto e^{A(x,y)/\eta}$——高优势轨迹指数级更高概率，但所有轨迹都保留非零概率。

### 两个关键分布

LAD 把这个等式**具体化**为两个分布：

$$\mathcal{P}_{\pi_\theta}(y|x) = \frac{\pi_\theta(y|x)/\pi_{old}(y|x)}{Z_\pi(x)} \quad \text{（策略诱导分布）}$$

$$\mathcal{P}_A(y|x) = \frac{e^{A(x,y)/\eta}}{Z_A(x)} \quad \text{（优势诱导目标分布）}$$

**Lemma 3.1 的推论**：$\pi^*$ 时，$\mathcal{P}_{\pi_\theta} = \mathcal{P}_A$。

### LAD 目标函数

最小化两个分布之间的 f-divergence：

$$\mathcal{L}_{LAD}^{theory} = \mathbb{E}_{x} \sum_y \frac{e^{A(x,y)/\eta}}{Z_A(x)} f\left(\frac{\pi_\theta(y|x)/\pi_{old}(y|x)}{e^{A(x,y)/\eta}} \cdot \frac{Z_A(x)}{Z_{\pi_\theta}(x)}\right)$$

**问题**：$Z_A(x)$ 和 $Z_{\pi_\theta}(x)$ 对大 action space（自然语言）不可计算。

### Lemma 3.2 — 等价代理 Loss

关键发现：f-divergence 目标的最优解仅取决于权重函数的**比值结构**，而非绝对值。

令 $g_1(x,y) = g_2(x,y) = \pi_{old} \cdot e^{A(x,y)/\eta}$，则最优策略不变，但目标变为：

$$\boxed{\mathcal{L}_{LAD} = \mathbb{E}_{x, y \sim \pi_{old}} \left[ e^{A(x,y)/\eta} \cdot f\left(\frac{\pi_\theta(y|x)/\pi_{old}(y|x)}{e^{A(x,y)/\eta}}\right) \right]}$$

这个**实用目标**：
- 无需计算 $Z_A(x)$（归一化项被消除）
- 与理论目标有相同的最优解 $\pi^* \propto \pi_{old} \cdot e^{A/\eta}$
- 训练代价与 GRPO 相同（单次 generation）

---

## 梯度的隐式正则化

LAD 梯度：

$$\mathbb{E}_{y \sim \pi_\theta} \left[ f'\!\left(\frac{\pi_\theta/\pi_{old}}{e^{A(x,y)/\eta}}\right) \cdot \nabla_\theta \log \pi_\theta(y|x) \right]$$

关键机制：**f' 的参数同时依赖 advantage 和 likelihood ratio**。

- 当某轨迹 $y$ 有大 advantage（$A$ 大）但 $\pi_\theta/\pi_{old}$ 已经很大（已经被多次强化）：
  - 分子 $\pi_\theta/\pi_{old}$ 大，分母 $e^{A/\eta}$ 大
  - 比值 $\to 1$ 时 $f'(1) = 0$（f-divergence 在 P=Q 时梯度为零）
  - **自动抑制**：不会继续 overfit 这条轨迹

- 对比 PPO clip：硬截断（不连续）；LAD：软抑制（连续，由 f 的凸性驱动）

这就是"无需显式熵正则"的原因——正则化是 f-divergence 结构的内生性质。

---

## 与 FlowRL 的关系

FlowRL（GFlowNet trajectory balance loss）也是"离开期望奖励最大化"的方法，但：

- LAD 证明 FlowRL 是 LAD 框架在**更严格条件**下的特例（需要更强的 policy update 约束和参数缩放假设）
- LAD 更一般，直接匹配优势诱导分布的全结构
- 实验对比：1.5B 模型，LAD Avg@32 48.88 vs FlowRL 41.13（MATH series）

---

## 实验结果

### 数学推理（Qwen2.5-7B, DAPO-MATH）

| Method | AIME 2024 | AIME 2025 | OlympiadBench | Avg |
|--------|-----------|-----------|---------------|-----|
| GRPO | 14.03 | 4.86 | 39.47 | 37.28 |
| FlowRL | 14.67 | 7.51 | 40.80 | 37.28 |
| LAD | **19.60** | **8.84** | **41.25** | **40.08** |

AIME 2024 +3.31（绝对值），AIME 2025 +1.33，Average +2.8

### 代码推理（DeepSeek-R1-Distill-7B, DeepCoder）

| Method | CodeForces Rating | Percentile |
|--------|------------------|------------|
| GRPO | 1355.35 | 70.30% |
| ClipCov | 1473.09 | 79.20% |
| LAD | **1533.64** | **82.50%** |

### 多样性指标（AIME 24/25）

| Method | dist-3 ↑ | dist-4 ↑ | GPT4-Judge ↑ |
|--------|----------|----------|-------------|
| GRPO | 0.2306 | 0.2902 | 2.04 |
| FlowRL | 0.2878 | 0.3654 | 2.38 |
| LAD | **0.3498** | **0.4442** | **2.58** |

dist-3/4 = 不同 3/4-gram 的比例（越高越多样）；LAD 多样性提升显著。

---

## 关键洞察与评价

### 理论贡献的价值

LAD 的理论框架做了一件重要的事：**指出了 trust-region RL 的最优解形式蕴含了一个分布匹配问题**，但这个结构长期被忽略（文献只取了它的 gradient 形式）。把这个隐含结构显式化，就得到了一个更好的优化目标。

这不是 trick，是对 RL for LLM 基本问题的 **重新建模**。

### 与 OAPL 的对比（同类工作）

| | OAPL（2602.19362）| LAD（2602.20132）|
|--|--|--|
| 核心方向 | off-policy，regression 到 V* | on-policy，分布匹配 $\mathcal{P}_A$ |
| 主要 benefit | 训练速度 2x，内存 -30% | 多样性保留，对 difficult tasks 提升大 |
| 场景 | 异步分布式训练，单轮 | 标准 on-policy，多轮兼容 |
| f-divergence 选择 | 不涉及 | JS divergence 效果最佳 |

两篇方向正交，可以组合：OAPL 解决 off-policy 问题，LAD 解决分布 collapse 问题。

### 局限

- η（temperature）和 f-divergence 选择需要 ablation，工程门槛略高
- 主要在数学/代码任务验证，agent 任务（多轮、稀疏 reward）效果待验证
- 理论上对 multimodal advantage 的恢复依赖足够多的 samples 被良好覆盖

---

## 与现有体系的关系

GRPO 改进全景中，LAD 位于**目标函数范式转移**这一支：

```
GRPO 问题树：
├── Advantage 估计 → Blockwise / GiGPO / AgentPRM
├── Trust region → MASPO / DAPO / DEEP-GRPO
├── 训练稳定性 → STAPO / VESPO / Goldilocks
├── Off-policy 问题 → OAPL / REBEL / Stable Asynchrony
└── 目标函数范式 → FlowRL（GFlowNet TB loss）← LAD 的更一般框架
                    LAD（f-divergence 分布匹配）← 本文 ★★★★☆
```

LAD 是当前"目标函数范式转移"方向里理论最干净的一篇。

---

## 结论

LAD 把 RLVR 的基础问题从"最大化期望 advantage"重新表述为"匹配 advantage 诱导分布"——这不只是换了个损失函数，而是对优化目标本质的理论重建。实用代理 loss 的推导（Lemma 3.2）使得这个理论想法以零额外计算代价落地。结果：多样性显著提升，难题（AIME）性能大幅改善。

**推荐给想理解 RL for LLM 优化目标本质、或者在 GRPO 上做工程改进的人。**

---

*写于 2026-02-25 学者 HB35，morpheus-vault/AI/LLM/RL/Other-Algorithms/*

---

## See Also

**目标函数范式转移方向（LAD 的直接对话对象）：**
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO改进全景]] — LAD 在"目标函数范式"这一支的完整定位
- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO深度理解]] — 理解 LAD 改进的是什么

**同期相关工作（互补视角）：**
- [[AI/3-LLM/RL/算法/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] — 正交方向：off-policy regression；LAD 是 on-policy 分布匹配（可组合）
- [[AI/3-LLM/RL/算法/REBEL-Regret-Based-RL-LLM-Alignment|REBEL]] — 同为"离开期望奖励最大化"的重新建模，regret vs distribution matching
- [[AI/3-LLM/RL/算法/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — Trust region 方向的改进，与 LAD 正交可叠加
- [[AI/3-LLM/RL/算法/SAPO-Soft-Adaptive-Policy-Optimization|SAPO]] — **三维正交可叠加**：SAPO 改 clip 函数（sigmoid 软衰减），LAD 改目标函数范式；OAPL+LAD+SAPO 三者覆盖 off-policy/collapse/clip 三个独立问题维度

**GFlowNet 理论前驱：**
- LAD 证明 FlowRL（GFlowNet trajectory balance）是 LAD 在更严格条件下的特例；LAD 更一般

## 推荐阅读

1. **原文**：[arXiv:2602.20132](https://arxiv.org/abs/2602.20132) — Learning Advantage Distribution for Reasoning
2. **前置**：[[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO深度理解]] — 理解 GRPO 的结构性缺陷才能理解 LAD 的动机
3. **对比**：[[AI/3-LLM/RL/算法/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] — 同期两篇，一个解决 off-policy，一个解决 collapse；组合使用是当前理论上最完整的方案
