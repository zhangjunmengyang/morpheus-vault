---
title: "SSPO: Soft Sequence Policy Optimization — 统一 GMPO 与 SAPO"
tags: [RL, GRPO, off-policy, ImportanceSampling, SoftGating, PolicyOptimization]
created: 2026-02-27
status: permanent
rating: ★★★☆
arxiv: "2602.19327"
related:
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解]]"
  - "[[AI/3-LLM/RL/实践/DAPO-verl实践]]"
  - "[[AI/2-Agent/Agentic-RL/SORL-Stabilizing-Off-Policy-RL-Long-Horizon-Agent]]"
---

# SSPO: Soft Sequence Policy Optimization

> arXiv:2602.19327 | 2026-02-22 | ★★★☆  
> 全名：Soft Sequence Policy Optimization: Bridging GMPO and SAPO

---

## 一句话定位

GRPO 用 token-level 硬截断（clip），存在梯度信号丢失和 entropy collapse 风险。两条独立改进路线——GSPO（sequence-level IS）和 SAPO（soft gate 替代 clip）——各解决了一半问题。**SSPO 统一这两条路线**：在 sequence-level IS 框架下引入 token-level soft gating，既保留 token 粒度的自适应性，又对齐 reward 的 sequence-level 语义。

---

## 背景：GRPO 的两个已知问题

### 问题 1：token-level vs sequence-level reward 的量纲错配

GRPO 用 token-level 重要性比率（IS ratio），但 reward 是 sequence-level 的（整条输出的 reward）：

```
GRPO 目标：
J = E_{τ} [ Σ_t min(r_t · A, clip(r_t, 1-ε, 1+ε) · A) ]
         where r_t = π_θ(y_t|x,y_<t) / π_old(y_t|x,y_<t)
```

问题：sequence reward A 分配给每个 token 的 IS correction 是 r_t（单 token ratio），但 sequence-level 的 off-policy 程度应该是整条序列的 IS 乘积，而不是单 token 的。这导致对 off-policy 程度的校正不准确。

**GSPO 的修复**：用 geometric mean 的 token ratios 近似 sequence-level IS：

```
r_seq = (Π_t r_t)^{1/T}   # 几何均值 = length-normalized sequence IS
```

### 问题 2：硬截断（clip）的梯度消失

PPO/GRPO 的 clip 操作：当 r_t > 1+ε 或 r_t < 1-ε 时，梯度为 0——高偏差样本的信号被完全截断，导致训练效率下降和 entropy collapse（policy 倾向于高概率但保守的输出）。

**SAPO 的修复**：用 sigmoid-based soft gate 替代硬 clip：

```
soft_gate(r_t) = σ(α · (r_t - r_threshold))
```

Sigmoid 渐进式衰减：偏差大时 gate → 0（但不是突然截为 0），保留一定梯度信号。

---

## SSPO 的统一框架

SSPO 把两个修复组合进同一个目标：

### 核心公式（概念级）

```
SSPO 目标：

J_SSPO = E_{τ} [
    r_seq_soft · A
]

其中：
r_seq_soft = geometric_aggregate(soft_gate(r_1), soft_gate(r_2), ..., soft_gate(r_T))
           = (Π_t σ(f(r_t)))^{1/T}

f(r_t) = α · (r_t - r_threshold)   # α 是温度参数
```

**关键对应关系**：

| 极限情况 | SSPO 退化为 |
|---------|-----------|
| σ → Heaviside（硬截断），geometric mean | GSPO |
| token-level aggregation，soft gate | SAPO |
| 既有 geometric mean 又有 soft gate | **SSPO**（完整形式）|

**理论联系**：不带 clip 时，token 比率的几何均值 = length-normalized sequence IS（GSPO 的形式）。这证明 GMPO（Geometric Mean Policy Optimization）是 GSPO 的 token-level 等价。SSPO 在此基础上加 soft gate，使得不需要硬截断就能控制极端 IS ratio 的影响。

---

## 与相关工作的关系

### GRPO 算法谱系定位

```
GRPO
├── 修改 IS 粒度（token→sequence）
│   ├── GSPO（geometric mean，sequence-level IS）
│   └── GMPO（token ratio 几何均值，理论等价 GSPO）
├── 修改截断机制（hard clip→soft）
│   ├── SAPO（sigmoid gate，全序列）
│   └── DAPO（decoupled clip：生成/截断解耦）
└── 统一（IS粒度 + 软截断）
    └── SSPO ←（本文）
```

### vs SORL（arXiv:2511.20718）

SORL 解决 **multi-turn** off-policy 问题（turn-level IS，防止长 horizon IS ratio 指数爆炸）。
SSPO 解决 **单轮** off-policy 问题（mini-batch reuse 的 token-level IS 不准确）。

两者是不同层面的 off-policy 校正：SORL 是时间维度（跨 turn），SSPO 是空间维度（token vs sequence）。

### vs VESPO（arXiv:2602.10693）

VESPO（Variational Sequence-Level Soft Policy Optimization）也是"sequence-level + soft"方向，用变分框架推导。SSPO 用 geometric aggregation 框架推导，理论视角不同，目标类似。两者可视为同一时期的平行工作。

---

## 关键洞察

### "几何均值 = sequence IS"的等价性

这是论文的核心理论贡献：

**定理**（非正式）：在无 clip 的情况下，

$$\left(\prod_{t=1}^T r_t\right)^{1/T} = \exp\left(\frac{1}{T}\sum_{t=1}^T \log r_t\right) \approx \frac{\pi_\theta(y|x)}{\pi_{old}(y|x)}^{1/T}$$

这最后一项恰好是 sequence-level IS ratio 的 length-normalized 形式（log-domain 平均 = geometric mean）。

**意义**：GRPO 的 token-level IS 和 GSPO 的 sequence-level IS 之间的差距，被精确量化为"是否取几何均值"。这为统一提供了数学基础。

### Soft Gate 的优势：attenuation vs truncation

PPO-clip：$r_t$ 超出 $[1-\epsilon, 1+\epsilon]$ → 梯度为 0（硬截断）
SSPO soft gate：$r_t$ 偏大 → gate 值渐近 0，但**仍有非零梯度**

这对 entropy collapse 的缓解机制：
- 硬截断使高偏差 token 完全不参与更新 → policy 倾向于低 entropy（保守）
- 软衰减允许高偏差 token 有小梯度 → 一定程度维持 entropy

---

## 局限与批判

**实验结果未知**：目前检索到的信息未包含 SSPO 的 benchmark 数字。理论框架清晰，但实证验证的说服力待查。

**超参数敏感性**：soft gate 引入了额外超参（α 温度参数，threshold），相比 GRPO 的单一 ε，调参负担更重。

**计算成本**：geometric aggregation 需要对每个 token 的比率取 log 再求平均，与 GRPO 相近，增量成本低——这是优点。

**和 VESPO/SAPO 的区分度**：三个方向都在做"sequence-level + soft gate"，区别在于数学框架。目前缺少系统对比，不清楚哪个在实践中更优。

---

## 在 Policy Optimization 算法谱系中的定位（MEMORY.md 补充）

```
RL 目标函数大家族（截至 2026-02-27）

期望奖励最大化：
  GRPO / PPO / RLOO / REINFORCE

IS 粒度优化（off-policy 校正）：
  GSPO（sequence-level geometric mean）
  GMPO（token ratio 几何均值，GSPO 等价）
  **SSPO（geometric mean + soft gate，统一框架）**
  VESPO（变分推导的 sequence-level soft）

截断机制改进（anti-entropy-collapse）：
  DAPO（decoupled clip + dynamic sampling）
  SAPO（sigmoid soft gate 替代 hard clip）
  Dr. GRPO（非对称 clip + 去 std 归一化）

Multi-turn 稳定性：
  SORL（turn-level IS + CTN，时间维度 off-policy）

分布匹配范式：
  LAD（f-divergence matching，最一般框架）
  FlowRL（GFlowNet trajectory balance）
```

---

## See Also

- [[AI/3-LLM/RL/算法/GC-RL-Second-Order-Rollout-Generation-Critique]] — 同日发现，另一类 RL 信号利用效率提升
- [[AI/2-Agent/Agentic-RL/SORL-Stabilizing-Off-Policy-RL-Long-Horizon-Agent]] — multi-turn 维度的 off-policy 校正
- [[AI/3-LLM/RL/实践/DAPO-verl实践]] — 同一 anti-clip 方向的工程实现

> 注：SSPO 的 benchmark 数字（在 MATH/GSM8K 等标准评测上的具体性能）未能获取（网络限制）。机制分析基于 abstract + snippet + VESPO 论文中对 SSPO 的引用推导，置信度高；实验部分待补充。

