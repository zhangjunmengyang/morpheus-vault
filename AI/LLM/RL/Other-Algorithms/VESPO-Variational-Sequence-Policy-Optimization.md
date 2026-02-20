---
title: "VESPO: Variational Sequence-Level Soft Policy Optimization"
type: note
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - off-policy
  - variational-inference
  - grpo-improvement
  - type/paper
date: 2026-02-20
---

# VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training

**arXiv**: 2602.10693  
**提交日期**: 2026-02-11  
**作者**: Guobin Shen, Chenxiao Zhao, Xiang Cheng, Lei Huang, Xing Yu  
**代码**: https://github.com/FloyedShen/VESPO  
**评分**: ★★★★★

---

## 一句话

VESPO 把 RL 中 importance weight 的 reshaping 统一为一个变分问题，推导出**闭合形式的 soft kernel** `ϕ(W) = W^α · exp(-λW)`，理论基础比所有 heuristic clip 方法（GRPO/GSPO/SAPO）都更严格——在 staleness ratio 高达 64×、全异步训练环境下依然稳定。

---

## 核心问题：为什么 off-policy 在 LLM RL 中如此普遍？

三个来源（论文明确列举）：
1. **mini-batch 分割**：把一个大 rollout batch 分成多个 mini-batch 顺序更新 → 后面的 mini-batch 用的是过时的参数（staleness ratio N = gbs/mbs）
2. **异步训练**：rollout engine 和 training engine 并行运行，rollout 永远比 training 慢几步
3. **train-inference mismatch**：训练引擎（FSDP/Megatron）和推理引擎（vLLM/SGLang）实现不同 → 相同输入产生略微不同的 logits。在 MoE 模型中路由决策不同会放大这个误差

**共同机制**：behavior policy μ ≠ current policy π → importance weight W = π/μ 产生偏差 → 训练不稳定或 collapse。

---

## 现有方法的问题

### 序列级 IS 的方差爆炸

序列级 IS weight 是 token 级 IS 之积：
```
W(τ) = ∏_{t=1}^{T} ρ_t   (ρ_t = π_θ(y_t|x,y<t) / μ(y_t|x,y<t))
```
而梯度是 token 级之和。这个**积-和结构**（product-sum）导致即使每个 token 只有微小偏差，W(τ) 的方差随序列长度 T 指数增长。

### 现有 ϕ(W) 都是 heuristic

把任何对 importance weight 的 reshaping 统一表示为 `ϕ(W)`：

```
∇J̃(θ) = E_{τ~μ}[ϕ(W(τ)) · R(τ) · ∇log π_θ(τ)]
```

- **GRPO**：token 级 clip → `ϕ_GRPO(ρ_t; A) = ρ_t（在 clip 范围内）or 0`。只是一阶近似，破坏 token 间依赖
- **GSPO**：序列级 1/T 长度归一化（几何均值）→ 引入长度偏差（长序列更不容易被 clip）
- **SAPO**：soft adaptive gating → 设计启发式，对负优势样本的 W<1 情况抑制不足，导致 length explosion

VESPO 的核心洞察：**任何 ϕ(W) 都隐式定义了一个 proposal 分布 Q**：
```
Q(τ) = (1/Z) · μ(τ) · ϕ(W(τ))
```
因此与其手工设计 ϕ，不如反过来——指定 Q 应该满足什么性质，然后推导对应的 ϕ。

---

## VESPO 的理论推导

### 变分目标：双近邻约束

Q 应该满足：
- 离 μ 近（采样效率）
- 离 π 近（减少估计偏差）
- 控制方差（有限样本下的可靠性）

形式化为带约束的优化：

```
min_Q  (1-α)·D_KL(Q‖μ) + α·D_KL(Q‖π)
s.t.   E_Q[W(τ)] ≤ C,   ∫Q = 1
```

α 控制 μ-π 之间的权衡。方差约束 E_Q[W] ≤ C 与经典 ESS（Effective Sample Size）指标直接相关。

### 闭合形式解

对拉格朗日函数取泛函导数 δℒ/δQ = 0，得：

```
Q*(τ) ∝ μ(τ)^(1-α) · π(τ)^α · exp(-λW(τ))
```

代入 Q(τ) ∝ μ(τ)·ϕ(W)，识别出 **reshaping kernel**：

$$\boxed{\phi(W) = W^\alpha \cdot \exp(-\lambda W)}$$

两个成分的直觉：
- **W^α**：power term，α 控制小 W 的行为（低概率样本的权重）
- **exp(-λW)**：soft exponential suppression，λ 控制大 W 的衰减速度

**与 hard clipping 的本质区别**：
- Hard clip：W 超过阈值直接截断（不连续，一阶不可导）
- VESPO kernel：光滑可微，在 W → ∞ 时自然饱和

### 代理目标

VESPO 隐式优化的是下不完全 Gamma 函数形式的代理目标：

```
f(W) = (1/λ^α) · γ(α, λW)   其中 γ(a,x) = ∫₀ˣ t^(a-1)·e^(-t) dt
```

光滑、无穷可微、W→∞ 时饱和——这正是 hard clip 的 smooth approximation。

### 实际算法

使用 shifted form 确保 ϕ(1) = 1（on-policy 样本得到单位权重）：

```
ϕ(W) = W^(c₁) · exp(c₂·(1-W))
```

**非对称超参**：正负优势分别用不同的 (c₁, c₂)：
- A > 0：(c₁, c₂) = (2.0, 3.0)
- A < 0：(c₁, c₂) = (3.0, 2.0)

理由：负优势样本 W < 1 时更容易被过度惩罚，需要更强的指数抑制（c₂=2 vs c₂=3 提供更温和保护）。

**实现**：全程在 log-space 计算，避免极端 IS 权重的数值溢出：
```
log ϕ = c₁·log W + c₂·(1-W)
```
W 本身也是 log-space 求和后再 exp。

---

## 实验结果

### 主要结果（Qwen3-30B-A3B-Base，gbs/mbs=8）

| Method | AIME25 | AIME24 | AMC23 | MATH500 | Avg |
|--------|--------|--------|-------|---------|-----|
| GRPO | 28.2 | 40.0 | 81.4 | 69.9 | 54.9 |
| GSPO | 24.6 | 34.1 | 80.5 | 68.8 | 52.0 |
| SAPO | 21.4 | 27.9 | 73.0 | 68.7 | 47.7 |
| **VESPO** | **34.2** | **44.1** | **80.3** | **70.2** | **57.2** |

### Staleness 鲁棒性（Qwen3-30B-A3B-Base）

| N (gbs/mbs) | GRPO Avg | GSPO Avg | SAPO Avg | **VESPO Avg** |
|------------|---------|---------|---------|------------|
| 4 | 49.8 | 56.1 | 61.5 | **66.4** |
| 8 | 54.9 | 55.3 | 47.7 | **66.9** |
| 16 | 47.7 | 55.1 | 46.5 | **63.9** |
| 32 | 49.2 | 47.5 | 19.8（collapse）| **61.4** |
| 64 | 44.7 | 45.8 | 18.4（collapse）| **58.5** |

**关键**：SAPO 在 N≥8 时完全 collapse；GRPO 和 GSPO 随 N 增大性能持续下降；**VESPO 在 N=64 时仍保持 58.5%，仅比 N=4 时的 66.4% 下降约 8 点**。

### 全异步训练

- GRPO：rollout log-perplexity > 2.0，PG loss 和梯度范数频繁大幅 spike，response length 剧烈震荡
- GSPO：稳定但收敛到较低 reward
- SAPO：早期 collapse
- **VESPO**：rollout KL、log-perplexity、PG loss、梯度范数全部接近零，方差最小，AIME25/24 最高

### MoE 架构的 Train-Inference Mismatch

| 方法 | 稳定性 | 训练 Reward |
|-----|-------|-----------|
| Vanilla GRPO | plateau ~0.60 | 最低 |
| GRPO + TIS | 改善 | - |
| GRPO + R2（路由回放）| 改善 | - |
| VESPO（无任何 fix）| ~= GRPO+R2 | - |
| **VESPO + R2** | 最稳定 | **最高** |

VESPO 无需工程 hack 就能匹配 GRPO+R2；加上 R2 后进一步提升（两者互补）。

### 消融：长度归一化是有害的

| 方法 | 训练稳定性 |
|-----|---------|
| VESPOlin（÷T）| 约 step 350 爆炸 collapse |
| VESPOsqrt（÷√T）| reward 约 0.58 后缓慢下降 |
| **VESPO（无归一化）**| **全程稳定，最高 reward** |

这解释了 GSPO 在某些 staleness 下 collapse 的原因：1/T 归一化让长序列更难被 clip，创造了正反馈循环 → 模型输出越来越长 → 最终 collapse。

---

## 我的分析

### 理论贡献的格局

VESPO 提供了**统一视角**：把 GRPO/GSPO/SAPO/DAPO 所有的 clip/norm 操作都归结为隐式定义不同的 proposal distribution Q。这个"measure-change 视角"是真正的理论贡献，不只是方法改进。

它给出了一个判断标准：**任何 ϕ(W) 的好坏，最终取决于它隐式定义的 Q 是否平衡了（proximity to μ）×（proximity to π）×（variance control）**。Hard clipping（GRPO）定义了一个对 W>1+ε 完全截断的 Q，从这个视角看是一种极端选择。

### ϕ(W) = W^α · exp(-λW) 的直觉

这个 kernel 的形状：
- W 很小（< 1）：主要由 W^α 决定，下权重 unlikely 的样本
- W ≈ 1（on-policy）：ϕ ≈ 1，保留完整梯度
- W 很大（>> 1）：exp 项迅速压制，比 hard clip 更平滑

这正是 RL 中 importance weight 应该有的行为：关注 near-policy 样本，平滑地忽略 far-off-policy 样本，而不是突然截断。

### 与 Jet-RL / VCPO 的关系

| | Jet-RL | Stable Asynchrony (VCPO) | VESPO |
|---|---|---|---|
| 解决方式 | **消除** off-policy 来源（精度统一）| **适应** off-policy（ESS-based LR）| **纠正** off-policy（soft IS reshaping）|
| 应用层面 | 系统工程层 | 优化算法层 | 策略梯度算法层 |
| 依赖 | H100+ FP8 | Song Han lab（细节未知）| 通用（任何 RL 框架）|

**三者正交**，可以叠加：Jet-RL 消除精度引入的 off-policy + VESPO 纠正残余 off-policy + R2 处理 MoE 路由不一致。

### 对 GRPO 全景综述的补充

我在 GRPO 全景综述的 Off-Policy 维度里，之前只有 Jet-RL 和 VCPO（待确认）。VESPO 提供了第三个视角：**算法层面的 soft correction**，而不是消除来源或限制学习率。

更重要的是：VESPO 的变分框架给"为什么 hard clip 是次优的"提供了理论证明——GRPO 的 ϕ(W) = clip 实际上是在隐式定义一个非常极端的 proposal Q（完全截断尾部）。理论上，任何 soft、differentiable 的 Q 都比这更好。

### 限制与质疑

1. **仅测数学推理**：同类问题，实际 production 代码/agent 场景下的表现未知
2. **(c₁, c₂) 超参需要调**：论文用 (2,3)/(3,2)，这些是从变分推导中涌现的还是实验调出来的？论文说"treat as tunable"——说明偏离了纯粹的变分推导，接受了工程 trade-off
3. **Q3-30B-A3B MoE 具体数字**：部分实验 baseline 差异较大，需要检查 implementation 是否 fair（SAPO 直接 collapse 让人怀疑超参）
4. **计算开销**：序列级 log W 的计算需要存储所有 token 的 log-prob（μ 和 π），比 token 级 clip 略多内存，但论文说"无额外参数开销"

---

## 与 GRPO 全景综述的连接

- **Off-Policy 维度**：VESPO 是第三种路径（算法纠正），补充 Jet-RL（系统消除）和 VCPO（动态适应）
- **Trust Region 维度**：VESPO 的 kernel 是 GRPO hard clip 的理论推广，提供了 soft trust region 的数学基础
- 与 **MASPO** 的关系：MASPO 关注正/负样本概率质量差异（token 级），VESPO 关注序列级 IS variance（sequence 级）——不同层次，但都在说"固定 clip 是次优的"

---

## 关键词连接

- [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 消除 off-policy vs 纠正 off-policy，互补；Jet-RL 从精度角度保证 on-policy，VESPO 从理论角度纠正 off-policy 梯度
- [[AI/LLM/RL/Other-Algorithms/Stable-Asynchrony-VCPO-Off-Policy-RL|VCPO]] — **同类问题（异步/off-policy），不同路径**：VCPO=LR scaling（系统工程），VESPO=变分 IS reshaping（算法理论）；两篇互读理解 off-policy 全貌
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — VESPO 归属 Off-Policy 稳定性维度，是该维度理论最强一篇
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 都在解决 fixed ε 的局限，token 级 vs sequence 级

---

## 总结

VESPO 是这批 GRPO 改进论文里**理论深度最高**的一篇。它不只是提出了一个新方法，而是提供了一个统一框架来分析所有现有方法。

`ϕ(W) = W^α · exp(-λW)` 这个 kernel 来自变分原理，不是拍脑袋设计的——这让它在理论上比 GRPO/GSPO/SAPO 所有的 heuristic clip 都更有说服力。

实验上，N=64 staleness 下仍保持 58.5%（GRPO 只有 44.7%）、全异步训练下唯一稳定——这些数字说明这个 kernel 确实有效，不只是理论上优雅。

如果要在 production RL 系统里选一个 off-policy 修复方案：VESPO 是算法层面最 principled 的选择，可以和 Jet-RL（系统层）+ R2（MoE 工程层）叠加使用。
