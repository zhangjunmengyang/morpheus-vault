---
title: "SAPO: Soft Adaptive Policy Optimization"
brief: 用 sigmoid 软门控替代 GRPO/GSPO 硬截断，非对称温度控制正负 advantage token 的梯度衰减速率；在正常训练条件下理论等价于 GSPO 连续版，在 MoE 异构场景下自动退回 token 级精确控制；Alibaba Qwen 团队 Qwen3-VL 系列实际生产方法。
arxiv: "2511.20347"
date: 2025-11-25
updated: 2025-12-01
venue: Preprint
rating: ★★★★☆
authors:
  - Chang Gao
  - Chujie Zheng
  - Bowen Yu
  - An Yang
  - Junyang Lin
tags:
  - RL
  - policy-optimization
  - trust-region
  - GRPO
  - MoE
  - Qwen
  - soft-gating
  - training-stability
type: paper-note
sources:
  - arXiv:2511.20347 — https://arxiv.org/abs/2511.20347
  - "GSPO (前驱): arXiv:2507.18071"
  - Qwen3-VL 技术报告 (生产验证)
related:
  - "[[GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]]"
  - "[[GSPO-Group-Sequence-Policy-Optimization|GSPO]]"
  - "[[OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]]"
  - "[[LAD-Learning-Advantage-Distribution|LAD]]"
  - "[[VESPO-Variational-Sequence-Policy-Optimization|VESPO]]"
  - "[[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]]"
---

# SAPO: Soft Adaptive Policy Optimization

**arXiv**: 2511.20347 (v2: 2025-12-01)  
**机构**: Alibaba Qwen Team (Chang Gao, Chujie Zheng, Bowen Yu, An Yang, Junyang Lin 等)  
**评分**: ★★★★☆  
**关键词**: policy optimization, soft gating, trust region, MoE RL, Qwen3-VL

---

## 一句话 TL;DR

用 **sigmoid 形软门控函数**取代 GRPO/GSPO 的硬截断（hard clipping），同时对正/负 advantage token 使用**非对称温度**，在保持学习信号的同时平滑控制 off-policy 程度。Alibaba Qwen 团队实际用于 **Qwen3-VL 系列**训练的生产方法。

---

## 动机与问题

### GRPO 的 hard clipping 困境

GRPO 在 token 级别用 $\varepsilon$-band 硬截断：band 外梯度为零，band 内梯度不变。这是二元信号：
- **太紧**：有效样本减少，学习信号丢失
- **太松**：off-policy 噪声过大，不稳定

### GSPO 的 sequence-level clipping 困境

GSPO 在序列级别截断（$s_i(\theta)$ = token ratio 的几何平均）——能避免 token 级高方差，但**全有全无**：只要序列中有几个高 off-policy token，整个序列的梯度都被抹零，浪费了序列中其余 near-on-policy token 的信号。

### MoE 的特殊挑战

MoE 模型中 expert routing 的异构性会**放大** token 级 importance ratio 的方差。Qwen3-30B-A3B（MoE）的 $\text{Var}_i(\theta)$ 分布比 Qwen3-4B（dense）宽得多——hard clipping 在 MoE 上更容易失效。

---

## 核心方法：SAPO

### 目标函数

$$\mathcal{J}(\theta) = \mathbb{E}_{q,\{y_i\}} \left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} f_{i,t}(r_{i,t}(\theta)) \cdot \hat{A}_{i,t} \right]$$

### 软门控函数 $f_{i,t}$

$$f_{i,t}(x) = \sigma(\tau_{i,t}(x-1)) \cdot \frac{4}{\tau_{i,t}}$$

其中 $\sigma$ 是 sigmoid，$\tau_{i,t}$ 是温度参数。

**三点关键性质**：
1. 在 $r_{i,t}(\theta)=1$（on-policy）时，$f_{i,t}(1) = 2/\tau_{i,t}$，梯度权重 $w_{i,t}=4p(1-p)=1$——与无截断目标完全等价
2. 随 $r$ 偏离 1，梯度平滑衰减（非突然截零）
3. 衰减速率由 $\tau$ 控制：$\tau$ 越大，衰减越快

### 梯度权重

$$w_{i,t}(\theta) = 4p_{i,t}(\theta)(1-p_{i,t}(\theta)), \quad p_{i,t}(\theta) = \sigma(\tau_{i,t}(r_{i,t}(\theta)-1))$$

这是个 bell-shaped 函数（sech² 形），peak 在 $r=1$，双边平滑衰减。

### 非对称温度（关键设计）

$$\tau_{i,t} = \begin{cases} \tau_{\text{pos}}, & \hat{A}_{i,t} > 0 \\ \tau_{\text{neg}}, & \hat{A}_{i,t} \leq 0 \end{cases}, \quad \tau_{\text{neg}} > \tau_{\text{pos}}$$

**为什么负 advantage 需要更快衰减？**

分析梯度传播：负 advantage 的梯度会**提升**词表中**所有未采样 token** 的 logit（词表大小 $|\mathcal{V}|$ 通常数十万）。大量无关 token 的 logit 被同时上推，导致不稳定性远大于正 advantage。故 $\tau_{\text{neg}} > \tau_{\text{pos}}$，对负梯度更快压制。

---

## 统一框架：gating function 视角

**关键洞察**：GRPO、GSPO、SAPO 可以用统一代理目标 $\mathcal{J}(\theta) = \mathbb{E}[\sum f_{i,t}(r_{i,t}(\theta)) \hat{A}_{i,t}]$ 表达，区别只在 $f_{i,t}$ 的选择：

| 算法 | $f_{i,t}$ 类型 | 特征 |
|------|----------------|------|
| GRPO | 硬截断（token 级）| 二元权重，band 外梯度=0 |
| GSPO | 硬截断（序列级）| $f$ 不依赖 $t$（token 无差别）|
| SAPO | 软门控 sigmoid（token 级）| 连续衰减，非对称温度 |

**GRPO 梯度**：$f^{\text{GRPO}'}(r) \in \{0, 1\}$（分段常数，不可微）

**SAPO 梯度**：$f^{\text{SAPO}'}(r) = \text{sech}^2\!\left(\frac{\tau}{2}(r-1)\right)$（连续，处处可微）

---

## SAPO 与 GSPO 的理论等价（Theorem）

在两个温和假设下：
- **(A1) 小步/on-policy**：$r_{i,t}(\theta) \approx 1$，故 $\log r \approx r-1$
- **(A2) 序列内低方差**：$\text{Var}_i(\theta) = \frac{1}{|y_i|}\sum_t (\log r_{i,t} - \log s_i)^2$ 较小

SAPO 的 token 级软门控的序列平均近似为：

$$\frac{1}{|y_i|}\sum_t f_{i,t}^{\text{SAPO}'}(r_{i,t}) \approx g_{\tau_i}(\log s_i(\theta)) = \text{sech}^2\!\left(\frac{\tau_i}{2}\log s_i(\theta)\right)$$

误差上界：$D_i(\theta) \leq \frac{\tau_i^2}{4} \text{Var}_i(\theta)$

**结论**：正常训练步骤中（A1、A2 均满足，实验验证 $>99\%$ 样本），SAPO ≡ GSPO（连续版）。当序列中存在异构 outlier token 时，SAPO 自动退回 token 级处理，保留好的梯度——这是 GSPO 做不到的。

---

## 实验结果

### 数学推理

| 方法 | AIME 2024 Pass@1 | 稳定性 |
|------|------------------|--------|
| GRPO | baseline | 最早崩溃 |
| GSPO | +△ | 稳定时间更长 |
| SAPO | **最高** | **最稳定** |

SAPO 在所有方法中坚持学习最长时间后才出现不稳定迹象，Pass@1 最高。

### Qwen3-VL 系列（实际生产）

- 覆盖：text + multimodal 混合任务，不同模型规模和架构
- 结论：SAPO 在**所有 scale 和任务类型**上一致带来性能提升
- 这是论文最重要的 signal——不是 toy 实验，是 Qwen3-VL 的实际 post-training 方法

### 温度消融

- $\tau_{\text{neg}} > \tau_{\text{pos}}$（非对称）vs 对称温度：非对称设计是关键
- 对称（$\tau_{\text{neg}} = \tau_{\text{pos}}$）在 MoE 上更容易早期崩溃

---

## 我的评价

### 这篇论文 elegant 在哪

1. **统一 gating function 视角**：把 GRPO、GSPO、SAPO 的区别归结为一个 $f_{i,t}$ 的选择，认识极为清晰
2. **非对称温度的理论推导**：不是 hyperparameter tuning 的经验结论，而是从梯度传播的数学分析中推导出来的设计
3. **SAPO ≡ GSPO（连续版）的证明**：两个假设（A1、A2）都有实验验证，不是空洞声明

### 定位在 GRPO 改进谱系中

```
硬截断族：
  GRPO（token-level hard clip）
  GSPO（sequence-level hard clip）→ 解决 token-level 方差

软门控族：
  SAPO（token-level soft gate + 非对称温度）
      ↑ 在正常条件下 ≡ sequence-level，在异构条件下 ≡ token-level
      → 两全其美
```

### 对比 OAPL 和 LAD

| 方法 | 改什么 | 核心思路 | 正交性 |
|------|--------|----------|--------|
| OAPL（2602.19362）| 替换 IS ratio | KL-reg closed-form → squared loss | 与 SAPO 正交 |
| LAD（2602.20132）| 目标函数范式 | distribution matching vs scalar expectation | 与 SAPO 正交 |
| SAPO（2511.20347）| 替换 hard clip | sigmoid 软门控 + 非对称温度 | 与两者正交 |

**可叠加性**：OAPL（off-policy 稳定）+ LAD（分布匹配）+ SAPO（软门控）理论上可叠加，覆盖不同维度的改进。

### 局限

1. **实验细节未完全披露**：Qwen3-VL 生产训练的超参、数据配比等无公开
2. **温度超参敏感性**：$\tau_{\text{pos}}$ 和 $\tau_{\text{neg}}$ 的选择对 MoE vs dense 有差异，没有给出鲁棒的调参指导
3. **理论最优性**：SAPO 的软门控函数形式（sigmoid）为何比其他平滑函数（如 softplus、Gaussian）更好？没有严格证明，是经验选择

---

## See Also

**直接技术谱系：**
- [[GSPO-Group-Sequence-Policy-Optimization|GSPO（Qwen3正式版）]] — **SAPO 的前驱**：GSPO 发现 sequence-level IS 可避免 token 高方差，但 hard clip 全有全无浪费梯度；SAPO 是 GSPO 的软门控改进版（理论证明在 A1+A2 假设下等价）
- [[GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — SAPO 归属"Trust Region 软化"维度（维度三），与 MASPO/DAPO/VAPO 同属信任域改进族
- [[VESPO-Variational-Sequence-Policy-Optimization|VESPO]] ⭐ — **理论上界**：VESPO 用变分推导给出最优 IS kernel（`ϕ(W)=W^α·exp(-λW)`），在高 staleness（64×）异步场景仍稳定；SAPO 在 N≥8 时 collapse，VESPO 是 SAPO 的理论升级

**正交可叠加方向（改不同维度）：**
- [[OAPL-Off-Policy-RL-LLM-Reasoning|OAPL]] — 改 IS ratio 本身（KL-reg closed-form → squared loss）；SAPO 改 clip 函数形状；两者正交，可叠加
- [[LAD-Learning-Advantage-Distribution|LAD]] — 改目标函数范式（f-divergence 分布匹配 vs 标量期望最大化）；与 SAPO 正交，可叠加
- [[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 同为 trust region 改进，但用 probability mass 自适应调整 ε；SAPO 用 sigmoid 软衰减
- [[NoRD-Dr-GRPO-Reasoning-Free-VLA-Autonomous-Driving|NoRD（CVPR 2026）]] — 改 std 归一化（移除 → Dr. GRPO）；SAPO 改 clip 形状；两者都修改 advantage 计算但路径完全不同——SAPO 解决 trust region 过硬问题，NoRD 解决 difficulty bias；**组合设想**：NoRD 消除 bias + SAPO 软门控 = 无偏 + 稳定的训练信号

## 推荐阅读

1. **原文**：[arXiv:2511.20347](https://arxiv.org/abs/2511.20347) — SAPO: Soft Adaptive Policy Optimization
2. **前驱**：[[GSPO-Group-Sequence-Policy-Optimization|GSPO]] — 理解 SAPO 改进了什么（sequence-level IS → soft gate）
3. **理论上界**：[[VESPO-Variational-Sequence-Policy-Optimization|VESPO]] — 变分推导的理论最优，SAPO 在高 staleness 下的理论天花板
4. **全景定位**：[[GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] — 了解 SAPO 在整个 GRPO 改进谱系中的位置

---

## Tags

`#RL` `#policy-optimization` `#trust-region` `#GRPO` `#MoE` `#Qwen` `#soft-gating` `#training-stability`
