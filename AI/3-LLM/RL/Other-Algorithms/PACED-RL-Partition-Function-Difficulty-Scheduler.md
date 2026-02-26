---
brief: "PACED-RL——用 GFlowNet 配分函数作为课程学习的难度调度器；将任务难度分布建模为能量函数，自适应采样训练边界任务；在 RLVR 任务上样本效率提升 2x，比固定难度课程学习更鲁棒。"
title: "PACED-RL: 配分函数作为难度调度器"
type: paper
domain: ai/llm/rl
tags:
  - rl
  - GFlowNet
  - curriculum-learning
  - sample-efficiency
  - RLVR
  - difficulty-scheduling
  - partition-function
created: 2026-02-21
status: v1
---

# PACED-RL: 配分函数作为难度调度器

**论文**: Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR  
**arXiv**: 2602.12642 (cs.CL)  
**投稿**: ICML 2026  
**提交**: 2026-02-13  
**作者**: Dohyung Kim 等（共 6 人）  
**评分**: ★★★★★

---

## 核心 Insight：配分函数 = 在线准确率信号

GFlowNet 在 LLM post-training 中需要学一个可学习的配分函数 Z_φ(x)。之前所有工作都把 Z_φ 当作纯粹的**归一化项**——必须的计算开销，但别无用处。

**PACED-RL 发现**：Z_φ(x) 不只是归一化项，它**编码了每道题的在线准确率信息**，可以直接用来做难度调度，且**零额外计算开销**（代价已经被 GFlowNet 训练摊平）。

---

## 数学基础

### RLVR 的最优策略

KL 正则化 RLVR 目标：

```
max_θ E_{x~D, y~π_θ(·|x)}[r(x,y)] - β·D_KL(π_θ(·|x) || π_ref(·|x))
```

最优策略的闭合形式：

```
π*(y|x) = π_ref(y|x) · exp(β⁻¹·r(x,y)) / Z(x)
```

其中 Z(x) = Σ_y π_ref(y|x)·exp(β⁻¹·r(x,y)) 是 intractable 的配分函数。

### GFlowNet Trajectory Balance Loss

GFlowNet 用可学习的 Z_φ(x) 逼近 Z(x)，通过 Trajectory Balance (TB) 目标训练：

```
L_TB(x,y; θ,φ) = [log(Z_φ(x)·π_θ(y|x) / (π_ref(y|x)·exp(β⁻¹·r(x,y))))]²
```

最小化 L_TB ⟺ 最小化 D_KL(π_θ || π*)。

### PACED-RL 的修改：将 π_ref 换成 π_old

```
L_ours(x,y; θ,φ) = [log(Z_φ(x)·π_θ(y|x) / (π_old(y|x)·exp(β⁻¹·r(x,y))))]²
```

这个修改的含义：KL 正则化锚定在当前步的策略 π_old 而非固定的参考模型 π_ref，更接近 on-policy 训练动态。

### 核心定理（Proposition 4.1）

在上述修改的 TB loss 下，最优配分函数满足：

```
p_old(x) = β·log Z*(x) - β·D_KL(π_old(·|x) || π_θ(·|x))
```

其中 p_old(x) 是问题 x 在当前策略 π_old 下的在线准确率。

### 实用近似

训练中由于 small learning rate + gradient norm clipping，参数更新很小。实验验证 β·D_KL(π_old || π_θ) 的均值在整个训练过程始终小于 4×10⁻³（接近零）。

因此：

```
p_old(x) ≈ β·log Z_φ(x)
```

**每道题的在线准确率直接由 Z_φ 给出，零额外开销。**

---

## PACED-RL 两大组件

### 组件 1：Adaptive Prompt Selection（难度感知采样）

已知中间难度题（准确率 ≈ 0.5）给出最高的样本效率（来自 Goldilocks 等工作的观察）。

PACED-RL 用 Z_φ 估计每道题的当前准确率，**优先选 accuracy ≈ 0.5 的题**进行训练。

对比其他方法：
- 过采样 + 过滤（Yu et al., Foster et al.）：需要生成更多 rollout，计算开销大
- 历史准确率 + Bayesian 估计（Zheng et al., Qu et al.）：对大数据集有 off-policy bias（策略变了但历史记录还是旧的）
- **PACED-RL**：Z_φ 是 on-policy 的（每步更新），无 staleness，无额外 rollout

### 组件 2：Accuracy Estimation Error-Prioritized Replay

**原理**：利用 GFlowNet 本身的 off-policy 容忍度做 replay。

优先级排序：estimation error = |estimated accuracy - observed accuracy|

直觉：
- 估计误差大的样本 → Z_φ 对这道题的学习还不到位
- 优先 replay 这些样本 → 加速 Z_φ 的收敛 → 准确率估计更精准 → prompt selection 更有效

两个组件**协同增强**：更准确的 Z_φ → 更好的 prompt selection；更有针对性的 replay → 更准确的 Z_φ。

---

## 实验结果

### 数学推理

| 方法 | AIME pass@1 |
|------|------------|
| GRPO | baseline |
| FlowRL | +? |
| **PACED-RL** | **GRPO +29.1%，FlowRL +40.0%** |

### 多样性（pass@k 指标）

| 方法 | pass@k vs GRPO | pass@k vs FlowRL |
|------|---------------|-----------------|
| PACED-RL | +14.2% | +9.1% |

**关键**：PACED-RL 在 pass@1（准确率）和 pass@k（多样性）上**同时优于** GRPO 和 FlowRL。这说明 GFlowNet 的 diversity 保持被成功延续，且 PACED-RL 进一步提升了效率。

---

## 与 Goldilocks 的比较

这是两篇同时解决"中间难度样本选择"问题的论文，但框架完全不同：

| 维度 | Goldilocks (2602.14868, Apple+EPFL) | PACED-RL (2602.12642, ICML) |
|------|-------------------------------------|------------------------------|
| **框架** | RLVR（GRPO 改进） | GFlowNet（分布匹配） |
| **难度估计** | Teacher LM 预测 utility（外部模型） | Z_φ 内部推导（零额外开销） |
| **数学基础** | ||∇L_PG|| = √(p_q(1-p_q))（梯度范数） | p_old(x) ≈ β·log Z_φ(x)（配分函数） |
| **目标** | 最大化梯度信号 | 最大化样本效率+保持多样性 |
| **off-policy** | 不处理（on-policy GRPO） | 天然支持（GFlowNet tolerance） |

**根本区别**：Goldilocks 在 reward-maximizing 框架内做课程，PACED-RL 在 distribution-matching 框架内做课程。两者不是竞争关系——回答的是不同的问题。

---

## 批判性分析

### 亮点

1. **理论推导严格**：Proposition 4.1 是严格的数学命题，不是 heuristic
2. **零额外开销的 insight 很 elegant**：把"已有的开销"变废为宝，而不是增加新的计算
3. **KL 项可以丢弃的实证验证充分**：4×10⁻³ 的上界让近似可信
4. **两个组件协同**：Prompt selection 和 replay 相互强化，形成正向循环
5. **ICML 级别工作**：理论+实验+对比完整

### 疑问

1. **β 的选择敏感性**：`p_old(x) ≈ β·log Z_φ(x)` 中 β 是 KL 惩罚系数，不同任务最优 β 不同，这会影响准确率估计的尺度
2. **GFlowNet 的计算开销**：GFlowNet 本身比 GRPO 更复杂，PACED-RL 在 GFlowNet 的基础上"零开销"地加了这些组件，但和直接用 GRPO 相比总计算量如何？
3. **中间难度的定义**：accuracy ≈ 0.5 是 Goldilocks 实验的结论，但 GFlowNet 的 distribution-matching 目标是否也有同样的最优难度点？
4. **大规模验证**：实验用的是 Qwen2.5-Math-1.5B，在更大模型上是否成立？

### 我的判断

**理论贡献是真实的**：Z_φ 编码准确率这一发现不是凑出来的，而是从数学中推导出来的。这种"重新挖掘已有变量的信息"的思路在机器学习中有很强的传统（如 attention weights 作为 saliency）。

局限：论文完全依赖 GFlowNet 框架，对于只用 GRPO/PPO 的场景没有直接帮助。但 GFlowNet for RLVR 是一个增长中的方向，PACED-RL 大概率会成为这个方向的标配组件。

---

## 与 GRPO 改进全景的关系

PACED-RL 属于 GRPO Panorama 里的 **Sample 维度**（样本选择/课程学习），但它是在 GFlowNet 框架下，而非 GRPO 框架下。这提示 Sample 维度有两条路：

```
Sample 维度
├── GRPO 框架内：Goldilocks（Teacher LM → utility → intermediate difficulty）
└── GFlowNet 框架内：PACED-RL（Z_φ → accuracy estimate → intermediate difficulty）
```

更深的 insight：**两者都发现了同一个 empirical 规律**（中间难度样本最有效），但用了完全不同的数学工具来检测"中间难度"。这种多路验证反而让这个规律更可信。

---

## 元数据

- **Tags**: #GFlowNet #curriculum-learning #sample-efficiency #RLVR #difficulty-scheduling #partition-function
- **关联笔记**: [[AI/3-LLM/RL/Other-Algorithms/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] ⭐ — **独立多路验证同一规律**：两者都发现中间难度（accuracy≈0.5）最优，但工具完全不同——Goldilocks 用 Teacher LM 预测难度，PACED-RL 用 GFlowNet Z_φ 估计准确率；两篇合读让这个规律从 empirical 升级为 robust finding | [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景 2026]] | [[AI/3-LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] | [[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR]] — TSR 提升 rollout 质量，PACED-RL 提升 batch 质量，两者正交
- **写于**: 2026-02-21

---

## Curriculum Learning 谱系对照（从新版合并，2026-02-23）

PACED-RL、Goldilocks、Dynamic Sampling、LILO 都在实现同一个核心 idea——在 RLVR 训练中优先选择中等难度样本：

| 方法 | 难度估计方式 | 额外开销 | 框架 |
|------|------------|---------|------|
| Dynamic Sampling | 实际 oversample | 2x rollout | GRPO |
| LILO | 实际 oversample | 4x rollout | GRPO |
| Goldilocks | reward 分布统计量分析 | 无额外 rollout | GRPO |
| **PACED-RL** | **Z_φ 配分函数（已有）** | **零额外开销** | **GFlowNet** |

**结论**：PACED-RL 在 GFlowNet 框架内是目前最优雅的 curriculum 方案——零额外开销 + 有理论保证。
