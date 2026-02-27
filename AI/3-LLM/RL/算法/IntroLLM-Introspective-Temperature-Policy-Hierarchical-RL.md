---
brief: "IntroLLM（arXiv:2602.13035）——分层 RL 自适应调节采样温度；上层策略根据当前状态动态控制探索温度，解决 GRPO 固定温度导致 entropy collapse 或过度随机的问题；在多样性与 accuracy 之间自动平衡。"
title: "IntroLLM: Introspective Temperature Policy via Hierarchical RL"
date: 2026-02-21
arxiv: "2602.13035"
domain: AI/LLM/RL/Other-Algorithms
tags:
  - rl
  - grpo
  - temperature
  - hierarchical-rl
  - exploration
  - diversity
  - hidden-state
  - ICML-2026
  - type/paper
rating: 4
status: permanent
---

# IntroLLM: Introspective Temperature Policy via Hierarchical RL

**arXiv**: 2602.13035  
**机构**: 未注明（ICML 投稿）  
**作者**: Yixiao Zhou, Yang Li, Dongzhou Cheng, Hehe Fan, Yu Cheng  
**提交**: 2026-02-13  
**投稿**: ICML  
**评分**: ★★★★☆  
**一句话**: 不再把采样温度当固定超参——用 LLM 自身的隐状态学一个 temperature policy，hierarchical RL 联合优化，推理时让模型自己决定"现在该探索还是利用"。

---

## 核心问题

RLVR（Reinforcement Learning from Verifiable Rewards）的效果高度依赖于采样多样性，而采样多样性由**温度 τ** 控制。但现有方法一律用固定温度——对所有 prompt、所有 token 位置、训练的所有阶段用同一个 τ。

**三个已知问题**（实证支持来自 POLARIS 等工作）：
1. **不同 token 位置需要不同温度**：推理转折点需要高 τ（探索），数值计算/事实检索需要低 τ（利用）
2. **不同 prompt 需要不同温度**：困难问题需要高熵，简单问题不需要
3. **训练过程中探索需求变化**：RL 训练自然压低模型熵，固定 τ 越来越不合适

问题本质：**温度是个决策，但被当成了超参**。没人让它从 reward 里学习。

---

## 方法：IntroLLM

### 形式化

把 temperature selection 定义为一个 **hierarchical RL** 问题，两个 policy 联合优化：

```
π(y, τ | x) = ∏ᵢ πϕ(τₜ | hₜ, τₜ₋₁) · πθ(yₜ | hₜ, τₜ)
```

- **Temperature policy** `πϕ(τₜ | hₜ, τₜ₋₁)`：根据当前隐状态 hₜ 和上一步温度 τₜ₋₁，决定本步用什么温度
- **Token policy** `πθ(yₜ | hₜ, τₜ)`：用选定的温度 τₜ 从分布中采样下一个 token

联合目标：`J(θ, ϕ) = E[R(x, y)]`，两个 policy 共享同一个 task-level reward 信号。

### Temperature Policy 架构

**轻量 MLP head**，从最后一层 decoder 的隐状态 hₜ 分支出来：

```
uₜ = W₂ · ReLU(W₁hₜ + b₁) + b₂    # 3维控制向量 [uᶜ, uα, uβ]
```
参数量：`W₁ ∈ R^{d/2 × d}`，`W₂ ∈ R^{3 × d/2}`。极小的额外参数开销。

**混合离散-连续动作空间**（关键设计）：

**Stage 1 — 是否更新？**（离散）
```
cₜ ~ Bernoulli(σ(uᶜ))
```
如果 cₜ = 0，保持 τₜ = τₜ₋₁（不需要每步都调整，减少方差）

**Stage 2 — 调多少？**（连续，仅当 cₜ = 1）
```
zₜ ~ Beta(softplus(uα) + ε, softplus(uβ) + ε)    # β ∈ [0,1]
τₜ = τ_min + zₜ · (τ_max - τ_min)                  # 映射到范围
```
用 Beta 分布而非 Gaussian：天然有界在 [0,1]，支持双峰（高温/低温偏好）和单峰模式。

### 优化：坐标上升

两个 policy 各自用 GRPO 更新，轮流优化（**coordinate ascent**）：
- 固定 πϕ，用 GRPO 更新 πθ（token policy）
- 固定 πθ，用 GRPO 更新 πϕ（temperature policy）

联合 log-likelihood：
```
log πϕ(τₜ | hₜ) = log P(cₜ) + cₜ · log p(zₜ | α, β)
```

---

## 实验结果

**基准**：数学推理（MATH, AIME 等）

| 方法 | 特点 |
|------|------|
| 固定 τ（baseline） | 全局同一温度 |
| Heuristic adaptive | 基于 token entropy 或预定义规则 |
| **IntroLLM** | learned temperature policy |

结论：IntroLLM **一致优于**固定温度和启发式自适应方法。

**Interpretability 分析**（亮点）：

学出来的温度模式与先验经验高度吻合：
- **高温区**：多步推理中的不确定性转折点（"reasoning pivots"）
- **低温区**：数值计算、事实检索、最终答案合成

这些模式**从 reward 信号自然涌现**，没有任何人工设计的规则。这是 learned exploration 方法相比 heuristic 方法的根本优势。

---

## 我的分析

### 真正 novel 的是什么？

**不是 hierarchical RL 本身**，这个框架在 robotics/game AI 里用了很久。

**真正 novel 的是**：把 temperature 从超参变成 policy，并且从 **LLM 的内部隐状态**（hidden state）读取控制信号，而不是从外部统计量（如 token entropy）。

核心洞察：**hₜ 包含比 token entropy 更丰富的信息**。entropy 是分布的标量统计，hₜ 是模型当前对整个问题状态的全信息编码。直接从 hₜ 读信号，相当于让模型用它自己的"感知"来决定探索强度——真正的 introspection。

### Beta 分布的设计价值

用 Beta(α, β) 而不是直接输出均值是有意义的：
- Beta(α≫β)：高温偏好（探索模式）
- Beta(α≪β)：低温偏好（利用模式）  
- Beta(1,1)：均匀分布（无倾向）

这种参数化让 policy 能表达"置信地选高温"和"不确定随机选"的区别——普通 Gaussian 参数化做不到这个。

### 与 GRPO 全景七维框架的关系

这篇论文完全属于第七维 **Diversity（探索多样性）**，也是目前该维度最精细的工作：
- VAM（Verbalized Action Masking）：粗粒度，在词表层面控制探索
- DEEP-GRPO：在 trajectory 层面重采样
- **IntroLLM**：在 **token 层面**，用隐状态动态调节，最细粒度

层次：trajectory-level（粗）→ token-level（中）→ **internal-state-conditioned token-level（最细）**

### 局限与质疑

1. **额外计算开销**：每个 token 都要跑 MLP head，理论上有一定 overhead（论文未给具体数字）
2. **稳定性问题**：coordinate ascent 两个 policy 交替优化，有 non-convergence 风险，特别是当两者目标高度耦合时
3. **温度范围的边界设置**：τ_min 和 τ_max 仍是人工指定的，论文中未给出敏感性分析
4. **只在数学推理上验证**：代码生成、agent 任务等是否同样有效？
5. **Bernoulli gate 的解释**：为什么学出来的 c=1 比率是多少？论文有没有分析？——摘要没提，可能是 ablation

### 与盾卫 Phase 3 的连接

IntroLLM 从 **hₜ（hidden state）预测最优温度**，盾卫 Phase 3 要从 **hₜ 检测注入攻击**。两者都依赖同一个核心假设：**LLM 的内部表示包含任务表面看不到的丰富信息**。

IntroLLM 证明了这个假设在生成控制上是可行的——这是 Phase 3 激活探针设计的间接实验支持。如果 hₜ 能学会区分"探索性推理步骤 vs 利用性计算步骤"，它大概率也能区分"正常生成 vs 被注入后的异常生成"。

---

## 关键公式

**联合生成过程**：
```
π(y, τ | x) = ∏ₜ πϕ(τₜ | hₜ, τₜ₋₁) · πθ(yₜ | hₜ, τₜ)
```

**Temperature policy（混合动作空间）**：
```
cₜ ~ Bernoulli(σ(uᶜ))                                    # 是否更新
zₜ ~ Beta(softplus(uα)+ε, softplus(uβ)+ε)  if cₜ=1      # 连续采样
τₜ = τ_min + zₜ·(τ_max - τ_min)                         # 映射到范围
```

**联合对数概率**：
```
log πϕ(τₜ | hₜ) = log P(cₜ) + cₜ · log p(zₜ | α, β)
```

---

## Tags
#RLVR #Temperature #HierarchicalRL #Exploration #IntrospectiveLLM #GRPO #ICML2026 #隐状态 #探索利用 #Diversity维度

---

## See Also

- [[AI/3-LLM/RL/算法/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO（深密探索Pivot重采样）]] — 同为 Diversity 维度，方法粒度不同：DEEP-GRPO 在 trajectory 层面重采样 error-prone pivot，IntroLLM 在 token 层面用 hₜ 动态调温——前者选"在哪个节点多探索"，后者让模型自己决定"现在该高温还是低温"
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 七维改进全景]] ⭐ — IntroLLM 属于第七维 Diversity 层的最细粒度实现；全景框架提供定位：trajectory-level → token-level → internal-state-conditioned token-level（由粗到精）
- [[AI/5-AI 安全/AutoInject-RL-Prompt-Injection-Attack|AutoInject（RL自动化Prompt Injection）]] ⭐ — 同一技术假设的正反两面：IntroLLM 用 hₜ 预测最优采样温度（生成质量提升），盾卫 Phase 3 用 hₜ 检测注入攻击（安全防御）——hₜ 包含比表面 token 更丰富信息这一命题，IntroLLM 是正向验证
- [[AI/3-LLM/RL/Theory/REMuL-CoT-Faithfulness-Multi-Listener-RL|REMuL（CoT Faithfulness多听众RL）]] — 同为 GRPO 框架的精细化：REMuL 在 reward 信号上精细化（faithfulness），IntroLLM 在采样策略上精细化（adaptive temperature）——两者合用=更可信的推理链 + 更有效的探索
- [[AI/3-LLM/RL/算法/ProGRPO-Probabilistic-Advantage-Reweighting|ProGRPO（概率优势重加权）]] — Diversity 维度的另一路：ProGRPO 重加权 advantage 估计（从 reward 信号层面增加探索），IntroLLM 从采样温度层面增加探索——同目标，前者改 training，后者改 inference-time sampling
- [[AI/3-LLM/RL/算法/LACONIC-Length-Constrained-RL|LACONIC（Primal-Dual长度约束RL）]] — 约束 RL 的互补维度：IntroLLM 约束采样**温度**（控探索多样性），LACONIC 约束输出**长度**（控推理成本）——两者从不同维度对 RL 生成过程施加结构性控制，可互补叠加
