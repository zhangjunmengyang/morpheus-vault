---
title: "SORL: Stabilizing Off-Policy RL for Long-Horizon LLM Agent"
brief: "诊断 multi-turn agent RL off-policy 崩溃的两个根因：①粒度错配（token-level IS ratio 在 multi-turn 下指数爆炸）②方差累积（mini-batch reuse 的 off-policy 程度随轮次升高）。提出 Turn-Level IS（turn 内 IS ratio 求均值替代乘积）+ Clipping-Triggered Normalization（clip 触发率越高 KL 惩罚越重）。实例化为 SO-PPO 和 SO-GRPO，消除 training collapse，适用 Search/Tool-Use agent。"
arxiv: "2511.20718"
date: 2025-11-28
updated: 2026-02-24
rating: ★★★★☆
tags:
  - agent-rl
  - multi-turn-rl
  - off-policy
  - ppo-stability
  - training-stability
  - importance-sampling
sources:
  - "arXiv:2511.20718 (Chenliang Li et al., 2025-11-28, v2 2026-02-24, Texas A&M + GE HealthCare + Univ. Minnesota)"
related:
  - "[[AI/2-Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution]]"
  - "[[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]]"
  - "[[AI/2-Agent/Agentic-RL/Search-R1plus-Tool-Use-RL-Ablation]]"
  - "[[AI/2-Agent/Agentic-RL/LOOP-Leave-One-Out-PPO-Long-Horizon-Agent-RL]]"
  - "[[Projects/MA-RLHF/lc8-PPO/lc8-01-PPO-手撕实操]]"
  - "[[AI/2-Agent/Agentic-RL/Dr-MAS-Stable-RL-Multi-Agent-LLM-Systems]]"
  - "[[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL]]"
---

# SORL: Stabilizing Off-Policy RL for Long-Horizon LLM Agent

> **一句话**：诊断 multi-turn agent RL 中 off-policy 训练崩溃的两个根本原因（粒度错配 + 方差累积），提出 Turn-Level IS 和 Clipping-Triggered Normalization，实例化为 SO-PPO 和 SO-GRPO，防止 training collapse。

**作者**：Chenliang Li, Adel Elmahdy 等（Texas A&M + GE HealthCare + Univ. Minnesota）
**arXiv**：2511.20718（2025-11-28，v2 更新于 2026-02-24）
**Venue**：preprint（正在审）

---

## 问题诊断：Off-Policy Multi-Turn RL 的两个崩溃根因

Multi-turn agent 训练中，PPO/GRPO 在 off-policy pipeline 下经常崩溃。为什么？

### 根因一：粒度错配（Granularity Mismatch）

**现象**：标准 PPO 在 token 粒度做 credit assignment，但 multi-turn 交互的自然单位是 **turn**（一轮推理/检索/回答）。

```
Multi-turn trajectory:
Turn 1: [problem analysis tokens]         ← 推理阶段
Turn 2: [query formulation tokens]        ← 查询形成
Turn 3: [information processing tokens]  ← 信息整合  
Turn 4: [final answer tokens]             ← 答案生成

Token-level advantage: 每个 token 都从末尾 reward 反向传播
→ 中间 turn 的 value 估计严重不准确（distant supervision）
→ 错误的 token-level importance ratio
```

**关键问题**：token 粒度的 importance sampling ratio $w_t = \pi_\theta(y_t|x,y_{<t}) / \pi_{\theta_\text{old}}(y_t|x,y_{<t})$ 在 multi-turn 场景下会**乘积爆炸**——turn 内所有 token 的 IS 比值相乘，导致 extreme ratio。

### 根因二：Off-Policy 方差累积（Variance Accumulation）

**Mini-batch reuse 的 off-policy 程度随训练加深**：
- 第 1 次梯度更新：on-policy（刚采样的数据）
- 第 2, 3, ..., K 次更新：越来越 off-policy（policy 已漂移）

**PPO clip 的失效**：clip 设计假设 policy 漂移有界，但 long-horizon multi-turn 的累积漂移超出 clip 范围：
- OOD token 的 advantage 估计不准
- 梯度方差随轮次累积增大
- 最终 gradient spike → policy collapse

---

## SORL 方法：两个核心机制

### 机制一：Turn-Level Importance Sampling

将 importance ratio 从 token 粒度提升到 **turn 粒度**：

$$w^k(\theta) = \frac{\pi_\theta(y^k | x, y^{<k})}{\pi_{\theta_\text{old}}(y^k | x, y^{<k})} = \prod_{t=t_k^\text{start}}^{t_k^\text{end}} w_t(\theta)$$

但不直接用上面这个乘积（会爆炸），而是用 **turn-level 聚合**：

$$\bar{w}^k(\theta) = \frac{1}{|y^k|} \sum_{t=t_k^\text{start}}^{t_k^\text{end}} w_t(\theta)$$

**为什么这样更好**：
- 对 turn 内所有 token 的 IS ratio 做均值，不做乘积 → 避免指数爆炸
- 保留 turn 结构信息：每个 turn 得到一个统一的重要性权重
- Turn 内所有 token 共享同一个 advantage（来自 turn-level value 估计）

**Turn-Level advantage 估计**：

$$\hat{A}^k = R_k - V_\phi(s_k)$$

其中 $V_\phi(s_k)$ 是 turn 开始时的 value（而非每个 token 的 value）。

### 机制二：Clipping-Triggered Normalization

**问题**：off-policy updates 越来越多时，被 clip 的 token 比例升高（说明 policy 漂移严重），但 PPO 不对这种情况做额外处理，导致梯度偏差积累。

**SORL 的解法**：检测 clip 触发情况，触发时施加**归一化惩罚**：

$$\mathcal{J}_\text{SORL}(\theta) = \mathcal{J}_\text{PPO}(\theta) - \lambda \cdot \mathbb{E}\left[\frac{|\{t : |w_t-1| > \epsilon\}|}{|y|} \cdot \text{KL}(\pi_\theta \| \pi_{\theta_\text{old}})\right]$$

通俗理解：
- 当大量 token 的 IS ratio 超出 clip 边界（说明 policy 已经漂移太远）
- 额外施加一个 KL 惩罚，强制 policy 更保守
- clip 触发比例越高，惩罚越强 → 自适应控制 off-policy 程度

**工程简化版**：
```python
clip_fraction = (torch.abs(ratio - 1.0) > epsilon).float().mean()
kl_penalty = kl_divergence(pi_theta, pi_theta_old)
loss = ppo_loss + lambda * clip_fraction * kl_penalty
```

---

## SO-PPO 和 SO-GRPO

### SO-PPO 完整目标

$$\mathcal{J}_\text{SO-PPO}(\theta) = \mathbb{E}\left[\frac{1}{n}\sum_{k=1}^n \min\left(\bar{w}^k \hat{A}^k, \text{clip}(\bar{w}^k, 1-\epsilon, 1+\epsilon)\hat{A}^k\right)\right] - \lambda \cdot \text{CTN penalty}$$

- 第一项：turn-level PPO 目标
- 第二项：Clipping-Triggered Normalization（CTN）penalty

### SO-GRPO

SORL 框架对 GRPO 同样适用：把 GRPO 的 group advantage 也从 token 粒度提到 turn 粒度：

$$A_k^\text{SO-GRPO} = \frac{R_k - \bar{R}_\text{group}}{\sigma_\text{group}}$$

其中 $\bar{R}_\text{group}$ 是同组 K 个 rollout 的 turn-level reward 均值。

**关键改变**：advantage 不再是每个 token 的独立估计，而是每个 turn 的统一值。

---

## 实验结果

**评测基准**（全为 multi-turn search 场景）：
- General QA：NQ, TriviaQA, PopQA
- Multi-Hop QA：HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle
- Medical QA：MedQA（多选题）

**关键结论**：
1. SO-PPO 和 SO-GRPO 均**消除 training collapse**（标准 PPO/GRPO 在 multi-turn 下频繁崩溃）
2. **Clipping ratio 更低**：说明 policy 更新更保守，漂移更小
3. **Gradient norm 更平稳**：无 spike（论文 Figure 1 右图）
4. 性能与或超过标准方法（不崩溃 = 更好的最终性能）

---

## 核心洞察

### 1. Off-Policy 程度是 Multi-Turn RL 特有的系统性风险

单轮任务（数学推理）的 PPO 通常稳定，因为：
- 单轮 response 不长
- IS ratio 的乘积不多
- Mini-batch reuse 的 off-policy 程度有限

Multi-turn 任务恰好相反：长 trajectory × 多 mini-batch reuse = 累积 off-policy + 高方差。

### 2. Turn 是 Multi-Turn RL 的自然优化粒度

Token 粒度太细（噪声高），Sequence 粒度太粗（无法区分好坏 turn）。
**Turn 粒度是 Goldilocks**：与 reasoning 的认知结构对齐（每 turn = 一个推理/检索步骤）。

### 3. CTN 是 PPO clip 的"自适应加强版"

PPO clip 是静态约束（epsilon 固定），CTN 是动态约束（clip 越多 = 惩罚越重）。
两者互补：clip 控制 per-step 偏差，CTN 控制累积 off-policy 漂移。

---

## 与相关工作对比

| 方法 | 粒度 | Off-Policy 处理 | 崩溃防御 |
|------|------|----------------|---------|
| PPO（标准） | Token | IS clip（静态） | 有限 |
| GRPO | Token（group norm） | 无 | 差（multi-turn 尤其） |
| LOOP（LOO） | Trajectory | 无（on-policy 为主） | 好（on-policy）|
| SeeUPO | Turn（backward） | 无 | 好（理论保证）|
| **SORL（本文）** | **Turn-level IS** | **CTN 自适应** | **好（off-policy 专设计）** |
| Search-R1++ | Token（REINFORCE） | 无 | 好（REINFORCE 天然稳）|

### 与 RAGEN/StarPO Echo Trap 对比

RAGEN 的 Echo Trap = variance collapse → entropy drop → gradient spike 三联征。
SORL 诊断：gradient spike 来自 IS ratio 爆炸（off-policy 累积）。
两者描述同一现象的不同角度，SORL 给出更精确的工程解。

---

## 批判性评估

**优点**：
- 诊断精准：两个根因（粒度 + off-policy）是真实的工程痛点
- 方法简洁：turn-level averaging + CTN，实现成本低
- Algorithm-agnostic：PPO/GRPO 都能用

**待确认**：
- CTN 的 lambda 超参如何选？论文未详细报告
- Turn 边界如何自动检测？（`</search>` `<answer>` 等 special token 划分？）
- 在 non-search agent（code agent/GUI agent）上的效果？

**与 SeeUPO 的比较**：SeeUPO 从理论出发证明 GRPO 在 multi-turn 的 variance normalization 破坏单调性（需要 backward induction），SORL 是工程取向的修复方案——两者互补而非竞争。

---

## 落地应用

**适用场景**：
- Search agent、tool-use agent 的 GRPO/PPO 训练（特别是 mini-batch reuse 的 off-policy pipeline）
- 出现 training collapse / gradient spike 时的第一个排查方向

**诊断工具**：
```python
# 监控两个指标：
# 1. Clipping fraction（clip 触发率，>50% 说明 off-policy 严重）
clip_fraction = (torch.abs(ratio - 1.0) > epsilon).float().mean()
# 2. Gradient norm（>X 说明将崩溃）
grad_norm = sum(p.grad.norm() for p in model.parameters())
```

---

## See Also

- [[AI/2-Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution]] — Echo Trap 三联征（variance collapse → entropy drop → gradient spike），与本文诊断互补
- [[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]] — backward induction 理论解（不同角度解决 multi-turn GRPO 问题）
- [[AI/2-Agent/Agentic-RL/Search-R1plus-Tool-Use-RL-Ablation]] — REINFORCE > PPO > GRPO 稳定性（Search-R1++ 的经验验证与本文机制分析互为佐证）
- [[AI/2-Agent/Agentic-RL/LOOP-Leave-One-Out-PPO-Long-Horizon-Agent-RL]] — LOOP 用 LOO baseline 替代 critic，避免 off-policy 问题（on-policy 方案）
- [[AI/3-LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性 2026 统一分析]] — 本文所在的稳定性图谱，Off-Policy 级别章节（反向链接）
