---
title: "OAPL: LLMs Can Learn to Reason Via Off-Policy RL"
brief: 质疑 on-policy 假设的必要性：现实的分布式 RL 训练架构天然使数据变成 off-policy（trainer vs vLLM 的 log-prob 不一致 + 异步延迟）。OAPL 基于 KL-regularized RL 的 closed-form 解推导出 squared regression loss，不需要 importance sampling，允许 400 步 policy lag，outperform GRPO+IS on AIME25/HMMT25/BRUMO25，LiveCodeBench v5 用 1/3 生成量匹配 DeepCoder。
date: 2026-02-25
type: paper-note
rating: ★★★★★
venue: arXiv (cs.LG)
arxiv: "2602.19362"
authors: Daniel Ritter, Owen Oertell, Bradley Guo, Jonathan Chang, Kianté Brantley, Wen Sun
affiliation: Cornell University, Databricks, Harvard University
tags:
  - off-policy-RL
  - LLM-reasoning
  - policy-optimization
  - async-training
  - GRPO
  - KL-regularization
  - math
  - code
related:
  - "[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[REBEL-Regret-Based-RL-LLM-Alignment|REBEL]]"
  - "[[AI/LLM/RL/GRPO/GRPO-Improvement-Panorama-2026|GRPO-Improvement-Panorama-2026]]"
  - "[[GRPO 深度理解|GRPO深度理解]]"
  - "[[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]]"
---

# OAPL: LLMs Can Learn to Reason Via Off-Policy RL

> arXiv:2602.19362 | Cornell + Databricks + Harvard | 2026-02-22

---

## 一句话 TL;DR

**现实的 LLM RL 训练本来就是 off-policy 的**（trainer 和 vLLM 的 log-prob 天然不一致）——与其用 importance sampling 把 off-policy 数据伪装成 on-policy，不如直接设计一个原生 off-policy 算法。OAPL 从 KL-regularized RL 的 closed-form 解出发，推导出一个纯 squared regression loss，不需要任何 IS ratio，允许 400 步 policy lag，效果比 GRPO+IS 更好。

---

## 动机：RL 训练天然就是 Off-Policy

### 两个来源的 Off-Policyness

**来源 1：Trainer 和 Inference Engine 的 log-prob 不一致**

即使 HuggingFace trainer 和 vLLM 共享相同权重，它们对同一序列计算出的 log-probability 可能不同（kernel 实现差异）。Liu et al. (2025) 测量到这种 KL 散度的突然增大直接导致了 GRPO 训练不稳定和 policy collapse。

**来源 2：异步训练中的 Policy Lag**

在分布式异步 RL 架构中，vLLM 推理引擎可能比 trainer 落后多个梯度步骤。Fu et al. (2025) 的 Stable Asynchrony 框架就是为此设计的——但用的还是 IS 来弥补。

**现有解法的问题**：
- **Importance Sampling 方案**：IS ratio $\frac{\pi_{old}(y|x)}{\pi_{vllm}(y|x)}$ 引入额外方差，policy lag 大时 IS 不可靠
- **修改 inference engine 方案**：让 vLLM 和 trainer 完全同步会降低推理速度，且在异步场景下依然无法完全消除 gap

---

## 方法：OAPL

### 核心推导

**Step 1：KL-regularized RL 目标**

$$\max_{\pi} \mathbb{E}_{x,y\sim\pi(\cdot|x)} r(x,y) - \beta \text{KL}(\pi || \pi_{vllm})$$

注意：KL 是对当前 inference engine $\pi_{vllm}$ 的约束，而不是对固定 reference policy $\pi_{ref}$。

**Step 2：closed-form 最优解**

KL-regularized RL 有解析解：

$$\pi^*(y|x) \propto \pi_{vllm}(y|x) \exp(r(x,y)/\beta)$$

$$V^*(x) = \beta \ln \mathbb{E}_{y \sim \pi_{vllm}} \exp(r(x,y)/\beta)$$

**Step 3：关键变换**

$$\beta \ln \frac{\pi^*(y|x)}{\pi_{vllm}(y|x)} = r(x,y) - V^*(x) = A^*(x,y)$$

即：最优 policy 和 inference policy 的 log-ratio = **最优 advantage**。

**Step 4：实用估计量**

给定 $G$ 条来自 $\pi_{vllm}$ 的 rollout，估计 $V^*$：

$$\hat{V}^*(x) = \beta \ln \frac{1}{G} \sum_{i=1}^G \exp(r(x, y_i)/\beta)$$

- $\beta \to 0$：退化为 $\max_i r(x, y_i)$（只看最高 reward）
- $\beta \to \infty$：退化为平均 reward（等同于 GRPO 的 group baseline）

**Step 5：OAPL 的训练目标**（squared regression loss）

$$\min_\pi \sum_x \sum_{i=1}^G \left(\beta \ln \frac{\pi(y_i|x)}{\pi_{vllm}(y_i|x)} - (r(x,y_i) - \hat{V}^*(x))\right)^2$$

这是一个最小二乘回归：让 $\ln \frac{\pi}{\pi_{vllm}}$ 对齐 optimal advantage $A^*$。

**没有 IS ratio，没有 clipping，没有删除 stale tokens。**

### OAPL 算法（Algorithm 1）

```
初始化：同步 π 和 π_vllm
for t = 1 → T:
  1. π_vllm 异步生成数据 {x, {y_i}} → 存入 buffer D
  2. π 对 D 中数据做梯度下降（目标：Eq.3）
  if t mod L == 0:
    3. 同步 π_vllm ← π（权重更新）
    4. 清空 buffer D
```

**L（policy lag）** 是关键超参数：L=400 时依然稳定运行（100x more off-policy than prior approaches）。

### OAPL vs GRPO 的根本差异

| 维度 | GRPO | OAPL |
|------|------|------|
| 核心假设 | on-policy（数据来自当前 π）| off-policy（数据来自 π_vllm，可 lagged）|
| 如何处理 policy lag | IS ratio $\frac{\pi_{old}}{\pi_{vllm}}$ 矫正 | KL 约束内化到目标函数 |
| 稳定性机制 | clipping（不总是有效）| squared loss（天然稳定）|
| Entropy collapse | 风险较高 | 无（不限制 π 偏离 π_vllm）|
| 需要保存 $\pi_{old}$? | 是 | 否（只需 π 和 π_vllm）|
| Policy lag 容忍度 | <5 步（通常）| **400 步**（实验验证）|

**关于 clipping 的批判**：GRPO 的 clip(r_t, 1-ε, 1+ε) 理论上防止 π 偏离 π_old，但在第一步梯度时（π=π_old，比值=1），clipping 无效——一个大梯度步骤已经让 π 远离了 π_old，而 clipping 无法拉回来。OAPL 的 squared loss + KL regularization 直接在目标函数层面解决了这个问题。

---

## 实验结果

### 数学竞赛推理

基础模型：Deepscaler（基于 DeepSeek R1 系列）
评估：AIME 25, HMMT 25 (Feb + Nov), BRUMO 25
对比：GRPO + IS（考虑了 trainer-vLLM log-prob 差异）

| Benchmark | Pass@1 | Pass@5 | Pass@10 |
|-----------|--------|--------|---------|
| OAPL 优势 | ✅ | ✅ | ✅ |

（OAPL 在三个 benchmark 的所有 Pass@k 指标上 outperform GRPO+IS）

### 代码生成（LiveCodeBench v5）

**对比 DeepCoder**（用 GRPO + 额外启发式：clip-high, overlong filtering 等）

- OAPL：**1/3 生成量** 匹配或超越 DeepCoder
- Policy lag：**400 步**，无需任何 IS

### 熵不崩溃（Anti-Entropy-Collapse）

OAPL 明确不会引起 entropy collapse：
- Pass@k 指标（k=1 到 256）稳定提升
- 说明 OAPL 不只是在做 distribution sharpening（GRPO 有时会过度 sharpen）

---

## 理论视角：为什么 Off-Policy 可行？

### 与经典 RL 的类比

论文指出：在传统 RL 领域（机器人控制、视频游戏），off-policy 算法（DDPG、SAC）通常比 on-policy 算法（PPO、REINFORCE）更高效。LLM RL 一直以来对 on-policy 的执着，更多源于历史沿袭而非理论必然。

### OAPL 作为"value learning"

OAPL 可以被理解为一种 value function 方法：$\ln\frac{\pi}{\pi_{vllm}}$ 作为函数近似器，直接估计 optimal advantage $A^*$。这和 Q-learning 系列方法的思路一致——不通过 policy gradient 而是通过 value estimation 学习。

### 关键理论性质

**命题**：当 $\hat{V}^* = V^*$ 时，目标函数 Eq.3 的最优解正好是 KL-regularized RL 的最优 policy $\pi^*$，无论 rollout 的采样分布是什么（包括 off-policy 的 $\pi_{vllm}$）。

这是 OAPL 容忍大量 policy lag 的理论基础：只要 $\hat{V}^*$ 的估计足够准确，off-policy rollout 同样能产生有效的学习信号。

---

## 在 GRPO 改进生态中的位置

```
GRPO 改进谱系（截至 2/25）：

【修改采样过程】
├── TSR — training-time tree search，提升 rollout 质量
└── RAGEN Instance Filtering — 不确定性采样

【修改优化目标·稳定性】
├── DEEP-GRPO — 梯度裁剪
├── STAPO — 多维度 trust region
├── Goldilocks / LACONIC — 长度约束
└── OAPL — 直接放弃 GRPO 改用 off-policy squared loss ← 这里

【Off-Policy 方向】（OAPL 的相关生态）
├── Stable Asynchrony（Fu et al.）— IS for async RL
├── VCPO — variance-constrained off-policy
├── MASPO — mass-adaptive soft policy opt
└── OAPL — KL-regularized closed-form，无 IS，400步 lag ← 最激进

【Credit Assignment（与 OAPL 正交）】
├── GiGPO — step-level anchor
├── HiPER — subgoal-segment level
└── SeeUPO — 逆序更新理论保证
```

OAPL 是 off-policy 方向中**理论最干净**的方案：没有 heuristic（不 clip IS、不 delete stale tokens），直接从 KL-regularized RL 的最优解出发。

---

## 批判性评估

### ★★★★★ 的理由

1. **理论基础最干净**：从 KL-regularized RL closed-form 解直接推导，不是 GRPO 的 trick 叠加
2. **解决真实工程痛点**：400 步 policy lag 在大规模分布式训练中是真实需求
3. **Entropy collapse 问题的根本解决**：GRPO 的 entropy collapse 被频繁报告，OAPL 的 squared loss 天然规避
4. **来自顶尖学术机构**：Cornell + Databricks + Harvard，学术严谨性高
5. **Pass@k 改进说明真正提升能力**：不只是 distribution sharpening

### 保留意见

1. **只测了 math 和 code**：两类 verifiable reward 任务。对 open-ended tasks（工具调用、agent RL）是否同样有效尚未验证
2. **$\beta$ 的调参敏感性**：两个 beta（$\beta_1$, $\beta_2$）的选择对结果的影响需要更系统的 ablation
3. **与 credit assignment 的结合**：OAPL 是 sequence-level 目标，和 step-level credit assignment（GiGPO/HiPER）的结合尚未探索

---

## 工程价值

**对 LLM RL 基础设施的直接意义**：

1. **异步 RL 框架的天然选择**：VeRL/SGLang/Slime 等异步框架不再需要为 IS 做特殊处理
2. **inference engine 可以用更旧的权重**：训练 400 步后再同步 vLLM，throughput 大幅提升
3. **去掉 clip-high/overlong-filtering 等 heuristic**：OAPL 的 squared loss 天然稳定，不需要这些补丁

**最佳实践组合**：
- 异步 RL 架构 + OAPL（稳定的 off-policy 训练）
- \+ TSR（高质量 rollout 生成）
- \+ HiPER（hierarchical credit assignment）
= 理论上目前最完整的 multi-turn agent RL 系统

---

## 开放问题

1. **是否可以推广到 multi-turn RL？** OAPL 目前是 sequence-level，在长 horizon agent 任务中（每步都有 reward）如何处理？
2. **$\beta$ 与任务难度的关系**：hard math 问题（通过率低）和 easy 问题，最优的 $\beta$ 应该不同。$\hat{V}^*$ 的估计在 hard cases 下是否准确？
3. **OAPL + GiGPO**：OAPL 给出 sequence-level loss，GiGPO 给出 step-level signal——两者能否在同一框架内结合？

---

## See Also

**Off-Policy RL 生态：**
- [[MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 另一种自适应软策略优化，同属 off-policy 改进方向
- [[GRPO-Improvement-Panorama-2026|GRPO改进全景]] — OAPL 在 off-policy 方向的完整定位
- [[REBEL-Regret-Based-RL-LLM-Alignment|REBEL]] — Regret-based 目标，与 OAPL 同为 critic-free 替代
- [[SAPO-Soft-Adaptive-Policy-Optimization|SAPO]] — **正交可叠加**：OAPL 改 IS ratio（squared loss），SAPO 改 clip 函数（sigmoid 软衰减）；解决不同维度，理论上可组合

**代码实操（理解 off-policy 的工程含义）：**
- [[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]] — GRPO on-policy 实现对照
- [[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操]] — IS ratio 在 PPO 中的具体位置
- [[AI/LLM/Infra/Ray-分布式RL训练实操|Ray分布式RL训练实操]] — 异步训练架构，OAPL 解决的工程问题来源

**理论相关：**
- [[GRPO 深度理解|GRPO深度理解]] — 理解 GRPO 局限是理解 OAPL 动机的前提
- [[SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees|SeeUPO]] — 同样质疑 GRPO 的收敛性，但从 multi-turn 角度
- [[iStar-Implicit-Step-Rewards-Agentic-RL|iStar]] — **KL-reg 框架同根**：iStar 用 $\beta\log\frac{\pi_\phi}{\pi_\text{old}}$ 作 step-level reward，OAPL 用 $\beta\log\frac{\pi^*}{\pi_\text{ref}}$ 推导闭合 squared loss；两者共享 KL-reg RL 的最优解结构，分别解决 credit assignment 和 off-policy 稳定性；详见 [[Long-Horizon-Credit-Assignment专题|Long-Horizon CA 专题 § 11-B]]

## 推荐阅读

1. **原文**：[arXiv:2602.19362](https://arxiv.org/abs/2602.19362) — OAPL: LLMs Can Learn to Reason Via Off-Policy RL
2. **前置阅读**：[[GRPO 深度理解|GRPO深度理解]] — 理解 OAPL 在改进什么
3. **工程对照**：[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray分布式RL训练实操]] — 异步架构中 OAPL 的应用场景
4. **姊妹论文**：[[SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees|SeeUPO]] — 同期从不同角度批判 GRPO 收敛性
