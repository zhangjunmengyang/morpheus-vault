---
title: "REBEL: Reinforcement Learning via Regressing Relative Rewards"
aliases: ["REBEL", "Regret-Based RL"]
brief: "REBEL 将 LLM policy optimization 简化为回归 relative reward 问题——给定两个 completion 的 reward 差，直接回归 policy logits，无需 clipping/value network/多个 heuristic。理论上等价于 Natural Policy Gradient，匹配 RL 文献最强收敛和样本复杂度保证。在 Llama-3-8B-Instruct 上达到 AlpacaEval 2.0/MT-Bench 强竞争性能。"
date: 2026-02-24
type: paper-note
tags: [rl, alignment, policy-optimization, regret-minimization, llm, image-generation]
rating: ★★★★☆
sources:
  - "arXiv:2404.16767v4 (Zhaolin Gao, Jonathan Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, J. Andrew Bagnell, Jason D. Lee, Wen Sun — CMU+Cornell+Princeton, Apr 2024 → Dec 2024)"
venue: preprint (2024)
related:
  - "[[AI/LLM/RL/PPO/PPO 原理]]"
  - "[[AI/LLM/RL/GRPO/GRPO 深度理解]]"
  - "[[AI/LLM/RL/DPO/DPO-TRL实践]]"
---

# REBEL: Reinforcement Learning via Regressing Relative Rewards

## 一、核心问题

PPO 在 LLM alignment 中的根本问题不是"它不 work"，而是 **工程复杂度过高**：
- Clipping heuristic 的理论基础薄弱
- Value network 的训练不稳定
- 对实现细节极度敏感（gradient clipping, reward normalization, entropy bonus 等）

**REBEL 的问题**：能否设计一个**最小主义** RL 算法，没有任何 heuristic，但有 NPG 级别的理论保证？

---

## 二、核心方法

### 2.1 从 Regret Minimization 到 Policy Optimization

REBEL 基于 **regret minimization**（后悔最小化）框架：

给定 prompt $x$，两个 completions $y_1, y_2 \sim \pi_\theta$，定义：

$$f_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

REBEL 的 loss 直接**回归 relative reward**：

$$\mathcal{L}_{\text{REBEL}}(\theta) = \mathbb{E}_{x, y_1, y_2} \left[ \left( (f_\theta(x, y_1) - f_\theta(x, y_2)) - (r(x, y_1) - r(x, y_2)) \right)^2 \right]$$

**直觉**：如果 $y_1$ 比 $y_2$ 好（reward 差 = +5），那么 policy 的 log-prob 差也应该 = +5。

### 2.2 为什么这等价于 Natural Policy Gradient？

**Theorem 1**（arXiv:2404.16767, §3）：REBEL 的更新在 KL-regularized 设定下等价于 Natural Policy Gradient（NPG）：

$$\pi_{t+1}(y|x) \propto \pi_t(y|x) \cdot \exp\left(\frac{r(x,y) - b(x)}{\beta}\right)$$

NPG 是已知收敛保证最强的 RL 算法之一（$O(1/T)$ 收敛到最优策略）。

```mermaid
graph LR
    A[Prompt x] --> B[Sample y1, y2]
    B --> C[Reward Model\nr(x,y1), r(x,y2)]
    C --> D[Relative Reward\nΔr = r(y1)-r(y2)]
    D --> E[MSE Regression\nf_θ(y1)-f_θ(y2) → Δr]
    E --> F[Updated Policy π_θ]
    style E fill:#ffa,stroke:#333
```

### 2.3 关键简洁性

**REBEL 不需要**：
- ❌ Value network
- ❌ Clipping（PPO 的核心 heuristic）
- ❌ GAE（advantage estimation 的复杂步骤）
- ❌ Entropy bonus
- ❌ Multiple rollout epochs

**REBEL 只需要**：
- ✅ 一个 reward model
- ✅ 一个 reference model（同 DPO）
- ✅ 回归 loss（MSE）

### 2.4 支持 Online/Offline 混合

REBEL 天然支持 offline 数据（已有的 preference pair）：

$$\mathcal{L}_{\text{offline}}(\theta) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \left( f_\theta(x, y_w) - f_\theta(x, y_l) - \hat{r}(x, y_w, y_l) \right)^2 \right]$$

其中 $\hat{r}$ 可以是 Bradley-Terry reward 或直接的 preference margin。

### 2.5 处理 Intransitive Preferences

现实中偏好不一定满足传递性（$A > B, B > C \not\Rightarrow A > C$）。REBEL 可以扩展到 **general preference function**，不假设 Bradley-Terry model——这是 DPO 做不到的（DPO 硬依赖 BT 假设）（arXiv:2404.16767, §4.3）。

---

## 三、实验结果

### Llama-3-8B-Instruct 微调

| 方法 | AlpacaEval 2.0 LC WR | MT-Bench | Open LLM LB |
|------|---------------------|----------|-------------|
| DPO | 25.8% | 7.82 | 67.3 |
| PPO | 28.4% | 7.91 | 67.8 |
| **REBEL (online)** | **30.2%** | **7.95** | **68.1** |

### 图像生成（Stable Diffusion）

REBEL 在图像生成的 RLHF 中也表现强劲，证明其跨模态通用性（arXiv:2404.16767, §5.3）。

---

## 四、批判性分析

### 我的评价

REBEL 是一篇 **理论驱动** 的优秀工作，但实际工程影响力有限：

**优点**：
1. **理论最强**：NPG 等价性 → 自动继承收敛和样本复杂度最强保证
2. **概念最简**：整个算法就是一个 MSE 回归，没有任何 heuristic
3. **跨模态**：在语言和图像上都 work
4. **处理 intransitive preference**：DPO 的理论盲区

**局限**：
1. **实际增益有限**：vs PPO/DPO 的提升在 2-3%，不够"颠覆性"
2. **需要 reward model**：仍然需要 RM，不像 DPO 可以 reference-free（SimPO）
3. **成对 rollout**：每次需要 2 个 completion 做对比，inference 开销不小
4. **社区采用率低**：GRPO/DAPO 的实际采用远超 REBEL，因为 DeepSeek R1 的推广效应
5. **未在 reasoning 任务验证**：论文聚焦 chat alignment，未测数学/代码推理

### 与 DPO 的理论关系

DPO 假设 preference 来自 Bradley-Terry model：$P(y_w > y_l) = \sigma(r(y_w) - r(y_l))$。

REBEL 不做此假设——它直接回归 reward 差值。当 BT 假设成立时，REBEL 和 DPO 在理论上等价；当 BT 假设不成立时（intransitive preferences），REBEL 仍然 work，DPO 则可能 fail。

这正是 IPO（arXiv:2310.12036）试图解决的同一问题——但 IPO 用 identity mapping 替代 sigmoid，REBEL 用 regression 框架。

---

## 五、落地应用

### 工程要点

- **实现**：约 50 行核心代码——比 PPO 简单一个量级
- **超参**：$\beta$（KL 系数）是唯一关键超参
- **适用场景**：chat alignment、图像生成 RLHF、非传递性偏好场景

### 面试高频问法

1. **"REBEL 和 PPO 的区别？"** → REBEL 将 RL 简化为回归 relative reward，无 clipping/value net/GAE；理论等价于 NPG
2. **"REBEL 和 DPO 的区别？"** → DPO 假设 BT model，REBEL 不做此假设；REBEL 支持 online data，DPO 天然 offline
3. **"为什么 REBEL 没有广泛采用？"** → 实际增益有限（2-3%），且 GRPO/DAPO 被 DeepSeek R1 推广，先发优势

---

## 六、启发思考

**So What**：REBEL 证明了一个深刻的点——**复杂的 RL 算法（PPO）在 LLM alignment 中是不必要的**。一个简单的回归 loss 就能达到同等甚至更好的效果，且有更强的理论保证。这与 GRPO、REINFORCE++、ReMax 等工作形成共识：**LLM RLHF 不需要经典 RL 的全部机制**。

**未解问题**：
- REBEL 在 multi-turn reasoning（AIME/MATH）上的表现？
- REBEL + PRIME 的 process reward 如何交互？
- 成对 regression 在 agent 场景下的 scaling 行为？

---

## 推荐阅读

- **原始论文**：[arXiv:2404.16767](https://arxiv.org/abs/2404.16767)
- **关联笔记**：
  - [[AI/LLM/RL/PPO/PPO 原理|PPO]] — REBEL 试图简化的对象
  - [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO]] — 另一种 critic-free 简化
  - [[AI/LLM/RL/DPO/DPO-TRL实践|DPO]] — 理论近亲（BT 假设下等价）
  - [[AI/LLM/RL/Preference-Optimization/IPO-Identity-Preference-Optimization|IPO]] — 同一问题的另一种解法
