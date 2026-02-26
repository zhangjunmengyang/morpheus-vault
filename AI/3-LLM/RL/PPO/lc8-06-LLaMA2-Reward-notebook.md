---
title: lc8 — LLaMA2 Reward Model 从零手写
brief: 从零用 LLaMA2 实现 Reward Model：在 CausalLM 基础上替换 lm_head 为 scalar reward head，用 Bradley-Terry 损失训练，实现 pairwise 偏好评分。是 PPO 四模型体系中 RM 的完整工程实现，面试常考：RM 和 Policy 的架构差异、reward normalization 的必要性。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl-alignment
  - reward-model
  - llama2
  - bradley-terry
  - lc8
related:
  - "[[AI/3-LLM/RL/PPO/LLaMA2-Reward-Model实现]]"
  - "[[AI/3-LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
  - "[[RLHF-DPO-2026-技术全景]]"
---

# LLaMA2 Reward Model 从零手写

> MA-RLHF Batch B / lc8-LLaMA2-Reward
> Source: `notebook/reward/LLaMA2-reward.ipynb`
> Author: xiaodongguaAIGC / dhcode-cpp
> 对应: LLaMA2 论文 §3.2.2 Reward Modeling
> 评分: ★★★★☆

---

## TL;DR

LLaMA2 RM 的完整实现细节：**模型结构（LM Head 换 Regression Head）→ Margin Loss 训练 → 双 RM 选择机制 → Reward Whitening（逆 Sigmoid + Z-score）→ KL Penalty 组合**。这是 PPO-based RLHF 中 RM 部分的工程化参考实现。

---

## 模型结构

```
LlamaForCausalLM（SFT 初始化）
    └── lm_head: Linear(hidden, vocab_size)  ← 换掉

LlamaForSequenceClassification（RM）
    └── score: Linear(hidden, 1)              ← regression head
```

```python
model = LlamaForCausalLM(config)
model.save_pretrained('./lm_pretrained')

# 直接从 LM checkpoint 加载，替换 head
rm_model = LlamaForSequenceClassification.from_pretrained(
    './lm_pretrained', num_labels=1   # num_labels=1 → scalar reward
)
```

**关键**：`num_labels=1` 使得输出是 scalar（回归），不是概率（分类）。RM 的最后一层是 `Linear(hidden_size, 1)`，取最后一个 token 的 hidden state 过这个线性层得到 reward score。

---

## Margin Loss（LLaMA2 改进版）

**标准 BT Loss**：
$$L = -\log\sigma(r_w - r_l)$$

**LLaMA2 Margin Loss**（考虑偏好强度）：
$$L = -\log\sigma(r_w - r_l - m(r))$$

其中 $m(r)$ 是离散的 margin，由标注者给出的偏好强度决定（"明显更好" → 大 margin，"稍微更好" → 小 margin）。

```python
X_chosen   = torch.randint(0, 100, (1,10))
X_rejected = torch.randint(0, 100, (1,10))
margin = 3.0  # Margin Large: "Significantly Better"

rm_chosen   = rm_model(input_ids=X_chosen).logits    # scalar
rm_rejected = rm_model(input_ids=X_rejected).logits  # scalar

# 标准 BT loss
loss = -torch.sigmoid(rm_chosen - rm_rejected).log()

# 带 margin 的 loss（LLaMA2）
loss_with_margin = -torch.sigmoid(rm_chosen - rm_rejected - margin).log()
```

**Margin 的意义**：
- 当 chosen 只是"稍微好一点"时，`margin=0`（或小），标准 BT 即可
- 当 chosen 是"显著更好"时，`margin=large`，要求 reward 差距更大才能让 loss 足够低
- 防止模型对"明显好坏"的样本产生不够大的 reward 差异

---

## 双 RM 选择机制（LLaMA2 Safety）

LLaMA2 同时训练两个 RM：
- `R_s`：Safety Reward Model（安全性）
- `R_h`：Helpfulness Reward Model（有用性）

选择规则：

$$R_c(g|p) = \begin{cases}
R_s(g|p) & \text{if is\_safety}(p) \text{ or } R_s(g|p) < 0.15 \\
R_h(g|p) & \text{otherwise}
\end{cases}$$

```python
def llama2_reward_select(reward_safety, reward_helpfulness):
    # 安全相关 prompt 或安全分低 → 用安全 RM
    return reward_safety if reward_safety < 0.15 else reward_helpfulness

# 安全分低（-0.3）→ 用安全分：-0.3
rc = llama2_reward_select(reward_safety=-0.3, reward_helpfulness=0.7)  # → -0.3

# 安全分高（1.3）→ 用有用性分：0.4
rc = llama2_reward_select(reward_safety=1.3, reward_helpfulness=0.4)   # → 0.4
```

**设计哲学**：安全是底线，当回复安全分不够高（< 0.15）时，直接用安全分作为信号，无论帮助性多高。这确保了不会因为"非常有帮助的危险回复"而获得高奖励。

---

## Reward Whitening

**问题**：RM 输出的 raw scalar 量级不稳定，会影响 PPO 中 KL penalty 的相对强度。

**LLaMA2 解法**：先做逆 Sigmoid（logit 变换），再做 Z-score 归一化（whitening）：

$$\hat{R}_c = \text{WHITEN}(\text{LOGIT}(R_c))$$

### 步骤 1：逆 Sigmoid（LOGIT）

```python
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))  # logit(x) = log(x/(1-x))

# sigmoid_output=0.9 → logit ≈ 2.20
# sigmoid_output=0.5 → logit = 0.0  (中性点)
# sigmoid_output=0.01 → logit ≈ -4.6
```

**意义**：把 $(0,1)$ 范围的 reward 映射到 $(-\infty, +\infty)$，拉伸两端的区分度，压缩中间（接近 0.5 的）差异。

> 注：LLaMA2 论文注释"实际 reward 已输出 scalar，无需这步"——代码仅为展示完整流程。

### 步骤 2：Z-score Whitening

```python
def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)  # (x - μ) / σ
    if not shift_mean:
        whitened += mean   # 只缩放方差，不移动均值
    return whitened

# 效果：无论原始范围是 [0.8, 4.6] 还是 [100.8, 104.6]
# 输出都是均值≈0、方差≈1 的标准化数值
```

**Whitening 的作用**：
- 消除 reward 的绝对量级，使 reward 信号的"单位"与 KL penalty 可比
- 防止某批样本 reward 整体偏高或偏低，导致 PPO 更新不稳定
- `rsqrt(var + 1e-8)`：数值稳定（防止 var=0 时除零）

---

## KL Penalty 组合（Final Reward）

$$R(g|p) = \hat{R}_c(g|p) - \beta \cdot D_{KL}(\pi_\theta(g|p) \| \pi_0(g|p))$$

```python
# KL 近似计算（per-token）
output = model(X)['logits'][:, -1, :].sigmoid()
prob   = torch.gather(output, dim=1, index=index_old)   # π_θ(a)
prob_old = old_policy_prob                               # π_0(a)

kl = F.kl_div(torch.log(prob), prob_old)   # KL(π_0 || π_θ) 近似

# 最终 PPO reward
beta = 0.01
rm_ppo = rm_score - beta * kl
```

**注意代码 bug**：`F.kl_div` 的参数顺序是 `kl_div(log_input, target)`，计算的是 $KL(target \| input)$，这里实际上是 $KL(\pi_0 \| \pi_\theta)$（而不是通常所说的 $KL(\pi_\theta \| \pi_0)$）。生产实现中应仔细确认方向。

---

## 与 RLHF-PPO notebook 的关联

在 `RLHF_PPO_Pytorch.ipynb` 中，reward 是简化的：
```python
rm_xy = get_reward(models.rm, xy)  # 直接用 RM score
```

LLaMA2 实际做法（本 notebook）是完整的 4 步 pipeline：
```
RM score → [逆Sigmoid] → [Whitening] → [- β*KL] → final reward for PPO
```

Whitening 对应 PPO notebook 里的 `rewards_kl` 中的归一化处理，KL penalty 对应 `compute_rewards_kl` 函数。

---

## Batch B 完整知识链总结

| notebook | 核心内容 | 关键知识点 |
|---------|---------|---------|
| `RLHF_PPO_Pytorch` | PPO 完整 pipeline | 四模型架构 / GAE / Clipped Loss |
| `GRPO` | GRPO 训练 | Group advantage / KL as loss term |
| `GRPO_KL` | KL 近似分析 | K3 = r-log(r)-1 恒非负且均值最准 |
| `DPO + IPO + BT` | 偏好优化 | BT 过拟合 / IPO 二次目标 / 4×forward |
| `KTO` | 单标签偏好 | 前景理论 / z_ref baseline |
| `o1_prm_search` | PRM + Search | SEP token / LM Head 复用 / 逐步验证 |
| `LLaMA2-reward` | RM 工程化 | Margin Loss / 双RM / Whitening |

**核心 insight**：PPO 的 reward 不是"原始 RM score"，而是经过双 RM 选择 + whitening + KL penalty 后的复合信号。每一步都有工程原因。

---

## 面试高频考点

**Q: LLaMA2 为什么要训练两个 RM？**
A: 安全性和有用性是两个独立的、有时相互冲突的维度。单一 RM 很难同时优化两者（安全性高的回复可能没那么有用）。双 RM + 选择规则确保了安全是硬约束（safety < 0.15 时优先安全），有用性在安全的前提下最大化。

**Q: Reward Whitening 为什么要做逆 Sigmoid 再 Z-score？**
A: 逆 Sigmoid（logit 变换）把 $(0,1)$ 的概率空间映射到实数空间，拉伸两端差异；Z-score 消除绝对量级，使 reward 与 KL penalty 的量纲可比。不做 whitening 会导致 KL penalty 的 β 需要随 RM 输出的量级手动调整，训练极不稳定。

**Q: Margin Loss 相比标准 BT Loss 有什么改进？**
A: 标准 BT 只要 chosen > rejected 就满足，不区分偏好强度。Margin Loss 对"明显更好"的样本施加更大的 margin，要求 reward 差距更大，迫使 RM 学到更精细的偏好区分，尤其对区分"还行"和"非常好"的回复有帮助。

**Q: LLaMA2 RM 的模型结构是怎么初始化的？**
A: 从 SFT checkpoint 初始化，把 lm_head（`Linear(hidden, vocab_size)`）换成 score（`Linear(hidden, 1)`），其他参数保持不变。SFT 的语言理解能力直接迁移，只需微调最后一层学习打分。

---

## 关联笔记

- `AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写.md` — PPO 完整 pipeline，RM 作为 reward 来源
- `AI/LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写.md` — BT loss 的理论基础
- `AI/LLM/MA-RLHF课程/lc8-KTO-PRM-Search-从零手写.md` — 偏好优化替代方案

## See Also

- [[AI/3-LLM/RL/PPO/LLaMA2-Reward-Model实现]] — 同内容 Batch A 版
- [[AI/3-LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写]] — 同系列：PPO 四模型完整训练
- [[AI/3-LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写]] — 同系列：BT 模型理论基础
- [[AI/3-LLM/RL/PPO/PPO 原理]] — PPO 算法理论（RM 是 RLHF 的前置模块）
- [[RLHF-DPO-2026-技术全景]] — RLHF 全链路：RM 在四模型体系中的位置
