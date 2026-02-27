---
title: lc8 — DPO / IPO / Bradley-Terry 偏好优化从零手写
brief: 从零实现 DPO（对比偏好优化）、IPO（迭代偏好优化）、Bradley-Terry 模型和 GRPO KL 散度三种近似。掌握 BT 模型的最大似然估计、DPO 的 policy gradient 闭式推导，以及 KL 散度 forward/reverse/JSD 三种实现对训练稳定性的影响。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl-alignment
  - dpo
  - ipo
  - bradley-terry
  - kl-divergence
  - lc8
related:
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-01-DPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-03-DPO-完整Notebook实现]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-04-GRPO-KL散度三种近似]]"
  - "lc8-RL×LLM-MOC"
  - "RLHF-DPO-2026-技术全景"
---

# DPO / IPO / Bradley-Terry 偏好优化从零手写

> MA-RLHF Batch B / lc8-DPO + BT_model + GRPO_KL
> Source: `notebook/DPO/DPO.ipynb` + `BT_model.ipynb` + `GRPO/GRPO_KL.ipynb`
> Author: xiaodongguaAIGC / dhcode-cpp
> 评分: ★★★★★

---

## TL;DR

三个 notebook 联合精读，覆盖 RLHF 偏好优化的完整理论链：

```
Bradley-Terry 模型（偏好概率建模）
    ↓ RM training
Reward Model（sequence-level scalar）
    ↓ RLHF → PPO（KL-constrained RL）
    ↓ 闭式解（离线 dataset）
DPO（implicit reward + closed-form update）
    ↓ 过拟合问题（π_l → 0）
IPO（二次目标，防止 policy collapse）
```

GRPO_KL notebook 则深挖了三种 KL 近似的统计性质，解释了为什么 DeepSeek GRPO 选择了 Schulman (2020) 的 K3 近似。

---

## Part 1：Bradley-Terry 模型

### 直觉

BT 模型解决一个问题：**给定两两对比结果（谁打败谁），推算每个选手的绝对实力**。对 RLHF 的意义：人类只能给出 "A 比 B 好"（pairwise preference），BT 将其转化为连续的 reward score。

### 两种等价形式

**形式一：exp ratio（MLE 形式）**

$$P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)}$$

```python
def forward_exp(self, chosen_id, rejected_id):
    reward_chosen = torch.exp(self.reward[chosen_id])
    reward_rejected = torch.exp(self.reward[rejected_id])
    return reward_chosen / (reward_chosen + reward_rejected)

def loss_exp(self, pred, label):
    return -torch.log(pred) if label == 1 else -torch.log(1-pred)
```

损失 = `-log P(winner)`，即最大似然估计。

**形式二：sigmoid（logistic 回归形式）**

$$P(i \succ j) = \sigma(s_i - s_j)$$

$$L = -\log \sigma(s_i - s_j)$$

```python
def forward_sigmoid(self, chosen_id, rejected_id):
    return torch.sigmoid(self.reward[chosen_id] - self.reward[rejected_id])

def loss_sigmoid(self, pred, label):
    loss = -(label * log(pred) + (1-label) * log(1-pred))
    return loss
```

**两种形式数学等价**：$\frac{e^{s_i}}{e^{s_i}+e^{s_j}} = \sigma(s_i - s_j)$（softmax with 2 items = sigmoid of difference）

### 过拟合实验

```python
# 100 epochs → reward 收敛到合理值
# 100000 epochs → reward 发散到 ±∞（分数差距趋于无穷）
```

关键观察：BT 损失的全局最优解可以是 $s_i - s_j \to +\infty$（只要排序正确，分数差越大 loss 越小）。这是 DPO 过拟合的根源。

---

## Part 2：DPO（Direct Preference Optimization）

### 数据格式

```python
# Preference pair: (prompt, chosen, rejected)
prompt_chosen   = [5,8,9,10,5,3,  16,29,18,17]  # prompt + good response
prompt_rejected = [5,8,9,10,5,3,  26,14,31, 0]  # prompt + bad response
label           = [0,0,0,0, 0,0,   1, 1, 1, 1]  # 只对 response tokens 算 loss
```

**关键**：prompt 部分 label=0（不参与 loss），只在 response tokens 位置计算 DPO loss。

### per-token log-prob 提取

```python
def get_probs(logits, labels):
    per_token_logps = torch.gather(
        logits.log_softmax(-1),    # [B, T, V]
        dim=2,
        index=labels.unsqueeze(2)  # [B, T, 1]
    ).squeeze(2)                   # → [B, T]
    return per_token_logps
```

即：对每个位置 t，从 vocab 维度取出对应真实 token 的 log 概率。

### DPO Loss 推导

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

代码实现：

```python
# 四次 forward（ref & model，chosen & rejected）
logits_chosen_ref  = ref_model(**x_chosen).logits    # no_grad
logits_rejected_ref = ref_model(**x_rejected).logits  # no_grad
logits_chosen   = model(**x_chosen).logits            # grad
logits_rejected = model(**x_rejected).logits          # grad

probs_chosen_ref  = get_probs(logits_chosen_ref,  prompt_chosen)
probs_chosen      = get_probs(logits_chosen,       prompt_chosen)
probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
probs_rejected    = get_probs(logits_rejected,      prompt_rejected)

beta = 0.1
pi_logratios  = probs_chosen  - probs_rejected      # log π_θ(y_w) - log π_θ(y_l)
ref_logratios = probs_chosen_ref - probs_rejected_ref  # log π_ref(y_w) - log π_ref(y_l)
logits = pi_logratios - ref_logratios               # implicit reward difference

losses = -F.logsigmoid(beta * logits) * label       # mask prompt
loss = losses.sum(-1) / attention_mask.sum()         # normalize by total tokens
```

**数学展开**：

$$\beta\left(\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)$$

$= \beta\left[\underbrace{(\log\pi_\theta(y_w) - \log\pi_\theta(y_l))}_{pi\_logratios} - \underbrace{(\log\pi_{ref}(y_w) - \log\pi_{ref}(y_l))}_{ref\_logratios}\right]$

计算时以 **per-token 累加的 log-prob 之差** 作为近似（sum of token log-probs ≈ sequence log-prob）。

### DPO 训练行为（实验观察）

```python
# 100 epochs SGD, lr=0.1
# β 的大小不影响最终收敛点——π(y_l) 趋于 0
```

**DPO 的过拟合问题（IPO 论文指出）**：

- BT 目标的全局最优解可以是 $\pi_\theta(y_l) \to 0$（rejected response 的概率趋于 0）
- 当训练数据中 $y_l$ 被明确 100% 劣于 $y_w$ 时，最优策略要求 $r(y_w) - r(y_l) \to +\infty$
- 代入 DPO 闭式解：$\frac{\pi^*(y_l)}{\pi^*(y_w)} = 0$，即 rejected policy **collapse 到 0**
- KL 正则化的效果随着策略趋于确定性而**逐渐消失**（KL regularization strength decreases）

这在 **finite data** 情况下尤其严重：即使真实偏好是 0.8，empirical 估计可能是 1.0，导致策略 collapse。

---

## Part 3：IPO（Identity Preference Optimization）

### IPO 的核心改动

将 DPO 的 sigmoid CE loss 换成**二次（MSE）目标**：

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(h_\pi(y_w, y_l, x) - \frac{\tau^{-1}}{2}\right)^2\right]$$

其中：
$$h_\pi(y, y', x) = \log\frac{\pi(y|x)}{\pi_{ref}(y|x)} - \log\frac{\pi(y'|x)}{\pi_{ref}(y'|x)}$$

推导（复用 DPO 的 logits 计算）：
$$h_\pi = \log\frac{\pi(y_w|x)\pi_{ref}(y_l|x)}{\pi(y_l|x)\pi_{ref}(y_w|x)} = \underbrace{(probs\_chosen - probs\_rejected)}_{pi\_logratios} - \underbrace{(probs\_chosen\_ref - probs\_rejected\_ref)}_{ref\_logratios}$$

即：**`logits` 变量本身就是 $h_\pi$，可以直接复用 DPO 代码**。

```python
constant = 1.0 / (beta * 2.0)    # τ^{-1}/2

if loss_type == 'DPO':
    losses = -F.logsigmoid(beta * logits) * label
elif loss_type == 'IPO':
    losses = torch.square(logits - constant) * label    # MSE 目标
```

### IPO vs DPO 收敛对比

```
DPO（β=0.1, lr=0.01）: π_l → 0（policy collapse，与 β 无关）
IPO（β=0.1, lr=0.0001）: π_l → 某个正值（由 β/τ 控制，不 collapse）
IPO（β=0.5, lr=0.0001）: π_l → 更大的正值（β 控制收敛点）
```

**关键差异**：
- DPO：β 不影响最终 π_l（只影响收敛速度），最终一定 collapse
- IPO：β（即 τ^{-1}/2）直接控制 policy 的收敛点，**KL 正则化真正有效**

**IPO 的代价**：需要更小的学习率（lr=0.0001 vs DPO 的 0.01），Adam 在大 lr 下会不稳定。

---

## Part 4：三种 KL 近似（GRPO_KL notebook）

背景：GRPO 中需要对每个 token 计算 KL，加入 loss 作为正则化。

### 三种近似（Schulman 2020）

设 $r = \frac{\pi_{ref}(a)}{\pi_\theta(a)}$（或 log 形式的差值），令 $\text{logr} = \log r$：

| 近似 | 公式 | 特点 |
|------|------|------|
| **K1** | $-\text{logr}$ | 有**负值**，不保证非负，均值=真实KL但方差极大 |
| **K2** | $\text{logr}^2 / 2$ | 恒非负，泰勒展开二阶项，均值偏小 |
| **K3** | $r - \log r - 1 = (r-1) - \log r$ | 恒非负（因为 $x-1 \geq \log x$），均值最接近真实KL，**方差最小** |

```python
logr = p.log_prob(x) - q.log_prob(x)    # log(π_ref/π_θ)，从 q=π_θ 采样

k1 = -logr                               # K1
k2 = logr**2 / 2                        # K2
k3 = (logr.exp() - 1) - logr            # K3: r - log(r) - 1
```

### K3 恒非负的数学证明

因为 $\forall x > 0$：$\log x \leq x - 1$（等号当且仅当 $x=1$）

令 $x = r = \pi_{ref}/\pi_\theta$：
$$\log r \leq r - 1 \implies 0 \leq r - 1 - \log r$$

所以 K3 = $r - 1 - \log r \geq 0$，且仅当 $\pi_\theta = \pi_{ref}$ 时等于 0（与真实 KL 的非负性一致）。

### 为什么 GRPO 选 K3

```
真实 KL ≈ 0.125（高斯分布实验）

K1: 均值=0.125, std >> truekl  ← 方差太大，梯度不稳定
K2: 均值=0.062, std 中等        ← 系统性低估
K3: 均值=0.126, std < K1        ← 均值最准，恒非负，方差可接受
```

K3 是三者中**偏差最小、恒非负、方差最低**的近似，用于 per-token loss 计算最合适。

代码（GRPO.ipynb 中）：
```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    # K3: r - log(r) - 1，其中 r = π_ref / π_θ
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1
    # = exp(log π_ref) / exp(log π_θ) - log(π_ref/π_θ) - 1
    # = π_ref/π_θ - log(π_ref/π_θ) - 1
```

---

## 知识体系：偏好优化方法谱系

```
人类偏好数据 (y_w > y_l)
         ↓ Bradley-Terry 建模
Reward Model r(x, y) = scalar
         ↓ RLHF
PPO: max E[r(x,y)] - β KL(π||π_ref)     ← 需要 RM + critic，在线，4模型
         ↓ 闭式解推导
DPO: r_θ(x,y) = β log(π_θ/π_ref) + C   ← implicit reward，离线，2模型
         ↓ 过拟合问题
IPO: 二次目标，h_π → τ^{-1}/2           ← 防止 policy collapse
         ↓ verifiable reward 替代人类偏好
GRPO/RLVR: rule reward + group advantage  ← 去掉 RM，去掉 BT 假设
```

---

## 面试高频考点

**Q: DPO 和 PPO 的最本质区别？**
A: PPO 显式训练 RM，用 RM 分数 + critic 做 RL 更新（在线）。DPO 利用 RLHF 最优解的闭式形式，把 RM 隐含进 log(π/π_ref)，直接从偏好数据 offline 训练（无需 RM 推理、无需 critic、无需 rollout）。代价：受 distribution shift 约束，数据分布必须覆盖好。

**Q: Bradley-Terry 模型的损失函数是什么？**
A: $L = -\log\sigma(r_w - r_l)$，即 logistic 回归的 negative log-likelihood。等价于 BT 模型的最大似然估计。RM 训练用的正是这个损失。

**Q: DPO 为什么会过拟合？IPO 怎么解决？**
A: DPO 的目标函数（BT 损失）全局最优解可以是 $\pi(y_l) \to 0$，此时 KL 正则化失效。IPO 换成二次 MSE 目标，把隐式 reward 差（h_π）回归到一个固定值（τ^{-1}/2），KL 正则化变得真实有效，不会 collapse。

**Q: DPO 中为什么要做 per-token log-prob 的 sum，而不是整体 sequence log-prob？**
A: Causal LM 的 sequence log-prob = sum of per-token log-probs（joint = product of conditional = sum of log conditionals）。实现上，先 log_softmax 再 gather 取真实 token 的概率，是 per-token 形式；sum 之后就是 sequence-level log-prob。两者等价，per-token 实现更灵活（可以 mask prompt）。

**Q: GRPO 为什么用 K3 近似 KL，而不是直接用 KL 散度？**
A: 直接计算精确 KL 需要对整个 vocab 求和（$\sum_a \pi_{ref}(a)\log(\pi_{ref}/\pi_\theta)$），计算量大且不能 per-sample 地加入 loss。K3 近似只需要当前 token 的 logprob，高效且恒非负，均值最接近真实 KL，方差比 K1/K2 更小。

---

## 关联笔记

- `AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写.md` — PPO 完整 pipeline
- `AI/LLM/MA-RLHF课程/lc8-GRPO-notebook-Pytorch从零手写.md` — GRPO，含 K3 近似使用
- `AI/LLM/RL/RLHF/DPO.md` — DPO 论文精读（Rafailov et al. 2023）
- `AI/LLM/RL/Other-Algorithms/GRPO-Improvement-Panorama-2026.md` — GRPO 改进全景

## See Also

- [[Projects/MA-RLHF/lc8-DPO/lc8-01-DPO-手撕实操]] — 同算法手撕实操版（MA-RLHF lc8 Batch A）
- [[Projects/MA-RLHF/lc8-DPO/lc8-03-DPO-完整Notebook实现]] — 同算法 Notebook 端到端版
- [[Projects/MA-RLHF/lc8-GRPO/lc8-04-GRPO-KL散度三种近似]] — KL 散度三种近似的专题版
- [[Projects/MA-RLHF/lc8-PPO/lc8-03-RLHF-PPO-完整Pytorch实现]] — 同系列：PPO 完整实现
- RLHF-DPO-2026-技术全景 — 理论全景：DPO/IPO/ORPO 等全图谱
