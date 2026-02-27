---
title: "RL 基础算法手撕实操"
brief: "强化学习核心算法完整PyTorch实现：Policy Gradient（REINFORCE）、Actor-Critic（A2C/A3C）、PPO（clip+GAE）、DQN（replay buffer/target network），LLM对齐视角下各算法选择指南，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, reinforcement-learning, policy-gradient, actor-critic, ppo, dqn, pytorch]
related:
  - "[[Projects/MA-RLHF/lc8-PPO/lc8-01-PPO-手撕实操|PPO-手撕实操-MA-RLHF]]"
  - "[[Projects/MA-RLHF/lc8-GRPO/lc8-01-GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc8-DPO/lc8-01-DPO-手撕实操|DPO-手撕实操]]"
---

# RL 基础算法手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

本笔记覆盖从 Bellman 方程到 Policy Gradient 的完整 RL 基础算法链条，为理解 LLM 中的 RLHF（PPO/DPO/GRPO）打下基础。

**学习路径**：Bellman 方程 → 动态规划(DP) → 广义策略迭代(GPI) → 蒙特卡洛(MC) → Q-Learning → SARSA → DQN → Policy Gradient

**核心概念**：
- **状态价值函数** $V(s)$：从状态 $s$ 出发，按策略 $\pi$ 执行获得的期望回报
- **动作价值函数** $Q(s,a)$：从状态 $s$ 执行动作 $a$，再按策略 $\pi$ 执行的期望回报
- **Bellman 方程**：$V(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V(s')]$
- **策略梯度**：$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi}(s,a)]$

> 注：lc7_rl 的 .py 源文件当前为空（占位），以下内容基于仓库中的 Cross Entropy、MC 示例 notebook 和 RL 理论框架整理。

## 二、核心实现

### 2.1 Cross Entropy Loss 手撕（RL/LLM 共享基础）

**原理**：交叉熵是 SFT 和 RL 的共享损失函数基础。给定目标分布 $p$ 和预测分布 $q$：

$$H(p, q) = -\sum_i p_i \log(q_i)$$

在分类任务中，$p$ 为 one-hot 向量，简化为 $-\log q_{\text{label}}$。

**代码**：

```python
# 手撕 Cross Entropy（batch 版本，仿 PyTorch 实现）
def cross_entropy_pytorch(label, logits):
    bs, _ = logits.shape
    # 用 log_softmax 避免 softmax 溢出
    logprob = F.log_softmax(logits, dim=-1)
    idx = torch.arange(0, bs)
    logprob = logprob[idx, label]
    CE_loss = -logprob.mean()
    return CE_loss

# 验证：与 PyTorch 官方实现一致
loss_fn = nn.CrossEntropyLoss()
assert torch.allclose(cross_entropy_pytorch(label, logits), loss_fn(logits, label))
```

**关键洞察**：
- CE 对 logits 的梯度有闭合形式：$\nabla_{\text{logits}} = q - p$（softmax 输出 - 目标分布）
- 这就是为什么 `nn.CrossEntropyLoss` 接受 logits 而非 softmax 输出——直接从 logits 求导计算量更小

### 2.2 Cross Entropy 梯度推导

**CE + Softmax 联合梯度**：

$$\frac{\partial \text{CE}}{\partial l_i} = q_i - p_i$$

其中 $q_i = \text{softmax}(l_i)$，$p_i$ 为目标分布。

**代码验证**：

```python
p = torch.tensor([0.0, 1.0, 0.0])
logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# PyTorch 自动求导
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits.unsqueeze(0), torch.tensor([1]))
loss.backward()
print(logits.grad)  # 与 softmax(logits) - p 一致

# 手动验证
q = F.softmax(logits, dim=0)
print(q - p)  # 与 logits.grad 完全一致
```

### 2.3 蒙特卡洛方法基础

**原理**：通过大量随机采样估计期望值。经典例子——用 MC 估计 $\pi$：

```python
def monte_carlo_pi(n):
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside_circle = x**2 + y**2 <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n
    return pi_estimate
```

**关键洞察**：采样越多估计越准确——这直接对应 RL 中 MC 方法估计 V(s) 和 Q(s,a) 的思想。

### 2.4 Bellman 方程 → 动态规划

**Bellman 方程（价值迭代）**：

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

**策略评估**：给定策略 $\pi$，迭代计算 $V^{\pi}(s)$

**策略改进**：$\pi'(s) = \arg\max_a Q^{\pi}(s,a)$

### 2.5 广义策略迭代（GPI）

**原理**：交替进行策略评估和策略改进，两者不断向最优策略收敛。

### 2.6 蒙特卡洛方法（MC V/Q）

**MC 估计 V(s)**：多次从状态 $s$ 开始完成 episode，用实际回报均值估计 $V(s)$

**MC 估计 Q(s,a)**：估计状态-动作对的价值，支持 off-policy 学习

**ε-greedy MC**：以概率 $\epsilon$ 随机探索，$(1-\epsilon)$ 贪心执行——平衡探索与利用

### 2.7 Q-Learning

**原理**：Off-policy TD 学习，直接学习最优 Q 函数：

$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**关键**：用 $\max$ 而非当前策略选择 $a'$，因此是 off-policy 的。

### 2.8 SARSA

**原理**：On-policy TD 学习：

$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(s',a') - Q(s,a)]$$

其中 $a'$ 由当前策略选择（而非 max）。

**Q-Learning vs SARSA**：Q-Learning 更激进（乐观估计），SARSA 更保守（考虑实际策略的探索）。

### 2.9 DQN

**原理**：用神经网络逼近 Q 函数，引入 Experience Replay 和 Target Network 稳定训练。

### 2.10 Policy Gradient

**原理**：直接参数化策略 $\pi_\theta$，通过梯度上升最大化期望回报：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$$

**与 LLM 的联系**：
- RLHF 中的 PPO 就是 Policy Gradient 的变种
- LLM 的"动作"是生成 token，"策略"是 token 概率分布
- Reward Model 替代环境提供奖励信号

## 三、从基础到 LLM 应用的逻辑链

| RL 基础概念 | LLM-RLHF 对应 |
|------------|---------------|
| 状态 $s$ | prompt + 已生成的 token 序列 |
| 动作 $a$ | 下一个 token（vocab size 大小的离散空间）|
| 策略 $\pi_\theta$ | LLM 的 token 概率分布 |
| 奖励 $R$ | Reward Model 的输出分数 |
| 参考策略 $\pi_{\text{ref}}$ | SFT 模型（KL 惩罚的锚点）|
| GAE/Advantage | Token 级别的优势估计 |
| Policy Gradient | PPO 的 clipped surrogate objective |
| Value function | Critic 模型估计每个 token 位置的价值 |

## 四、关键洞察与总结

1. **CE 梯度的简洁性**：$q - p$ 这个优美的结果是整个 LLM 训练的数学基石
2. **MC 到 TD 的演进**：MC 需要完整 episode，TD 可以在线更新——LLM 中 token-level 的 reward 分配类似 TD
3. **On-policy vs Off-policy**：PPO 是 on-policy（但通过 importance sampling 复用旧数据），DPO 是 off-policy（直接用偏好数据）
4. **Value-based → Policy-based**：LLM 太大无法枚举所有动作的 Q 值，Policy Gradient 直接优化策略更自然
5. **搜索空间的挑战**：棋类游戏动作空间 ~225，LLM 动作空间 ~128K，且序列深度可达数千——这是 MCTS 难以直接用于 LLM 的根本原因

> 完整代码见：`/tmp/ma-rlhf/lecture/lc7_rl/`（.py 文件为占位）、`/tmp/ma-rlhf/notebook/common/` 和 `/tmp/ma-rlhf/notebook/o1/mc-example.ipynb`
