---
brief: "HuggingFace 官方深度强化学习课程——系统覆盖基础 RL 到高级 RLHF，配套 Colab 实践；是从工程角度入门 LLM 强化学习的推荐路径，兼顾理论与 HF transformers 生态实践。"
title: "Deep RL Course"
type: tutorial
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/tutorial
---
# HF Deep RL Course

> Hugging Face 出品的免费 RL 课程，从零到可以看懂 PPO/GRPO 论文的水平。

课程地址：https://huggingface.co/learn/deep-rl-course/unit7/hands-on

## 课程结构与学习路径

| Unit | 主题 | 核心收获 |
|------|------|---------|
| 1 | Introduction to RL | MDP, reward, policy, value function |
| 2 | Q-Learning | Bellman equation, exploration vs exploitation |
| 3 | Deep Q-Learning | DQN, experience replay, target network |
| 4 | Policy Gradient | REINFORCE, 为什么要 PG |
| 5 | Actor-Critic | A2C, advantage function, GAE |
| 6 | PPO | Clipping, 多 epoch 训练 |
| 7 | Multi-Agent RL | 合作/竞争, centralized training |
| 8 | RLHF | 人类反馈, reward model |

## 对 LLM 工程师最重要的概念

不需要把整个课程啃完。做 LLM alignment 需要深入理解的核心概念：

### 1. Policy Gradient Theorem

```python
# 核心公式：
# ∇J(θ) = E_τ~π_θ [∑_t ∇log π_θ(a_t|s_t) * R(τ)]

# 在 LLM 语境下：
# π_θ(a_t|s_t) → 模型在位置 t 生成 token a_t 的概率
# R(τ) → 整条 response 的 reward
# 直觉：好 response 中每个 token 的概率都被推高
```

### 2. Advantage Function 与 GAE

```python
# Advantage = Q(s,a) - V(s)
# "这个 action 比平均好多少"

# GAE (Generalized Advantage Estimation)
# A_t^GAE = ∑_l (γλ)^l * δ_{t+l}
# δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)

# λ 控制 bias-variance tradeoff:
# λ=0 → 纯 TD(0), low variance, high bias
# λ=1 → 纯 MC,    high variance, low bias
# 通常取 λ=0.95
```

### 3. KL Divergence 在 RLHF 中的角色

```python
# RLHF 的 reward 通常加 KL penalty：
# R_total = R_reward_model - β * KL(π_θ || π_ref)

# 为什么需要 KL penalty:
# 1. 防止 reward hacking（模型找到 RM 的漏洞）
# 2. 保持模型的通用能力不退化
# 3. 保证输出多样性

# β 的选择很关键：
# β 太大 → 模型几乎不更新
# β 太小 → reward hacking
# 实践中通常 β ∈ [0.01, 0.1]，或用 adaptive KL controller
```

### 4. Value Function 的 token 级别理解

在经典 RL 中，V(s) 是状态的价值估计。在 LLM RLHF 中：

```python
# 状态 s_t = (prompt, token_1, ..., token_t)
# V(s_t) = "从当前已生成的 tokens 开始，预期的 total reward"
# 
# Critic model 需要对每个 token 位置输出一个 value
# 这实际上是一个 sequence labeling 问题
# 
# 训练 Critic 的 loss:
# L = (V_θ(s_t) - R_t)^2  where R_t = 实际 return
```

## 学习建议

1. **不需要实现经典 RL 环境**（CartPole、Atari），但要理解 MDP 的语言
2. **重点关注 Unit 4-6**：Policy Gradient → Actor-Critic → PPO 这条主线
3. **Unit 8 (RLHF)** 直接和 LLM 工作相关，但课程内容偏浅，需要补充论文
4. **数学不需要完全推导**，但要理解每个公式的直觉含义

推荐补充材料：
- Spinning Up in Deep RL (OpenAI) — 更数学化
- 李宏毅的 RL 课程 — 中文讲解，深入浅出
- InstructGPT 论文 — RLHF 的开山之作

## 相关

- [[AI/3-LLM/RL/Fundamentals/为什么 PPO 优于 PG|为什么 PPO 优于 PG]] — PPO 的核心改进
- [[AI/3-LLM/RL/Fundamentals/On-Policy vs Off-Policy|On-Policy vs Off-Policy]] — 基础分类
- [[AI/3-LLM/RL/实践/GRPO-verl实践|GRPO-verl实践]] — 实际动手
- [[AI/3-LLM/RL/实践/RLOO-TRL实践|RLOO-TRL实践]] — TRL 框架的 RL 训练
