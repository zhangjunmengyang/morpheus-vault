---
title: "为什么 PPO 优于 PG(Policy Gradient)"
brief: "从 REINFORCE 到 PPO 的演进路径：PG 高方差+数据利用率低的问题，通过 baseline/GAE/clip ratio 逐步解决；理解 PPO 优于朴素 PG 的根本原因是 LLM alignment 面试必备知识点。"
type: concept
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/concept
---
# 为什么 PPO 优于 PG(Policy Gradient)

> 从 REINFORCE 到 PPO 的演进路径，理解这条线就理解了 LLM alignment 的基础。

参考：
- https://zhuanlan.zhihu.com/p/342150033
- https://www.zhihu.com/question/357056329
- https://mp.weixin.qq.com/s/ESrn3Dlmu1Q0QHcUNTm_Sg

## REINFORCE 的问题

Policy Gradient (PG) 的最基础形式就是 REINFORCE：

```python
# REINFORCE
# J(θ) = E[R(τ)]  最大化期望回报
# ∇J(θ) = E[∑_t ∇log π_θ(a_t|s_t) * G_t]

# 伪代码
for episode in episodes:
    trajectory = rollout(policy)
    returns = compute_returns(trajectory.rewards)
    for t, (state, action, G_t) in enumerate(trajectory):
        loss = -log_prob(action | state) * G_t
        loss.backward()
```

三个致命问题：

### 1. 方差爆炸

`G_t`（return）的方差非常大。想象一个 LLM 生成任务，同一个 prompt 不同 response 的 reward 可以从 -5 到 +5。直接用 G_t 做梯度权重，训练会非常不稳定。

**解决方案：Baseline（基线）**

```python
# 减去一个 baseline b(s)，不改变梯度期望但降低方差
# ∇J(θ) = E[∑_t ∇log π_θ(a_t|s_t) * (G_t - b(s_t))]

# 最常用的 baseline：Value Function V(s)
# A(s,a) = G_t - V(s_t)  ← 这就是 Advantage
```

### 2. 采样效率极低

REINFORCE 是纯 on-policy：每次更新后数据必须丢弃。对于 LLM，生成一条 response 的计算成本很高，用一次就扔太浪费。

**解决方案：Importance Sampling**

```python
# 用旧策略 π_old 的数据来更新当前策略 π_θ
# ∇J(θ) = E_{π_old}[π_θ(a|s)/π_old(a|s) * A(s,a) * ∇log π_θ(a|s)]
#                    ↑ importance weight

# 这就是 TRPO/PPO 的基础
```

### 3. 步长难调

学习率太大 → 策略崩溃（catastrophic forgetting）；太小 → 收敛慢。PG 没有机制保证每次更新的幅度合理。

## TRPO → PPO 的演进

### TRPO (Trust Region Policy Optimization)

核心想法：限制每次更新的策略变化幅度。

```python
# TRPO 的约束优化
# max_θ E[π_θ(a|s)/π_old(a|s) * A(s,a)]
# s.t. KL(π_old || π_θ) ≤ δ

# 问题：约束优化需要二阶导（Fisher information matrix），计算昂贵
```

### PPO: TRPO 的工程友好版

PPO 用 **clipping** 替代 KL 约束，把约束优化变成无约束优化：

```python
def ppo_loss(log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 两个目标取最小值
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    return -torch.min(surr1, surr2).mean()

# 直觉：
# 如果 advantage > 0（好动作），ratio 最多到 1+ε，不会过度鼓励
# 如果 advantage < 0（坏动作），ratio 最多到 1-ε，不会过度惩罚
```

## PPO 相对 PG 的优势总结

| 问题 | PG | PPO |
|------|-----|-----|
| 高方差 | 原始 return | GAE (Generalized Advantage Estimation) |
| 采样效率 | 单次使用 | 多 epoch 复用 |
| 步长控制 | 无 | Clipping 约束 |
| 稳定性 | 差 | 好 |

## 在 LLM Alignment 中

InstructGPT 率先用 PPO 做 RLHF，核心流程：

```
1. SFT Model（监督微调后的模型）
2. Reward Model（从人类偏好训练的奖励模型）  
3. PPO Training:
   - Actor: 生成 response
   - Critic: 估计 V(s)，计算 GAE advantage
   - Reward: RM 打分
   - 用 PPO loss 更新 Actor
   - 加 KL penalty: reward = RM_score - β * KL(π_θ || π_ref)
```

PPO 在 LLM 场景的特殊挑战：
1. **Critic 很难训好**：token 级别的 value estimation 本身就是个难题
2. **Reward hacking**：模型学会利用 RM 的漏洞拿高分
3. **工程复杂度高**：需要同时维护 4 个模型

这也是为什么 GRPO、DPO 等 "去 critic" 方法受到关注 — 它们试图绕过 PPO 的工程复杂度。

## 相关

- [[AI/3-LLM/RL/Fundamentals/On-Policy vs Off-Policy|On-Policy vs Off-Policy]] — PPO 的 on-policy 属性
- [[Projects/HF Deep RL Course|HF Deep RL Course]] — 系统学习 RL
- [[AI/3-LLM/RL/实践/GRPO-verl实践|GRPO-verl实践]] — 去掉 Critic 的 PPO 替代
- [[AI/3-LLM/RL/实践/DAPO-verl实践|DAPO-verl实践]] — DAPO 改进
