---
title: "On-Policy vs. Off-Policy"
brief: "On-Policy vs Off-Policy 的核心区分：采样策略与更新策略是否为同一个；On-Policy（PPO）数据利用率低但稳定，Off-Policy（SAC/Q-Learning）可复用数据但有分布偏移风险；直接影响 LLM RL 训练效率。"
type: concept
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/concept
---
# On-Policy vs. Off-Policy

> RL 中最基础的分类维度之一。在 LLM alignment 语境下，这个区分直接影响训练效率和稳定性。

## 核心区别

**一句话**：采样策略和更新策略是不是同一个？

```
On-Policy:   用当前策略 π_θ 采样 → 用这批数据更新 π_θ → 数据丢弃 → 重新采样
Off-Policy:  用旧策略 π_old 采样 → 可以多次复用数据更新 π_θ → 数据不用丢
```

| | On-Policy | Off-Policy |
|---|-----------|------------|
| 采样效率 | 低（每次更新后数据作废） | 高（数据可复用） |
| 稳定性 | 高（数据分布匹配） | 低（分布偏移风险） |
| 实现复杂度 | 简单 | 需要 importance sampling 等修正 |
| 代表算法 | REINFORCE, PG | Q-Learning, SAC |

## 在 LLM 中的情况

这里有个常见误解需要澄清：

### PPO 到底是 On-Policy 还是 Off-Policy？

**严格来说 PPO 是 On-Policy 算法**。但它用了一个 trick：在一次采样后做**多个 epoch 的更新**（通常 1-4 个 mini-batch epoch），通过 **clipping** 来限制策略变化幅度，确保不会偏离太远。

```python
# PPO 的 clipping 机制
ratio = π_θ(a|s) / π_old(a|s)  # importance sampling ratio
clipped_ratio = torch.clamp(ratio, 1 - ε, 1 + ε)
loss = -min(ratio * A, clipped_ratio * A)

# ε 通常取 0.2
# 当 ratio 超出 [0.8, 1.2] 范围时，梯度被截断
```

所以 PPO 是 **"带点 off-policy 味道的 on-policy"** — 它复用了同一批数据多次，但通过 clipping 保证策略不会偏离太远。

### GRPO 的设计

GRPO (Group Relative Policy Optimization) 更接近纯 on-policy：对同一个 prompt 采样一组 response，用组内相对排序作为 advantage，不需要 value model。

```python
# GRPO 的核心思想
responses = [model.generate(prompt) for _ in range(G)]  # G 个采样
rewards = [reward_fn(r) for r in responses]
advantages = (rewards - mean(rewards)) / std(rewards)  # 组内标准化
# 用 advantages 做 policy gradient
```

### 真正的 Off-Policy：DPO

DPO (Direct Preference Optimization) 是真正的 off-policy — 它直接用**离线的偏好数据**训练，不需要在线采样：

```python
# DPO 的数据格式
# (prompt, chosen_response, rejected_response)
# 这些数据可以是任何策略生成的

# DPO loss
log_ratio_chosen = log π_θ(y_w|x) - log π_ref(y_w|x)
log_ratio_rejected = log π_θ(y_l|x) - log π_ref(y_l|x)
loss = -log(σ(β * (log_ratio_chosen - log_ratio_rejected)))
```

## 实际影响

| 方面 | On-Policy (PPO/GRPO) | Off-Policy (DPO) |
|------|----------------------|-------------------|
| GPU 需求 | 高（需要在线推理） | 低（纯训练） |
| 数据需求 | 低（自己生成） | 高（需要偏好标注） |
| 训练时间 | 长（采样是瓶颈） | 短 |
| 效果上限 | 高（持续探索） | 受限于数据质量 |
| reward hacking | 容易出现 | 不太会 |

我的观点：**On-policy 的效果天花板更高，但工程成本也更高**。如果 reward model 质量好且有足够算力，PPO/GRPO 优于 DPO。资源受限时 DPO 是很好的起步方案。

## verl 中的处理

verl 默认是 on-policy 流程（rollout → reward → update）。但也支持 off-policy 异步训练：

```python
# verl off-policy: 采样和训练解耦
# 采样 worker 持续生成数据放入 buffer
# 训练 worker 从 buffer 中取数据更新
# 用 importance sampling ratio 修正分布偏移
```

参见 [[AI/LLM/Frameworks/verl/Off Policy 异步训练器|Off Policy 异步训练器]]

## 相关

- [[AI/LLM/RL/Fundamentals/为什么 PPO 优于 PG|为什么 PPO 优于 PG]] — PPO 的设计动机
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO-verl实践]] — GRPO 在 verl 中的使用
- [[AI/LLM/RL/DPO/DPO-Unsloth实践|DPO-Unsloth实践]] — Off-policy 的 DPO
- [[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO-verl实践]] — DAPO 变体
- [[AI/LLM/Frameworks/verl/Off Policy 异步训练器|Off Policy 异步训练器]] — verl 的 off-policy 支持
