---
brief: "GPG verl 实践——Group Policy Gradient 的 verl 工程实现；GPG 是 GRPO 的早期变体，组内策略梯度的基础形式；verl 中的训练配置参考和与 GRPO 的算法差异对比。"
title: "GPG"
type: project
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/project
---
# GPG verl 实践

> Group Policy Gradient — verl 中的另一种 group-based RL 算法。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/gpg.html

## GPG 概述

GPG（Group Policy Gradient）是 GRPO 的一个变体，主要区别在 advantage 的计算方式和策略更新的机制。

核心差异：

```python
# GRPO: 标准化的 advantage
advantage_grpo = (reward - mean) / std

# GPG: 直接使用 reward 差值，不做标准化
advantage_gpg = reward - baseline
# 其中 baseline 可以是 group mean 或 learned baseline
```

GPG 的设计哲学是**更接近原始 REINFORCE**，减少对 advantage 的人工干预（标准化），让梯度信号更直接。

## 什么时候用 GPG

GPG 适合的场景：
- 奖励分布比较稳定（不需要标准化来平滑）
- 奖励的绝对值有意义（标准化会丢失这个信息）
- 探索性要求更高的任务

不推荐的场景：
- 奖励分布方差很大 — 不标准化会导致训练不稳定
- 数学推理等有明确正确答案的任务 — GRPO/DAPO 更好

## verl 配置

```yaml
algorithm:
  name: gpg
  group_size: 8
  clip_range: 0.2
  kl_coef: 0.01        # GPG 通常需要 KL 惩罚来防止策略偏移
  normalize_advantage: false  # 关键区别

actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 5e-7

rollout:
  engine: vllm
  temperature: 0.8
  max_new_tokens: 1024
```

## 训练

```bash
torchrun --nproc_per_node=8 \
    -m verl.trainer.main_ppo \
    --config-path config \
    --config-name gpg
```

## 我的观点

GPG 在 verl 的算法矩阵中属于「可用但不是首选」的位置。大多数场景下，GRPO（可验证任务）或 DAPO（需要稳定性）是更好的选择。GPG 更适合研究探索，或者在奖励设计非常精细的场景下使用。

## 相关

- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO verl 实践]] — 更常用的方案
- [[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO verl 实践]] — GRPO 增强版
- [[AI/LLM/RL/Other-Algorithms/OPO-verl实践|OPO verl 实践]]
- [[AI/LLM/RL/Other-Algorithms/SPPO-verl实践|SPPO verl 实践]]
- [[AI/LLM/Frameworks/verl/实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
