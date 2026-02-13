---
title: "OPO"
type: project
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/project
---
# OPO verl 实践

> Online Preference Optimization — 在线版的 DPO。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/opo.html

## OPO 是什么

OPO（Online Preference Optimization）结合了 DPO 和 on-policy RL 的优点：

```
DPO: 离线偏好数据 → 一次性训练
OPO: 在线生成 → 自动构造偏好对 → 实时训练
```

工作流程：

```python
# OPO 训练循环
for prompt_batch in dataset:
    # 1. 当前策略生成多个回答
    responses = policy.generate(prompt_batch, num_samples=K)
    
    # 2. 奖励模型评分
    rewards = reward_model.score(prompt_batch, responses)
    
    # 3. 自动构造偏好对（最高分 vs 最低分）
    chosen = responses[rewards.argmax()]
    rejected = responses[rewards.argmin()]
    
    # 4. DPO loss 更新
    loss = dpo_loss(policy, ref_policy, chosen, rejected)
    loss.backward()
```

**OPO 的优势**：
- 不需要人工标注偏好数据
- On-policy 生成，数据分布更匹配当前策略
- 避免了 PPO 的复杂实现

## verl 配置

```yaml
algorithm:
  name: opo
  group_size: 8          # 每个 prompt 采样数
  beta: 0.1              # DPO 温度
  pair_strategy: best_worst  # 配对策略: best_worst / adjacent

actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 1e-6

rollout:
  engine: vllm
  temperature: 0.7
  max_new_tokens: 1024

reward:
  type: model             # 需要 reward model
  model_name: "reward-model-path"
```

## OPO vs DPO vs GRPO

| 维度 | DPO | OPO | GRPO |
|------|-----|-----|------|
| 数据 | 离线偏好对 | 在线生成偏好对 | 在线 + function reward |
| 需要 RM | ❌ | ✅ | ❌（用规则函数） |
| On-policy | ❌ | ✅ | ✅ |
| 适用任务 | 通用对齐 | 通用对齐 | 可验证任务 |

## 我的观点

OPO 是 DPO 的自然进化 — 解决了 DPO 最大的问题（离线数据分布不匹配）。但它引入了对 reward model 的依赖，如果 RM 质量不高，OPO 的效果也会受限。在有高质量 RM 的场景下，OPO 是一个很好的选择。

## 相关

- [[AI/LLM/RL/DPO/DPO-Unsloth实践|DPO Unsloth 实践]] — 离线版
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO verl 实践]] — function reward 方案
- [[AI/LLM/RL/Other-Algorithms/SPPO-verl实践|SPPO verl 实践]]
- [[AI/LLM/Frameworks/verl/实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
