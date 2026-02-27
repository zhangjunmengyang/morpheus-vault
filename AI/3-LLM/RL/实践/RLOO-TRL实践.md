---
brief: "RLOO TRL 实践——REINFORCE Leave-One-Out 的 TRL 工程指南；比 GRPO 实现更简单，无需独立 Critic，相比 PPO 显存需求降低 50%；适合资源受限下的在线 RL 快速实验。"
title: "RLOO"
type: project
domain: ai/llm/rl/rloo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/rloo
  - type/project
---
# RLOO TRL 实践

> REINFORCE Leave-One-Out — PPO 的轻量替代方案。
> 参考：https://huggingface.co/docs/trl/v0.21.0/en/rloo_trainer

## RLOO 是什么

RLOO（REINFORCE Leave-One-Out）是一种 policy gradient 方法，核心思想：

```
PPO: 需要 Critic Model 来估计 baseline（价值函数）
GRPO: 用 group 内的均值/标准差作为 baseline
RLOO: 用 "留一法" 估计 baseline — 每个样本的 baseline 是组内其他样本的均值
```

RLOO 的优势：
- **无需 Critic Model** — 比 PPO 省一半显存
- **低方差估计** — 比朴素 REINFORCE 稳定得多
- **实现简单** — 比 PPO 代码量少很多

## 数学直觉

```python
# RLOO 的核心：Leave-One-Out baseline
def rloo_advantage(rewards):
    """
    rewards: [r1, r2, r3, r4] — 同一个 prompt 的 K 个采样
    
    对于 r1 的 baseline: mean(r2, r3, r4)
    对于 r2 的 baseline: mean(r1, r3, r4)
    ...
    """
    K = len(rewards)
    total = sum(rewards)
    advantages = []
    for i, r in enumerate(rewards):
        baseline = (total - r) / (K - 1)  # 排除自己
        advantages.append(r - baseline)
    return advantages

# 示例
rewards = [1.0, 0.0, 1.0, 0.0]  # 4 个采样，2 对 2 错
# r1 的 advantage: 1.0 - mean(0, 1, 0) = 1.0 - 0.33 = +0.67
# r2 的 advantage: 0.0 - mean(1, 1, 0) = 0.0 - 0.67 = -0.67
```

对比 GRPO 的 advantage：
```python
# GRPO: 全组均值和标准差
mean = np.mean(rewards)   # 0.5
std = np.std(rewards)     # 0.5
advantages_grpo = [(r - mean) / std for r in rewards]
# 全部是 +1 或 -1
```

RLOO 的 advantage 更细腻，因为每个样本的 baseline 不同。

## TRL 实践

```python
from trl import RLOOTrainer, RLOOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 需要一个 reward model 或 reward function
reward_model = AutoModelForSequenceClassification.from_pretrained("reward-model")

config = RLOOConfig(
    output_dir="./rloo-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_ppo_epochs=2,        # 每批数据训几个 epoch
    rloo_k=4,                # 每个 prompt 采样 K 个回答
    kl_coef=0.05,            # KL 惩罚
    max_new_tokens=512,
)

trainer = RLOOTrainer(
    config=config,
    model=model,
    ref_model=None,           # 自动创建
    reward_model=reward_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

trainer.train()
```

## RLOO vs GRPO vs PPO

| 维度 | PPO | GRPO | RLOO |
|------|-----|------|------|
| 需要 Critic | ✅ | ❌ | ❌ |
| Baseline | 学习的价值函数 | 组均值/标准差 | Leave-one-out 均值 |
| 显存开销 | 最大 | 中等 | 中等 |
| 实现复杂度 | 高 | 中 | 低 |
| 方差 | 低 | 中 | 低 |
| 适用场景 | 大规模训练 | 可验证任务 | 通用 |

## 调参建议

- `rloo_k`：4-8 比较好，太大计算浪费，太小 baseline 估计不准
- `kl_coef`：RLOO 建议保留 KL 惩罚（0.01-0.1），不像 GRPO/DAPO 那样设为 0
- `learning_rate`：1e-7 到 5e-6，需要根据模型大小调整

## 我的观点

RLOO 是 PPO 和 GRPO 之间的一个甜蜜点。如果你有 reward model 但不想折腾 PPO 的复杂实现，RLOO 是最佳选择。如果你的任务可以用 rule-based reward（如数学/代码），GRPO/DAPO 更直接。

## 相关

- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]] — 对比算法
- [[AI/3-LLM/RL/实践/DPO-Unsloth实践|DPO Unsloth 实践]] — 离线偏好对齐
- [[AI/3-LLM/RL/实践/DAPO-verl实践|DAPO verl 实践]] — GRPO 改进版
- [[AI/3-LLM/Frameworks/verl/实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[AI/3-LLM/RL/算法/PPO 原理|PPO 原理]]
- [[AI/3-LLM/Frameworks/TRL/TRL 概述|TRL 概述]]
