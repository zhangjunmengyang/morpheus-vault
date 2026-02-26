---
brief: "verl 实现其他 RL 方法——在 verl 框架中扩展非标准 RL 算法的工程指南；自定义 advantage 计算/loss function/采样策略的接口；DAPO/GRPO 变种在 verl 中的实现参考。"
title: "实现其他 RL 方法"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# 实现其他 RL 方法

> 参考：https://verl.readthedocs.io/en/latest/advance/dpo_extension.html

## verl 不只是 PPO/GRPO

verl 的架构设计比较灵活，核心抽象是 **Worker + DataFlow**，并不绑死某个特定算法。官方文档用 DPO 作为扩展示例，展示如何在 verl 里实现非 on-policy 的算法。

## 扩展机制

verl 的算法扩展有三个层次：

### 1. 修改 Reward Function（最简单）

不改训练流程，只改 reward 计算逻辑：

```python
# 自定义 reward function
def my_reward_fn(data_batch):
    """
    data_batch 包含:
    - prompts: 输入 prompt
    - responses: 生成的 response
    - old_log_probs: 旧策略的 log prob
    """
    # 比如加入格式奖励
    format_reward = check_format(data_batch["responses"])
    # 加入正确性奖励
    correctness_reward = verify_answer(data_batch["responses"])
    
    return 0.3 * format_reward + 0.7 * correctness_reward
```

### 2. 修改 Loss Function（中等）

保持 on-policy rollout 流程，但换 loss 计算方式。比如从 PPO 换成 RLOO 或 ReMax：

```python
from verl.trainer.ppo import core_algos

class CustomAlgorithm:
    @staticmethod
    def compute_policy_loss(old_log_probs, new_log_probs, advantages, clip_ratio):
        """替换 PPO clip loss 为你的自定义 loss"""
        # 例如实现 RLOO
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # RLOO: 用 leave-one-out baseline
        # 对 group 内每个 response，baseline 是其他 response 的平均 reward
        group_size = advantages.shape[1]
        baseline = (advantages.sum(dim=1, keepdim=True) - advantages) / (group_size - 1)
        adjusted_advantages = advantages - baseline
        
        loss = -(ratio * adjusted_advantages).mean()
        return loss
```

### 3. 自定义 Trainer（最灵活）

完全重写训练循环，适合 DPO / KTO 这类 off-policy 算法：

```python
from verl.trainer.base import BaseTrainer

class DPOTrainer(BaseTrainer):
    """DPO 不需要 rollout，直接用离线数据"""
    
    def __init__(self, config, actor_model, ref_model, tokenizer):
        super().__init__(config)
        self.actor = actor_model
        self.ref = ref_model
        self.beta = config.get("beta", 0.1)
    
    def training_step(self, batch):
        # DPO 核心: 偏好对 (chosen, rejected)
        chosen_ids = batch["chosen_input_ids"]
        rejected_ids = batch["rejected_input_ids"]
        
        # 计算 log probs
        pi_chosen = self.actor.forward_log_prob(chosen_ids)
        pi_rejected = self.actor.forward_log_prob(rejected_ids)
        ref_chosen = self.ref.forward_log_prob(chosen_ids)
        ref_rejected = self.ref.forward_log_prob(rejected_ids)
        
        # DPO loss
        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = pi_logratios - ref_logratios
        
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss
    
    def fit(self):
        """DPO 不走 rollout -> reward -> update 循环"""
        for epoch in range(self.config.epochs):
            for batch in self.train_dataloader:
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
```

## 已支持的算法列表

verl 官方和社区已经实现的算法：

| 算法 | 类型 | 是否需要 Critic | 实现复杂度 |
|------|------|----------------|-----------|
| PPO | On-policy | ✅ | 高 |
| GRPO | On-policy | ❌ | 中 |
| RLOO | On-policy | ❌ | 中 |
| ReMax | On-policy | ❌ | 低 |
| DPO | Off-policy | ❌ | 低 |
| DAPO | On-policy | ❌ | 中 |

## 实战建议

1. **先用 GRPO 跑通**，再考虑换算法。GRPO 不需要 Critic，省资源且效果不差
2. **DPO 在 verl 里不如直接用 TRL**：verl 的优势在分布式 on-policy 训练，off-policy 场景用 TRL / Unsloth 更方便
3. **自定义 loss 时注意梯度裁剪**：RL loss 天然不稳定，`max_grad_norm=1.0` 是标配
4. **DAPO 的 clip_higher 和 clip_lower 区分开**：DAPO 对正负 advantage 用不同 clip ratio，实现时容易搞混

```python
# DAPO 双侧 clip
def dapo_loss(ratio, advantages, clip_low=0.2, clip_high=0.28):
    # 正 advantage 用更大的 clip (鼓励探索)
    clip_upper = torch.where(advantages > 0, 1 + clip_high, 1 + clip_low)
    clip_lower = torch.where(advantages > 0, 1 - clip_low, 1 - clip_high)
    
    clipped_ratio = torch.clamp(ratio, clip_lower, clip_upper)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
```

## 相关

- [[verl 概述|verl 概述]]
- [[算法概述|算法概述]]
- [[HybridFlow|HybridFlow]]
- [[AI/3-LLM/RL/目录|RL 算法总览]]
- [[AI/3-LLM/RL/DPO/DPO|DPO]]
- [[AI/3-LLM/RL/GRPO/GRPO|GRPO]]
