---
brief: "DAPO verl 实践——基于 verl 框架的 DAPO（Dynamic sAmpling Policy Optimization）训练指南；DAPO 去掉 KL clip 改用动态采样，verl 实现细节和超参配置；比 GRPO 更激进的探索策略工程落地。"
title: "DAPO"
type: project
domain: ai/llm/rl/dapo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/dapo
  - type/project
---
# DAPO verl 实践

> Decoupled Clip and Dynamic Sampling Policy Optimization — GRPO 的增强版。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/dapo.html

## DAPO 是什么

DAPO（Decoupled Alignment Policy Optimization）是在 GRPO 基础上的改进，核心解决了 GRPO 训练中的几个痛点：

1. **Clip 解耦**：将 upper clip 和 lower clip 分开控制
2. **动态采样**：根据训练进度动态调整采样策略
3. **Token-Level 策略**：更细粒度的策略优化
4. **过长惩罚移除**：不再因为回答太长而惩罚模型

## DAPO vs GRPO 的关键差异

```python
# GRPO 的 clipping
ratio = new_prob / old_prob
clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

# DAPO 的 decoupled clipping
# 分别控制正向和负向 clip
if advantage > 0:
    # 正优势：鼓励探索，用较大的 upper clip
    clipped = torch.clamp(ratio, max=1 + clip_range_upper)
else:
    # 负优势：快速远离差回答，用较小的 lower clip
    clipped = torch.clamp(ratio, min=1 - clip_range_lower)
```

**为什么解耦有效？**
- 正优势时 clip 太紧会抑制好回答的强化（模型不敢偏离太多）
- 负优势时 clip 太松会导致训练不稳定
- 解耦后可以分别调参，兼顾探索和稳定性

## verl 配置

```yaml
# config/dapo.yaml
algorithm:
  name: dapo
  group_size: 16          # DAPO 建议更大的 group_size
  clip_range_upper: 0.28  # 正优势的 clip 上界
  clip_range_lower: 0.22  # 负优势的 clip 下界
  kl_coef: 0.0            # 无 KL 惩罚
  entropy_coef: 0.001     # 轻微的熵正则，鼓励多样性
  
  # DAPO 特有参数
  dynamic_sampling: true
  token_level: true        # token-level advantage
  remove_length_penalty: true
  
actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 5e-7     # DAPO 建议比 GRPO 更低的 lr
  
rollout:
  engine: vllm
  temperature: 1.0        # DAPO 用更高的温度采样
  top_p: 0.95
  max_new_tokens: 2048
```

## 动态采样策略

DAPO 的一个核心创新是动态采样：

```python
# 训练初期：高温度 + 大 group_size → 充分探索
# 训练后期：降低温度 → 稳定策略

def dynamic_sampling_config(step, total_steps):
    progress = step / total_steps
    
    if progress < 0.3:
        return {"temperature": 1.0, "group_size": 16}
    elif progress < 0.7:
        return {"temperature": 0.8, "group_size": 12}
    else:
        return {"temperature": 0.6, "group_size": 8}
```

另一个关键点是**过滤全对/全错的 group**：

```python
# 如果一个 group 内所有回答都对了（或都错了），advantage 全为 0
# DAPO 会跳过这些 group，只训练有区分度的数据
def filter_groups(rewards, group_size):
    groups = rewards.reshape(-1, group_size)
    # 保留有差异的 group
    valid_mask = (groups.std(dim=1) > 0)
    return valid_mask
```

## 运行

```bash
# 与 GRPO 类似，只是换配置
torchrun --nproc_per_node=8 \
    -m verl.trainer.main_ppo \
    --config-path config \
    --config-name dapo
```

## 效果对比

在数学推理任务上（论文数据）：

| 算法 | MATH-500 | GSM8K | 训练稳定性 |
|------|----------|-------|----------|
| GRPO | 58.2 | 82.1 | 偶尔崩溃 |
| DAPO | 62.5 | 85.3 | 稳定 |

DAPO 的提升主要来自：
1. 更好的 clip 策略减少了训练震荡
2. 动态采样避免了浪费算力在没有区分度的数据上
3. Token-level advantage 提供了更细粒度的信号

## 我的观点

DAPO 是 GRPO 的工程改进版——原理上没有本质突破，但在实践中效果更好、更稳定。如果你已经在用 GRPO 且遇到训练不稳定的问题，直接切换到 DAPO 是最快的解决方案。

核心 takeaway：RL for LLM 目前的瓶颈不在算法创新，而在工程细节的打磨。Clip 怎么设、温度怎么调、哪些数据该跳过，这些「小」决定对最终效果影响巨大。

## 相关

- [[GRPO 深度理解|GRPO 深度理解]] — 基础算法
- [[GRPO-verl实践|GRPO verl 实践]] — 对比参考
- [[verl 训练参数|verl 训练参数]]
- [[实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[硬件资源预估|硬件资源预估]]
- [[PPO 原理|PPO 原理]]
- [[verl 概述|verl 概述]]

> **see-also（Agent 训练场景）**：DAPO 的 clip-higher（非对称 clip）和动态采样是 Agent RL 训练最常用的稳定性工具，在以下 Agent 场景中频繁被引用：
> - [[Agent-RL-训练实战指南|Agent RL 训练实战指南]] — DAPO clip-higher 在 Agent 场景的完整实践指南，含 entropy collapse 的诊断和修复配方
> - [[AI/2-Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong（极长 horizon Agent）]] — 极长任务场景下 DAPO 思路的延伸：如何在 100+ step 的训练中维持探索熵
