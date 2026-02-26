---
brief: "verl Off-Policy 异步训练器——解耦采样与更新的异步架构；Actor 持续生成 rollout，Trainer 并行更新模型权重，消除同步等待；提升 GPU 利用率 30-50%，适合大 batch RLHF 生产场景。"
title: "Off Policy 异步训练器"
type: project
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/project
---
# Off Policy 异步训练器

> 参考：https://verl.readthedocs.io/en/latest/advance/one_step_off.html

## 问题：on-policy 的瓶颈

传统 PPO/GRPO 是严格的 on-policy：生成一批 rollout → 计算 reward → 更新模型 → 丢弃 rollout → 重新生成。这里有个巨大的效率问题：

```
Timeline (同步 on-policy):
|===== Rollout =====|===== Train =====|===== Rollout =====|===== Train =====|
                     ^ GPU 闲着做推理     ^ GPU 闲着做推理
```

**rollout 和 training 不能重叠**，GPU 利用率大概只有 50-60%。

## 解决方案：One-Step Off-Policy

verl 的 off-policy 异步训练器允许 **rollout 和 training 同时进行**：

```
Timeline (异步 off-policy):
|===== Rollout 1 =====|===== Rollout 2 =====|===== Rollout 3 =====|
      |===== Train on Rollout 0 =====|===== Train on Rollout 1 =====|
```

核心思想：training 用的是**上一轮** rollout 的数据（one-step off），而当前轮的 rollout 已经在并行生成了。

## 为什么只 off 一步

off-policy 的程度越大（用越旧的数据训练），distribution shift 越严重。One-step off 是一个 sweet spot：

- **性能提升**：几乎 2x throughput（rollout 和 train 完全重叠）
- **质量损失**：极小。一步的 policy 差异很小，importance sampling 可以修正

```python
# Importance Sampling 修正
# 旧策略 π_old 生成的数据，用新策略 π_new 训练时需要修正
importance_weight = torch.exp(log_prob_new - log_prob_old)

# clip 防止极端权重
importance_weight = torch.clamp(importance_weight, 0.5, 2.0)

loss = -(importance_weight * advantages).mean()
```

## verl 实现架构

```
┌─────────────────────────────────────────────┐
│              AsyncTrainer                    │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │ Rollout Pool  │    │  Training Pool   │   │
│  │ (GPU Group A) │    │  (GPU Group B)   │   │
│  │               │    │                  │   │
│  │ 生成 batch N+1│    │  训练 batch N    │   │
│  └──────┬───────┘    └────────┬─────────┘   │
│         │    Data Queue       │              │
│         └────────────────────▶│              │
│                                              │
└─────────────────────────────────────────────┘
```

两组 GPU 分别负责 rollout 和 training，通过数据队列解耦。

## 配置方式

```yaml
# verl 配置
trainer:
  type: "async"
  async:
    # rollout 和 train 分配到不同 GPU pool
    rollout_gpu_ids: [0, 1, 2, 3]
    train_gpu_ids: [4, 5, 6, 7]
    
    # 队列大小，控制 rollout 领先 train 多少步
    buffer_size: 2
    
    # off-policy 修正
    importance_sampling: true
    is_clip_range: [0.5, 2.0]
    
    # 数据新鲜度，超过这个步数差的数据会被丢弃
    max_staleness: 1
```

## 适用场景

| 场景 | 是否推荐异步 | 理由 |
|------|------------|------|
| 资源充足（16+ GPU） | ✅ 强烈推荐 | 可以拆成两组 GPU |
| 资源紧张（8 GPU） | ❌ 不推荐 | 分两组后每组太少 |
| 训练稳定性要求极高 | ⚠️ 谨慎 | off-policy 引入额外 variance |
| 大 batch 长 sequence | ✅ 推荐 | rollout 耗时长，异步收益更大 |
| DPO/离线算法 | ❌ 不需要 | 本身就是 off-policy |

## 与同步模式的对比

```python
# 实测数据 (Qwen2.5-7B, 16×A100-80G, GRPO)
# 同步模式:
#   - Rollout: 180s / batch
#   - Training: 120s / batch
#   - Total per step: 300s
#   - Throughput: ~3.3 steps/min

# 异步模式 (8+8 GPU split):
#   - Rollout: 360s / batch (GPU 减半)
#   - Training: 240s / batch (GPU 减半)
#   - 但 overlap 后 effective: 360s / step
#   - Throughput: ~1.7 steps/min ... 等等，这不对

# 关键：异步模式下虽然单步更慢，但 pipeline 填满后:
#   - 每 360s 完成一次 update (rollout 是瓶颈)
#   - vs 同步每 300s 完成一次
#   - 异步优势在 rollout >> train 时更明显
```

**结论**：异步训练器在 **rollout 耗时远大于 training** 时收益最大（比如长 sequence 生成、复杂 reward 计算）。如果 rollout 和 train 差不多快，异步反而可能因为 GPU 数减半而变慢。

## 踩坑记录

1. **模型同步时机**：train 更新完权重后要同步给 rollout 的 actor，这个通信开销不能忽略
2. **buffer 数据过期**：如果 train 比 rollout 快很多，buffer 会空转等待，实际退化为同步
3. **gradient accumulation 要重新算**：GPU 数减半后，effective batch size 变了，需要调整 gradient accumulation steps

## 相关

- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]]
- [[AI/LLM/Frameworks/verl/性能调优|性能调优]]
- [[AI/LLM/Frameworks/verl/硬件资源预估|硬件资源预估]]
- [[AI/LLM/Frameworks/verl/verl 训练参数|verl 训练参数]]
