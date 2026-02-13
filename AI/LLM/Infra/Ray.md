---
title: "Ray"
type: concept
domain: ai/llm/infra
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/infra
  - type/concept
---
# Ray

> UC Berkeley 出品的分布式计算框架，从 RL 训练起家，现在是 LLM 训练/推理/serving 的基础设施之一。

文档：https://docs.ray.io/en/latest/ray-overview/index.html

## 核心抽象

Ray 的设计哲学是**把分布式变成函数调用**。三个核心 primitive：

### Remote Functions (Tasks)

```python
import ray

@ray.remote
def train_step(batch):
    # 这个函数会在集群中某个 worker 上执行
    return model.forward(batch)

# 异步调用，返回 ObjectRef（future）
futures = [train_step.remote(b) for b in batches]
results = ray.get(futures)  # 阻塞等待结果
```

### Remote Classes (Actors)

```python
@ray.remote
class PPOTrainer:
    def __init__(self, model_config):
        self.model = load_model(model_config)
    
    def train_step(self, batch):
        return self.model.update(batch)

# 创建 actor，有状态，生命周期跨多次调用
trainer = PPOTrainer.remote(config)
result = ray.get(trainer.train_step.remote(batch))
```

### Object Store (Plasma)

分布式共享内存，零拷贝读取。`ray.put()` 写入，`ray.get()` 读取。对于大 tensor 传输非常关键 — 避免序列化/反序列化开销。

## Ray 生态与 LLM

| 组件 | 用途 | 备注 |
|------|------|------|
| **Ray Core** | 分布式原语 | Task, Actor, Object Store |
| **Ray Train** | 分布式训练 | 封装 PyTorch DDP/FSDP/DeepSpeed |
| **Ray Serve** | 模型服务 | 支持动态 batching |
| **Ray Tune** | 超参搜索 | 和训练框架无缝集成 |
| **Ray Data** | 数据处理 | 流式 ETL，替代 Spark 在 ML 场景的角色 |

## 在 RLHF/RL 训练中的角色

Ray 在 LLM post-training 中的杀手级应用是**编排异构 worker**。一个 RLHF pipeline 需要：

```
Actor Model (生成) → Reward Model (打分) → Critic Model (估值) → Actor Update (训练)
```

这些模型可能分布在不同的 GPU 组上，Ray Actor 天然适合这种编排：

```python
@ray.remote(num_gpus=4)
class ActorWorker:
    """负责 rollout 生成"""
    pass

@ray.remote(num_gpus=2) 
class CriticWorker:
    """负责 value estimation"""
    pass

@ray.remote(num_gpus=1)
class RewardWorker:
    """负责 reward 计算"""
    pass
```

verl 就是基于 Ray Actor 来编排整个 RL 训练流程的，参见 [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]]。

## Placement Group：资源隔离

```python
from ray.util.placement_group import placement_group

# 确保 4 张卡在同一节点（TP 需要 NVLink）
pg = placement_group([{"GPU": 1}] * 4, strategy="STRICT_PACK")
ray.get(pg.ready())

# 在 placement group 内创建 actor
worker = TrainWorker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg
    )
).remote()
```

这个对大模型训练特别重要：TP 组必须在同一节点，PP 组可以跨节点，`STRICT_PACK` vs `SPREAD` 直接决定了通信效率。

## 实用经验

1. **Ray 的 overhead 不可忽视**：每次 remote call 有 ~100μs 的调度开销。热路径上不要频繁 `.remote()`
2. **Object Store 大小要配够**：默认是系统内存的 30%，训练大模型时经常不够，用 `--object-store-memory` 调
3. **GCS (Global Control Store) 是单点**：head node 挂了全挂。生产环境考虑 Ray HA（Redis-based GCS）
4. **和 NCCL 的关系**：Ray 管编排和数据搬运，GPU 间的 collective communication 还是走 NCCL。Ray 不替代 NCCL

## Ray vs 纯 torchrun

```
torchrun:  简单暴力, 所有进程跑同一份代码, SPMD 模式
Ray:       异构编排, 不同 actor 跑不同代码, MPMD 模式
```

纯预训练用 torchrun 就够了。一旦涉及 **多模型交互**（RLHF、MCTS、multi-agent），Ray 的优势就体现出来了。

## 相关

- [[AI/LLM/Infra/Megatron-LM|Megatron-LM]] — 模型并行框架
- [[AI/LLM/Infra/分布式训练|分布式训练]] — 分布式训练全景
- [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] — 基于 Ray 的混合编排
- [[AI/LLM/Frameworks/verl/训练后端|verl 训练后端]] — Ray 在 verl 中的应用
