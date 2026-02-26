---
title: "Ray"
brief: "UC Berkeley 的分布式计算框架，核心抽象是 Task/Actor/Object Store——把分布式变成函数调用；在 LLM post-training（RLHF）中的杀手级应用是编排异构 worker（Actor/Critic/Reward 分布在不同 GPU 组）。"
type: concept
domain: ai/llm/infra
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/llm/infra
  - type/concept
status: complete
sources:
  - "Ray: A Distributed Framework for Emerging AI Applications — arXiv:1712.05889"
  - "https://docs.ray.io/en/latest/"
  - "https://github.com/ray-project/ray"
related:
  - "[[AI/LLM/Infra/Megatron-LM|Megatron-LM]]"
  - "[[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]]"
  - "[[AI/LLM/Infra/模型并行策略|模型并行策略]]"
---
# Ray

> UC Berkeley 出品的分布式计算框架，从 RL 训练起家，现在是 LLM 训练/推理/serving 的基础设施之一。

文档：https://docs.ray.io/en/latest/ray-overview/index.html

## 核心抽象

Ray 的设计哲学是**把分布式变成函数调用**。三个核心 primitive：

> 来源：Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications" arXiv:1712.05889, Sec. 3

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

> 来源：verl 论文 "HybridFlow: A Flexible and Efficient RLHF Framework" arXiv:2409.19256 — Ray Actor 编排 RLHF 的工业实践

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

| 维度 | torchrun | Ray |
|------|----------|-----|
| 编程模型 | SPMD（所有进程跑同一份代码） | MPMD（不同 actor 跑不同代码） |
| 适用场景 | 纯预训练 / 单模型训练 | 多模型交互（RLHF、MCTS、multi-agent） |
| 调度开销 | 极低 | 每次 `.remote()` ~100μs |
| 异构 worker | 不支持 | 天然支持 |

纯预训练用 torchrun 就够了。一旦涉及 **多模型交互**（RLHF、MCTS、multi-agent），Ray 的优势就体现出来了。

## 📚 推荐阅读

### 原始论文
- [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/abs/1712.05889) — Ray 的架构设计与性能分析
- [Anyscale Blog: Scaling LLM Training with Ray](https://www.anyscale.com/blog) — 工业级 Ray 集群管理经验

### 深度解读
- [Ray 官方文档 — Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html) — Task/Actor/Object Store 完整教程 ⭐⭐⭐⭐
- [Ray Train 文档](https://docs.ray.io/en/latest/train/train.html) — 封装 PyTorch DDP/FSDP/DeepSpeed 的分布式训练

### 实践资源
- [ray-project/ray GitHub](https://github.com/ray-project/ray) — 官方仓库（30K+ stars）
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — 基于 Ray 做分布式推理的高性能框架

## 🔧 落地应用

### 直接可用场景
- **RLHF 异构编排**：Actor/Critic/Reward 模型分布在不同 GPU 组，Ray Actor 天然适配
- **分布式推理**：vLLM 基于 Ray 编排多 GPU 推理，实现动态 batching
- **超参搜索**：Ray Tune 支持 ASHA、PBT 等高效搜索算法

### 工程实现要点
- **Placement Group**：TP 组用 `STRICT_PACK`（同一节点 NVLink），PP 组可用 `SPREAD`
- **Object Store 大小**：默认系统内存 30%，训练大模型时用 `--object-store-memory` 扩大
- **GCS 单点问题**：生产环境启用 Ray HA（Redis-backed GCS），否则 head node 挂全挂
- **热路径避免 `.remote()`**：每次调度 ~100μs 开销，在 inner loop 中会成为瓶颈

### 面试高频问法
- Q: Ray 和 NCCL 是什么关系？
  A: Ray 管编排和数据搬运（调度 Task/Actor、Object Store 传输），GPU 间的 collective communication（AllReduce、AllGather）走 NCCL。Ray 不替代 NCCL，两者互补。

## 💡 启发与思考

### So What？对老板意味着什么
- Ray 是 RLHF/RL 训练的"胶水层"——理解 Actor 模型和 Placement Group 是使用 verl/OpenRLHF 的前提
- Ray 的价值不在"快"，而在"灵活"——能编排复杂的多模型工作流（这是 torchrun 做不到的）

### 未解问题与局限
- Ray 的调度开销在超大规模（10K+ worker）下是否会成为瓶颈？
- Ray 对故障恢复的支持（checkpoint + actor restart）在长时间 RL 训练中的可靠性待验证

### 脑暴：如果往下延伸
- 如果把 [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] 的 SPMD + MPMD 混合思路推广，能否构建一个"通用 RL 编排标准"？
- Ray Serve + vLLM 的组合能否成为 LLM Serving 的事实标准？

## 相关

> 🔗 See also: [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] — 基于 Ray Actor 的 RLHF 混合编排
> 🔗 See also: [[AI/LLM/Infra/Megatron-LM|Megatron-LM]] — Ray 编排 + Megatron 做模型并行是 verl 的核心架构
> 🔗 See also: [[AI/LLM/Infra/模型并行策略|模型并行策略]] — DP/TP/PP 如何与 Ray 配合

- [[AI/LLM/Infra/分布式训练|分布式训练]] — 分布式训练全景
- [[AI/LLM/Frameworks/verl/训练后端|verl 训练后端]] — Ray 在 verl 中的应用
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]]
