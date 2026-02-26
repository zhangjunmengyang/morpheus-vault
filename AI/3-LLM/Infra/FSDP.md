---
title: FSDP (Fully Sharded Data Parallel)
brief: PyTorch 原生的分布式训练方案，本质是 ZeRO-3 的官方实现——将参数/梯度/优化器状态全部分片到 N 张 GPU，通信量增加 50% 换取 N 倍显存节省；FSDP2 进一步支持 torch.compile 和 FP8。
type: concept
domain: ai/llm/infra
created: 2026-02-13
updated: 2026-02-22
tags:
  - ai/llm/infra
  - type/concept
  - interview/hot
status: complete
sources:
  - "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel — arXiv:2304.11277"
  - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — arXiv:1910.02054"
  - https://pytorch.org/docs/stable/fsdp.html
related:
  - "[[DeepSpeed|DeepSpeed]]"
  - "[[Megatron-LM|Megatron-LM]]"
  - "[[模型并行策略|模型并行策略]]"
---
# FSDP

## 概述

FSDP（Fully Sharded Data Parallel）是 PyTorch 原生的分布式训练方案，核心思想是**将模型参数、梯度和优化器状态在所有 GPU 之间分片（shard）**，每个 GPU 只持有完整模型的 1/N。

FSDP 本质上是 Microsoft ZeRO（Zero Redundancy Optimizer）的 PyTorch 原生实现，对标 [[DeepSpeed]] ZeRO Stage 3。但因为是 PyTorch 官方维护，与 PyTorch 生态（torch.compile、FSDP2 等）的集成更好。

> 来源：Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" arXiv:2304.11277

## 为什么需要 FSDP

### DDP 的瓶颈

标准的 DDP（DistributedDataParallel）每个 GPU 持有**完整**的模型副本：

```
GPU 0: [完整模型参数] + [完整梯度] + [完整优化器状态]
GPU 1: [完整模型参数] + [完整梯度] + [完整优化器状态]
GPU 2: [完整模型参数] + [完整梯度] + [完整优化器状态]
GPU 3: [完整模型参数] + [完整梯度] + [完整优化器状态]
```

对于一个 7B 参数的模型（FP32）：
- 参数：7B × 4 bytes = 28 GB
- 梯度：28 GB
- 优化器状态（Adam）：28 × 2 = 56 GB
- 总计：**112 GB/GPU** → 单卡 A100 80G 根本放不下

### FSDP 的方案

```
GPU 0: [参数分片 1/4] + [梯度分片 1/4] + [优化器分片 1/4]
GPU 1: [参数分片 2/4] + [梯度分片 2/4] + [优化器分片 2/4]
GPU 2: [参数分片 3/4] + [梯度分片 3/4] + [优化器分片 3/4]
GPU 3: [参数分片 4/4] + [梯度分片 4/4] + [优化器分片 4/4]
```

每个 GPU 显存占用：112 / 4 = **28 GB** ✓

## 工作原理

### 前向传播

```
对于每个 FSDP 包装的模块:
    1. all-gather: 从所有 GPU 收集完整参数  ← 通信开销
    2. 用完整参数做前向计算
    3. 丢弃其他 GPU 的参数分片          ← 释放显存
```

### 反向传播

```
对于每个 FSDP 包装的模块（逆序）:
    1. all-gather: 收集完整参数
    2. 计算梯度
    3. reduce-scatter: 每个 GPU 只保留自己那份梯度
    4. 丢弃完整参数
```

### 通信模式

FSDP 的两个核心通信原语：
- **All-Gather**：每个 GPU 广播自己的分片，所有 GPU 得到完整数据
- **Reduce-Scatter**：先 reduce（求和）再 scatter（分片）

通信量分析（相比 DDP）：
- DDP：一次 All-Reduce = $2 \times \text{model\_size}$
- FSDP：每层 All-Gather（前向 + 反向各一次）+ Reduce-Scatter = 约 $3 \times \text{model\_size}$
- FSDP 通信量增加约 50%，但**换来了 N 倍的显存节省**

> 来源：arXiv:2304.11277, Sec. 3 — 通信量 vs 显存的 trade-off 分析；ZeRO arXiv:1910.02054, Sec. 3.2

## PyTorch FSDP 使用

### 基本用法

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = MyLargeModel()

# 用 FSDP 包装
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
    device_id=torch.cuda.current_device(),
)
```

### Sharding Strategy

```python
# FULL_SHARD: 参数+梯度+优化器全部分片（ZeRO-3）
# SHARD_GRAD_OP: 只分片梯度和优化器（ZeRO-2）
# NO_SHARD: 不分片，退化为 DDP（ZeRO-0）
# HYBRID_SHARD: 节点内 FULL_SHARD，节点间复制（减少跨节点通信）
```

`HYBRID_SHARD` 在多节点训练时特别有用——节点内 NVLink 带宽高（600 GB/s），分片开销小；节点间网络带宽低（100-400 Gbps），减少跨节点通信。

> 来源：arXiv:2304.11277, Sec. 4.2 — HYBRID_SHARD 在多节点场景的性能对比

### Auto Wrap Policy

FSDP 需要指定哪些模块作为分片单元：

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 按 Transformer layer 粒度分片（最常用）
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
```

## FSDP2（PyTorch 2.x）

PyTorch 正在推进 FSDP2，对 FSDP1 的主要改进：

1. **per-parameter sharding**：更细粒度的分片，不再要求 flat parameter
2. **与 torch.compile 兼容**：FSDP1 和 compile 配合有各种问题
3. **更灵活的混合精度**：支持 FP8 训练
4. **DTensor 支持**：基于 PyTorch 的 DTensor 抽象，统一多种并行策略

```python
# FSDP2 API（更简洁）
from torch.distributed._composable.fsdp import fully_shard

for layer in model.layers:
    fully_shard(layer)  # 对每一层做分片
fully_shard(model)       # 对整个模型做分片
```

## 与 HuggingFace Accelerate 的集成

实际项目中很少直接用 FSDP API，通常通过 Accelerate 封装：

```yaml
# accelerate config (fsdp_config.yaml)
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_offload_params: false
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_state_dict_type: SHARDED_STATE_DICT
mixed_precision: bf16
```

```bash
accelerate launch --config_file fsdp_config.yaml train.py
```

## FSDP vs DeepSpeed

| 维度 | FSDP | DeepSpeed |
|------|------|-----------|
| 维护 | PyTorch 官方 | Microsoft |
| ZeRO Stage | 2, 3 | 1, 2, 3 |
| CPU Offload | 支持 | 支持（更成熟） |
| NVMe Offload | 不支持 | 支持 |
| torch.compile | FSDP2 支持 | 兼容性较差 |
| 配置复杂度 | 中等 | 较高 |
| 社区 | PyTorch 原生生态 | HuggingFace 深度集成 |

**选择建议**：
- PyTorch 原生生态、需要 torch.compile → FSDP
- 需要 CPU/NVMe offload、更多调优选项 → DeepSpeed
- 两者都行时 → 看团队熟悉度

## 📚 推荐阅读

### 原始论文
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) — Meta 团队的工业级经验总结
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — FSDP 背后的理论基础

### 深度解读
- [PyTorch FSDP 官方教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — 最权威的入门指南 ⭐⭐⭐⭐
- [FSDP vs DeepSpeed 详细对比（知乎）](https://zhuanlan.zhihu.com/p/667455728) — 中文社区实测对比

### 实践资源
- [PyTorch FSDP2 RFC](https://github.com/pytorch/pytorch/issues/114299) — FSDP2 设计讨论和进展
- [HuggingFace Accelerate FSDP 文档](https://huggingface.co/docs/accelerate/usage_guides/fsdp) — 实际项目首选集成方式

## 🔧 落地应用

### 直接可用场景
- **7B-70B 模型微调/SFT**：FSDP FULL_SHARD + Accelerate 是最主流的方案
- **verl RL 训练**：[[verl 概述|verl]] 同时支持 FSDP 和 Megatron 后端，FSDP 更易调试
- **需要 torch.compile 加速**：FSDP2 是目前唯一与 compile 良好兼容的分布式方案

### 工程实现要点
- **auto_wrap_policy**：按 `TransformerDecoderLayer` 粒度分片最常用，粒度太细反而增加通信
- **HYBRID_SHARD**：多节点必开——节点内 NVLink 分片，跨节点复制，性价比最优
- **Checkpoint 保存**：用 `SHARDED_STATE_DICT` 避免 OOM（比 `FULL_STATE_DICT` 安全得多）
- **常见坑**：FSDP 包装后不能直接 `model.module` 访问原始模型；用 `FSDP.summon_full_params()` 临时聚合

### 面试高频问法
- Q: FSDP 和 DeepSpeed ZeRO-3 本质区别是什么？
  A: 算法层面等价（都是参数+梯度+优化器全分片）。差异在工程实现：FSDP 是 PyTorch 原生（torch.compile 兼容、DTensor 支持）；DeepSpeed 更成熟（CPU/NVMe offload、更多调优旋钮）。选择取决于生态：需要 compile → FSDP，需要 offload → DeepSpeed。

## 💡 启发与思考

### So What？对老板意味着什么
- FSDP 是"PyTorch 亲儿子"——长期来看会成为默认分布式训练方案，投入学习的 ROI 最高
- FSDP2 + DTensor + torch.compile 的组合可能统一 DP/TP/PP 的编程模型，是 2026 年最值得关注的基础设施演进

### 未解问题与局限
- FSDP1 不支持 NVMe Offload，极端资源受限场景仍需 DeepSpeed
- FSDP2 尚未完全稳定（截至 PyTorch 2.5），生产环境需谨慎

### 脑暴：如果往下延伸
- FSDP2 的 per-parameter sharding + [[Megatron-LM|Megatron-Core]] 的 TP 能否在同一模型上自由组合？
- 如果 PyTorch 原生支持 ZeRO-1/2（不仅仅是 3），DeepSpeed 的存在意义是否会被侵蚀？

## 相关

> 🔗 See also: [[DeepSpeed]] — FSDP 的主要竞品，ZeRO 系列的原始实现
> 🔗 See also: [[Megatron-LM]] — TP/PP 并行方案，与 FSDP 的 DP 方向互补
> 🔗 See also: [[模型并行策略|模型并行策略]] — 从 DP 到 5D 并行的全景选型

- [[分布式训练]] — 分布式训练概览
- [[Ray]] — 分布式计算框架
- [[verl 概述|verl 概述]] — verl 对 FSDP 的集成
- [[TRL 概述|TRL 概述]]
- [[OpenRLHF|OpenRLHF]]
