---
title: "DeepSpeed"
type: concept
domain: ai/llm/infra
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/infra
  - type/concept
---
# DeepSpeed

## 概述

DeepSpeed 是 Microsoft 开源的深度学习优化库，核心是 **ZeRO（Zero Redundancy Optimizer）** 系列技术。它解决的问题很直接：**怎么用有限的 GPU 训练更大的模型**。

ZeRO 的思路是消除数据并行中的冗余——标准 DDP 在每个 GPU 上都存一份完整的参数、梯度、优化器状态，这在大模型时代是巨大的浪费。

## ZeRO 三个阶段

ZeRO 的核心是分阶段消除冗余，每个阶段分片更多的内容：

### 显存占用分析（以 7B FP16 模型 + Adam 为例，4 GPU）

每个 GPU 上的显存（不含激活值）：

| 组件 | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|-----|--------|--------|--------|
| 参数 (FP16) | 14 GB | 14 GB | 14 GB | **3.5 GB** |
| 梯度 (FP16) | 14 GB | 14 GB | **3.5 GB** | **3.5 GB** |
| 优化器状态 (FP32) | 56 GB | **14 GB** | **14 GB** | **14 GB** |
| **总计** | **84 GB** | **42 GB** | **31.5 GB** | **21 GB** |

### ZeRO Stage 1: 优化器状态分片

只分片 optimizer states（Adam 的 momentum 和 variance）。通信量与 DDP 相同（一次 All-Reduce），但显存减半。

### ZeRO Stage 2: + 梯度分片

在 Stage 1 基础上分片梯度。反向传播时用 Reduce-Scatter 替代 All-Reduce，每个 GPU 只保留自己需要的梯度分片。通信量不变，但显存进一步减少。

### ZeRO Stage 3: + 参数分片

所有东西都分片。前向和反向传播时需要 All-Gather 收集完整参数，计算完再丢弃。通信量增加 50%，但显存降到极致。

**等价于 FSDP 的 FULL_SHARD。**

## 配置方式

DeepSpeed 通过 JSON 配置文件控制：

### ZeRO Stage 2（最常用的平衡点）

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### ZeRO Stage 3 + Offload

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

## ZeRO-Offload 和 ZeRO-Infinity

### ZeRO-Offload

把优化器状态和梯度卸载到 CPU 内存：
- GPU 做前向和反向计算
- CPU 做优化器更新（Adam step）
- 通过 PCIe 传输数据

代价：训练速度降低 20-40%（取决于 PCIe 带宽），但能在单卡上训练更大的模型。

### ZeRO-Infinity

在 Offload 基础上进一步卸载到 **NVMe SSD**：
- CPU 内存有限？用 SSD 扩展
- 理论上可以在单 GPU 上训练任意大的模型（只要 SSD 够大）
- 速度会更慢，但对于极端资源限制下的训练是唯一选择

## DeepSpeed 的其他功能

### 1. Mixture of Experts (MoE) 支持

DeepSpeed 提供了 MoE 训练的基础设施：
- Expert Parallelism：不同 GPU 上放不同的 expert
- 与 ZeRO 结合使用

### 2. 通信优化

- **Gradient Compression**：梯度量化后再通信
- **1-bit Adam**：用 1-bit 量化的梯度做通信，极大减少带宽需求

### 3. Activation Checkpointing

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "cpu_checkpointing": true
    }
}
```

用时间换空间：不保存中间激活值，反向传播时重算。显存节省显著，计算量增加约 33%。

### 4. DeepSpeed-Chat

针对 RLHF 训练的完整 pipeline（SFT → Reward Modeling → PPO），但社区使用率不如 TRL/verl。

## 与 HuggingFace 的集成

通过 Accelerate 或直接在 Trainer 中使用：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # 指定配置文件
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)
```

```bash
# 启动
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
# 或
accelerate launch --config_file accelerate_ds.yaml train.py
```

## 选择建议

```
单卡训练 → 不需要 DeepSpeed
多卡、模型 < 13B → ZeRO Stage 2（性价比最高）
多卡、模型 13B-70B → ZeRO Stage 3
显存极度紧张 → ZeRO Stage 3 + CPU Offload
只有一两张卡训大模型 → ZeRO Stage 3 + CPU/NVMe Offload
```

## 相关

- [[FSDP]] — PyTorch 原生的竞品方案
- [[分布式训练]] — 分布式训练概览
- [[Megatron-LM]] — Tensor/Pipeline Parallel
- [[Ray]] — 分布式计算框架
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL 概述]] — 集成 DeepSpeed 的训练框架
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]] — verl 对分布式训练的支持
