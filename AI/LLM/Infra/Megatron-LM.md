---
title: "Megatron-LM"
type: concept
domain: ai/llm/infra
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/infra
  - type/concept
---
# Megatron-LM

> NVIDIA 出品的大规模语言模型训练框架，核心卖点是**高效的模型并行**。

官方仓库：https://github.com/NVIDIA/Megatron-LM

## 为什么需要 Megatron-LM

单卡放不下大模型，这是最朴素的动机。PyTorch 原生的 `DistributedDataParallel` 只解决了数据并行，模型本身得塞进一张卡。当参数量到 7B 以上，fp16 下光权重就要 14GB，加上 optimizer states、activations，一张 80GB A100 都捉襟见肘。

Megatron-LM 的核心贡献：**把 Tensor Parallelism 做到了极致**，并且和 Pipeline Parallelism、Data Parallelism 无缝组合。

## 三维并行 (3D Parallelism)

```
┌─────────────────────────────────────────────┐
│              Data Parallelism (DP)           │
│  ┌──────────────┐  ┌──────────────┐         │
│  │   Replica 0  │  │   Replica 1  │         │
│  │ ┌──────────┐ │  │ ┌──────────┐ │         │
│  │ │  Stage 0 │ │  │ │  Stage 0 │ │  ← PP   │
│  │ │ (TP=2)   │ │  │ │ (TP=2)   │ │         │
│  │ ├──────────┤ │  │ ├──────────┤ │         │
│  │ │  Stage 1 │ │  │ │  Stage 1 │ │         │
│  │ │ (TP=2)   │ │  │ │ (TP=2)   │ │         │
│  │ └──────────┘ │  │ └──────────┘ │         │
│  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────┘
```

### Tensor Parallelism (TP)

将单个 Transformer 层内的矩阵运算切分到多卡。以 MLP 为例：

```python
# 原始: Y = GeLU(XA) * B
# TP 切分: A 按列切, B 按行切
# GPU 0: Y_0 = GeLU(X @ A_0) @ B_0
# GPU 1: Y_1 = GeLU(X @ A_1) @ B_1
# AllReduce: Y = Y_0 + Y_1
```

关键设计：每个 TP 组内只需要 **2 次 AllReduce**（forward 一次，backward 一次），通信量和单层参数量成正比。TP 适合放在 **NVLink 连接的同机卡间**，因为通信密集。

### Pipeline Parallelism (PP)

将模型按层分成多个 stage，不同 stage 放在不同节点。朴素的 PP 会有严重的 bubble（空闲时间），Megatron 用了 **1F1B schedule**（interleaved）来减少 bubble：

```
# 4 micro-batches, 2 stages
# 朴素:  [F0 F1 F2 F3] [B3 B2 B1 B0]  -- 大量 bubble
# 1F1B:  [F0 F1] [F2 B0] [F3 B1] [B2 B3]  -- bubble 显著减少
```

PP 适合放在 **跨节点** 场景，因为只需要在 stage 边界传递 activations，通信量相对小。

### Data Parallelism (DP)

最外层包一圈 DP。Megatron 同时支持传统 DP 和 ZeRO-style 的分布式优化器（类似 DeepSpeed ZeRO-1）。

## 核心组件

| 组件 | 功能 |
|------|------|
| `megatron/core/transformer` | Transformer 层实现，内建 TP 支持 |
| `megatron/core/pipeline_parallel` | PP schedule（1F1B, interleaved） |
| `megatron/core/distributed` | 分布式通信、梯度 AllReduce |
| `megatron/core/optimizer` | 分布式优化器 |
| `megatron/training` | 训练主循环、checkpoint 管理 |

## Megatron-Core vs Megatron-LM

从 v0.5 开始，NVIDIA 把核心并行逻辑抽成了 **Megatron-Core**（`megatron/core/`），可以独立安装使用。上层的 Megatron-LM 是基于它的完整训练方案。很多框架（NeMo、verl）直接依赖 Megatron-Core。

## 与 DeepSpeed 的对比

```
Megatron-LM:  TP 做得好, PP 调度优, 但上手门槛高
DeepSpeed:    ZeRO 系列省显存, API 友好, 但 TP 要额外配置
实践中:       两者经常混用 (Megatron-DeepSpeed)
```

我的观点：**训练 70B+ 模型，Megatron 的 TP + PP 基本是标配**。DeepSpeed ZeRO-3 虽然也能跑，但通信效率在大规模集群上不如 Megatron 的手动并行。小模型（< 13B）直接用 DeepSpeed ZeRO-2/3 更省心。

## 踩坑记录

1. **TP size 必须整除 attention heads 数量**：比如 32 heads 只能用 TP=1/2/4/8/16/32
2. **PP 的 num_layers 必须能被 PP size 整除**：否则直接报错
3. **混合精度**：Megatron 默认用自己的 `Float16Module`，和 PyTorch AMP 不是一回事
4. **Checkpoint 格式**：Megatron 的 checkpoint 和 HuggingFace 格式不通用，需要转换脚本

## 相关

- [[AI/LLM/Infra/分布式训练|分布式训练]] — 并行策略全景
- [[AI/LLM/Infra/Ray|Ray]] — 另一个分布式计算框架
- [[AI/LLM/Frameworks/verl/训练后端|verl 训练后端]] — verl 中使用 Megatron 后端
- [[AI/LLM/Frameworks/verl/HybridFlow|HybridFlow]] — verl 的混合并行编排
