---
title: P1：xtrain——分布式预训练基础设施从零实现
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, pre-training, distributed-training, TP, PP, ZeRO]
brief: 从零手撕分布式训练基础设施，覆盖通信原语、数据并行、ZeRO、张量并行、流水线并行、MoE专家并行全栈，作为预训练工程能力的技术背书项目。
related:
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操]]"
  - "[[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Infra/Pipeline-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Architecture/DeepSeek-V3-手撕实操]]"
---

# P1：xtrain——分布式预训练基础设施从零实现

> **一句话定位**：系统性地从零实现了现代大模型预训练的全套分布式技术栈，不是调 API，是真正理解每一层并能手写出来。

---

## 背景故事（面试口径）

> 怎么引入：

"在学习后训练的时候，发现一个问题：大家都在讨论 GRPO 怎么改、reward 怎么设计，但当你真正想跑一个实验——比如复现 DeepSeek-R1——你会发现卡在了分布式训练上：为什么显存不够？TP 的 AllReduce 怎么工作？ZeRO-3 为什么能把显存压到这么低？这些问题不搞清楚，工程上就会一直踩坑。

所以我花了一段时间，系统地把这些东西从零实现了一遍——xtrain 这个项目。不是为了写个教程，是为了真正理解。"

---

## 项目内容（技术全栈）

### 阶段一：通信原语（xtrain-01）

**做了什么**：从 socket 开始，手写 AllReduce 的几种实现：
- Ring AllReduce（Bandwidth-optimal）
- Tree AllReduce（Latency-optimal）
- 理解什么时候用哪种，以及 NCCL 底层在做什么

**面试展开点**：
- AllReduce 的 Ring 实现：为什么时间复杂度是 2(N-1)/N × data_size？
- 同步 vs 异步通信的 trade-off（梯度 staleness）
- NCCL 为什么能比手写实现快——RDMA、NVLink 的利用

### 阶段二：数据并行 + ZeRO（xtrain-02/03）

**做了什么**：实现了三个版本的数据并行：
- 朴素 DDP（每个 GPU 保存完整参数）
- ZeRO-1（分片 optimizer states）
- ZeRO-2（+ 分片 gradients）
- ZeRO-3（+ 分片 parameters）

**核心理解**：ZeRO 不是减少计算，是减少冗余存储。用 AllGather 换显存。

**面试展开点**：

显存分析——一个 7B 模型在不同 ZeRO 级别下的显存占用：

```
全精度（fp32）参数占用：7B × 4 bytes = 28 GB

训练时显存组成（AdamW）：
  参数：28 GB
  梯度：28 GB
  Optimizer States（m + v + master weights）：28 × 3 = 84 GB
  总计：~140 GB（单卡完全不可能）

ZeRO-1（N=8 GPU）：
  参数/梯度不分片：56 GB
  Optimizer States 分片：84/8 = 10.5 GB
  总计：~66 GB（A100 80GB 勉强能放一个 7B）

ZeRO-3（N=8 GPU）：
  全部分片：140/8 = 17.5 GB
  + 通信 buffer 额外开销
  代价：每个 forward/backward 需要 AllGather/ReduceScatter
```

**为什么不一直用 ZeRO-3**：通信开销，当节点间带宽低时（跨机器）ZeRO-3 可能比 ZeRO-1 慢。

### 阶段三：张量并行（xtrain-04）

**做了什么**：手实现 Megatron-LM 风格的张量并行：
- Column Parallel Linear（前向 AllGather，反向 AllReduce）
- Row Parallel Linear（前向 AllReduce，反向 AllGather）
- Attention Head 并行（Q/K/V 分割到不同 GPU）

**核心认知**：TP 的通信在每个 transformer layer 里都有，适合节点内（NVLink 速度）不适合跨节点。

**面试展开点**：

```
为什么 Column → Row 的组合只需要两次通信？

Column Parallel：
  输入 X 复制到每个 GPU
  每个 GPU 计算 Y_i = X × W_i（W 按列分割）
  输出 Y_i 在各 GPU，不需要通信（下一层是 Row）

Row Parallel：
  输入 Y_i 已经在各 GPU（上层输出）
  每个 GPU 计算 Z_i = Y_i × W_i（W 按行分割）
  需要 AllReduce：Z = sum(Z_i)

一个 MLP block 只需要 2 次 AllReduce，而不是 O(L) 次
```

### 阶段四：流水线并行（xtrain-05）

**做了什么**：实现了 GPipe 和 1F1B（One-Forward-One-Backward）调度：
- 理解 bubble ratio 的计算和优化
- Micro-batch 的设计

**面试展开点**：

```
Bubble Ratio（GPipe）= (p-1) / (m + p - 1)
  p = pipeline stages
  m = micro-batches

当 m >> p 时，bubble 趋近于 0
所以大模型训练 micro-batch 数量都很大（32、64 等）

1F1B 为什么更好：同一时刻，GPU 不全部 forward 或全部 backward
  → 显存峰值更低（不用同时存所有 micro-batch 的 activation）
  → Bubble 比例与 GPipe 相同，但显存从 O(p×m) 降到 O(p)
```

### 阶段五：MoE 专家并行（xtrain-07）

**做了什么**：实现了 MoE 的 Expert Parallel：
- 每个 GPU 放一部分 Expert
- Token routing（Top-K gating）的分发逻辑
- All-to-All 通信（每个 GPU 的 token 发给对应的 expert GPU）

**面试展开点**：
- MoE 的 All-to-All 为什么比 AllReduce 难优化（点对点通信，负载不均匀问题）
- Expert Load Balancing：auxiliary loss 为什么必要，不加会发生 expert collapse
- DeepSeek MoE 的 Fine-Grained Expert + Shared Expert 设计思路

---

## 一句话总结（面试结尾）

"xtrain 让我真正理解了大模型训练的工程本质——不是调参，是理解每一层显存和通信的 trade-off，然后根据硬件拓扑（NVLink 带宽、节点间带宽）做出正确的并行策略组合。这个认知直接帮助我在后训练 verl 实验中少踩了很多坑。"

---

## See Also

- [[Projects/项目故事/P2-后训练大项目-MA-RLHF工程实战]] — 预训练基础支撑后训练
- [[AI/3-LLM/Infra/ZeRO-手撕实操]]
- [[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]]
- [[AI/3-LLM/Infra/Pipeline-Parallel-手撕实操]]
- [[AI/3-LLM/Infra/分布式训练通信原语-手撕实操]]
- [[AI/3-LLM/Infra/MoE-Context-Parallel-手撕实操]]
- [[AI/3-LLM/Architecture/DeepSeek-V3-手撕实操]]
