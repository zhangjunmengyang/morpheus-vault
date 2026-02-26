---
title: SSM/Mamba：线性复杂度的序列建模新范式
brief: Mamba 通过 input-dependent selection mechanism 解决了传统 SSM 无法选择性聚焦信息的问题，实现 O(L) 线性复杂度替代 Transformer 的 O(L²)。核心创新：B、C、Δ 参数依赖输入动态生成 + selective scan 硬件高效实现。在长序列（>16K）场景显著优于 Transformer，但复杂推理和 ICL 能力仍有差距，实践中 Jamba 等混合架构是更务实的选择。
type: concept
domain: ai/llm/architecture
created: 2026-02-14
updated: 2026-02-22
tags:
  - ai/llm/architecture
  - ai/llm/ssm
  - type/concept
status: complete
sources:
  - "Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* arXiv:2312.00752"
  - Gu et al. *Efficiently Modeling Long Sequences with Structured State Spaces (S4)* arXiv:2111.00396
  - "Dao & Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* arXiv:2405.21060 (Mamba-2)"
  - "Lieber et al. *Jamba: A Hybrid Transformer-Mamba Language Model* arXiv:2403.19887"
related:
  - "[[Transformer 通识|Transformer 通识]]"
  - "[[AI/3-LLM/Architecture/Attention 变体综述|Attention 详解]]"
  - "[[AI/3-LLM/Architecture/长上下文处理|长上下文处理]]"
  - "[[AI/3-LLM/Architecture/架构范式对比|架构范式对比]]"
---

# SSM/Mamba 架构

## 概述

State Space Model (SSM) 是一种新兴的神经网络架构，特别是 Mamba 的提出，为解决 Transformer 在长序列建模上的二次复杂度问题提供了一个线性复杂度的替代方案。Mamba 通过选择性状态空间机制，在保持线性计算复杂度的同时，实现了与 Transformer 相当甚至更好的性能。

> 来源：Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* arXiv:2312.00752

## State Space Model 基础

### 从 S4 到 Mamba 的演进

Structured State Space (S4)（arXiv:2111.00396）是现代 SSM 的起点，其核心思想是将序列建模问题转化为连续状态空间方程的离散化：

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t \qquad y_t = C \cdot h_t + D \cdot x_t$$

其中 $\bar{A} = \exp(\Delta A)$，$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$，$\Delta$ 是离散化步长。

S4 的关键创新在于：
1. **HiPPO 初始化**：使用 HiPPO (High-order Polynomial Projection Operator) 理论初始化矩阵 A，确保对历史信息的有效记忆
2. **结构化参数化**：通过对角化和低秩分解，使参数数量从 N² 降至 O(N)
3. **卷积视图**：将递归计算转化为卷积操作，支持并行训练

### Mamba 的核心创新：选择性状态空间

Mamba 的突破性创新是引入了 **input-dependent selection mechanism**，解决了传统 SSM 无法选择性聚焦重要信息的问题：

1. **选择性参数**：$B$、$C$、$\Delta$ 参数不再是固定的，而是依赖于输入 $x$：

$$B_t = W_B \cdot x_t, \quad C_t = W_C \cdot x_t, \quad \Delta_t = \text{softplus}(W_\Delta \cdot x_t + b_\Delta)$$

2. **门控机制**：引入类似 LSTM 的门控，但更高效：

$$y = \text{SSM}(x) \odot \text{SiLU}(W_z \cdot x)$$

3. **硬件高效实现**：通过 selective scan 算法，避免在 GPU 内存中物化大的状态矩阵（$O(BLN)$ → 分块扫描减少 HBM 访问）

> 来源：arXiv:2312.00752, Sec. 3.1 — Selection Mechanism

## 与 Transformer 的对比

| 维度 | Transformer | Mamba |
|------|------------|-------|
| **计算复杂度** | O(L²) 二次 | O(L) 线性 |
| **内存使用** | O(L²) | O(L) |
| **长序列性能** | 受二次复杂度限制 | 线性扩展 |
| **并行训练** | 高度并行 | 需要 selective scan |
| **推理效率** | KV Cache 占用大 | 固定状态大小 |
| **表达能力** | 全局注意力 | 选择性状态传播 |

### 实际性能对比

在 LLM Evaluation 基准测试中：
- **短序列** (< 2K)：Transformer 略优
- **中等序列** (2K-16K)：性能相当
- **长序列** (> 16K)：Mamba 显著优势，特别是在文档理解和代码生成任务

## Mamba-2 和 Jamba

### Mamba-2 改进

Mamba-2（arXiv:2405.21060）进一步优化了架构：
1. **State Space Duality (SSD)**：统一了状态空间和注意力的数学形式
2. **更高效的硬件实现**：减少 I/O 操作，提升实际推理速度
3. **更好的数值稳定性**：改进初始化和归一化策略

### Jamba：混合架构

Jamba（arXiv:2403.19887）是 AI21 Labs 提出的 Mamba + Attention 混合架构，周期性地交替使用 Mamba 层和 Attention 层：

优势：
- **兼顾长短序列**：Mamba 处理长依赖，Attention 处理复杂推理
- **渐进式采用**：可以从 Transformer 平滑迁移
- **更好的 [[Prompt-Engineering-2026实战全景#1.1 In-Context Learning（ICL）：为什么 Prompt 能工作|In-Context Learning]]**：Attention 层保证少样本学习能力

## 实际效果与局限性

### 优势

1. **推理效率**：固定大小的状态 (几百KB) vs Transformer 的线性增长 KV Cache
2. **长序列建模**：100K+ token 序列处理无压力
3. **内存友好**：特别适合资源受限的部署环境

### 局限性

1. **训练并行度**：selective scan 限制了并行训练效率
2. **复杂推理任务**：在需要全局信息比较的任务上仍不如 Transformer
3. **生态系统**：相比 Transformer，工具链和优化技术还不够成熟
4. **理论理解**：选择性机制的理论基础仍在探索中

### 实际部署考虑

- **混合使用**：很多实际应用采用 Mamba + Transformer 混合架构
- **任务特性**：文档处理、代码生成等长序列任务优先考虑 Mamba
- **硬件要求**：需要支持 selective scan 的推理框架

## 面试常见问题

### Q1: Mamba 相比 Transformer 的核心优势是什么？

**答案**：
1. **线性复杂度**：计算和内存复杂度都是 O(L) vs Transformer 的 O(L²)
2. **推理效率**：固定大小状态 vs 线性增长的 KV Cache
3. **长序列处理**：可以处理 100K+ token 的序列而不会有性能急剧下降
4. **选择性建模**：通过 input-dependent selection 机制，可以选择性地聚焦重要信息

### Q2: 什么是 selective scan，为什么它对 Mamba 很重要？

**答案**：
Selective scan 是 Mamba 的核心算法创新，解决了状态空间模型的并行化问题。传统 SSM 的递归性质使其难以并行训练，而 selective scan 通过以下方式解决：

1. **避免物化大矩阵**：不在 GPU 内存中存储完整的状态转移矩阵
2. **分块并行**：将长序列分成块，在块内递归，块间并行
3. **硬件友好**：针对 GPU 内存层次结构优化，减少 I/O 瓶颈

### Q3: Mamba 在哪些场景下可能不如 Transformer？

**答案**：
1. **复杂推理任务**：需要全局信息比较和多步推理的任务，如复杂的逻辑推理
2. **短序列任务**：序列长度 < 2K 时，Transformer 的性能通常更好
3. **Few-shot learning**：[[Prompt-Engineering-2026实战全景#1.1 In-Context Learning（ICL）：为什么 Prompt 能工作|In-Context Learning]] 能力可能不如 Transformer
4. **需要全局注意力的任务**：如文档问答中的跨段落信息整合

### Q4: 如何理解 Jamba 这种混合架构的设计思路？

**答案**：
Jamba 的设计基于"取长补短"的思路：

1. **Mamba 层**：负责长序列建模和高效的局部信息传播
2. **Attention 层**：周期性插入，负责全局信息整合和复杂推理
3. **渐进式采用**：允许从现有 Transformer 模型平滑过渡
4. **任务适应性**：根据任务特点调整混合比例

这种设计在保持 Mamba 线性复杂度优势的同时，保证了模型的表达能力。

### Q5: 从工程角度，部署 Mamba 模型需要考虑哪些因素？

**答案**：
1. **推理框架支持**：需要支持 selective scan 算子的框架，如 [[AI/3-LLM/Inference/TensorRT-LLM|TensorRT-LLM]]、vLLM 等
2. **内存规划**：虽然推理时内存占用更少，但训练时的选择性计算可能需要更多临时存储
3. **硬件兼容性**：某些优化需要特定 GPU 架构支持（如 H100 的 FP8）
4. **混合部署**：考虑与 Transformer 模型的混合使用策略
5. **性能调优**：根据序列长度分布选择合适的批处理策略

## 📚 推荐阅读

### 原始论文
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Mamba 原文，选择性 SSM 的核心论文
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) — SSM 系列的奠基之作，HiPPO + 结构化参数化
- [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) — State Space Duality，统一 SSM 和 Attention 的数学形式
- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) — 混合架构的实践验证

### 深度解读
- [The Annotated S4](https://srush.github.io/annotated-s4/) — Sasha Rush 的 S4 逐行注释实现 ⭐⭐⭐⭐⭐
- [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html) — 同作者的 Mamba 深度解读

### 实践资源
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Mamba 官方实现
- [mamba-ssm PyPI](https://pypi.org/project/mamba-ssm/) — 官方 Python 包，包含 selective scan CUDA kernel

## 🔧 落地应用

### 直接可用场景
- **长文档处理**：100K+ token 序列处理无压力，适合法律文档、代码库分析等长序列场景
- **流式推理**：固定大小状态（~几百 KB）vs Transformer 线性增长的 KV Cache，极适合实时流式生成
- **边缘部署**：推理时内存占用恒定，不随序列长度增长，$O(1)$ 内存推理

### 工程实现要点
- **selective scan kernel**：必须使用 CUDA 实现才能发挥性能，纯 Python 实现会极慢
- **混合架构更实用**：Jamba 实验表明，每 8 个 Mamba 层插入 1 个 Attention 层是一个好的起点
- **序列长度阈值**：经验上 $L < 2K$ 时 Transformer 更快，$L > 16K$ 时 Mamba 优势明显

### 面试高频问法
- Q: SSM 的连续状态空间方程和 RNN 有什么关系？
  A: SSM 本质是线性 RNN 的连续时间推广。离散化后 $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ 就是 RNN。关键差异：S4/Mamba 通过 HiPPO 初始化和结构化参数化解决了梯度消失，通过卷积视图实现并行训练。

## 💡 启发与思考

### So What？对老板意味着什么
- **Mamba 代表了"不靠 Attention 也能做 LLM"的可能性**。这打破了 Transformer 一家独大的格局，对硬件设计（不需要大规模矩阵乘优化）和模型架构（混合范式）都有深远影响
- **实际落地的最佳实践是混合架构**：纯 Mamba 在推理任务上不够强，但 Mamba + Attention 的组合既保留长序列效率，又不丢失 ICL 能力

### 未解问题与局限
- 选择性机制的理论基础仍不完善——为什么 input-dependent 的 $B$, $C$, $\Delta$ 就能"选择"重要信息？
- Mamba-2 的 SSD 框架声称统一了 SSM 和 Attention，但实际性能差距在复杂推理任务上仍然存在
- 训练并行度问题：selective scan 的并行效率仍不如标准 Attention 的矩阵乘

### 脑暴：如果往下延伸
- 如果把 Mamba 的状态压缩思想和 [[AI/3-LLM/Architecture/长上下文处理|长上下文处理]] 中的 Ring Attention 结合，能否实现百万 token 级别的高效分布式推理？
- [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] 的推理能力依赖长链 CoT，而 Mamba 擅长长序列——二者结合可能产生"推理效率极高"的新范式

> 🔗 See also: [[AI/3-LLM/Architecture/架构范式对比|架构范式对比]] — SSM vs Transformer vs MoE 全景对比