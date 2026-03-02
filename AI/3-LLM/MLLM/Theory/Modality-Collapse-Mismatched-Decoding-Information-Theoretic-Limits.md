---
title: Modality Collapse = Mismatched Decoding：多模态 LLM 的信息论上限（GMI）
arxiv: 2602.23136
venue: arXiv
date: 2026-03-01
source: https://arxiv.org/abs/2602.23136
tags:
  - mllm
  - vlm
  - speech
  - theory
  - information-theory
  - decoding
  - modality-collapse
rating: ★★★★☆
brief: 论文提出“多模态 LLM 看不见纹理/听不出音色”并非编码失败，而是 decoder 的 scoring rule 与训练目标导致的 mismatched decoding：decoder 只能沿 text-aligned 方向提取信息；其它 modality-specific 方向对 loss 来说是噪声，甚至移除 64–71% 方差会让 decoder loss 更好。作者用 Generalized Mutual Information (GMI) 给出“可访问信息”上限，且该上限与架构无关（projection/codebook/无 adapter 都适用）。结论：修复必须改训练目标/对齐 decoder，而不是一味换 encoder。
related:
  - "[[AI/3-LLM/MLLM/Evaluation/Reporting-Bias-Impairs-Vision-Language-Reasoning-Scale-Cant-Overcome-Pragmatics]]"
  - "[[AI/3-LLM/MLLM/Evaluation/Reporting-Bias-Impairs-Vision-Language-Reasoning-Scale-Cant-Overcome-Pragmatics|MLLM Evaluation（Reporting Bias）]]"
---

# Modality Collapse as Mismatched Decoding（arXiv:2602.23136）

## 1) 他们在解释什么现象？
多模态 LLM 常见的“表面能力”是：
- 能转写（speech → text）
- 能 caption（image → text）

但“深层失败”是：
- 听不出 speaker identity / emotion（音色、情绪）
- 看不出 texture / 细粒度视觉属性

论文把这种“只保留与文本高度相关的模态信息，其余坍缩”的现象称为 **modality collapse**。

## 2) 最关键的反直觉证据：不是 encoder 不会编码
他们用 linear probes 显示：
- speaker identity / emotion / visual attributes 等信息**穿过每一层 LLM 都还在**
- probe accuracy 甚至达到 **3–55× above chance**

但同时：
- 移除 **64–71% 的 modality-specific variance** 会 **improve decoder loss**

=> 信息“存在但不可用”。这不是表征缺失，而是 **decoder 没学会用**。

## 3) 核心机制：Mismatched Decoder（错配解码）
把 multimodal LLM 的 decode 看成“通信系统的解码器”：
- encoder/projection 把非文本输入映射到 LLM embedding space
- decoder（语言模型头）用一个 scoring rule（本质是为 text 训练出来的）来打分/生成

**错配点**：decoder 是为 text 建模训练的，只能提取沿 text-aligned 方向的信号。
- modality-specific directions 对它来说不提高 likelihood
- 于是成了噪声：对 loss 反而有害，模型倾向于压掉/忽略

关键一句（我认为可写进“多模态失败原因”的金句）：
> bottleneck is the decoder’s scoring rule, not the encoder or projection

## 4) 理论框架：GMI 上界（可访问信息）
他们用 **Generalized Mutual Information (GMI)** 来界定“给定 decoder scoring rule，最多能访问多少信息”。

结论形态（从摘要/框架描述抽象）：
- accessible information ≤ GMI
- degradation 随两件事增长：
  1) 分布距离（non-text 表征分布与 decoder 假设的偏离）
  2) decoder sensitivity（对这些偏离有多敏感，类似 Lipschitz）

更重要的是：
- 这个 bound 是 scoring rule 的性质，**与具体架构无关**
- 无论非文本输入来自 learned projection / 离散 codebook / 甚至无 adapter，都逃不掉

这点很“杀招”：把大量工程尝试（换 encoder、更复杂 adapter）直接归因到“不碰 decoder 就无解”。

## 5) 实验验证：Prismatic 控制实验 + LoRA 修复
### 5.1 Prismatic 控制实验
他们做了一个对照：
- 两个 Prismatic VLM 只在 encoder 的 text-alignment 上不同

结论：瓶颈仍在 decoder scoring rule，而非 encoder/projection。

### 5.2 LoRA intervention：证明“训练目标决定可访问性”
他们用 LoRA 加一个 emotion objective：
- emotion accessibility 提升 **+7.5%**
- 其他属性不受影响

=> 这是非常强的因果证据：
- 不是“信息不存在”
- 是“目标函数没要求 decoder 学会读这个方向”

## 6) 我对这篇论文的判断（机制 + 边界）
### 6.1 机制价值
它把“多模态能力缺失”从经验观察升级成：
- **可证伪的机制假设（mismatched decoding）**
- **可计算的上界（GMI）**
- **可干预的修复路径（改目标/对齐 decoder）**

而且与我刚入库的 reporting bias（2602.23351）形成一个很漂亮的对偶：
- 2602.23351：数据语用学偏置 → 缺监督信号类型
- 2602.23136：decoder scoring rule 偏置 → 信号虽在但不可访问

### 6.2 边界
- LoRA 实验说明“可修”，但规模化到通用多模态推理还需要更系统的 objective 设计（不仅是单一 emotion attribute）。
- “移除 64–71% 方差提升 loss”提示可能存在 trade-off：
  - 让 decoder 学会用更多方向，可能会牺牲原本的文本语言建模稳定性（需要进一步看实验）。

## Links
- arXiv:2602.23136 — https://arxiv.org/abs/2602.23136
- HTML — https://arxiv.org/html/2602.23136v1
- Code — https://github.com/jb1999/modality_collapse_paper
