---
title: "Attention 详解"
type: concept
domain: ai/foundations/dl-basics
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/foundations/dl-basics
  - type/concept
---
# Attention 详解

### 你能解释一下自注意力机制的工作原理吗？

- Attention的变种知道吗？ 
- 自注意力和交叉注意力这俩的区别？ 
https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84.md

由于 NLP 任务所需要处理的文本往往是序列，因此之前专用于处理序列、时序数据的 RNN 往往能够在 NLP 任务上取得最优的效果。

RNN 及 LSTM 虽然具有捕捉时序信息、适合序列生成的优点，却有两个难以弥补的缺陷：

1. 序列依序计算的模式能够很好地模拟时序信息，但限制了计算机并行计算的能力。由于序列需要依次输入、依序计算，图形处理器（Graphics Processing Unit，GPU）并行计算的能力受到了极大限制，导致 RNN 为基础架构的模型虽然参数量不算特别大，但计算时间成本却很高；
1. RNN 难以捕捉长序列的相关关系。在 RNN 架构中，距离越远的输入之间的关系就越难被捕捉，同时 RNN 需要将整个序列读入内存依次计算，也限制了序列的长度。虽然 LSTM 中通过门机制对此进行了一定优化，但对于较远距离相关关系的捕捉，RNN 依旧是不如人意的。
针对这样的问题，Vaswani 等学者参考了在 CV 领域被提出、被经常融入到 RNN 中使用的注意力机制（Attention）（注意，虽然注意力机制在 NLP 被发扬光大，但其确实是在 CV 领域被提出的），创新性地搭建了完全由注意力机制构成的神经网络——Transformer，也就是大语言模型（Large Language Model，LLM）的鼻祖及核心架构，从而让注意力机制一跃成为深度学习最核心的架构之一。

注意力机制有三个核心变量：Query（查询值）、Key（键值）和 Value（真值）。

包含手写 attention 的代码：https://github.com/datawhalechina/llms-from-scratch-cn

## 相关

- [[AI/Foundations/DL-Basics/Transformer 通识|Transformer 通识]]
- [[AI/LLM/Architecture/BERT|BERT]]
- [[AI/LLM/Architecture/GPT|GPT]]
- [[AI/LLM/Architecture/LLaMA|LLaMA]]
