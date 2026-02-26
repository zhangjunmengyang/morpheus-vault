---
title: "lc1 · 基础组件专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc1_base"
tags: [moc, ma-rlhf, tokenizer, embedding, positional-encoding, lc1]
---

# lc1 · 基础组件专题地图

> **目标**：理解语言模型的两大基石——离散化（Tokenizer）与连续化（Embedding），为 lc2 Transformer 打基础。  
> 语言模型的核心在于理解文本语义，机器表示语言分两步：**离散化**（文本 → token 序列）和**连续化**（token → 向量空间）。

---

## 带着这三个问题学

1. **BPE 算法如何从字符出发构建词表？** merge 规则的贪心策略会带来什么问题？
2. **为什么需要 Token Embedding + Position Embedding 两个独立的嵌入？** 少一个行不行？
3. **位置编码从 Absolute PE → RoPE → ALiBi → YaRN 的演进动机是什么？** 每一代解决了上一代的什么痛点？

---

## 学习顺序

```
Step 1  字符级分词器              ← 从最简单的 char-level 理解分词本质
   ↓
Step 2  BPE Tokenizer 算法       ← 词表构建：统计频次 → 贪心 merge → 迭代
   ↓
Step 3  Word2Vec & 词嵌入         ← 连续表示的起源，CBOW/Skip-gram
   ↓
Step 4  Token Embedding           ← 查表操作，nn.Embedding 的本质
   ↓
Step 5  位置编码家族              ← Absolute PE → Sinusoidal → RoPE → ALiBi → YaRN
```

---

## 笔记清单

### Step 1-2：Tokenizer — 离散化

**[[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操|Tokenizer-Embedding 手撕实操]]**

- **BPE 算法核心步骤**：初始化字符级词表 → 统计相邻 pair 频次 → 合并最高频 pair → 加入词表 → 重复至目标词表大小
- **merge 规则**：贪心策略，每次合并全局最高频 pair；`merges` 列表定义了编码时的 merge 优先级
- **encode 流程**：字符拆分 → 按 merges 顺序尝试合并 → 得到 token 序列
- **decode 流程**：token id → 词表查找 → 拼接还原

课程代码：`tokenizer_basic.ipynb` — 从零手撕字符级分词器

深入阅读：[[AI/LLM/Architecture/Tokenizer 深度理解|Tokenizer 深度理解]] · [[AI/LLM/Architecture/Tokenizer|Tokenizer 原理]]

---

### Step 3-4：Embedding — 连续化

**[[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操|Tokenizer-Embedding 手撕实操]]**（同一篇，Embedding 部分）

- **Word2Vec**：通过上下文预测目标词（Skip-gram）或用上下文预测中心词（CBOW），训练出词向量
- **Token Embedding**：`nn.Embedding(vocab_size, d_model)` 本质是查表，每个 token id 对应一行权重向量
- **为什么需要两个 Embedding**：Token Embedding 编码「是什么词」，Position Embedding 编码「在哪个位置」；Transformer 的 Self-Attention 是 permutation-invariant 的，不加位置信息无法区分词序

课程代码：`embedding.ipynb` — Word2Vec 实现 + 文本语义表示 + 上下文词表征

---

### Step 5：位置编码家族

**[[AI/LLM/Architecture/Transformer 位置编码|位置编码]]**

演进逻辑：

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **Absolute PE** | 固定/可学习的位置向量，直接加到 token embedding | 简单直观 | 无法外推到训练长度之外 |
| **Sinusoidal PE** | 用 sin/cos 不同频率编码位置 | 理论上可外推 | 实际外推效果有限 |
| **RoPE** | 旋转矩阵编码相对位置，作用在 Q/K 上 | 天然相对位置，外推性好 | 超长序列仍需扩展 |
| **ALiBi** | 不用位置编码，直接在 Attention Score 上加线性偏置 | 零额外参数，长度泛化好 | 表达力受限 |
| **YaRN** | 插值 + NTK 混合策略，动态扩展 RoPE 频率 | 4K→128K 可控扩展 | 需要少量微调 |

**关键洞察**：RoPE 将位置信息编码进 Q·K 的内积中，`<q_m, k_n>` 只依赖相对距离 `m-n`，不依赖绝对位置 → 天然支持变长序列。

---

## 与后续课时的关系

- **lc2 Transformer**：Embedding + PE 是 Transformer 的输入层，直接衔接
- **lc3 GPT**：BPE Tokenizer 是 GPT 系列的标配分词器
- **lc4 Llama**：RoPE 是 Llama 的核心组件，NTK-RoPE 解决长文本
- **lc5 DeepSeek V3**：YaRN 是 V3 的上下文扩展方案

---

## 面试高频场景题

**Q：BPE 和 WordPiece 的区别？**  
A：两者都是子词分词算法。BPE 基于**频率**贪心合并最高频 pair；WordPiece（BERT 用）基于**互信息**，选择合并后使语言模型似然提升最大的 pair。BPE 的 merge 是确定性的（频率排序），WordPiece 需要语言模型打分。实践中 BPE 更常用于 GPT 系列，WordPiece 用于 BERT 系列。

**Q：RoPE 为什么比绝对位置编码泛化性更好？**  
A：绝对位置编码将每个位置映射为固定向量，模型只见过训练长度内的位置 → 推理时遇到更长序列，位置向量超出训练分布。RoPE 把位置编码为旋转角度，Q·K 的内积只依赖相对距离 `m-n`（旋转矩阵的乘法性质），不依赖绝对位置 → 天然支持未见过的绝对位置，只要相对距离在训练范围内。

**Q：为什么 Transformer 需要位置编码？**  
A：Self-Attention 的计算对输入 token 的排列是不变的（permutation invariant）——打乱输入顺序，输出不变。但自然语言是有序的，「猫吃鱼」和「鱼吃猫」含义完全不同。位置编码为每个 token 注入位置信息，打破排列不变性。

**Q：Token Embedding 和 Word2Vec 的关系？**  
A：Word2Vec 是独立训练的词向量（浅层模型），Token Embedding 是 Transformer 内部的可学习参数（随整个模型端到端训练）。现代 LLM 不再使用预训练 Word2Vec，而是直接从随机初始化的 Embedding 开始训练。但 Word2Vec 的核心思想（分布式语义假说：上下文相似的词有相似向量）贯穿至今。
