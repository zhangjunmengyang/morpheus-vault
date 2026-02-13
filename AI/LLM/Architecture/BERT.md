---
title: "BERT"
type: paper
domain: ai/llm/architecture
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/architecture
  - type/paper
---
# BERT

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型，开创了 NLP 的 "预训练 + 微调" 范式，在 11 项 NLP 基准上刷新了 SOTA。虽然在 LLM 时代 BERT 不再是主角，但它的设计思想至今深刻影响着整个领域。

## 核心创新

### 双向上下文建模

在 BERT 之前，GPT 使用单向（从左到右）的语言建模，ELMo 虽然是双向的但只是简单拼接两个方向。BERT 的关键洞察是：**真正的语言理解需要同时看到左右两侧的上下文**。

但双向建模有一个致命问题——标准的语言模型目标函数不适用（每个词能"看到自己"）。BERT 用了一个巧妙的解决方案：**Masked Language Model（MLM）**。

### Masked Language Model

随机遮盖输入中 15% 的 token，让模型预测被遮盖的 token：

$$L_{\text{MLM}} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})\right]$$

但直接 mask 会导致预训练和微调的不匹配（微调时没有 [MASK] token），所以 BERT 对被选中的 15% token 做了三种处理：
- 80% 替换为 [MASK]
- 10% 替换为随机 token
- 10% 保持不变

这个设计看似简单，实际上是非常精巧的正则化策略。

### Next Sentence Prediction（NSP）

BERT 还引入了句对级别的预训练任务：判断两个句子是否连续。后来的研究（RoBERTa、ALBERT）证明 NSP 的价值有限，甚至有害。这提醒我们：**不是所有看起来合理的预训练任务都有用**。

## 架构细节

BERT 使用标准的 Transformer Encoder：

| 版本 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

关键设计选择：
- **WordPiece 分词**：30000 词汇表
- **位置编码**：可学习的绝对位置编码（最大 512）
- **特殊 token**：[CLS] 用于分类任务，[SEP] 用于分隔句子
- **Segment Embedding**：区分句对中的两个句子

输入表示 = Token Embedding + Segment Embedding + Position Embedding

## 微调范式

BERT 的微调设计极其优雅：不同的下游任务只需要在 [CLS] token 的输出上接不同的分类头：

- **文本分类**：[CLS] → 线性层 → softmax
- **序列标注（NER）**：每个 token 的输出 → 线性层 → softmax
- **问答（SQuAD）**：预测 answer span 的起止位置
- **句对任务**：两个句子用 [SEP] 分隔输入

这种统一的微调框架让 BERT 成为了 NLP 的"万能基座"。

## BERT 的后继者们

BERT 催生了大量改进工作：

- **RoBERTa**（Facebook）：去掉 NSP，更大的 batch size，动态 masking，训练更久 → 效果显著提升
- **ALBERT**：参数共享 + Factorized Embedding，大幅减少参数量
- **DistilBERT**：知识蒸馏，6 层保留 97% 的性能
- **SpanBERT**：mask 连续 span 而非随机 token，对抽取式任务更好
- **DeBERTa**（微软）：解耦注意力机制，分别处理内容和位置，在 SuperGLUE 上首次超过人类

## BERT vs GPT：两条路线的分野

BERT 和 GPT 代表了两种根本不同的预训练哲学：

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder | Decoder |
| 方向 | 双向 | 单向（左到右） |
| 预训练任务 | MLM + NSP | Causal LM |
| 擅长 | 理解（分类、NER、QA） | 生成（对话、创作、推理） |
| 规模天花板 | ~1B | ~1T |

历史的选择是 GPT 路线赢了。原因不是 BERT 不好，而是：
1. **生成能力是刚需**——ChatGPT 的出现证明了这一点
2. **Decoder-Only 更容易 scale**——自回归训练天然支持无限长序列
3. **In-Context Learning 只在自回归模型中涌现**

但 BERT 的 Encoder 架构在嵌入模型（sentence-transformers）、信息检索、NER 等任务中依然不可替代。

## 在 LLM 时代的价值

虽然 GPT 系列抢占了聚光灯，BERT 架构在 2026 年依然活跃：

1. **Embedding 模型**：大部分 RAG 系统的 retriever 基于 BERT 变体
2. **Reranker**：cross-encoder 本质上就是 BERT 的句对分类
3. **Token 级任务**：NER、POS tagging 等仍然首选 BERT 架构
4. **轻量部署**：BERT-Base 只有 110M 参数，适合边缘设备

## 相关

- [[GPT]]
- [[AI/CV/ViT|ViT]]
- [[AI/MLLM/CLIP|CLIP]]
- [[AI/Foundations/DL-Basics/深度学习|深度学习]]
- [[AI/LLM/SFT/LoRA|LoRA]]
- [[AI/LLM/Architecture/T5|T5]]
- [[AI/LLM/Architecture/LLaMA|LLaMA]]
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]
- [[AI/Foundations/DL-Basics/Transformer 通识|Transformer 通识]]
- [[AI/Foundations/DL-Basics/Attention 详解|Attention 详解]]
- [[AI/LLM/Application/Embedding/Embedding|Embedding]]
