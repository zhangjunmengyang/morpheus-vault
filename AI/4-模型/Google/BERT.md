---
title: BERT：双向预训练语言模型的开创者
brief: BERT 通过 Masked Language Model 实现真正的双向上下文建模，开创了 NLP 预训练+微调范式。核心洞察是 15% token masking 的 80/10/10 策略巧妙解决了双向训练的信息泄露问题。虽然 LLM 时代 Decoder-Only 胜出，BERT 架构在 Embedding/Reranker/NER 等场景仍不可替代。
type: paper
domain: ai/llm/architecture
created: 2026-02-13
updated: 2026-02-22
tags:
  - ai/llm/architecture
  - type/paper
status: complete
sources:
  - "Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* arXiv:1810.04805"
  - "Liu et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach* arXiv:1907.11692"
  - "He et al. *DeBERTa: Decoding-enhanced BERT with Disentangled Attention* arXiv:2006.03654"
related:
  - "[[AI/4-模型/OpenAI/GPT|GPT]]"
  - "[[AI/3-LLM/Architecture/Tokenizer|Tokenizer]]"
  - Transformer 通识
  - "[[Embedding|Embedding]]"
---
# BERT

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型，开创了 NLP 的 "预训练 + 微调" 范式，在 11 项 NLP 基准上刷新了 SOTA。虽然在 LLM 时代 BERT 不再是主角，但它的设计思想至今深刻影响着整个领域。

> 来源：Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* arXiv:1810.04805

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

- **RoBERTa**（Facebook, arXiv:1907.11692）：去掉 NSP，更大的 batch size，动态 masking，训练更久 → 效果显著提升
- **ALBERT**：参数共享 + Factorized Embedding，大幅减少参数量
- **DistilBERT**：知识蒸馏，6 层保留 97% 的性能
- **SpanBERT**：mask 连续 span 而非随机 token，对抽取式任务更好
- **DeBERTa**（微软, arXiv:2006.03654）：解耦注意力机制，分别处理内容和位置，在 SuperGLUE 上首次超过人类

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

## 📚 推荐阅读

### 原始论文
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 必读原文，MLM + NSP 的设计哲学
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 证明 NSP 无用，训练策略比架构更重要
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — SuperGLUE 首次超人类的 BERT 变体

### 深度解读
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — Jay Alammar 经典可视化解读 ⭐⭐⭐⭐⭐
- [BERT Research Series (Chris McCormick)](https://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/) — 系统性的 BERT 研究系列

### 实践资源
- [HuggingFace Transformers BERT](https://huggingface.co/docs/transformers/model_doc/bert) — 官方文档，快速上手
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — 基于 BERT 的 Embedding 模型库

## 🔧 落地应用

### 直接可用场景
- **语义搜索/RAG Retriever**：BERT 变体（如 BGE、E5）是 2026 年 RAG 系统中最主流的 embedding 模型，将文本编码为稠密向量后做 ANN 检索
- **Reranker/Cross-Encoder**：接收 query + document 对，用 [CLS] token 输出相关性分数，比 bi-encoder 精度更高
- **NER / 序列标注**：token 级别的分类任务仍首选 BERT 架构，110M 参数在边缘设备也能跑

### 工程实现要点
- **微调学习率**：BERT 微调典型学习率 $2 \times 10^{-5}$ 到 $5 \times 10^{-5}$，过大容易灾难性遗忘
- **[CLS] vs Mean Pooling**：sentence-transformers 证明 mean pooling 比 [CLS] 在句子级任务上效果更好
- **最大长度 512**：BERT 的绝对位置编码限制了最大长度，长文档需截断或分块

### 面试高频问法
- Q: BERT 的 MLM 为什么用 80/10/10 的 mask 策略而不是全部替换为 [MASK]？
  A: 全部替换会导致预训练和微调的分布不匹配（微调时没有 [MASK] token）。10% 保持不变让模型学会对正常 token 也建模；10% 随机替换防止模型"偷懒"只在看到 [MASK] 时才预测。

## 💡 启发与思考

### So What？对老板意味着什么
- **BERT 架构是 RAG 系统的地基**：选 Embedding 模型时，绝大多数高质量选项（BGE、E5、GTE）底层都是 BERT 变体。理解 BERT 就理解了 RAG retriever 的能力边界
- **预训练目标决定模型能力**：MLM 让 BERT 擅长"理解"，Causal LM 让 GPT 擅长"生成"。这个设计选择在 2018 年就决定了两条路线的分野

### 未解问题与局限
- BERT 的绝对位置编码无法外推到训练长度之外（对比 [[AI/3-LLM/Architecture/长上下文处理|长上下文处理]] 中的 RoPE）
- MLM 的 15% mask 比例是否最优？SpanBERT 等工作在探索，但没有定论
- Encoder 架构为何难以 scale 到 1T+ 参数？（对比 [[AI/4-模型/OpenAI/GPT|GPT]] 的 Decoder-Only 路线）

### 脑暴：如果往下延伸
- 如果把 BERT 的双向理解能力和 [[AI/3-LLM/Architecture/Mamba-SSM|Mamba]] 的线性复杂度结合，能否做出效率更高的 Embedding 模型？
- 结合 [[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek-R1]] 的蒸馏经验，1.5B 的 BERT 变体是否能在边缘设备上替代 7B 的 Decoder 模型做理解任务？

## 相关

- [[AI/4-模型/OpenAI/GPT|GPT]] — Decoder-Only 路线对比，理解两条技术路线的分野
- [[Projects/MA-RLHF/lc6/CLIP-ViT-LLaVA-手撕实操|ViT]] — BERT 思想迁移到视觉领域
- [[AI/3-LLM/MLLM/CLIP|CLIP]] — 多模态对比学习，Encoder 架构的跨模态应用
- [[AI/1-Foundations/DL-Basics/深度学习|深度学习]]
- [[AI/3-LLM/SFT/LoRA|LoRA]] — BERT 微调的高效方法
- [[AI/4-模型/Google/T5|T5]] — Encoder-Decoder 路线的代表
- [[AI/4-模型/Meta/LLaMA|LLaMA]]
- [[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek-R1]]
- Transformer 通识
- [[AI/3-LLM/Architecture/Attention 变体综述|Attention 详解]]
- [[Embedding|Embedding]] — BERT 在 RAG 中的核心应用
