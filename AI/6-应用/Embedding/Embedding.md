---
brief: "Embedding 基础——文本向量化的原理和主流模型（OpenAI text-embedding/BGE/E5/GTE）；Embedding 质量评估方法（MTEB benchmark）；RAG 和语义搜索场景的 Embedding 工程基础。"
title: "Embedding"
type: concept
domain: ai/llm/application/embedding
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/application/embedding
  - type/concept
---
# Embedding

[Embedding Model Fine-Tuning 案例](https%3A%2F%2Fninehills.tech%2Farticles%2F118.html)

[Embedding 模型在 RAG 场景下的评估和微调](https%3A%2F%2Fninehills.tech%2Farticles%2F104.html)

# Embedding 模型在 RAG 场景下的评估和微调

为检验 Embedding 模型在 RAG 应用中的性能，我们引入 [C-MTEB](https%3A%2F%2Fgithub.com%2FFlagOpen%2FFlagEmbedding%2Fblob%2Fmaster%2FC_MTEB%2FREADME.md) 评测用来评估 Embedding 模型的性能。

已有的 Embedding 模型的 C-MTEB 分数在 [MTEB Leaderboard](https%3A%2F%2Fhuggingface.co%2Fspaces%2Fmteb%2Fleaderboard) 上可以通过选择 `Chinese` 选项卡查看。

而针对有明确数据集的场景，我们也可以复用 C-MTEB 的评估方法，评估 Embedding 模型在特定数据集上的性能，从而为后续微调提供参考。

## C-MTEB 评估任务

C-MTEB 有多种任务，其中和 RAG 能力相关是 Reranking 和 Retrieval 任务，其数据集格式如下：

Reranking 任务的数据集格式为：

```python
{
    **"query"**: "大学怎么网上选宿舍",
    **"positive"**: ["long text", "long text"],
    **"negative"**: ["long text", "long text"]
}
```

目前通用场景的 Reranking 数据集主要是 [T2Reranking](https%3A%2F%2Fhuggingface.co%2Fdatasets%2FC-MTEB%2FT2Reranking)。

在评测分数中，我们主要关心 `map` 分数，这是因为任务不涉及排序，而是看是否命中 positive 。

而 Retrieval 任务的数据集以 [T2Retrieval](https%3A%2F%2Fhuggingface.co%2Fdatasets%2FC-MTEB%2FT2Retrieval) 为例，分为三个部分：

- corpus：是一个长文本集合。
- queries：是检索用的问题。
- qrels: 将检索问题和长文本对应起来，通过 score 进行排序。不过目前数据集中的 score 都为 1。（这可能是规避了数据错误标注，但是也影响了评测效果）。
在评测分数中，我们主要关心 `ndcg@10` 分数，是检验 top10 检索结果中排序是否一致的指标。

此外由于 Retrieval 数据集比较难构造，所以一般自定义数据集都是用 Reranking 数据集。Reranking 数据集的格式还和 [FlagEmbedding fine-tune](https%3A%2F%2Fgithub.com%2FFlagOpen%2FFlagEmbedding%2Fblob%2Fmaster%2Fexamples%2Ffinetune%2FREADME.md) 所需的数据格式相同，方便用于微调后的评估。

## 自定义模型的通用任务评测

选择 T2Reranking 进行评测，评测目标是文心千帆上的 [Embedding-V1](https%3A%2F%2Fcloud.baidu.com%2Fdoc%2FWENXINWORKSHOP%2Fs%2Falj562vvu) 模型。

参见 [Colab Notebook](https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1PcJcgWZ-B5AQUZ2FsRYd6inQ42_NqnUr%3Fusp%3Dsharing)。

取 1000 条测试数据（总数据的 1/6，为了降低 token 使用），评测 map 得分为 66.54。超过了 Leaderboard 上的 SOTA 分数 66.46 分。（不过并不是全部数据，这个分数仅供参考，如果是全部数据，得分可能会低于 SOTA）

## 自定义数据集微调和评测

使用 T2Reranking 数据集拆分出训练集和测试集，对 [BAAI/bge-small-zh-v1.5](https%3A%2F%2Fhuggingface.co%2FBAAI%2Fbge-small-zh-v1.5) 模型进行微调和测试。

参见 [Colab Notebook](https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1dAAVssdWNin47e2xeGsEpWnArU6Nx4eu%3Fusp%3Dsharing)。

可以看到微调效果不尽如人意（第一个 checkpoint 就提升了 3% 的效果，然后后续无明显提升）。这可能是因为数据集的质量不高，导致模型无法学到有效的信息。

社区相关讨论：[https://github.com/FlagOpen/FlagEmbedding/issues/179](https%3A%2F%2Fgithub.com%2FFlagOpen%2FFlagEmbedding%2Fissues%2F179)

微调资源占用：small 模型，4090 显存占用 20G。

---

## See Also

- [[AI/6-应用/Embedding/Embedding 选型|Embedding 选型]] — 同方向实践版，与本篇互补
- [[AI/6-应用/RAG/RAG 检索策略|RAG 检索策略]] — Embedding 在 RAG 中的核心角色
- Transformer 通识 — Embedding 的架构基础
-  — 大语言模型知识全图谱
