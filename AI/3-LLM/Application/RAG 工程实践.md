---
brief: "RAG 工程实践——生产级 RAG 系统搭建的完整工程指南；文档处理流水线/向量数据库选型/检索策略调优/幻觉检测/延迟优化；interview/hot，RAG 系统设计的面试标准答案框架。"
title: "RAG 工程实践"
date: 2026-02-13
tags:
  - ai/llm/application
  - ai/rag
  - type/practice
  - interview/hot
status: active
---

# RAG 工程实践

> Retrieval-Augmented Generation — Chunking 策略、Embedding 选型、Reranker、Hybrid Search 与实际踩坑

## 1. RAG 架构全景

```
                    ┌────────────────────────────────┐
                    │        RAG Pipeline            │
                    └────────────────────────────────┘

  ┌──────────────────── Indexing (离线) ────────────────────┐
  │                                                         │
  │  Documents → Chunking → Embedding → Vector DB           │
  │     ↓           ↓          ↓           ↓               │
  │  PDF/HTML    分块策略    向量化模型   Milvus/Qdrant     │
  │                                    + BM25 Index         │
  └─────────────────────────────────────────────────────────┘

  ┌──────────────────── Retrieval (在线) ───────────────────┐
  │                                                         │
  │  Query → Query Transform → Hybrid Search → Reranker     │
  │    ↓         ↓                ↓              ↓          │
  │  用户问题  HyDE/扩写     Dense+Sparse     交叉注意力    │
  │                              Top-K          Top-N       │
  └─────────────────────────────────────────────────────────┘

  ┌──────────────────── Generation (在线) ──────────────────┐
  │                                                         │
  │  Prompt + Retrieved Context → LLM → Response            │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

## 2. Chunking 策略

Chunking 是 RAG 中**最容易被忽视但影响最大**的环节。

### 2.1 策略对比

| 策略 | 原理 | 优点 | 缺点 | 推荐场景 |
|------|------|------|------|----------|
| Fixed-size | 按字符/token 数切分 | 简单快速 | 语义断裂 | 快速原型 |
| Recursive | 按分隔符层级切分 | 尊重文档结构 | 需要调参 | **通用首选** |
| Semantic | 按语义相似度切分 | 语义完整 | 计算开销大 | 高质量要求 |
| Page-level | 按页/段落 | 无信息丢失 | chunk 过大 | PDF 文档 |
| Agentic | LLM 辅助切分 | 最优质量 | 成本极高 | 关键场景 |

### 2.2 Recursive Chunking（推荐）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # chunk 大小（tokens）
    chunk_overlap=50,      # 重叠区域（防止上下文断裂）
    separators=[
        "\n\n",   # 优先按段落分
        "\n",     # 其次按换行
        "。",     # 然后按句号
        ".",
        " ",      # 最后按空格
        "",       # 兜底按字符
    ],
    length_function=len,
)

chunks = splitter.split_text(document_text)
```

### 2.3 Contextual Chunking（2025 最佳实践）

Anthropic 提出的**上下文化 Chunking**——为每个 chunk 添加上下文前缀：

```python
def contextual_chunk(document, chunks):
    """为每个 chunk 添加文档上下文"""
    results = []
    for chunk in chunks:
        # 用 LLM 生成该 chunk 在文档中的上下文描述
        context = llm.generate(f"""
        文档标题: {document.title}
        文档摘要: {document.summary}
        
        以下是文档的一个片段。请用 1-2 句话描述这个片段在文档中的位置和上下文：
        
        {chunk.text}
        """)
        # 将上下文与原文拼接
        results.append(f"{context}\n\n{chunk.text}")
    return results
```

### 2.4 关键参数调优

```yaml
chunk_size: 256-1024 tokens  # 太小：上下文不足; 太大：噪声多
  - 问答场景: 256-512（精准匹配）
  - 摘要场景: 512-1024（更多上下文）
  - 代码场景: 按函数/类切分

chunk_overlap: 10-20% of chunk_size
  - 0%: 完全不重叠（可能断裂）
  - 50+: 太多冗余
```

## 3. Embedding 选型

参见 [[AI/3-LLM/Application/Embedding/Embedding 选型|Embedding 选型]] 了解基础内容。

### 3.1 2025-2026 推荐 Embedding

| 模型 | 维度 | 语言 | MTEB 排名 | 适用场景 |
|------|------|------|-----------|----------|
| **voyage-3-large** | 1024 | 多语言 | Top 3 | 商业首选 |
| **text-embedding-3-large** | 3072 | 多语言 | Top 5 | OpenAI 生态 |
| **bge-m3** | 1024 | 多语言 | Top 10 | 开源首选 |
| **jina-embeddings-v3** | 1024 | 多语言 | Top 5 | Jina 生态 |
| **nomic-embed-text-v2** | 768 | 英文 | 高 | 本地/轻量 |
| **gte-Qwen2** | 1536 | 中英 | Top 5 | 中文场景 |

### 3.2 Embedding 使用注意事项

```python
# 最佳实践：查询和文档使用不同前缀
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

# 索引时
doc_embedding = model.encode("passage: " + chunk_text)

# 查询时
query_embedding = model.encode("query: " + user_question)

# 为什么需要前缀？因为查询通常是短问题，文档是长段落
# 不对称 embedding 模型通过前缀区分两种输入模式
```

### 3.3 踩坑：Embedding 维度 vs 检索质量

- 维度越高 ≠ 越好（nomic-768d 在某些任务超过 OpenAI-3072d）
- **Matryoshka Representation**：新模型支持截断维度（如 3072 → 512）而保持大部分质量
- 存储成本：3072d × float32 = 12KB/doc → 百万文档 = 12GB 纯向量

## 4. Reranker（重排序）

Reranker 是 RAG 质量提升最显著的单一组件。

### 4.1 为什么需要 Reranker？

```
Bi-Encoder (Embedding):
  Query → [Encoder] → q_vec
  Doc   → [Encoder] → d_vec    → cosine_sim(q_vec, d_vec)
  ✅ 快速（可预计算）  ❌ 无交互（query 和 doc 独立编码）

Cross-Encoder (Reranker):
  [Query + Doc] → [Encoder] → relevance_score
  ❌ 慢（每对都要计算）  ✅ 深度交互（query 和 doc 联合编码）
```

实际 pipeline：**Embedding 粗筛 Top-100 → Reranker 精排 Top-10**

### 4.2 推荐 Reranker

| 模型 | 类型 | 速度 | 质量 |
|------|------|------|------|
| **Cohere rerank-v3.5** | API | 快 | 最好 |
| **voyage-rerank-2** | API | 快 | 很好 |
| **bge-reranker-v2-m3** | 开源 | 中 | 好 |
| **jina-reranker-v2** | 开源 | 中 | 好 |
| **flashrank** | 开源/轻量 | 最快 | 够用 |

### 4.3 实现示例

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# 粗筛结果
candidates = vector_db.search(query_embedding, top_k=100)

# 精排
pairs = [[query, doc.text] for doc in candidates]
scores = reranker.compute_score(pairs)

# 按 reranker 分数重排
reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
top_results = reranked[:10]
```

## 5. Hybrid Search（混合检索）

### 5.1 Dense vs Sparse

```
Dense Search (向量检索):
  ✅ 语义理解（"猫" 能匹配 "喵星人"）
  ❌ 精确匹配弱（专有名词、ID、代码）

Sparse Search (关键词检索, BM25):
  ✅ 精确匹配强（关键词、ID）
  ❌ 无语义理解（"猫" 匹配不到 "喵星人"）

Hybrid = Dense + Sparse → 两全其美
```

### 5.2 融合策略

```python
# Reciprocal Rank Fusion (RRF)
def rrf_fusion(dense_results, sparse_results, k=60):
    """RRF: 将不同检索结果的排名融合"""
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

# 或者加权融合
def weighted_fusion(dense_scores, sparse_scores, alpha=0.7):
    """alpha * dense + (1-alpha) * sparse"""
    return alpha * normalize(dense_scores) + (1-alpha) * normalize(sparse_scores)
```

### 5.3 向量数据库选型

| 数据库 | Hybrid | 优势 | 适合 |
|--------|--------|------|------|
| **Qdrant** | ✅ 原生 | 性能好、过滤强 | 生产首选 |
| **Milvus/Zilliz** | ✅ 原生 | 大规模、分布式 | 企业级 |
| **Weaviate** | ✅ 原生 | GraphQL、多模态 | 灵活场景 |
| **pgvector** | ⚠️ 需扩展 | 已有 PG 生态 | 小规模 |
| **Pinecone** | ✅ 托管 | 零运维 | 快速上手 |
| **Chroma** | ❌ | 最简单 | 原型/实验 |

## 6. 高级 RAG 技术

### 6.1 Query Transformation

```python
# HyDE (Hypothetical Document Embedding)
# 让 LLM 先生成一个假设的回答，再用假设回答做检索
def hyde_retrieve(query, llm, retriever):
    hypothetical_doc = llm.generate(f"请回答：{query}")
    embedding = embed(hypothetical_doc)  # 用假设回答做检索
    return retriever.search(embedding)

# Multi-Query: 将一个查询改写为多个角度
queries = llm.generate(f"将以下问题改写为3个不同角度的查询：{query}")
results = [retriever.search(q) for q in queries]
final = deduplicate_and_merge(results)
```

### 6.2 Self-RAG & Corrective RAG

```
Self-RAG: 模型自己决定是否需要检索
  Query → LLM: "需要检索吗？" → Yes/No
    → Yes → Retrieve → LLM: "检索结果相关吗？" → Filter
    → No → 直接回答

CRAG (Corrective RAG):
  Query → Retrieve → Evaluator: "结果可靠吗？"
    → Correct: 使用检索结果
    → Ambiguous: 补充搜索
    → Incorrect: 丢弃，用 web search 替代
```

## 7. 实际踩坑经验

### 7.1 常见问题清单

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| Chunk 切断关键信息 | 回答不完整 | 增加 overlap、用语义切分 |
| 检索不相关 | 答非所问 | 加 Reranker、HyDE、调 chunk_size |
| 长文档 lost in middle | 中间内容被忽略 | 控制 context 长度、chunk 重排 |
| 精确匹配失败 | 搜不到 ID/名词 | 加 BM25 hybrid search |
| 幻觉 | 编造不存在的信息 | 加 citation、降温度、Few-shot |
| 多语言混合 | 跨语言检索差 | 用多语言 embedding (bge-m3) |
| 更新滞后 | 旧数据覆盖新数据 | 增量索引 + 时间衰减 |

### 7.2 调试流程

```python
# RAG 调试步骤
def debug_rag(query, pipeline):
    # 1. 检查检索结果是否包含答案
    retrieved = pipeline.retrieve(query, top_k=20)
    print("检索到的文档:", [d.text[:100] for d in retrieved])
    # → 如果答案不在检索结果中 → 优化 Chunking/Embedding/检索策略
    
    # 2. 检查 Reranker 后的排序
    reranked = pipeline.rerank(query, retrieved)
    print("重排后 Top-3:", [d.text[:100] for d in reranked[:3]])
    # → 如果答案在检索结果中但排名靠后 → 优化 Reranker
    
    # 3. 检查 LLM 生成
    response = pipeline.generate(query, reranked[:5])
    print("生成结果:", response)
    # → 如果上下文正确但生成错误 → 优化 Prompt/LLM
```

## 8. 面试常见问题

**Q1: RAG 和长上下文模型（如 128K/1M 上下文窗口）哪个更好？**
A: 不冲突。长上下文解决的是"能放多少"，RAG 解决的是"找什么放"。百万文档不可能全塞进上下文。最佳实践是 RAG 检索 + 长上下文承载更多 context。

**Q2: Chunk size 如何选择？**
A: 没有银弹。经验法则：问答 256-512，摘要 512-1024，代码按函数切。最终靠评估数据集 A/B 测试决定。

**Q3: 向量检索和 BM25 各自什么时候更好？**
A: 语义匹配（同义词、改写）→ 向量检索更好。精确匹配（关键词、人名、ID）→ BM25 更好。生产中用 Hybrid 兼顾。

**Q4: Reranker 放在哪个位置？为什么不直接用 Cross-Encoder 做检索？**
A: Cross-Encoder 需要每个 query-doc 对都做一次 forward，百万文档就要百万次 → 不现实。所以先用 Bi-Encoder 从百万文档筛到百条，再用 Cross-Encoder 精排。

**Q5: 如何评估 RAG 系统的效果？**
A: 分层评估：1) 检索质量（Recall@K, MRR），2) Reranker 质量（NDCG），3) 端到端生成质量（faithfulness, relevance）。工具：RAGAS、DeepEval、LlamaIndex evaluators。

## 相关链接

- [[AI/3-LLM/Application/Embedding/Embedding|Embedding]] — 向量化基础
- [[AI/3-LLM/Application/Embedding/Embedding 选型|Embedding 选型]] — 详细选型对比
- [[vLLM|vLLM]] — RAG 的 LLM 推理后端
- [[AI/3-LLM/Application/Prompt-Engineering-基础|Prompt Engineering]] — RAG prompt 设计
