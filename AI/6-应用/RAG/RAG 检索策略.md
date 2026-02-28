---
brief: "RAG 检索策略详解——Sparse（BM25）vs Dense（Embedding）vs Hybrid 三类检索策略对比；QueryExpansion/HyDE/Step-back Prompting 的 query 变换技巧；Interview 标注，RAG 检索优化的完整方法论。"
title: "RAG 检索策略详解"
date: 2026-02-14
tags: [rag, retrieval, embedding, reranker, interview]
type: note
---

# RAG 检索策略详解

## 1. 稠密检索（Dense Retrieval）

### 核心思想

用神经网络将 query 和 document 分别编码为高维向量（embedding），通过向量相似度（余弦相似度 / 内积）来衡量语义相关性。核心优势是能捕捉**语义相似性**——"汽车"和"轿车"在向量空间中距离很近，而传统关键词匹配完全无法处理。

### 双塔模型（Bi-Encoder）

```
Query: "如何提高RAG的检索质量"     Document: "改善检索增强生成效果的方法..."
         ↓                                    ↓
    Query Encoder                         Doc Encoder
         ↓                                    ↓
    q = [0.12, -0.34, ...]              d = [0.15, -0.31, ...]
                    ↓
              sim(q, d) = cosine(q, d)
```

**关键特性**：
- Query 和 Document 的编码**完全独立**，Document 可以离线预计算
- 检索时只需对 Query 编码一次，然后在向量库中做 ANN 搜索
- 这也是它的局限——query 和 document 之间没有交互，语义理解不如 Cross-encoder

### 主流 Embedding 模型

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| **BGE 系列** (BAAI) | 768/1024 | 中英文双语强，开源 SOTA 级别 | 通用中文 RAG |
| **E5** (Microsoft) | 768/1024 | instruction-tuned，加前缀 "query:" / "passage:" | 通用英文 |
| **GTE** (Alibaba) | 768/1024 | 多语言，长文本支持好 | 多语言场景 |
| **Jina Embeddings** | 768 | 支持 8K 长文本 | 长文档检索 |
| **OpenAI text-embedding-3** | 256-3072 | 维度可调，API 调用 | 快速接入 |
| **Cohere embed-v3** | 1024 | 支持压缩到 int8/binary | 大规模部署 |

### 实践要点

- **训练数据对齐**：选择与目标领域相近的预训练模型，或做 fine-tune
- **归一化**：大多数模型输出需要 L2 归一化后再算余弦相似度
- **维度权衡**：高维度精度更高但存储和计算成本也更高；OpenAI 和 Cohere 支持 Matryoshka（嵌套维度）
- **前缀指令**：部分模型需要区分 query/document 前缀（如 E5: `"query: ..."` / `"passage: ..."`）

---

## 2. 稀疏检索（Sparse Retrieval）

### BM25

经典的词频统计方法，至今仍是强 baseline：

```
BM25(q, d) = Σ IDF(t) × [TF(t,d) × (k1 + 1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

- **TF**：词在文档中出现的频率（有饱和效应，k1 控制）
- **IDF**：逆文档频率，罕见词权重更高
- **b**：文档长度归一化参数（通常 0.75）
- **k1**：TF 饱和参数（通常 1.2-2.0）

**BM25 的优势**：
- 精确关键词匹配，对专有名词、编号、代码片段效果好
- 无需 GPU，速度极快，Elasticsearch/Lucene 原生支持
- 不需要训练，零样本就能用

### TF-IDF

BM25 的前身，区别在于 BM25 有词频饱和和文档长度归一化。TF-IDF 现在主要用于特征工程和文本分析，检索场景基本被 BM25 替代。

### SPLADE

**Sparse Lexical and Expansion Model** — 用 BERT 预测每个词的权重，并进行词汇扩展：

```
Input: "如何训练大模型"
SPLADE output: {训练: 2.1, 大模型: 1.8, fine-tune: 1.5, LLM: 1.3, 预训练: 0.9, ...}
```

- 保持稀疏表示（可用倒排索引），但能做语义扩展
- "训练大模型" 会自动扩展出 "fine-tune"、"预训练" 等相关词
- 精度介于 BM25 和 Dense 之间，推理效率接近 BM25

---

## 3. 混合检索（Hybrid Retrieval）

### 为什么要混合

- **稠密检索**擅长语义匹配，但对精确关键词（产品编号、专有名词）可能漏检
- **稀疏检索**擅长精确匹配，但无法理解同义词和语义关系
- 混合两者可以取长补短，几乎所有生产级 RAG 系统都用混合检索

### RRF（Reciprocal Rank Fusion）

最常用的无参数融合方法：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

其中 `k` 通常取 60，`rank_i(d)` 是文档 d 在第 i 个检索器中的排名。

**优点**：不需要分数归一化，不需要训练，直接基于排名融合
**实现**：

```python
def rrf_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 线性加权融合

```
hybrid_score(d) = α × dense_score(d) + (1 - α) × sparse_score(d)
```

**注意**：需要对两个分数做归一化（min-max 或 z-score），否则尺度不同无法直接相加。α 通常在 0.5-0.7 之间，偏向稠密检索。

### 向量数据库的混合检索支持

- **Milvus**：原生支持 hybrid search（dense + sparse）
- **Qdrant**：支持多向量 + keyword filter
- **Weaviate**：内置 BM25 + vector hybrid
- **Elasticsearch 8.x**：dense_vector 字段 + 传统 BM25 原生融合

---

## 4. HyDE（Hypothetical Document Embeddings）

### 核心思想

Gao et al., 2022 — 用户的 query 通常很短且表达模糊，直接编码效果差。HyDE 的做法是：先用 LLM 生成一个"假设性回答文档"，再用这个假文档的 embedding 去检索。

### 流程

```
用户 Query: "量子计算的优势是什么"
         ↓
LLM 生成假文档: "量子计算利用量子叠加和纠缠态，能在特定问题上
实现指数级加速。例如在密码破解领域，Shor 算法可以在多项式时间
内分解大整数，而经典计算机需要指数时间..."
         ↓
Embedding(假文档) → 向量
         ↓
用此向量检索真实文档库
```

### 为什么有效

- **Query-Document 分布对齐**：假文档的 embedding 更接近真实文档的分布，而短 query 的 embedding 往往偏离
- **语义扩展**：LLM 会自动补充相关概念和术语
- 即使假文档有事实错误也没关系——重要的是它的语义方向对

### 注意事项

- 增加一次 LLM 调用的延迟和成本
- 对简单/精确查询可能反而引入噪声（如搜索特定编号）
- 可以与混合检索结合：HyDE 做 dense 部分，BM25 做 sparse 部分

---

## 5. Query Expansion / Rewriting

### 多查询生成（Multi-Query）

用 LLM 从不同角度重写原始查询，每个查询分别检索，合并去重：

```
原始 Query: "如何优化 RAG 系统的效果"
        ↓ LLM 改写
Query 1: "RAG 系统性能优化最佳实践"
Query 2: "提高检索增强生成准确率的方法"  
Query 3: "RAG pipeline 的常见瓶颈和解决方案"
        ↓
分别检索 → 结果合并（RRF 融合）
```

**LlamaIndex 实现**：`MultiQueryRetriever`
**LangChain 实现**：`MultiQueryRetriever`

### 子查询分解（Sub-Question Decomposition）

将复杂问题拆分为多个独立的子问题：

```
原始 Query: "比较 PyTorch 和 TensorFlow 在工业部署场景的优劣"
        ↓ 分解
Sub-Q 1: "PyTorch 在工业部署中的优势"
Sub-Q 2: "TensorFlow 在工业部署中的优势"
Sub-Q 3: "PyTorch 和 TensorFlow 的部署工具链对比"
        ↓
分别检索 → 分别回答 → 综合合成最终答案
```

### Step-Back Prompting

让 LLM 先提出一个更抽象、更高层的问题，用这个问题检索到更通用的背景知识：

```
原始 Query: "为什么我的 BERT fine-tune 在验证集上 loss 震荡"
Step-back: "BERT fine-tuning 中常见的训练不稳定问题及原因"
```

---

## 6. Reranker 重排序

### 为什么需要 Reranker

初始检索（BM25/Dense）追求**召回率**，从百万文档中快速选出 top-k（如 50-100 个）。但这些结果的排序不够精确。Reranker 的作用是对这些候选进行**精排**，大幅提升精确率。

### Cross-encoder vs Bi-encoder

```
Bi-encoder（初始检索用）：           Cross-encoder（Reranker）：
Query → Encoder → q_vec              [CLS] Query [SEP] Document [SEP]
Doc   → Encoder → d_vec                          ↓
sim = cosine(q_vec, d_vec)                      BERT
                                                  ↓
                                          relevance_score
```

| 维度 | Bi-encoder | Cross-encoder |
|------|-----------|---------------|
| 速度 | 快（独立编码，ANN 检索） | 慢（需要 query-doc 拼接过模型） |
| 精度 | 较低（无交互） | 高（query-doc 深度交互） |
| 用途 | 初始检索/召回 | 重排序/精排 |
| 规模 | 可处理百万级 | 通常只处理 top 50-100 |

### 主流 Reranker

| 模型 | 特点 |
|------|------|
| **BGE-Reranker-v2** (BAAI) | 开源最强之一，中英文支持好 |
| **Cohere Rerank** | API 服务，效果稳定，支持多语言 |
| **Jina Reranker** | 开源，支持长文本 |
| **bge-reranker-v2.5-gemma2** | 基于 Gemma2 的轻量级 reranker |
| **RankGPT** | 用 GPT 做 listwise 排序 |

### 实践建议

- **两阶段 pipeline**：Retriever（top 50-100）→ Reranker（重排，取 top 5-10）→ LLM
- **Reranker 对最终效果的提升通常 5-15%**，性价比非常高
- 注意 Reranker 的延迟：Cross-encoder 对每个候选都要过一次模型
- 可以用**蒸馏**（distillation）将 Cross-encoder 的知识蒸馏到 Bi-encoder

---

## 7. 文本分块策略（Chunking）

### 为什么分块很重要

- LLM 上下文窗口有限，不能把整篇文档塞进去
- 太大的 chunk → 引入噪声，稀释相关信息
- 太小的 chunk → 丢失上下文，语义不完整
- 分块质量直接决定了检索质量和最终回答质量

### Fixed-size Chunking

```python
# 按固定字符数/token数切分
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
```

- `chunk_size`：通常 256-1024 tokens
- `overlap`：重叠部分，通常 chunk_size 的 10-20%，防止语义被截断
- **优点**：简单、可预测
- **缺点**：可能从句子中间切断

### Recursive Character Splitting

LangChain 默认策略。按优先级尝试不同分隔符切分：

```python
separators = ["\n\n", "\n", ". ", " ", ""]
# 先按段落分，段落太长按换行分，再按句子分...
```

- 尽量保持语义完整性
- 在段落/句子边界切分
- 实践中最常用的通用策略

### Semantic Chunking

根据语义相似度决定切分点：

```
[sentence 1] ←→ [sentence 2] sim=0.92  → 不切
[sentence 2] ←→ [sentence 3] sim=0.45  → 切！语义跳变
[sentence 3] ←→ [sentence 4] sim=0.88  → 不切
```

- 计算相邻句子的 embedding 相似度
- 相似度骤降处作为切分点
- **优点**：最尊重语义边界
- **缺点**：需要额外 embedding 计算，chunk 大小不可控

### Parent-Child Chunks（层级分块）

```
Parent Chunk（大块，如 2000 tokens）
├── Child Chunk 1（小块，如 256 tokens） ← 用于检索
├── Child Chunk 2 ← 命中这个
└── Child Chunk 3

检索时匹配 Child，返回时返回 Parent（提供完整上下文）
```

- **检索粒度细**（小 chunk 精确匹配）+ **上下文完整**（返回大 chunk）
- LlamaIndex 的 `SentenceWindowNodeParser` 和 `HierarchicalNodeParser` 原生支持
- 实际效果非常好，推荐作为默认策略

### 分块参数经验值

| 场景 | chunk_size | overlap | 策略 |
|------|-----------|---------|------|
| 通用问答 | 512 tokens | 50-100 | Recursive |
| 法律/合同 | 1024 tokens | 200 | Recursive + 按条款结构 |
| 代码 | 按函数/类 | 0 | AST-based splitting |
| 对话记录 | 按轮次 | 1-2 轮 | 自定义 |

---

## 8. 向量数据库选型简评

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **FAISS** (Meta) | 库 | 纯 C++/Python 库，性能极高，无服务端 | 实验/小规模/嵌入式 |
| **Milvus** | 分布式服务 | 亿级向量，支持混合检索，云服务 Zilliz | 大规模生产部署 |
| **Qdrant** | 服务 | Rust 实现，payload filter 强，API 友好 | 中等规模，需要复杂过滤 |
| **Chroma** | 嵌入式 | Python-native，API 极简，开发体验好 | 原型开发、小项目 |
| **Pinecone** | 全托管 SaaS | 零运维，自动扩缩容 | 不想管基础设施的团队 |
| **Weaviate** | 服务 | 内置多模态、BM25+Vector hybrid | 需要混合检索原生支持 |
| **pgvector** | PG 扩展 | 在现有 PostgreSQL 上加向量能力 | 已有 PG 基础设施 |

### 选型决策

```
数据量 < 100万 + 不想部署服务  → FAISS / Chroma
数据量 < 100万 + 需要过滤/持久化 → Qdrant / Weaviate
数据量 > 1000万 + 生产环境     → Milvus / Pinecone
已有 PostgreSQL 不想引入新组件  → pgvector
预算充足 + 不想运维            → Pinecone
```

---

## 9. 面试常见问题及回答要点

### Q1: 稠密检索和稀疏检索各有什么优缺点？什么时候该用混合检索？

**回答要点**：
- **稠密检索优点**：语义理解强，能匹配同义词和近义表达
- **稠密检索缺点**：对精确关键词（型号、编号）可能漏召回；需要 GPU 编码；对 out-of-domain 数据泛化差
- **稀疏检索优点**：精确匹配强；无需 GPU；BM25 零样本效果好
- **稀疏检索缺点**：无法理解语义相似性；对同义词、改述无能为力
- **混合检索**：几乎所有生产系统都应该用，RRF 融合简单有效，稀疏补精确匹配，稠密补语义理解
- 用 α 参数调节两者权重，一般偏向稠密（0.6-0.7）

### Q2: 请解释 HyDE 的原理，它有什么局限性？

**回答要点**：
- **原理**：用 LLM 先生成一个假设性回答文档，用假文档的 embedding 去检索（因为假文档更接近真实文档的分布）
- **优势**：缩小 query-document 分布差距，尤其对短 query 效果好
- **局限性**：
  - 增加一次 LLM 调用的延迟和成本
  - 对精确查询（搜索特定 ID、编号）可能引入噪声
  - LLM 生成的假文档如果方向错误，会把检索带偏
  - 对于非常专业的领域，LLM 可能生成质量差的假文档
- **改进**：可以生成多个假文档取平均 embedding，或与 BM25 混合

### Q3: Reranker 在 RAG 中的作用是什么？为什么不直接用 Cross-encoder 做检索？

**回答要点**：
- **作用**：对初始检索的 top-k 结果做精排，提升排序精度
- **为什么不直接用 Cross-encoder 检索**：Cross-encoder 需要将 query 和每个文档拼接后过模型，百万级文档逐一比对的计算量不可接受（O(N) 次模型推理 vs Bi-encoder 的 O(1) 向量检索）
- **两阶段 pipeline 的必要性**：Bi-encoder 快速召回 → Cross-encoder 精排，兼顾效率和精度
- **实际效果**：Reranker 通常带来 5-15% 的指标提升，是 ROI 最高的优化手段之一
- **替代方案**：RankGPT（用 LLM 做 listwise 排序）、ColBERT（late interaction，介于 Bi-encoder 和 Cross-encoder 之间）

### Q4: 文本分块对 RAG 效果影响大吗？你会怎么选择分块策略？

**回答要点**：
- **影响非常大**，分块是 RAG 中最被低估的环节
- **分块太大**：引入无关信息，稀释关键内容，LLM 可能忽略重要部分
- **分块太小**：丢失上下文，语义不完整，检索结果碎片化
- **推荐策略**：
  - 通用场景用 Recursive Character Splitting（512 tokens + 10% overlap）
  - 进阶用 Parent-Child chunks（小块检索，大块返回）
  - 结构化文档（法律、论文）按文档结构切分（标题、章节）
  - 代码用 AST-based splitting
- **评估方法**：用 hit rate 和 MRR 评估不同分块策略的检索质量

### Q5: 如何评估和优化一个 RAG 系统的检索质量？

**回答要点**：
- **检索指标**：
  - Hit Rate / Recall@k：top-k 结果中是否包含正确答案
  - MRR（Mean Reciprocal Rank）：正确答案的排名倒数
  - NDCG@k：考虑相关性等级的排序质量
- **端到端指标**：
  - Faithfulness：回答是否忠于检索到的上下文
  - Answer Relevancy：回答是否切题
  - Context Relevancy：检索到的上下文是否与问题相关
- **评估框架**：RAGAS、TruLens、LlamaIndex Evaluation
- **优化顺序**（ROI 从高到低）：
  1. 分块策略调优
  2. 加 Reranker
  3. 混合检索
  4. Query Rewriting
  5. Embedding 模型微调
  6. 数据清洗和预处理

### Q6: 如果用户的查询很模糊或者很复杂，你会怎么处理？

**回答要点**：
- **模糊查询**：
  - Query Rewriting：用 LLM 改写为更清晰的表述
  - HyDE：生成假设文档扩展语义
  - 多轮对话澄清：让系统反问用户以明确意图
- **复杂查询**：
  - Sub-Question Decomposition：拆分为多个子问题分别检索
  - Multi-Query：从不同角度改写查询，扩大召回范围
  - Step-Back Prompting：先检索背景知识，再处理具体问题
- **实际 pipeline 设计**：
  - 先做意图分类（简单/模糊/复杂）
  - 简单查询直接检索
  - 模糊查询走 Rewrite + HyDE
  - 复杂查询走 Decomposition
- **关键原则**：检索质量 > 生成质量，garbage in garbage out

## See Also

-  — 检索增强生成全景索引
- [[AI/6-应用/RAG/Advanced RAG|Advanced RAG]] — 本文检索策略的进阶应用
- [[AI/6-应用/RAG/向量数据库选型|向量数据库选型]] — 实现稠密检索的基础设施
- RAG 2026 技术全景 — 宏观综述，包含检索策略在完整系统中的位置
- [[AI/3-LLM/Inference/量化综述|模型量化综述]] — 向量检索中 embedding 压缩的相关技术
