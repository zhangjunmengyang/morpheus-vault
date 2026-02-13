---
title: "Embedding 与向量检索：从模型选型到工程落地"
date: 2026-02-13
tags:
  - ai/llm/application
  - ai/embedding
  - ai/vector-search
  - ai/rag
  - type/practice
  - interview/hot
status: active
---

# Embedding 与向量检索：从模型选型到工程落地

> 向量检索是 RAG 的基石——选对 Embedding 模型、用对 ANN 算法、配好向量数据库，决定了检索质量的上限

## 1. Embedding 模型基础

### 什么是文本 Embedding？

```
文本 → Embedding 模型 → 固定维度的稠密向量 ∈ R^d

"机器学习很有趣" → [0.023, -0.156, 0.089, ..., 0.045]  (d=1024)

核心性质: 语义相近的文本 → 向量空间中距离近
  sim("机器学习", "深度学习") > sim("机器学习", "煮饭技巧")
```

### 训练范式演进

```
第一代: Word2Vec / GloVe → 词级别，无上下文
第二代: BERT → 句级别 (CLS token 或 mean pooling)
第三代: 对比学习 → SimCSE, E5, BGE
第四代: 指令感知 → Instructor, E5-Mistral, BGE-M3 (2024+)

关键转折: 第三代开始使用对比学习 (contrastive learning)
  → 正例对: (query, relevant_doc)
  → 负例: in-batch negatives + hard negatives
  → Loss: InfoNCE / triplet loss
```

## 2. 主流 Embedding 模型选型

### 2024-2025 主流模型对比

```
模型              维度    最大长度  语言     特点                   MTEB 排名
─────────────────────────────────────────────────────────────────────────
BGE-M3            1024    8192     多语言   多粒度检索(dense+sparse+colbert) 顶级
BGE-large-zh-v1.5 1024    512      中文     中文最强开源              中文第一
E5-Mistral-7B     4096    32K      多语言   基于 LLM 的 embedding     顶级
GTE-Qwen2-7B      3584    32K      多语言   Qwen2 backbone            顶级
Cohere embed-v3   1024    512      多语言   商业 API，压缩支持        商业第一
OpenAI text-3-large 3072  8191     多语言   商业 API，广泛使用        商业顶级
Jina-embeddings-v3 1024   8192     多语言   支持 Matryoshka 维度      开源优秀
─────────────────────────────────────────────────────────────────────────
```

### 选型决策树

```
场景判断:
  ├─ 纯中文 → BGE-large-zh-v1.5 (性能最优)
  ├─ 中英混合 → BGE-M3 (多语言+多粒度)
  ├─ 需要超长文档 → E5-Mistral-7B / GTE-Qwen2 (32K 上下文)
  ├─ 追求极致效果 → GTE-Qwen2-7B (最强开源)
  ├─ 资源受限 → BGE-small-zh / all-MiniLM (小模型)
  └─ 商业 API → Cohere embed-v3 / OpenAI text-3-large
```

### 使用示例

```python
# BGE-M3 使用示例
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

queries = ["什么是 RAG？"]
documents = [
    "RAG 是检索增强生成，结合检索和生成",
    "今天天气很好",
    "Retrieval-Augmented Generation combines search with LLMs",
]

# 编码（同时生成 dense + sparse + colbert 表示）
q_embeddings = model.encode(queries, return_dense=True, return_sparse=True)
d_embeddings = model.encode(documents, return_dense=True, return_sparse=True)

# Dense 检索
dense_scores = q_embeddings["dense_vecs"] @ d_embeddings["dense_vecs"].T
print(dense_scores)  
# [[0.85, 0.12, 0.78]]  → 语义检索

# Sparse 检索 (类似 BM25 的稀疏匹配)
# BGE-M3 的优势: dense + sparse 混合检索，效果最佳
```

```python
# Sentence-Transformers 通用方式
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
embeddings = model.encode(["机器学习", "深度学习", "做饭"])

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
```

## 3. 距离度量

```
度量              公式                          适用场景
────────────────────────────────────────────────────────────
余弦相似度        cos(a,b) = a·b/(|a||b|)       最常用，归一化向量
点积 (Inner Product) a·b                         已归一化时 = 余弦
欧氏距离 (L2)      √Σ(aᵢ-bᵢ)²                  对绝对距离敏感
曼哈顿距离 (L1)    Σ|aᵢ-bᵢ|                     高维空间更鲁棒
────────────────────────────────────────────────────────────

实践建议:
  - 大多数 Embedding 模型输出已归一化 → 用余弦或点积
  - 未归一化 → 用余弦相似度（自动归一化）
  - 向量数据库中，归一化后用 Inner Product 最快
```

## 4. ANN 算法深度解析

精确最近邻搜索 O(N·d) 在百万级数据上太慢 → 需要 **近似最近邻 (ANN)** 算法。

### 4.1 HNSW（Hierarchical Navigable Small World）

```
核心思想: 多层跳表 + 贪心搜索

层 2 (稀疏):    A ──────────── D ──────────── G
                │                              │
层 1 (中等):    A ──── C ──── D ──── F ──── G
                │      │      │      │      │
层 0 (稠密):    A  B  C  D  E  F  G  H  I  J

搜索过程:
  1. 从最高层的入口点开始
  2. 贪心找到当前层最近的节点
  3. 下到下一层，继续贪心搜索
  4. 在第 0 层做精确的 beam search

关键参数:
  M:       每层最大连接数 (默认 16)
  ef_construction: 构建时的 beam width (默认 200)
  ef_search:       搜索时的 beam width (默认 100)
```

```
HNSW 性能特征:
  构建时间:  O(N · log(N) · M)
  搜索时间:  O(log(N) · ef)
  内存占用:  O(N · M · d)  — 较高！
  
  优点: 搜索速度快、召回率高 (>99%)
  缺点: 内存占用大（索引需全部在内存中）
  适用: 百万到千万级数据，有充足内存
```

### 4.2 IVF（Inverted File Index）

```
核心思想: 先聚类，搜索时只查相关聚类

构建:
  1. 对所有向量做 K-Means 聚类 (nlist 个聚类中心)
  2. 每个向量归入最近的聚类

搜索:
  1. 找到 query 最近的 nprobe 个聚类
  2. 只在这些聚类内做精确搜索

           ┌─ cluster_0: [v1, v5, v8, ...]
  query → ├─ cluster_1: [v2, v3, v9, ...]  ← 搜索这些
           ├─ cluster_2: [v4, v7, ...]      ← 搜索这些
           └─ cluster_3: [v6, v10, ...]

关键参数:
  nlist:  聚类数量 (推荐 √N 到 4√N)
  nprobe: 搜索时查询的聚类数 (推荐 nlist 的 5-10%)
```

### 4.3 PQ（Product Quantization）

```
核心思想: 将高维向量切分+量化，极大压缩内存

原始向量 d=128:
  [0.12, -0.34, ..., 0.56]  → 128 × 4 bytes = 512 bytes

PQ 压缩 (m=8 子空间, nbits=8):
  子空间1: [0.12,-0.34,...] → 聚类ID: 42   (1 byte)
  子空间2: [0.08, 0.91,...] → 聚类ID: 137  (1 byte)
  ...
  子空间8: [0.56,-0.11,...] → 聚类ID: 89   (1 byte)
  → 总共只需 8 bytes (压缩 64 倍!)

常用组合:
  IVF + PQ: 先粗筛聚类，再用量化向量精排
  → 十亿级数据的标准方案
```

### 算法对比

```
算法      召回率   搜索速度   内存占用    构建速度   适用规模
──────────────────────────────────────────────────────────
Flat      100%    最慢       最大       无需构建   <100K
HNSW      >99%    最快       大(内存)    中        100K-10M
IVF       95-99%  快         中         快        1M-100M
IVF+PQ    90-98%  快         最小       快        100M-10B
ScaNN     >99%    很快       中         快        Google 推荐
──────────────────────────────────────────────────────────
```

## 5. 向量数据库对比

### 主流方案（2025）

```
数据库        语言    部署模式        ANN 引擎       特色功能
──────────────────────────────────────────────────────────────────
Milvus        Go/C++  分布式/云原生    HNSW/IVF/DiskANN  最全面,GPU加速
Qdrant        Rust    单机/分布式      HNSW             Payload过滤,速度快
Chroma        Python  嵌入式/CS       HNSW (hnswlib)   最易上手,RAG首选
Weaviate      Go      分布式          HNSW             模块化,GraphQL API
Pinecone      -       全托管 SaaS     专有             零运维,企业级
Faiss         C++     Library         全部             Meta出品,最灵活
pgvector      C       PG 扩展         IVFFlat/HNSW     PostgreSQL 生态
──────────────────────────────────────────────────────────────────
```

### 选型建议

```
场景                            推荐
──────────────────────────────────────────────
快速原型/个人项目                Chroma (pip install 即用)
RAG 生产环境 (百万级)            Qdrant 或 Milvus Lite
大规模生产 (亿级)                Milvus (分布式) 或 Pinecone
已有 PostgreSQL                 pgvector (少加一个组件)
需要极致灵活性                   Faiss (Library, 自己管理)
企业级 + 零运维                  Pinecone / Zilliz Cloud
```

### Qdrant 快速示例

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 连接
client = QdrantClient(":memory:")  # 或 url="http://localhost:6333"

# 2. 创建 collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=1024,                 # BGE-M3 维度
        distance=Distance.COSINE,
    ),
)

# 3. 插入向量
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": documents[i], "source": "wiki"},
    )
    for i in range(len(documents))
]
client.upsert(collection_name="docs", points=points)

# 4. 搜索
results = client.search(
    collection_name="docs",
    query_vector=query_embedding.tolist(),
    limit=5,
    query_filter={  # 元数据过滤
        "must": [{"key": "source", "match": {"value": "wiki"}}]
    },
)
```

## 6. 维度选择与优化

```
维度的影响:
  小维度 (256-512):  存储省、搜索快、但表达能力弱
  中维度 (768-1024): 平衡选择
  大维度 (2048-4096): 表达力强、但存储和搜索成本高

Matryoshka Representation Learning (MRL):
  → 训练时让前 k 维也有意义
  → 可以截断到任意维度使用
  → 如 OpenAI text-3-large: 3072 维可截到 256 维

# MRL 截断示例
embedding = model.encode("text")        # [3072]
embedding_small = embedding[:256]       # [256], 仍然有效!
embedding_small /= np.linalg.norm(embedding_small)  # 重新归一化

存储估算:
  100 万文档 × 1024 维 × 4 bytes = ~4 GB
  100 万文档 × 256 维 × 4 bytes = ~1 GB
  → 4 倍差距，在亿级数据时很关键
```

## 7. 面试高频题

### Q1: 为什么不直接用 BERT 的 CLS token 做句子 embedding？
**答**：BERT 的 CLS token 是为 NSP 任务训练的，没有被优化来表示句子级语义相似度。直接使用 CLS token 做余弦相似度，效果甚至不如 GloVe 平均。需要额外的对比学习训练（如 SimCSE、BGE）来让模型学会将语义相近的句子映射到向量空间中的邻近位置。mean pooling（对所有 token 取平均）通常比 CLS token 效果好。

### Q2: HNSW 和 IVF 各自适用什么场景？
**答**：HNSW 适合**高召回率、低延迟**的场景（百万到千万级），代价是内存占用大（索引需全部在内存中）。IVF 适合**大规模数据**（千万到十亿级），配合 PQ 量化可以极大压缩内存，但召回率略低。实际选择：内存充足 + 数据量 < 1000 万 → HNSW；数据量 > 1000 万 or 内存受限 → IVF+PQ。多数向量数据库（Milvus、Qdrant）默认使用 HNSW。

### Q3: 混合检索（Dense + Sparse）为什么比纯 Dense 效果好？
**答**：Dense retrieval 擅长语义匹配（"汽车" 匹配 "轿车"），但对精确关键词匹配差（如型号 "RTX 4090"）。Sparse retrieval（BM25）擅长精确匹配但不理解语义。混合检索结合两者优势：(1) Dense 找到语义相关的文档；(2) Sparse 找到关键词精确匹配的文档；(3) 通过 RRF (Reciprocal Rank Fusion) 或加权融合合并排名。BGE-M3 同时输出 dense + sparse + colbert 三种表示，是混合检索的最佳选择。

### Q4: Embedding 维度如何选择？是否越大越好？
**答**：不是越大越好，存在边际递减效应。768-1024 维是多数场景的甜点。维度增大带来：(1) 存储成本线性增长（亿级数据时差别显著）；(2) 搜索延迟增加（距离计算 O(d)）；(3) 高维空间的"维度灾难"——距离区分度下降。实践建议：先用支持 Matryoshka 的模型（如 Jina v3、OpenAI text-3），用大维度评估效果基线，再逐步截断到小维度找到效果-效率的平衡点。

### Q5: 如何评估 Embedding + 检索系统的效果？
**答**：多层次评估：(1) **Embedding 质量**——用 MTEB benchmark 的 retrieval 子集（NDCG@10、MRR@10）；(2) **检索召回**——在自己的数据上标注 query-doc 相关对，计算 Recall@K（K=5/10/20）；(3) **端到端效果**——RAG 场景中，测量最终生成答案的准确率，因为检索到的文档还要经过 LLM 理解和生成；(4) **线上指标**——用户满意度、点击率、问题解决率。常见坑：MTEB 分数高不代表你的场景好——领域特定数据的分布可能与 benchmark 差异大。

---

**相关笔记**：[[RAG 工程实践]] | [[Advanced RAG]] | [[Prompt Engineering 高级]]
