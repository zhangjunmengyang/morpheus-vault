---
title: "Advanced RAG 技术全景"
date: 2026-02-13
tags:
  - ai/llm/application
  - ai/rag
  - type/concept
  - interview/hot
status: active
---

# Advanced RAG 技术全景

> HyDE、Self-RAG、RAPTOR、Corrective RAG、多跳检索——超越 Naive RAG 的进阶方案

## 1. 从 Naive RAG 到 Advanced RAG

### Naive RAG 的局限性

基础 [[RAG 工程实践|RAG]] 流程（Query → Retrieve → Generate）存在多个痛点：

```
Naive RAG 的典型失败场景:

1. 查询与文档语义不匹配
   Q: "如何提升推理速度" → 文档中用 "inference optimization" → 检索不到

2. 检索到的内容质量差
   Top-K 中混入不相关文档 → LLM 被误导 → 幻觉

3. 需要多步推理
   Q: "对比 A 公司和 B 公司的营收" → 需要检索两次 → 单次检索不够

4. 问题需要全局理解
   Q: "这些文档的共同主题是什么" → 需要 summarization → 检索单个 chunk 不够

5. LLM 不知道检索结果是否可靠
   检索到过时/错误信息 → 无验证 → 直接引用
```

### Advanced RAG 全景图

```
                        Advanced RAG 技术栈
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    Pre-Retrieval        Retrieval           Post-Retrieval
    (检索前优化)          (检索优化)          (检索后优化)
          │                   │                   │
    ├── HyDE              ├── Hybrid Search   ├── Reranking
    ├── Query Rewriting   ├── Multi-hop        ├── Self-RAG
    ├── Query Expansion   ├── RAPTOR           ├── Corrective RAG
    ├── Step-back Prompt  ├── GraphRAG         ├── Compression
    └── Sub-query Decomp  ├── Parent-Doc       └── Citation
                          └── Recursive
```

## 2. Pre-Retrieval 优化

### HyDE (Hypothetical Document Embeddings)

**核心思想**：让 LLM 先生成一个"假设性回答"，用这个回答（而非原始查询）去检索。

```
问题: "FlashAttention 怎么工作的?"

Naive RAG:
  embed("FlashAttention 怎么工作的?") → 搜索
  问题: 查询是疑问句，与文档的陈述句风格不匹配

HyDE:
  Step 1: LLM 生成假设回答:
    "FlashAttention 是一种 IO-aware 的精确注意力算法，
     通过 tiling 技术将 Q/K/V 分块加载到 SRAM 中计算..."
  
  Step 2: embed(假设回答) → 搜索
  效果: 假设回答的语言风格与文档更匹配 → 召回率提升
```

```python
# HyDE 实现
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""请写一段话回答以下问题（不需要完全正确，只需要大致正确）：

问题：{question}

回答："""
)

async def hyde_retrieve(question: str, retriever, llm):
    # Step 1: 生成假设文档
    hypothetical_doc = await llm.generate(
        hyde_prompt.format(question=question)
    )
    # Step 2: 用假设文档检索
    results = retriever.search(hypothetical_doc, top_k=5)
    return results
```

**优缺点**：
- ✅ 无需修改索引，即插即用
- ✅ 对语义不匹配的查询特别有效
- ❌ 增加一次 LLM 调用（延迟+成本）
- ❌ 假设回答本身有误时可能误导检索

### Query Decomposition（查询分解）

```python
# 将复杂查询拆分为子查询
DECOMPOSE_PROMPT = """
将以下复杂问题拆解为 2-4 个简单子问题：

问题: {question}

子问题:
1. 
2. 
"""

async def decompose_and_retrieve(question, retriever, llm):
    # 分解
    sub_questions = await llm.generate(
        DECOMPOSE_PROMPT.format(question=question)
    )

    # 分别检索
    all_docs = []
    for sub_q in parse_sub_questions(sub_questions):
        docs = retriever.search(sub_q, top_k=3)
        all_docs.extend(docs)

    # 去重 + 合并
    unique_docs = deduplicate(all_docs)
    return unique_docs

# 示例:
# Q: "对比 FlashAttention v2 和 v3 在 H100 上的性能差异"
# Sub-Q1: "FlashAttention v2 的性能特点"
# Sub-Q2: "FlashAttention v3 在 H100 上的优化"
# Sub-Q3: "FlashAttention v2 vs v3 benchmark 数据"
```

### Step-Back Prompting

```
原始查询: "LLaMA 3 的 GQA 配置用了多少个 KV heads?"
  → 检索精确细节时可能检索不到

Step-back 查询: "LLaMA 3 的架构设计有哪些关键改进?"
  → 更宽泛，更容易命中相关文档
  → 从检索结果中提取具体细节
```

## 3. Self-RAG (Self-Reflective RAG)

### 核心思想

训练/提示 LLM 在生成过程中 **自我评估**：是否需要检索、检索结果是否相关、生成的回答是否被检索内容支持。

### 流程

```
Self-RAG 决策流:

输入查询 → [Retrieve?] ─── No ──→ 直接生成回答
                │
               Yes
                │
        检索 Top-K 文档
                │
        [IsRelevant?] ── No ──→ 丢弃，重新检索或直接生成
                │
               Yes
                │
        生成回答 (带引用)
                │
        [IsSupported?] ── No ──→ 重新生成 / 检索更多
                │
               Yes
                │
        [IsUseful?] ── No ──→ 重新生成
                │
               Yes
                │
        返回回答
```

### 实现

```python
# Self-RAG 实现（使用 LLM 做自我评估）
class SelfRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    async def generate(self, query: str) -> str:
        # Step 1: 判断是否需要检索
        need_retrieval = await self.llm.generate(f"""
判断以下问题是否需要外部知识来回答。
输出 [Retrieve] 或 [No Retrieve]。

问题: {query}
""")

        if "[No Retrieve]" in need_retrieval:
            return await self.llm.generate(query)

        # Step 2: 检索
        docs = self.retriever.search(query, top_k=5)

        # Step 3: 评估相关性 (批量)
        relevant_docs = []
        for doc in docs:
            relevance = await self.llm.generate(f"""
判断以下文档是否与问题相关。输出 [Relevant] 或 [Irrelevant]。
问题: {query}
文档: {doc.text[:500]}
""")
            if "[Relevant]" in relevance:
                relevant_docs.append(doc)

        if not relevant_docs:
            return await self.llm.generate(query)  # fallback

        # Step 4: 生成回答
        context = "\n".join([d.text for d in relevant_docs])
        answer = await self.llm.generate(f"""
基于以下上下文回答问题。如果上下文不足以回答，请说明。

上下文:
{context}

问题: {query}
""")

        # Step 5: 验证回答是否被上下文支持
        supported = await self.llm.generate(f"""
判断回答是否被上下文支持。输出 [Supported] 或 [Not Supported]。
上下文: {context[:1000]}
回答: {answer}
""")

        if "[Not Supported]" in supported:
            # 重新生成，强调只用上下文中的信息
            answer = await self.regenerate_with_citation(query, relevant_docs)

        return answer
```

## 4. Corrective RAG (CRAG)

### 核心思想

在生成前 **评估检索文档的质量**，根据评估结果决定后续动作：

```
Corrective RAG 流程:

检索 Top-K 文档
      │
  评估每篇文档质量
      │
  ┌───┴───────────────────────────┐
  │                               │
  ├─ CORRECT (相关)   → 提取关键信息，Knowledge Refinement
  │
  ├─ AMBIGUOUS (模糊) → 补充检索 (Web Search 兜底)
  │
  └─ INCORRECT (不相关) → 丢弃，用 Web Search 补充
      │
  合并所有可用知识
      │
  生成最终回答
```

### 实现

```python
from enum import Enum

class DocRelevance(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"  
    INCORRECT = "incorrect"

class CorrectiveRAG:
    def __init__(self, llm, retriever, web_search):
        self.llm = llm
        self.retriever = retriever
        self.web_search = web_search

    async def generate(self, query: str) -> str:
        # Step 1: 初始检索
        docs = self.retriever.search(query, top_k=5)

        # Step 2: 评估每篇文档
        refined_knowledge = []
        need_web_search = False

        for doc in docs:
            relevance = await self.evaluate_relevance(query, doc)

            if relevance == DocRelevance.CORRECT:
                # 提取关键信息 (knowledge refinement)
                key_info = await self.extract_key_info(query, doc)
                refined_knowledge.append(key_info)
            elif relevance == DocRelevance.AMBIGUOUS:
                need_web_search = True
                key_info = await self.extract_key_info(query, doc)
                refined_knowledge.append(key_info)
            else:  # INCORRECT
                need_web_search = True

        # Step 3: 如果需要，用 Web Search 补充
        if need_web_search:
            web_results = await self.web_search.search(query)
            refined_knowledge.extend(web_results)

        # Step 4: 生成回答
        context = "\n".join(refined_knowledge)
        return await self.llm.generate(
            f"基于以下知识回答问题:\n{context}\n\n问题: {query}"
        )

    async def evaluate_relevance(self, query, doc) -> DocRelevance:
        result = await self.llm.generate(f"""
评估文档与问题的相关性。输出: correct / ambiguous / incorrect

问题: {query}
文档: {doc.text[:500]}

评估:""")
        if "correct" in result.lower():
            return DocRelevance.CORRECT
        elif "ambiguous" in result.lower():
            return DocRelevance.AMBIGUOUS
        return DocRelevance.INCORRECT
```

## 5. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

### 核心思想

构建 **多层摘要树**：底层是原始 chunks，上层是 chunk 组的摘要，再上层是摘要的摘要。查询时从不同层检索，同时获得细节和全局信息。

```
RAPTOR 索引结构:

Level 3:    [全局摘要]
              /        \
Level 2:  [摘要A]    [摘要B]
           / \        / \
Level 1: [C1] [C2]  [C3] [C4]    ← 聚类后的 chunk 组摘要
          |    |     |    |
Level 0: c1  c2    c3   c4   c5  c6  c7  c8   ← 原始 chunks

构建过程:
1. 文档 → chunks (Level 0)
2. 对 chunk embeddings 做聚类 (GMM/K-Means)
3. 对每个 cluster 生成摘要 → Level 1
4. 递归重复: 对 Level 1 聚类 + 摘要 → Level 2
5. 直到只剩一个节点 → 根摘要

检索方式:
├── Tree traversal: 从根往下搜索
└── Collapsed tree: 把所有层 flatten，一起做 similarity search (推荐)
```

```python
# RAPTOR 简化实现
from sklearn.mixture import GaussianMixture

class RAPTORIndex:
    def __init__(self, llm, embedder, max_levels=3):
        self.llm = llm
        self.embedder = embedder
        self.max_levels = max_levels
        self.all_nodes = []  # 所有层级的节点

    def build(self, chunks: list[str]):
        current_level = chunks
        self.all_nodes.extend(chunks)

        for level in range(self.max_levels):
            if len(current_level) <= 1:
                break

            # 1. Embedding + 聚类
            embeddings = self.embedder.encode(current_level)
            n_clusters = max(1, len(current_level) // 5)
            gmm = GaussianMixture(n_components=n_clusters)
            labels = gmm.fit_predict(embeddings)

            # 2. 每个 cluster 生成摘要
            summaries = []
            for cluster_id in range(n_clusters):
                cluster_texts = [
                    current_level[i]
                    for i in range(len(current_level))
                    if labels[i] == cluster_id
                ]
                summary = self.llm.generate(
                    f"请综合摘要以下内容:\n{'---'.join(cluster_texts)}"
                )
                summaries.append(summary)

            self.all_nodes.extend(summaries)
            current_level = summaries

    def search(self, query: str, top_k=5):
        # Collapsed tree search: 在所有层级中检索
        query_emb = self.embedder.encode(query)
        all_embs = self.embedder.encode(self.all_nodes)
        similarities = query_emb @ all_embs.T
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.all_nodes[i] for i in top_indices]
```

**优势**：
- 细节查询 → 命中底层 chunks
- 全局查询 → 命中高层摘要
- "这些文档的共同主题是什么?" → 根摘要直接回答

## 6. Multi-Hop Retrieval（多跳检索）

### 核心思想

一些问题需要 **多步检索**，每步检索基于前一步的结果：

```python
# 多跳检索: IRCoT (Interleaving Retrieval with Chain-of-Thought)
class MultiHopRAG:
    def __init__(self, llm, retriever, max_hops=3):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops

    async def generate(self, query: str) -> str:
        collected_context = []
        current_query = query

        for hop in range(self.max_hops):
            # 1. 检索
            docs = self.retriever.search(current_query, top_k=3)
            collected_context.extend(docs)

            # 2. 用 CoT 推理下一步需要什么信息
            cot = await self.llm.generate(f"""
已有信息:
{self.format_context(collected_context)}

原始问题: {query}
当前思考步骤: {hop + 1}

思考: 基于已有信息，回答原始问题还需要什么信息？
如果信息已充足，输出 [SUFFICIENT]。
否则，输出需要搜索的下一个查询。
""")

            if "[SUFFICIENT]" in cot:
                break

            # 3. 更新查询
            current_query = extract_next_query(cot)

        # 最终生成
        return await self.llm.generate(f"""
基于以下所有收集的信息回答问题:

{self.format_context(collected_context)}

问题: {query}
""")

# 示例:
# Q: "FlashAttention 的作者现在在哪家公司？该公司的估值是多少？"
# Hop 1: 搜索 "FlashAttention author" → Tri Dao
# Hop 2: 搜索 "Tri Dao company" → Together AI
# Hop 3: 搜索 "Together AI valuation" → $X billion
```

## 7. 技术对比总结

| 技术 | 解决问题 | 额外延迟 | 实现复杂度 | 适用场景 |
|------|---------|---------|-----------|---------|
| **HyDE** | 查询-文档语义鸿沟 | +1 LLM call | 低 | 语义搜索质量差时 |
| **Self-RAG** | 检索/生成质量不可控 | +2-3 LLM calls | 中 | 高质量要求的场景 |
| **CRAG** | 检索结果不可靠 | +1-2 LLM calls | 中 | 数据质量参差不齐 |
| **RAPTOR** | 全局理解能力弱 | 索引构建慢 | 高 | 文档摘要、全局 QA |
| **Multi-Hop** | 复杂多步推理 | +N LLM calls | 中高 | 比较、因果分析 |
| **Query Decomp** | 复杂查询 | +1 LLM call | 低 | 多条件查询 |

## 8. 与其他主题的关系

- **[[RAG 工程实践]]**：本文是 RAG 工程实践的进阶，Naive RAG 的基础知识参见该文
- **[[Embedding]]**：检索质量依赖 Embedding 模型，Advanced RAG 通过查询变换减少对 Embedding 的依赖
- **[[FlashAttention]]**：长上下文 Attention 优化使 RAG 可以传入更多检索结果
- **[[LLMOps]]**：Advanced RAG 的监控和评估是 LLMOps 的重要组成部分
- **[[Prompt Engineering 高级]]**：Step-back prompting、CoT 等 prompt 技术在 Advanced RAG 中广泛使用

## 面试常见问题

### Q1: Naive RAG 有哪些局限性？Advanced RAG 如何解决？

四大局限：(1) **语义鸿沟**——查询和文档用词不同导致检索失败 → HyDE/Query Rewriting 解决；(2) **检索质量不可控**——无法判断检索到的内容是否相关 → Self-RAG/CRAG 加入自我评估；(3) **单步检索不够**——复杂问题需要多步推理 → Multi-Hop Retrieval；(4) **缺乏全局视角**——只能检索局部 chunk，无法回答全局问题 → RAPTOR 的多层摘要树。

### Q2: HyDE 的原理是什么？什么时候用它？

HyDE 让 LLM 先对查询生成一个假设性回答，再用该回答的 embedding 去检索（而非用原始查询）。原理是假设回答的语言风格与文档更接近，缩小了查询与文档之间的语义鸿沟。适用场景：当原始查询是简短的疑问句、与文档的陈述式风格差异大时效果明显。注意：增加一次 LLM 调用的延迟和成本，且假设回答本身不正确时可能误导检索。

### Q3: Self-RAG 和 Corrective RAG 有什么区别？

**Self-RAG** 强调全流程自我反思——决定是否检索、评估文档相关性、验证回答是否被支持。它是一个端到端的框架（原论文通过特殊 token 训练模型）。**CRAG** 聚焦在检索后的文档质量评估，将文档分为 correct/ambiguous/incorrect 三类，并用 Web Search 作为兜底。实践中两者思想可以结合：先用 CRAG 思想评估文档质量，再用 Self-RAG 思想验证生成结果。

### Q4: RAPTOR 适合什么场景？如何构建索引？

RAPTOR 适合需要 **多粒度理解** 的场景：既有细节查询（"第三章提到的具体数字是什么"），也有全局查询（"这本书的核心论点是什么"）。构建过程：(1) 文档 → chunks；(2) 对 chunk embeddings 聚类（GMM）；(3) 每个 cluster 生成摘要；(4) 递归重复直到根节点。检索时用 collapsed tree 方式——将所有层级的节点 flatten，一起做向量搜索。

### Q5: 生产环境中如何选择 Advanced RAG 策略？

**渐进式选择**：先用好 Naive RAG 基础（好的 chunking + embedding + reranker），然后根据失败模式选择进阶技术。检索质量差 → HyDE + Hybrid Search；结果不可靠 → CRAG；需要多步推理 → Query Decomposition + Multi-Hop；全局理解 → RAPTOR。**注意**：每增加一个组件都增加延迟和成本，不要过度设计。大部分场景下，好的 chunking + hybrid search + reranker 就能解决 80% 的问题。
