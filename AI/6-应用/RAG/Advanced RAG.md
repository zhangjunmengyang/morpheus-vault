---
brief: "Advanced RAG 进阶技术（RAG 域）——GraphRAG/Self-RAG/CRAG/HyDE 进阶检索策略；多跳推理/动态知识更新/RAG+RL 的前沿方向；Interview 标注，RAG 工程师进阶面试的深度参考。"
title: "Advanced RAG 进阶技术"
date: 2026-02-14
tags: [rag, graphrag, self-rag, advanced, interview]
type: note
---

# Advanced RAG 进阶技术

## 1. Naive RAG 的局限性

传统 RAG（Retrieve-then-Read）的基本流程：Query → Embedding → 向量检索 Top-K → 拼接到 Prompt → LLM 生成。

**核心问题：**

- **单次检索不够**：用户查询可能含糊或多意图，一次检索无法覆盖所有所需信息。对于需要多步推理的复杂问题（如"对比 A 和 B 的优缺点"），单次检索往往只命中其中一方。
- **检索质量不可控**：Top-K 结果可能全部不相关，但模型仍被迫基于这些噪声生成回答，导致 **幻觉（hallucination）** 加剧——模型会"编造"看似合理但无依据的内容。
- **上下文窗口浪费**：将大量检索到的文档片段塞入 prompt，其中可能 80% 是无关内容，浪费了宝贵的上下文窗口容量，同时引入噪声干扰生成质量。
- **Lost in the Middle**：研究表明 LLM 对上下文中间部分的信息关注度较低，大量堆砌文档反而降低答案质量。
- **缺乏自省能力**：Naive RAG 无法判断自己是否需要检索、检索结果是否有用、生成结果是否忠于证据。

---

## 2. Self-RAG：自反思检索增强生成

> 论文：*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (ICLR 2024)

**核心思想：** 训练模型输出特殊的 **反思 token（reflection tokens）**，让模型自己决定检索和生成的每一步。

**四种反思 token：**

| Token | 含义 | 触发时机 |
|-------|------|----------|
| `[Retrieve]` | 是否需要检索 | yes / no / continue |
| `[IsRel]` | 检索结果是否与问题相关 | relevant / irrelevant |
| `[IsSup]` | 生成内容是否被检索结果支持 | fully / partially / no support |
| `[IsUse]` | 生成结果是否有用 | 1-5 评分 |

**工作流程：**

```
输入 Query
  → 模型判断 [Retrieve]: yes
    → 检索文档 d1, d2, d3
    → 对每个文档分别生成候选答案
    → 模型对每个候选打分：[IsRel] × [IsSup] × [IsUse]
    → 选择综合得分最高的答案输出
```

**关键优势：**
- **按需检索**：简单的事实题无需检索，复杂问题才触发
- **细粒度控制**：推理时可调节各反思 token 的权重，在事实性和创造性之间灵活权衡
- **端到端训练**：通过 Critic Model 生成反思标签，再用这些标签微调生成模型

---

## 3. Corrective RAG (CRAG)

> 论文：*Corrective Retrieval Augmented Generation* (2024)

**核心思想：** 在检索和生成之间加一个 **检索评估器（Retrieval Evaluator）**，对检索结果质量做分级处理。

**三级处理策略：**

```
检索结果 → Evaluator 打分 → 
  ├─ Correct（高置信）  → 知识精炼（提取关键信息，去除噪声）→ 生成
  ├─ Ambiguous（不确定）→ 知识精炼 + 网络搜索补充 → 融合后生成
  └─ Incorrect（低置信）→ 丢弃检索结果 → 网络搜索兜底 → 生成
```

**知识精炼（Knowledge Refinement）过程：**
1. 将检索到的文档分割为细粒度的知识条（knowledge strips）
2. 对每个知识条评估其与查询的相关性
3. 过滤掉不相关的条目，只保留高相关内容
4. 重组精炼后的知识作为生成依据

**与 Self-RAG 的区别：**
- Self-RAG 是训练模型自身具备反思能力（内化）
- CRAG 是用外部评估器做质量控制（外挂），不需要改模型

---

## 4. GraphRAG（微软）

> 来源：Microsoft Research, *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (2024)

**解决的痛点：** 传统向量检索擅长回答局部的、具体的问题（"张三是谁？"），但对 **全局性/主题性问题** 无能为力（"这个数据集的主要主题是什么？""不同团队之间有什么共性？"）。

**构建流程：**

```
原始文档
  → LLM 抽取实体和关系 → 知识图谱
  → Leiden 社区检测算法 → 分层社区结构
  → LLM 对每个社区生成摘要 → 社区摘要索引
```

**查询流程：**

```
用户问题
  → 判断是局部问题还是全局问题
  ├─ 局部问题 → 实体检索 → 子图遍历 → 生成
  └─ 全局问题 → 相关社区摘要 → Map-Reduce 汇总 → 生成
```

**关键设计：**
- **分层社区**：不同粒度的社区层级，低层级=细节，高层级=宏观主题
- **Map-Reduce 汇总**：将多个社区摘要并行处理（Map），再聚合为最终答案（Reduce）
- **成本考量**：索引构建阶段需要大量 LLM 调用（实体抽取+摘要生成），成本较高

**适用场景：**
- 企业知识库的全局洞察
- 大规模文档集的主题分析
- 需要跨文档关联推理的场景

---

## 5. Agentic RAG

**核心思想：** 将 RAG 从固定 pipeline 升级为 **Agent 驱动的动态工作流**，用 LLM 作为中枢来规划和执行检索策略。

**典型架构模式：**

### 路由模式（Router）
```
用户问题
  → Router Agent 判断：
    ├─ 技术问题 → 搜索技术文档库
    ├─ 产品问题 → 搜索产品 FAQ
    ├─ 最新资讯 → 调用网络搜索
    └─ 简单闲聊 → 直接回复，不检索
```

### 多步推理模式（Multi-step）
```
"对比 GPT-4 和 Claude 3 在代码生成上的表现"
  → Step 1: 搜索 GPT-4 代码生成 benchmark
  → Step 2: 搜索 Claude 3 代码生成 benchmark
  → Step 3: 搜索两者的对比文章
  → Step 4: 综合所有检索结果，生成对比分析
```

### 核心组件：
- **规划器（Planner）**：分解复杂查询为子任务
- **检索器（Retriever）**：多种检索工具（向量搜索、BM25、知识图谱、Web 搜索）
- **反思器（Reflector）**：评估中间结果是否足够，决定是否继续检索
- **生成器（Generator）**：基于完整证据链生成最终答案

**典型框架：** LlamaIndex Workflows、LangGraph、CrewAI

---

## 6. Modular RAG

**核心思想：** 将 RAG 系统拆解为可组合的独立模块，每个模块可独立优化和替换。

**六大核心模块：**

```
┌─────────────┐
│  Indexing    │  文档分块、嵌入、索引构建
├─────────────┤
│  Pre-Retrieval│  查询改写、Query Expansion、HyDE
├─────────────┤
│  Retrieval   │  向量检索、BM25、混合检索、重排序
├─────────────┤
│  Post-Retrieval│  重排序(Reranking)、压缩、过滤
├─────────────┤
│  Generation  │  Prompt 构建、引用生成、格式控制
├─────────────┤
│  Orchestration│  流程编排、条件分支、循环控制
└─────────────┘
```

**关键优化技术：**

| 模块 | 优化技术 | 说明 |
|------|----------|------|
| Pre-Retrieval | HyDE | 先用 LLM 生成假设性回答，用该回答做检索 |
| Pre-Retrieval | Query Rewriting | 将口语化查询改写为适合检索的形式 |
| Pre-Retrieval | Step-back Prompting | 将具体问题抽象化，检索更泛化的知识 |
| Retrieval | Hybrid Search | BM25 (关键词) + Dense (语义) 融合 |
| Post-Retrieval | Cross-encoder Reranking | 用交叉编码器对初检结果重排序 |
| Post-Retrieval | Contextual Compression | 提取文档中与查询相关的句子，去除冗余 |

---

## 7. Multi-hop RAG

**场景：** 回答需要跨多个文档、多步推理才能得出的问题。

**示例问题：**"获得诺贝尔物理学奖的最年轻科学家毕业于哪所大学？"

**推理链：**
```
Step 1: 谁是最年轻的诺贝尔物理学奖得主？ → Lawrence Bragg (25岁)
Step 2: Lawrence Bragg 毕业于哪所大学？ → 剑桥大学
```

**关键策略：**

- **查询分解（Query Decomposition）**：将复杂问题拆解为可独立检索的子问题
- **迭代检索（Iterative Retrieval）**：每一步的检索结果作为下一步的查询条件
- **思维链引导（CoT-guided Retrieval）**：利用 LLM 的推理能力指导检索方向
- **证据链追踪（Evidence Chain Tracking）**：记录每一步的证据来源，确保可追溯

**与 Agentic RAG 的关系：** Multi-hop RAG 是 Agentic RAG 的一个典型应用场景，Agent 框架天然适合编排多跳检索流程。

---

## 8. RAG vs Fine-tuning

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| **知识更新** | 实时更新，改文档即可 | 需要重新训练 |
| **成本** | 检索基础设施 + 更长的 prompt | 训练成本（GPU 时间 + 数据标注） |
| **幻觉控制** | 有据可查，可追溯 | 知识内化，但可能编造 |
| **领域适配** | 提供领域文档即可 | 需要领域数据微调 |
| **推理延迟** | 额外的检索延迟 | 无额外延迟 |
| **适用场景** | 知识密集型、时效性强 | 风格/格式要求、推理能力增强 |

**混合策略（实践中最常见）：**
1. **Fine-tuning 学格式 + RAG 补知识**：微调模型学会如何使用检索结果，RAG 提供最新知识
2. **RAG + 轻量微调**：用 LoRA/QLoRA 在小规模数据上微调，提升模型对检索结果的利用能力
3. **RAFT（Retrieval Augmented Fine-Tuning）**：在微调数据中混入检索到的相关/不相关文档，训练模型学会"从噪声中找信号"

---

## 9. RAG 评测框架：RAGAS

> RAGAS = Retrieval Augmented Generation Assessment

**四大核心指标：**

### Faithfulness（忠实度）
- **定义**：生成的答案是否能被检索到的上下文所支持
- **计算**：将答案拆解为多个声明（claims），逐一检查是否有上下文支持
- **公式**：`Faithfulness = 被支持的声明数 / 总声明数`

### Answer Relevancy（答案相关性）
- **定义**：生成的答案是否与用户问题相关
- **计算**：用 LLM 根据答案反向生成 N 个可能的问题，计算这些问题与原始问题的相似度
- **直觉**：好的答案应该能"还原"出原始问题

### Context Precision（上下文精确度）
- **定义**：检索到的上下文中，相关内容是否排在前面
- **计算**：评估每个检索到的文档块是否包含回答所需的信息，并考虑排序位置
- **关注**：检索结果的**排序质量**

### Context Recall（上下文召回率）
- **定义**：回答问题所需的信息是否都被检索到了
- **计算**：将标准答案拆解为多个关键信息点，检查每个信息点是否能在检索结果中找到依据
- **公式**：`Recall = 被覆盖的信息点数 / 总信息点数`

**使用方式：**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=embedding_model,
)
```

---

## 10. 面试常见问题及回答要点

### Q1：你们的 RAG 系统遇到过什么问题？怎么优化的？

**回答框架：**
> 遇到的核心问题是**检索质量不稳定**。具体表现为：
> 1. **语义鸿沟**：用户口语化查询和文档之间的语义差距大 → 引入 Query Rewriting + HyDE
> 2. **召回率低**：纯向量检索会漏掉关键词匹配 → 混合检索（BM25 + Dense）
> 3. **排序不优**：Top-K 中相关文档不在前面 → Cross-encoder Reranking
> 4. **上下文噪声**：塞入太多不相关文档 → Contextual Compression
> 5. 最终在 RAGAS 指标上 Faithfulness 从 0.72 提升到 0.89

### Q2：Self-RAG 和 CRAG 有什么区别？各自适合什么场景？

**回答要点：**
> - Self-RAG 是**内化能力**：通过微调让模型自身学会反思，需要训练成本但推理时不依赖外部评估器，适合对延迟敏感的场景
> - CRAG 是**外挂机制**：用独立的评估模型做质量控制，不需要改生成模型，部署更灵活，适合不想动模型的场景
> - 两者可以结合：用 Self-RAG 做初步的检索决策，用 CRAG 的知识精炼和网络搜索做兜底

### Q3：GraphRAG 的主要优势和劣势是什么？

**回答要点：**
> **优势**：解决了传统 RAG 无法回答全局性/主题性问题的硬伤；知识图谱提供了显式的实体关系，推理路径可解释；分层社区结构支持不同粒度的查询
>
> **劣势**：索引构建成本高（需要大量 LLM 调用做实体抽取和摘要生成）；图谱维护成本高（文档更新需要增量更新图谱）；对于简单的事实查询，可能不如传统向量检索高效
>
> **适用场景**：企业内部知识管理、科研文献分析、需要全局洞察的决策支持

### Q4：如何评估一个 RAG 系统的好坏？

**回答要点：**
> 分三个层面评估：
> 1. **检索质量**：Context Precision（排序好不好）、Context Recall（找全了没有）、MRR/NDCG
> 2. **生成质量**：Faithfulness（有没有编造）、Answer Relevancy（答没答到点子上）
> 3. **端到端**：用户满意度、任务完成率、A/B 测试
>
> 工具上可以用 RAGAS 做自动化评测，但要注意 LLM-as-Judge 本身的偏差，关键场景需要人工评估兜底。

### Q5：RAG 中的 chunking 策略怎么选？

**回答要点：**
> 没有银弹，需要根据数据特点选择：
> - **固定大小分块**：简单通用，适合结构均匀的文档，注意设置 overlap（通常 10-20%）
> - **语义分块（Semantic Chunking）**：基于嵌入相似度的断点检测，同一语义单元不被切断
> - **递归分块**：按段落→句子→字符逐级分割，保持结构完整性
> - **文档结构分块**：利用 Markdown/HTML 标题层级，保留文档结构信息
> - **Agentic Chunking**：用 LLM 判断主题边界，效果最好但成本最高
>
> 实践建议：先从 512 token + 50 overlap 开始，根据评测结果调优。

### Q6：如何处理 RAG 系统中的多模态内容？

**回答要点：**
> 三种策略：
> 1. **多模态嵌入**：用 CLIP/SigLIP 等多模态模型统一编码文本和图像到同一向量空间
> 2. **图像转文本**：用 VLM（如 GPT-4V）将图表/图片转换为文本描述后索引
> 3. **混合索引**：文本和图像分别建索引，检索时融合排序
>
> 实际中方案 2 最实用：对文档中的图表用 VLM 生成详细描述，将描述文本和原始文本一起索引。检索到相关描述时，将原始图片一并传给多模态 LLM 生成答案。

## See Also

- [[AI/6-应用/RAG/_MOC|RAG MOC]] — 检索增强生成全景索引
- [[RAG-2026-技术全景|RAG 2026 技术全景]] — 宏观综述
- [[RAG 检索策略|RAG 检索策略]] — 本文进阶技术的基础层
- [[向量数据库选型|向量数据库选型]] — 稠密检索的基础设施选型
- [[LLM代码生成-2026技术全景|LLM 代码生成]] — RAG 的一个核心应用场景
