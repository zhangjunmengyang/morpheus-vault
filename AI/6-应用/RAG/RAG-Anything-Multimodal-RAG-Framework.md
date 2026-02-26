---
brief: RAG-Anything（Multi-modal RAG）——支持文本/图像/表格/音频/视频的统一多模态 RAG 框架；跨模态 chunk 对齐和检索的技术路线；是 Universal Multimodal Retrieval 的工程实现参考，多模态知识库的 RAG 解决方案。
title: "RAG-Anything: All-in-One Multimodal RAG Framework"
type: paper
domain: ai/rag
tags:
  - ai/rag
  - topic/multimodal
  - topic/graph-rag
  - topic/retrieval
  - type/paper
  - rating/4star
arxiv: "2510.12323"
created: 2026-02-20
authors: Zirui Guo, Xubin Ren et al.
affiliation: HKU HKUDS Lab
see-also:
  - "[[AI/6-应用/RAG/Advanced RAG|Advanced RAG]]"
  - "[[RAG-2026-技术全景|RAG 2026 技术全景]]"
  - "[[RAG 检索策略|RAG 检索策略]]"
  - "[[多模态大模型-2026-技术全景|多模态大模型 2026 全景]]"
  - "[[AI-Agent-2026-技术全景|AI Agent 2026 技术全景]]"
---

# RAG-Anything: All-in-One Multimodal RAG Framework

**arXiv**: 2510.12323 | **Date**: 2025-10-14 | **Code**: https://github.com/HKUDS/RAG-Anything  
**Authors**: Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang  
**Affiliation**: The University of Hong Kong (HKUDS Lab)  
**GitHub Stars**: 趋势榜周榜（2026-02-20 周）  
**Rating**: ★★★★☆

---

## 一句话

在 LightRAG 基础上扩展多模态能力：用 **dual-graph construction**（跨模态图 + 文本图融合）+ **cross-modal hybrid retrieval**（结构导航 + 语义匹配）处理含图表、公式、图像的复杂文档 RAG。

---

## 动机：文本 RAG 的模态盲区

现有 RAG 框架（LightRAG、GraphRAG）的隐含假设：知识库 = 纯文本。

现实中高价值文档恰恰是多模态的：
- 学术论文：实验结果图、数学公式是核心信息载体
- 金融报告：市场走势图、相关性矩阵、业绩表格
- 医疗文献：影像学图像、诊断图表、临床数据表

文本 RAG 的处理方式：要么丢弃非文本内容，要么「flatten」成低质量文本近似。两者都会造成严重信息损失。

**RAG-Anything 的核心主张**：将多模态内容视为「相互关联的知识实体」而非「孤立的数据类型」。

---

## 框架架构

### 第一阶段：Universal Representation（多模态知识统一表示）

**原子内容单元分解**：
```
k_i → Decompose → {c_j = (t_j, x_j)}_{j=1}^{n_i}
```
- `t_j ∈ {text, image, table, equation, ...}`：模态类型
- `x_j`：内容本体（模态感知方式提取）

每个原子单元保留其在文档中的上下文位置和层级关系：
- 图保留 caption 和交叉引用
- 公式保留周边定义
- 表格保留 header 和 explanatory narrative

### 第二阶段：Dual-Graph Construction（双图构建）

**关键设计决策**：不构建单一统一图，而是构建两个互补图再融合。原因：单图容易混淆模态特有的结构信号。

#### Cross-Modal Knowledge Graph（跨模态知识图）

非文本内容（图像、表格、公式）的处理：
1. 用 MLLM 为每个非文本原子单元生成两种文本表示：
   - `d_j^chunk`：面向跨模态检索的详细描述
   - `e_j^entity`：包含 entity name/type/description 的实体摘要
2. 以上下文窗口 `C_j = {c_k | |k-j| ≤ δ}` 为条件，确保表示准确反映文档结构语境
3. 构建图结构：
   - 非文本单元 `c_j` 对应 anchor node `v_j^mm`
   - 从 `d_j^chunk` 提取 intra-chunk 实体和关系 `(V_j, E_j)`
   - `belongs_to` 边将 intra-chunk 实体连接到 anchor node

```
Ṽ = {v_j^mm}_j ∪ ⋃_j V_j
Ẽ = ⋃_j E_j ∪ ⋃_j {u --belongs_to--> v_j^mm : u ∈ V_j}
```

#### Text-based Knowledge Graph（文本知识图）

对纯文本块，用 LightRAG / GraphRAG 方法构建：NER + 关系抽取 → 实体-关系图。不需要多模态上下文整合。

#### Graph Fusion（图融合）

两个图通过 **entity name 匹配** 进行对齐，合并重复实体：
```
G = (V, E)   ← 统一知识图（同时包含跨模态关系和文本语义关系）
```

同时构建 embedding table：
```
T = {emb(s) : s ∈ V ∪ E ∪ {c_j}_j}
```
所有实体、关系、原子内容块统一 embedding，形成完整检索索引 `I = (G, T)`。

### 第三阶段：Cross-Modal Hybrid Retrieval

**模态感知查询编码**：分析 query 中的模态线索（"figure", "chart", "table", "equation"），提取偏好信号。

**两路互补检索**：

#### 结构导航（Structural Knowledge Navigation）
- 基于 keyword matching + entity recognition 定位 graph 相关节点
- 从匹配实体出发，做 neighborhood expansion（多跳）
- 特别适合：多跳推理、跨模态关系、显式结构连接
- 返回 `C_stru(q)`：相关实体、关系及关联内容块

#### 语义相似度匹配（Semantic Similarity Matching）
- Query embedding `e_q` 与 embedding table `T` 做 dense vector 搜索
- 覆盖所有模态的原子内容块、实体、关系表示
- 适合：无显式结构连接但语义相关的内容
- 返回 Top-k `C_seman(q)`

**候选池融合**：
```
C(q) = C_stru(q) ∪ C_seman(q)
```
Multi-signal fusion scoring：结构重要性（图拓扑）+ 语义相似度 + 模态偏好（query 分析）→ 排序 `C*(q)`

### 第四阶段：合成响应

1. 构建文本上下文：concatenate 检索到的实体摘要、关系描述、内容块（含 modality 标注和层级来源）
2. 还原视觉内容：检索到的多模态块 dereference 回原始图像/表格 `V*(q)`
3. VLM 联合条件生成：
```
Response = VLM(q, P(q), V*(q))
```
文本 proxy 用于高效检索，原始视觉内容用于深度推理。

---

## 实验结果

**数据集**：
- DocBench：229 文档（5 领域），66 页均值，46K tokens 均值，1102 QA pairs
- MMLongBench：135 文档（7 类型），47.5 页均值，21K tokens 均值，1082 QA pairs

**DocBench 准确率**（%）：

| 方法 | Aca. | Fin. | Gov. | Law | News | Txt. | Mm. | Overall |
|------|------|------|------|-----|------|------|-----|---------|
| GPT-4o-mini | 40.3 | 46.9 | 60.3 | 59.2 | 61.0 | 61.0 | 43.8 | 51.2 |
| LightRAG | 53.8 | 56.2 | 59.5 | 61.8 | 65.7 | 85.0 | 59.7 | 58.4 |
| MMGraphRAG | 64.3 | 52.8 | 64.9 | 40.0 | 61.5 | 67.6 | 66.0 | 61.0 |
| **RAG-Anything** | 61.4 | **67.0** | 61.5 | 60.2 | 66.3 | **85.0** | **76.3** | **63.4** |

**MMLongBench 准确率**：RAG-Anything **42.8%** vs GPT-4o-mini 33.5% vs LightRAG 38.9% vs MMGraphRAG 37.7%

**长文档优势（DocBench，按页数）**：
- 1–50 页：RAG-Anything ~持平 MMGraphRAG
- 101–200 页：68.2% vs 54.6%（+13.6 点）
- 200+ 页：68.8% vs 55.0%（+13.8 点）

**Ablation**（DocBench）：
- **Chunk-only**（去掉 graph construction）：60.0%（vs 63.4%，-3.4 点）
- **w/o Reranker**：62.4%（vs 63.4%，-1 点）
- 结论：Graph construction 是主要贡献；reranker 边际收益

**多模态内容（Mm. 类型）**：最明显优势在此——76.3% vs MMGraphRAG 66.0%（+10 点）

---

## 我的评价

### 技术贡献

**Dual-graph 的设计逻辑是对的**：跨模态图和文本图分别捕获不同类型的语义结构，再通过 entity alignment 融合，比直接构建统一图更稳健。原因：图像/表格的结构语义（panel↔caption↔axis）和文本的语义关系（entity-relation）有本质差异，混在一起反而会互相干扰。

**`belongs_to` anchor node 设计** 保证了非文本内容在图中的 grounding：VLM 生成的 entity 描述通过显式边连接回原始多模态 chunk，检索路径可追溯。

**Hybrid retrieval 的两路互补** 是成熟的工程方案（参考 DPR/BM25 hybrid，GraphRAG 的 local/global retrieval），这里扩展到跨模态场景，思路清晰。

### 边界条件与不足

1. **VLM 依赖**：图像/表格理解质量取决于 MLLM 生成描述的质量。如果 VLM 描述不准确（这在复杂图表上很常见），图的质量也会下降——错误会传播到整个检索链。

2. **数字**：DocBench 上的绝对准确率 63.4% 并不算高（这是 100 页+的多模态文档，任务本身很难）。表中 RAG-Anything 并不是每个领域都最好（Law 60.2% vs LightRAG 61.8%）。

3. **计算成本**：双图构建 + entity alignment + 全量 embedding 的 indexing cost 较高，论文未报告 vs baseline 的时间/资源开销对比。

4. **LightRAG 基础**：这是在 LightRAG 上的扩展，核心 graph-based RAG 框架并非原创。创新在于多模态适配层。

5. **评测依赖 GPT-4o-mini 打分**：LLM-as-judge 评测有一定误差，尤其对图表类答案的评判可靠性有待验证。

### 与 Vault 的连接

Vault 里有 LightRAG 相关内容（参考 ICLR-2026-趋势分析）。RAG-Anything 是 LightRAG → 多模态扩展的直接路径。对于老板做的 AI 应用工程，**多模态 RAG 的工程实现参考价值高**：文档 parsing（MinerU）→ dual-graph → hybrid retrieval → VLM 合成，这是一套完整可落地的 pipeline。

### 研究价值定位

这不是 paradigm shift，是 solid engineering work。**Graph-based RAG 多模态扩展**这个方向是对的，但 63% 的准确率说明问题还很硬。长文档多模态理解是 2026 年重要战场，这篇论文的架构是一个好的起点。

★★★★☆（实用性强，工程扎实，创新偏工程侧）

---

## 核心流程（简化版）

```
文档 → MinerU 解析 → 原子单元(text/image/table/equation)
         ↓
    [图像/表格/公式] → VLM描述 → Cross-Modal Graph
    [文本] → NER+RE → Text-based Graph
         ↓
    Entity Alignment → Unified Graph G
    All units → Embedding Table T
         ↓
    Query: 结构导航(graph multi-hop) + 语义检索(dense)
         ↓
    Multi-signal fusion scoring → Top-k candidates
         ↓
    文本上下文 + 还原原始图像 → VLM 生成答案
```

---

## 相关工作

- **LightRAG** (HKUDS, 2024) — 本框架的基础，graph-based RAG
- **GraphRAG** (Microsoft, 2024) — 另一个 graph-based RAG，RAG-Anything 对比 baseline
- **MMGraphRAG** (Wan & Yu, 2025) — 多模态 graph RAG 的直接竞争工作
- **MinerU** — 文档解析工具（本框架 parsing 基础）

---

*笔记日期: 2026-02-20*
