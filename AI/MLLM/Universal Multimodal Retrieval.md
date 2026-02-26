---
brief: "Universal Multimodal Retrieval——跨图文音视频的统一多模态检索框架；单一 embedding 空间覆盖多种模态；对多模态 RAG（RAG-Anything）的底层检索能力的基础支撑。"
title: "Universal Multimodal Retrieval"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# Universal Multimodal Retrieval

> 统一的多模态检索框架 — 让文本、图像、视频在同一个向量空间中互相检索。

## 核心问题

传统检索系统是模态隔离的：文本搜文本用 BM25/embedding，图搜图用 CLIP，跨模态检索需要专门的 bridge model。Universal Multimodal Retrieval 的目标是：**一个模型、一个向量空间，搞定所有模态的检索**。

## 技术思路

```
Query (任意模态)          Document (任意模态)
       │                         │
       ▼                         ▼
  Unified Encoder            Unified Encoder
       │                         │
       ▼                         ▼
  query_embedding            doc_embedding
       │                         │
       └──── cosine similarity ──┘
```

关键设计点：

### 1. 统一的编码器

不是为每种模态训一个 encoder，而是用一个支持多模态输入的模型（通常基于 MLLM）生成统一的 embedding：

```python
# 伪代码：统一编码
class UniversalEncoder(nn.Module):
    def __init__(self, mllm_backbone):
        self.backbone = mllm_backbone  # 如 InternVL2, Qwen-VL
        self.projection = nn.Linear(hidden_dim, embed_dim)
    
    def encode(self, input):
        """input 可以是 text / image / image+text / video"""
        # MLLM 处理多模态输入
        hidden = self.backbone(input)
        # 取 [EOS] 或 mean pooling 作为 embedding
        embedding = self.projection(hidden[:, -1, :])
        return F.normalize(embedding, dim=-1)  # L2 归一化
```

### 2. 训练数据构建

多模态检索最难的是构建跨模态的正负样本对：

```
正样本对示例：
- (产品图片, 产品描述文本) — 电商数据
- (论文图表, 图表描述) — 学术数据
- (视频片段, 字幕文本) — 视频数据
- (图片A, 相似图片B) — 同模态检索

负样本挖掘：
- Batch 内负采样 (in-batch negatives)
- Hard negative mining（BM25 召回但不相关的文档）
- Cross-modal hard negatives（跨模态困难负样本）
```

### 3. 训练策略

通常采用对比学习（contrastive learning）：

```python
def contrastive_loss(query_embeds, doc_embeds, temperature=0.07):
    """InfoNCE loss"""
    similarity = query_embeds @ doc_embeds.T / temperature
    labels = torch.arange(len(query_embeds), device=similarity.device)
    loss = F.cross_entropy(similarity, labels)
    return loss
```

进阶策略：
- **课程学习**：先学简单的同模态检索，再学跨模态
- **知识蒸馏**：从强大的单模态模型（如 E5、CLIP）蒸馏
- **指令感知**：embedding 可以根据不同检索意图（相似性/相关性/分类）动态调整

## 与传统方案对比

| 维度 | 传统方案 | Universal Retrieval |
|------|---------|-------------------|
| 文本搜文本 | BM25 + text embedding | 统一模型 |
| 图搜图 | CLIP / DINOv2 | 统一模型 |
| 文搜图 | CLIP | 统一模型 |
| 图搜文 | 需要额外适配 | 统一模型 |
| 视频检索 | 专门模型 | 统一模型 |
| 混合查询 | 无法处理 | 原生支持 |
| 模型维护 | 多个模型 | 一个模型 |

## 应用场景

1. **电商搜索**：用户上传一张图 + 文字描述搜索商品
2. **知识库检索**：RAG 场景下，文档包含图文表混排内容
3. **视频搜索**：用自然语言检索视频中的特定片段
4. **多模态 RAG**：检索结果包含文本、图片、表格等多种类型

## 我的观点

Universal Multimodal Retrieval 是 RAG 系统的下一步演进。当前大多数 RAG 系统只能检索纯文本，对图表、公式、截图等内容要么忽略、要么勉强用 OCR 转成文字。统一检索框架让 RAG 真正能处理多模态知识库。

但也有挑战：
- **向量维度和索引成本**：多模态 embedding 维度通常比纯文本大（1024+），检索成本更高
- **模型大小**：基于 MLLM 的 encoder 推理成本远高于轻量级 text encoder
- **评测标准不统一**：跨模态检索的 benchmark 还不够成熟

## 相关

- [[AI/MLLM/DeepSeek-VL|DeepSeek-VL]] — 可用作 retrieval backbone 的 MLLM
- [[AI/MLLM/InternVL3|InternVL3]] — 强力视觉编码器
- [[AI/LLM/Application/Synthetic-Data/Synthetic Data|Synthetic Data]] — 检索训练数据合成
