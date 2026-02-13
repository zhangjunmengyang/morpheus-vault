---
title: "DeepSeek Engram — 条件记忆与可扩展查找"
date: 2026-02-13
tags:
  - ai/llm/architecture
  - ai/llm/inference
  - ai/memory
  - type/concept
  - interview/hot
status: active
---

# DeepSeek Engram — 条件记忆与可扩展查找

> Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
> 论文: arXiv:2601.07372 (2026-01-12) | DeepSeek-AI & 北京大学

## 1. 核心问题：为什么需要 Engram？

传统 Transformer 模型在推理时面临一个根本性的效率问题：**静态知识的重复重建**。

当你输入 "亚历山大大帝"，模型每次都要花费宝贵的计算资源从头重建这个常见短语的表示。这相当于一个数学家每次解方程前都要从头数一遍 0-9。模型本质上是用**条件计算（conditional computation）** 来模拟**记忆检索**——这是极其低效的。

> "This process essentially amounts to an expensive runtime reconstruction of a static lookup table, wasting valuable sequential depth on trivial operations that could otherwise be allocated to higher-level reasoning." — Engram 论文

### 与现有技术的区别

| 技术 | 类型 | 持久性 | 计算开销 |
|------|------|--------|----------|
| KV Cache | 短期上下文缓存 | 会话级 | O(n) |
| [[AI/LLM/Architecture/DeepSeek-R1\|MoE 路由]] | 条件计算 | 训练固定 | 按需激活 |
| **Engram** | **条件记忆** | **预计算持久化** | **O(1) 查找** |

## 2. 核心架构

Engram 引入了一个全新的稀疏性维度——**条件记忆（Conditional Memory）**，与 [[AI/Foundations/DL-Basics/MoE 基础\|MoE]] 的条件计算互补。

### 2.1 三大核心组件

#### (1) Tokenizer 压缩
- 使用 NFKC 归一化，将 "Apple" 和 "apple" 映射到同一概念
- 有效词汇量减少约 **23%**，提升哈希效率

#### (2) Multi-Head N-gram Hashing
- 将连续 token 序列（2-gram, 3-gram）通过**乘法-XOR 哈希**映射到 embedding 表
- 多头设计防止哈希碰撞（类似多本电话簿交叉验证）
- **O(1) 时间复杂度**，确定性寻址

```python
def hash_ngram(tokens, ngram_size, head_idx, table_size):
    """Engram 的核心哈希函数 — 乘法-XOR"""
    multipliers = [2 * i + 1 for i in range(ngram_size)]
    mix = 0
    for i, token in enumerate(tokens[-ngram_size:]):
        mix ^= token * multipliers[i]
    mix ^= head_idx * 10007  # head-specific variation
    return mix % table_size
```

#### (3) Context-Aware Gating
- 注意力式门控机制评估每个检索记忆的相关性
- 不相关的记忆 → gate 值趋近 0 → 被忽略
- 确保记忆注入不会干扰当前上下文推理

### 2.2 架构集成方式

```
Input Tokens
    ├── Transformer Layer（动态推理）
    │       ├── Self-Attention
    │       └── FFN / MoE Expert
    └── Engram Module（静态记忆查找）
            ├── Token 压缩 & 归一化
            ├── N-gram Hash → Embedding Lookup
            ├── Multi-Head 聚合
            └── Context-Aware Gating → 融合
```

Engram 模块可以嵌入到 Transformer 的**早期层**，让早期层从静态模式重建中解放出来，将有效深度留给复杂推理。

## 3. U 型缩放定律

Engram 最重要的发现之一是**资源分配的 U 型曲线**：

- 100% MoE + 0% Engram → 浪费计算重建静态知识
- 0% MoE + 100% Engram → 缺乏动态推理能力
- **最优点：~75-80% MoE + 20-25% Engram**

这个 U 型定律为未来模型架构设计提供了理论指导。

## 4. 性能表现

Engram-27B vs 同参数量 MoE 基线（iso-parameter, iso-FLOPs 约束）：

| 基准测试 | 提升 |
|----------|------|
| BBH (推理) | **+5.0 points** |
| MMLU (知识) | **+3.4 points** |
| HumanEval (代码) | **+3.0 points** |
| Multi-Query Needle-in-Haystack | **97.0 vs 84.2** |

尤其是长上下文场景下的 Needle-in-Haystack 测试，从 84.2 跃升到 97.0，说明 Engram 的记忆机制在信息检索上有本质性提升。

## 5. 系统效率：解耦计算与存储

这是 Engram 最具工程价值的特性：

- **确定性寻址** → embedding 表可以从 GPU HBM 卸载到**主机系统内存（DRAM / CXL）**
- 推理延迟仅有轻微增加，但大幅降低 GPU 显存压力
- 与 NVIDIA KV Cache offload to NVMe 不同，Engram 的数据是**预计算的持久化知识**

### 与 KV Cache 的本质区别

| 维度 | KV Cache | Engram |
|------|----------|--------|
| 存储内容 | 当前会话的注意力状态 | 预训练学到的静态知识 |
| 生命周期 | 会话级（用完即弃） | 模型级（跨会话持久） |
| 计算方式 | 缓存避免重复注意力计算 | O(1) 哈希查找避免重复推理 |
| 硬件位置 | GPU HBM / NVMe offload | 可以在主机 DRAM / CXL |

## 6. 与 DeepSeek V4 的关系

根据多方报道，Engram 技术预计将集成到 DeepSeek V4 中，形成 **MoE + Engram 双稀疏架构**：
- MoE 处理动态推理（条件计算）
- Engram 处理静态知识检索（条件记忆）
- 两者按 U 型缩放定律分配容量

## 7. 代码实践

Engram 已开源：[github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

```python
import numpy as np

# 简化版 Multi-Head Embedding Lookup
MAX_NGRAM = 3
NUM_HEADS = 4
EMBEDDING_DIM = 128

def compress_token(token_id, vocab_size=1000):
    """NFKC 归一化的简化模拟"""
    return token_id % (vocab_size // 2)

def multi_head_lookup(token_sequence, embedding_tables):
    """多头 N-gram 查找"""
    compressed = [compress_token(t) for t in token_sequence]
    embeddings = []
    for ngram_size in range(2, MAX_NGRAM + 1):
        for head_idx in range(NUM_HEADS):
            table = embedding_tables[ngram_size - 2][head_idx]
            hash_idx = hash_ngram(compressed, ngram_size, head_idx, table.shape[0])
            embeddings.append(table[hash_idx])
    return np.concatenate(embeddings)
```

## 8. 面试常见问题

**Q1: Engram 和 RAG 有什么区别？**
A: RAG 是在推理时从外部文档库检索，Engram 是在模型内部通过哈希查找预训练时编码的静态知识。Engram 是模型架构的一部分，RAG 是推理 pipeline 的一部分。

**Q2: Engram 的 O(1) 查找为什么比注意力机制高效？**
A: 注意力机制需要 O(n²) 或 O(n) 的计算来动态组合信息，而 Engram 用确定性哈希直接定位 embedding，不需要任何矩阵运算。

**Q3: 为什么最优分配是 ~80% MoE + 20% Engram？**
A: 语言中约 20-25% 的信息是静态可查找的（事实、常见短语），其余需要动态推理。U 型曲线反映了这种自然分布。

**Q4: Engram 的哈希碰撞问题如何解决？**
A: 通过多头哈希（多个独立哈希函数）+ Context-Aware Gating。即使某个头发生碰撞，其他头可以纠正，门控机制会降低不相关检索的权重。

**Q5: Engram 能用于微调吗？**
A: 理论上可以冻结 Engram 的 embedding 表只微调 Transformer 部分，也可以针对特定领域重新训练 Engram 的哈希表。

## 相关链接

- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]
- [[AI/Foundations/DL-Basics/MoE 基础|MoE 基础]]
- [[AI/LLM/Inference/vLLM|vLLM]] — 推理优化
- [[AI/LLM/Application/Embedding/Embedding|Embedding]] — 向量化技术
