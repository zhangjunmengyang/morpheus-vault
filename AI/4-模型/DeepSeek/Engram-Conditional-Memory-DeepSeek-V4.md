---
title: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"
brief: "DeepSeek-AI + PKU（arXiv:2601.07372）提出 Engram：用现代化 N-gram 嵌入表实现'记忆稀疏'，作为与 MoE 计算稀疏互补的第二轴。100B 参数表推理时卸载到主机内存开销<3%，在同等参数/FLOPs下全面优于纯 MoE 基线。预计纳入 DeepSeek V4。"
tags:
  - architecture
  - sparsity
  - memory
  - MoE
  - DeepSeek
  - knowledge-retrieval
  - inference-optimization
  - n-gram
date: 2026-02-20
arxiv: "2601.07372"
github: "deepseek-ai/Engram"
rating: ★★★★★
institution: DeepSeek-AI + Peking University
expected_in: DeepSeek V4
---

# Engram：记忆作为稀疏的第二轴——DeepSeek V4 架构核心

> **一句话**：LLM 被迫用昂贵的神经计算模拟本应是 O(1) 查找的知识检索。Engram 把 N-gram 嵌入表现代化，作为与 MoE（计算稀疏）互补的**记忆稀疏**第二轴，在同等参数和 FLOPs 下全面优于纯 MoE 基线——且推理时 100B 参数表卸载到主机内存开销 <3%。

---

## 问题：Transformer 在做低效的计算模拟

语言建模有两种质性不同的子任务：

| 子任务 | 特征 | 最佳处理方式 |
|--------|------|------------|
| **Compositional Reasoning** | 动态、上下文依赖、创造性 | 深度神经计算（MoE） |
| **Knowledge Retrieval** | 静态、局部、高度模板化 | O(1) 查找表 |

问题：标准 Transformer 没有原生的 knowledge lookup 原语。当模型遇到 "Barack Obama" 这样的命名实体或固定代码模式时，它必须用**多层 attention + FFN 来运行时重建一张本可直接查找的静态表**。这是巨大的计算浪费。

实验证明：解析一个常见的多 token 实体需要消耗多个早期 attention 和 FFN 层（用 LogitLens + CKA 确认）。这些早期层本可用于更高层次的推理。

**Engram 的核心论点**：把静态知识检索从神经计算中解耦出来，既节省计算，又增加有效推理深度。

---

## Engram 架构

### 整体设计

Engram 是一个**条件记忆模块**，以残差方式插入 Transformer backbone 的特定层：

```
H^(ℓ) ← H^(ℓ) + Y    # Engram 的输出作为残差
然后执行标准的 Attention + MoE
```

注意：不是每层都加，由系统延迟约束决定放在哪些层。

### 两阶段处理：检索 + 融合

**Phase 1: Sparse Retrieval via Hashed N-grams**

1. **Tokenizer Compression**：
   - 原始 token ID → 规范化 canonical ID（NFKC + 小写化）
   - 把语义等价的形式（如 "Apple" vs " apple"）合并
   - 实际效果：128k 词表压缩 23%
   
2. **Multi-Head Hashing**：
   - 对每个 N-gram order n，用 K 个不同哈希头
   - 每头独立映射到对应嵌入表 E_{n,k}（素数大小 M_{n,k}）
   - 哈希函数：轻量级 multiplicative-XOR hash
   - 最终记忆向量：所有头和阶次的嵌入拼接

公式：
```
z_{t,n,k} = φ_{n,k}(g_{t,n})
e_{t,n,k} = E_{n,k}[z_{t,n,k}]
e_t = concat(e_{t,n,k} for all n,k)
```

**Phase 2: Context-aware Gating（条件融合）**

纯静态检索有两个问题：无法感知上下文 + 哈希碰撞噪声 + 多义词问题。

Engram 用当前 hidden state h_t（已聚合了全局上下文）作为 Query 对检索结果做门控：

```
k_t = W_K · e_t,   v_t = W_V · e_t
α_t = σ(RMSNorm(h_t)ᵀ · RMSNorm(k_t) / √d)    # 标量门
ṽ_t = α_t · v_t
```

关键设计：如果检索到的记忆 e_t 与当前上下文矛盾，α_t → 0，噪声被自动抑制。

最后加一个 depthwise causal convolution 扩大感受野：
```
Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
```

### 与 Multi-branch 架构的集成

Engram 不依赖特定 backbone 拓扑，但在 Manifold-Constrained Hyper-Connections（mHC，M=4）中有专门优化：

- 共享嵌入表和 W_V 矩阵（跨所有 M 个分支）
- 每个分支有独立的 W_K^(m) 实现 branch-specific gating
- 把线性投影融合成单个稠密 FP8 矩阵乘：最大化 GPU 利用率

---

## U-shaped Scaling Law：记忆-计算最优分配

核心问题：给定固定参数预算，MoE experts 和 Engram memory 各分多少？

实验发现：这是一个 **U 形曲线**——
- 全部给 MoE：中等
- 全部给 Engram：差（静态记忆无法推理）
- **最优点：20-25% 参数给 Engram，75-80% 给 MoE**

这个 Sparsity Allocation 问题有明确的最优解，不是 hyperparameter 调参，而是由架构本质决定的。

---

## 实验结果（27B 参数模型）

### vs. 严格 iso-param + iso-FLOPs MoE 基线

| 领域 | 任务 | MoE 27B | Engram-27B | 提升 |
|------|------|---------|-----------|------|
| **知识检索** | MMLU | 60.6 | 64.0 | +3.4 |
| | CMMLU | 57.9 | 61.9 | +4.0 |
| | MMLU-Pro | — | — | +1.8 |
| **通用推理** | BBH | 50.9 | 55.9 | **+5.0** |
| | ARC-Challenge | 70.1 | 73.8 | +3.7 |
| | DROP | — | — | +3.3 |
| **代码/数学** | HumanEval | 37.8 | 40.8 | +3.0 |
| | MATH | 28.3 | 30.7 | +2.4 |
| | GSM8K | — | — | +2.2 |
| **长上下文** | Multi-Query NIAH | 84.2 | 97.0 | **+12.8** |
| | Variable Tracking | 77.0 | 89.0 | +12.0 |

**最震撼的发现**：推理（BBH +5.0）和长上下文（NIAH +12.8）的提升比知识检索（MMLU +3.4）更大。这违反直觉——记忆模块为什么能帮助推理？

### 机理解释（LogitLens + CKA 分析）

Engram 解放早期层：
- 原本：早期层负责重建静态知识（e.g., "Barack Obama" 的 entity embedding）
- 有 Engram：这些静态工作被 O(1) 查找替代，早期层可以做更高级的处理
- **等效于增加了模型的有效推理深度**

Attention 专注全局上下文：
- 原本：部分 attention 头处理局部 N-gram 依赖（本可用查找的）
- 有 Engram：这部分工作被转移，attention 全力处理长距离全局依赖
- **长上下文 NIAH 暴涨 +12.8 的根本原因**

---

## 系统效率：Hardware Bypass 设计

这是 Engram 最有战略意义的部分。

### 训练阶段
- 嵌入表跨 GPU 分片（All-to-All 通信）
- 总记忆容量线性扩展于 GPU 数量
- 可训练 100B+ 参数的记忆表

### 推理阶段（最关键）
- **完全确定性**：检索 index 只取决于输入 token 序列，无需运行时计算
- **异步预取**：在 GPU 执行前几个 Transformer block 时，CPU 通过 PCIe 提前把需要的 embedding 行预取到 GPU
- **通信与计算重叠**：实际开销 < 3%（即使把 100B 嵌入表卸载到主机内存）

**战略意义**：GPU HBM 是中国最难获得的硬件资源（也是美国芯片出口管制的核心目标）。Engram 的参数可以存在便宜的系统 DRAM 而非昂贵的 HBM——DeepSeek 正在构建一个**不依赖 H100 级别 HBM 的 frontier 模型栈**。

---

## 批判性分析

### 真正 Novel 的点

1. **Bi-axial Sparsity 框架**：把 MoE（计算稀疏）和 Engram（记忆稀疏）视为互补的两个轴，Sparsity Allocation 是新的设计空间维度
2. **U-shaped scaling law**：不是 claim，是系统性实验发现，有理论意义
3. **推理增益 > 知识增益**的反直觉结果，并有机理解释（早层解放 + attention 专注化）
4. **<3% 开销的 100B 参数 host memory 卸载**：engineering 突破，彻底解耦参数规模和 GPU HBM

### 疑问与待验证

1. **哈希碰撞**：N-gram → hash 有碰撞损失，对长尾知识效果如何？论文用 K 个哈希头缓解，但没有详细分析碰撞率
2. **上下文依赖知识**的处理：有些知识本身是上下文相关的（"the president" 指谁取决于年份），gate 能处理吗？
3. **V4 真实规模**：27B 的结果令人信服，但 V4 可能是 600B+ MoE，Engram 在这个规模的表现未知
4. **冷启动问题**：N-gram 嵌入表需要从预训练数据中学出来，这多少数据才够？有多少训练成本？

### 与 GLM-5 DSA 的比较

GLM-5 用的是 DeepSeek Sparse Attention（content-aware dynamic sparsity），解决的是 attention 计算的稀疏化。Engram 解决的是知识存储的稀疏化——两者互补，都是"找到不必要的计算并删掉"的哲学，但 target 不同。

### 历史定位

这是继 MoE 之后第一个真正意义上的新稀疏轴。MoE 让参数稀疏激活（只激活部分 expert），Engram 让知识稀疏存储（只查找需要的记忆）。如果 DeepSeek V4 验证了在大规模生产环境中的有效性，这将成为 LLM 架构的标准组件。

---

## DeepSeek V4 预期架构（三大创新）

根据已知信息（待 V4 正式发布后核实）：

1. **Engram Conditional Memory**（本文）— 记忆稀疏轴，O(1) lookup + host memory bypass
2. **Manifold-Constrained Hyper-Connections (mHC)**（arxiv 待查）— 多分支 backbone，替代 residual stream
3. **Dynamic Sparse Attention + Lightning Indexer** — 稀疏 attention 高效实现

三者结合：Bi-axial Sparsity（MoE+Engram）× 多分支信息流（mHC）× 高效 Attention（DSA）= 参数规模与计算效率的双重突破。

---

## 与 Vault 其他笔记的连接

- → [[AI/4-模型/GLM/GLM-5|GLM-5-技术报告精读]] (DSA attention 稀疏化，同类思路)
- → [[AI/3-LLM/RL/算法/Slime-RL-Framework|Slime-RL-Framework]] (DeepSeek post-training 框架)
- → [[AI/3-LLM/Architecture/ReFINE-Fast-Weight-RL-Next-Sequence-Prediction|ReFINE-Fast-Weight-RL-Next-Sequence-Prediction]] (Fast weight 的记忆轴，不同实现)
- → [[AI/3-LLM/Inference/MAGE-Block-Diffusion-LLM-Sparse-Attention|MAGE-Block-Diffusion-LLM-Sparse-Attention]] (稀疏注意力，同日读)
- → [[AI/3-LLM/Inference/KV Cache|KV Cache 优化]] (记忆 vs 计算的权衡)
- → [[AI/3-LLM/Architecture/MoE 深度解析|MoE 架构]] (Engram 的互补轴)
