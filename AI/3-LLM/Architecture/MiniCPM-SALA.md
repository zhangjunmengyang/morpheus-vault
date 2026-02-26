---
title: "MiniCPM-SALA: Sparse + Linear Attention 混合架构"
brief: "面壁智能（arXiv:2602.11761）提出 SALA：9B 参数，1:3 sparse:linear attention 层比例混合架构，单张 A6000D 实现 256K 上下文 3.5× 加速，支持 1M token；在 RULER/LongBench 上与纯 Sparse Attention 模型持平，效率大幅提升。"
date: 2026-02-20
tags: [论文, 架构, 注意力, 长上下文, 效率]
source: "arXiv:2602.11761"
---

# MiniCPM-SALA: Sparse + Linear Attention 混合架构

> **一句话**: 9B 参数混合架构，用 1:3 的 sparse:linear attention 层比例，在单张 A6000D 上实现 256K 长度 3.5× 加速，支持 1M token 上下文。

## 核心问题

标准 Transformer full attention 的两个瓶颈：
1. **计算瓶颈**: O(N²) 复杂度，序列长度翻倍计算量翻四倍
2. **内存瓶颈**: KV Cache 随序列长度线性增长，8B 模型在百万 token 时 OOM

现有方案各有缺陷：
- **Sparse Attention** (如 InfLLM-V2): 保真度高但仍需存大量 KV
- **Linear Attention** (如 Lightning Attention): O(N) 全局效率但信息压缩有损

## 方法

### 混合架构设计
- **Sparse Attention 层 (25%)**: 用 InfLLM-V2，处理需要精确长距离依赖的层
- **Linear Attention 层 (75%)**: 用 Lightning Attention，处理全局信息聚合
- **比例**: 1:3 (sparse:linear)，通过层选择算法确定哪些层用 sparse
- **HyPE (Hybrid Positional Encoding)**: 混合位置编码，适配两种注意力机制

### 层选择算法
不是随机分配，而是根据每层对长上下文任务的重要性分析，选出最关键的 25% 层用 sparse attention。

### 低成本迁移训练
- 从已有的预训练 Transformer 模型迁移，而非从头训练
- **训练成本降低约 75%**
- 关键: continual training framework，把 full-attention 模型转换为混合模型

## 关键结果

| 指标 | MiniCPM-SALA | Full Attention 基线 |
|------|-------------|-------------------|
| 通用能力 | ≈ 基线 | 基线 |
| 256K 推理速度 | **3.5× 加速** | 1× |
| 最大上下文 | **1M tokens** | OOM (单卡 A6000D) |
| 参数量 | 9B | 8B |

## 面试价值

### 可能的面试题
**Q: 如何在不损失模型能力的前提下支持超长上下文？**

要点:
- 不是所有层都需要 full attention，关键层用 sparse，其余用 linear
- 1:3 比例是实验确定的最优点（太多 sparse = 效率差，太多 linear = 质量降）
- HyPE 解决两种 attention 机制的位置编码兼容问题
- 从已训练模型迁移比从头训练节省 75% 成本

### 与其他方案的关系
- **vs DeepSeek V4 (Engram)**: DeepSeek 用条件记忆层分离静态/动态，SALA 用 attention 类型混合分离精确/高效
- **vs SLA2**: SLA2 (arXiv:2602.12675) 用 learnable router 动态选择每次计算用 sparse 还是 linear，更灵活但训练更复杂
- **vs Flash Attention**: Flash Attention 优化 IO 不改计算复杂度，SALA 从架构层面改变计算复杂度

## 个人评价

这是 Transformer 架构演进的重要信号——"混合 attention" 正在成为主流范式。不再是"选 sparse 还是 linear"，而是"怎么混合"。1:3 的发现也很实用：大部分计算不需要精确的全局注意力。

与 DeepSeek V4 的 Engram 思路异曲同工：都在尝试把"不是所有计算都同等重要"这个洞察编码进架构中。

---

*Source: arXiv:2602.11761, OpenBMB MiniCPM Team, 2026-02*

---

## See Also

- [[Attention 变体综述|Attention 变体综述]] — SALA 是 Attention 变体的一种
- [[SLA2-Learnable-Router|SLA2 Learnable Router]] — 同期 MiniCPM 架构论文，互补
- [[Transformer|Transformer 通识]] — SALA 的架构基础
- [[AI/3-LLM/目录|LLM MOC]] — 大语言模型知识全图谱
