---
title: "SideQuest：Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning"
date: 2026-03-01
updated: 2026-03-01
arxiv: 2602.22603
tags:
  - AI/3-LLM/Inference
  - KVCache
  - long-context
  - agentic-reasoning
  - systems
---

## TL;DR

长时 agentic 任务（deep research / 多网页多跳）里，context 往往被“外部检索 token”淹没，导致 KV cache 内存爆炸、decode 性能下降。
SideQuest 的关键点不是“再来一个启发式剪枝”，而是：**让 LRM 自己推理哪些 token 对后续推理有用**，用模型做 KV 压缩的“选择器”。

同时它把 KV 压缩框成一个 **parallel auxiliary task**，避免“管理过程的 token”反过来污染主任务的记忆（这是一个很 agent/harness 的视角）。

## 论文信息
- Paper: *SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning*
- arXiv:2602.22603 (2026-02-26)
- Authors: Sanjay Kariyappa, G. Edward Suh

## 核心问题（我认可的定义）

- Agentic long-horizon = 多轮工具调用 + 大量网页/文档 observation。
- 这些 observation token 在 context 中占比极高，KV cache 会随着 step 快速增长。
- 已有 KV 压缩/长上下文优化多为“通用启发式”（例如按位置/注意力/相似度等），但论文 claim：**这些 heuristics 对 multi-step reasoning models 支持不好**。

> 直观解释：multi-step reasoning 的“关键 token”不是局部显著性，而是跨步的因果/约束条件；启发式很容易把未来需要的“桥接 token”删掉。

## SideQuest 方法（从摘要抽取的最小机制）

1) **LRM-as-compressor**：让 Large Reasoning Model 自己判断 token usefulness → 指导 KV cache compression。
2) **Auxiliary task in parallel**：KV 管理作为并行辅助任务执行；
   - 目的：避免“管理 token”（例如总结、打标签、筛选解释）进入主推理轨迹导致记忆污染。

## 结果（摘要）
- 用仅 215 samples 训练出的模型，即可在 agentic tasks 上：
  - peak token usage ↓ up to **65%**
  - accuracy 几乎不掉（minimal degradation）
  - outperform heuristic-based KV cache compression

## 我对 Vault 主线的映射（判断）

- 这篇本质上把“KV cache 管理”从纯系统层，提升为 **Agent harness 的决策问题**：
  - 需要一个 *selector policy*：哪些 observation 值得留、以什么粒度留。
- “parallel auxiliary task 防污染”很关键：
  - 你如果用同一条对话历史去做管理，会把管理解释当成事实写进上下文，导致 distribution shift。
  - SideQuest 把管理与主任务解耦，等价于一种“memory metabolization 的隔离层”。

## Open Questions（待后续精读 PDF）

- usefulness 的形式化是什么？（分类？打分？基于 next-step 预测的可用性？）
- 并行辅助任务如何实现：
  - 是另开一条模型调用？还是同模型多头？
  - KV 压缩是 hard prune 还是低秩/量化/聚合？
- 与 PageKV / chunked prefill / prefix caching 的组合方式：SideQuest 更像“内容选择器”，可叠加底层分页与压缩。

## Related
- SMTL（2602.22675）：强调 context management；SideQuest 更聚焦 KV cache 的选择与压缩层。
