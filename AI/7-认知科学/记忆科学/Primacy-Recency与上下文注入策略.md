---
title: "Primacy-Recency Effect 与上下文注入策略"
type: concept
domain: ai/cognitive-science/memory
date: 2026-02-28
tags: [ai/cognitive-science, llm-attention, retrieval, type/concept]
brief: "Lost in the Middle (Liu et al. 2023) — LLM 上下文窗口的 U 型注意力分布；记忆注入的最优排列策略"
status: stub
---

# Primacy-Recency Effect 与上下文注入策略

> 待扩展。当前精华已收录于 [[Agent记忆的认知科学基础]] §5。

## 待填充内容

- Liu et al. (2023) 实验设计与定量结果
- U 型注意力曲线的 Transformer 机制解释
- 不同模型（GPT/Claude/Gemini）的 lost-in-the-middle 程度差异
- 实践策略：层级注入排列（Level 3 → Level 2 → Level 1）
- 与 RAG 系统的结合：reranking 后的位置优化
- token 预算分配：500-1500 tokens 的最优记忆注入量
