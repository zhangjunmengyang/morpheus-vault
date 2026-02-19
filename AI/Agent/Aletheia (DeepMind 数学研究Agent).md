---
tags: [ai-agent, reasoning, math, deepmind, inference-scaling]
created: 2026-02-15
source: https://www.marktechpost.com/2026/02/12/google-deepmind-introduces-aletheia-the-ai-agent-moving-from-math-competitions-to-fully-autonomous-professional-research-discoveries/
paper: https://github.com/google-deepmind/superhuman/blob/main/aletheia/Aletheia.pdf
status: complete
---

# Aletheia — DeepMind 数学研究 Agent

> [!note] 深度版本
> 本文件为早期概览版（2026-02-15）。完整深度分析见 [[AI/Agent/Aletheia-Math-Research-Agent]]（2026-02-19，163行，含架构细节、科研成果分类、设计启示）。

## 一句话

Google DeepMind 推出的数学研究 Agent，从竞赛解题跨越到**全自主科研论文生成**，在 Erdős 猜想数据库中自主解决了 4 个开放问题。

## 核心架构：Agentic Loop

三组件循环，显式分离生成与验证：

```
Generator → Verifier → Reviser → (循环直到通过)
```

- **Generator**: 基于 Gemini Deep Think，提出候选解法
- **Verifier**: 自然语言非形式化验证，检查逻辑缺陷和幻觉
- **Reviser**: 修正 Verifier 发现的错误，直到最终输出被批准

**关键洞察**: 将验证从生成中解耦是必要的——模型在生成时会忽略自己的错误，但作为独立 Verifier 时能发现这些缺陷。这与 self-consistency / best-of-N 不同，是结构化的角色分离。

## 核心技术发现

### 1. Inference-Time Scaling 的威力

- "Thinking longer" 显著提升准确率
- Deep Think 2026.01 版本：IMO 级问题所需计算量比 2025 版降低 **100x**
- IMO-Proof Bench Advanced: **95.1%** 准确率（上一记录 65.7%）

### 2. 工具使用防幻觉

- 引用幻觉是数学推理的大问题（编造不存在的定理/论文）
- 解决方案：集成 Google Search + Web Browsing，实时检索真实数学文献
- 这比纯 CoT 更可靠——数学研究需要与已有文献对话

### 3. FutureMath Basic

- DeepMind 内部基准：PhD 级数学练习
- Aletheia 达到 SOTA

## 里程碑成果

| 成果 | 自主程度 | 说明 |
|------|----------|------|
| **Feng26 论文** | 全自主 (Level A2) | 算术几何领域，计算 eigenweights 结构常数，无人类干预 |
| **LeeSeo26** | 协作 | Agent 提供高层策略 roadmap，人类完成严格证明 |
| **Erdős 猜想** | 全自主 | 700 个开放问题中找到 63 个技术正确解，**解决 4 个开放问题** |

## AI 自主性分类法

DeepMind 提出类似自动驾驶分级的数学 AI 分类：

| Level | 自主程度 | 研究意义 |
|-------|----------|----------|
| L0 | 主要靠人类 | 可忽略新颖性（竞赛级） |
| L1 | 人-AI 协作 | 较小新颖性 |
| L2 | 本质上自主 | 可发表质量 |

Feng26 被评为 **Level A2**（本质自主 + 可发表质量）。

## 面试视角

### Q: Aletheia 的 Agentic Loop 与 Tree of Thoughts / Self-Consistency 有何区别？

ToT 和 Self-Consistency 是在同一角色内做多路径探索或投票。Aletheia 的创新在于**结构化角色分离**：Generator 专注生成、Verifier 专注找错、Reviser 专注修正。这种分工让模型能发现自己在生成阶段忽略的错误——因为验证时的 attention pattern 和生成时不同。

### Q: Inference-Time Compute Scaling 的上限在哪？

Aletheia 表明 100x 计算量降低是可能的（通过模型迭代优化而非堆算力）。但当问题超出模型知识边界（需要真正的新洞察而非组合已有知识），scaling 的收益会递减。Erdős 猜想 700 个问题只解决 4 个（0.57%）说明当前上限。

### Q: 为什么工具使用对数学推理如此重要？

数学推理的幻觉不同于一般 LLM 幻觉——编造的定理可能看起来完全合理但逻辑上有缺陷。通过 Search + Browse 锚定到真实文献，可以：(1) 验证引用是否存在 (2) 确认前置定理的正确陈述 (3) 避免重新发明已有结果。

## 与其他工作的关系

- [[Inference-Time Scaling]] — Aletheia 是 test-time compute 的极致应用
- [[AI Agent 架构]] — Generator-Verifier-Reviser 模式可推广到其他研究领域
- [[数学推理]] — 从 AlphaProof (IMO 2025) 到 Aletheia，数学 AI 的自主性跃迁
- [[幻觉问题]] — 工具增强验证 vs 纯内部验证的优劣

## 启发

1. **验证与生成的分离是可推广的设计模式** — 代码生成、科学发现、量化策略回测都可以用
2. **Test-time compute 还远没到天花板** — 算法优化比堆硬件更有效
3. **Agent 做研究的范式正在成熟** — 不是替代科学家，而是 L1-L2 级的协作/自主研究工具
