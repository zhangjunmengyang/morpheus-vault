---
title: "Gemini 3 Deep Think"
brief: "Google DeepMind 2026-02-12 发布 Gemini 3 Deep Think：具备 Extended Thinking（链式推理扩展）能力的旗舰推理模型，在 AIME 2025/Codeforces 等高难度 benchmark 超越 o3；分析 Deep Think 模式的 token 使用特征和推理效率。"
type: model
domain: ai/llm/inference
tags:
  - ai/llm/inference
  - type/model
  - model/gemini
created: 2026-02-16
---

# Gemini 3 Deep Think

> 发布：2026-02-12 (Google DeepMind)
> 来源：[Google Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-deep-think/)
> 关键词：test-time compute, inference-time scaling, reasoning, self-verification

## 概述

Gemini 3 Deep Think 是 Google 对 Gemini 3 Pro 的 **reasoning mode 升级**，核心是 scaled inference-time compute：模型在生成答案前分配更多算力做内部验证和 self-correction。

## Benchmark 成绩

| Benchmark | 成绩 | 意义 |
|-----------|------|------|
| **ARC-AGI-2** | 84.6% | ARC Prize Foundation 验证；人类 ~60%，之前模型 <20% |
| **Humanity's Last Exam** | 48.4% (无工具) | 超过 GPT-5.2 和 Claude Opus 4.6 |
| **Codeforces** | 3455 Elo | Legendary Grandmaster 级别 |
| **IPhO/IChO/IMO 2025** | 金牌水平 | 物理/化学/数学奥赛 |
| **CMT-Benchmark** | 50.5% | 高级理论物理 |

## 核心技术：Scaled Inference-Time Compute

不是简单的 next token prediction，而是：
1. **Internal verification** — 生成推理路径后，内部检验逻辑一致性
2. **Self-correction** — 发现错误路径后回溯修正
3. **Extended thinking** — 允许模型 "想更久" 再输出

这是 o1/R1 路线的延续和升级：

```
o1 (OpenAI, 2024) → DeepSeek-R1 (2025) → Gemini 3 Deep Think (2026)
                                            ↑
                                     ARC-AGI-2 超人类平均
```

## 技术分析

### 为什么 ARC-AGI-2 84.6% 重要

ARC-AGI 不测记忆，测的是 **novel task generalization**：
- 每个 puzzle 都是新的 visual pattern reasoning 任务
- 模型必须从几个 example 中归纳规则并泛化
- 84.6% 超过人类平均（~60%），说明模型在做真正的 abstract reasoning

### 但要保持怀疑

1. ARC-AGI-2 主要是 **visual grid pattern reasoning**，不等于 general intelligence
2. Test-time compute 本质是 **用推理算力换准确率**，成本可能很高
3. "超过人类平均" ≠ "超过人类专家"
4. Self-verification 的可靠性在 open-ended 任务上存疑

### 与竞品对比

| 模型 | 路线 | 推理方式 |
|------|------|----------|
| o1/o3 | Chain-of-thought + verification | 显式 CoT |
| DeepSeek-R1 | GRPO + RL-trained reasoning | 长 CoT |
| **Gemini 3 DT** | **TTC + internal verification** | 内部验证（不一定输出 CoT） |

## 实际能力亮点

- **Agentic coding** — 接受高层目标，自主完成多文件复杂方案
- **3D 建模** — 从 2D sketch 生成 3D-printable 文件
- **科研辅助** — 可以解读实验数据、建模物理系统

## 面试要点

1. **Test-time compute scaling** 是什么？为什么 work？
   - 允许模型在推理时分配更多计算做验证/搜索
   - 类似人类 "多想一会儿" 的机制
   - 核心 trade-off：latency/cost vs accuracy

2. **与 training-time scaling 的关系**
   - Chinchilla scaling law 是 training-time 的
   - TTC 是 inference-time 的补充维度
   - 两者不矛盾，是 orthogonal 的

3. **ARC-AGI 为什么是 AGI 的 litmus test？**
   - 测 novel task generalization，不是 memorization
   - 但仍是受限的视觉推理域，不代表 full AGI

## See Also

**Aletheia 系列（Gemini 3 Deep Think 作为基础模型的最高难度实测）**
- [[Aletheia-Gemini3-DeepThink-FirstProof|Aletheia FirstProof（arXiv:2602.21201，2026-02-25）]] ⭐ — **Gemini 3 Deep Think 迄今最强能力展示**：基于本模型的 Aletheia agent 在 FirstProof 挑战赛自主解决 6/10 研究级数学题，包括 Weinberger 书中的开放问题；P7 推理计算量比以往观测高一个数量级，验证 TTC scaling 对真正困难问题有质的影响
- [[Aletheia-Math-Research-Agent|Aletheia（前作：Erdős 数据库）]] — 同一基础模型的前一阶段挑战

**推理体系**
- [[LLM-推理优化-2026-全景|LLM 推理优化 2026 全景]] — TTC scaling 在推理优化整体框架中的位置
- [[AI/3-LLM/Inference/SpeculativeDecoding|Speculative Decoding]] — 推理加速体系
- [[AI/3-LLM/Inference/KV-Cache优化|KV Cache 优化]] — 推理内存优化

---
*Created: 2026-02-16 by Scholar heartbeat | See Also updated: 2026-02-25 by Librarian*
