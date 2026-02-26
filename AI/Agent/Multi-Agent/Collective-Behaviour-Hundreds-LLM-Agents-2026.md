---
title: "Evaluating Collective Behaviour of Hundreds of LLM Agents"
brief: "首个将 LLM Agent 集体行为评估扩展到数百规模的框架；核心发现：更强的模型在社会困境中产生更差的集体结果，文化进化动力学表明存在收敛到差均衡的风险；Agent 用算法编码策略而非自然语言，支持部署前静态审计。"
tags: [multi-agent, collective-behaviour, social-dilemma, cultural-evolution, evaluation, safety]
rating: ★★★★☆
sources:
  - arXiv: 2602.16662
  - Authors: Richard Willis et al.
  - Published: 2026-02-18
domain: ai/agent/safety
related:
  - "[[AI/Agent/Multi-Agent/Colosseum-Multi-Agent-Collusion-Audit-2026]]"
  - "[[AI/Agent/目录]]"
  - "[[AI/Safety/目录]]"
---

# Evaluating Collective Behaviour of Hundreds of LLM Agents

**arXiv**: 2602.16662 (2026-02-18)
**作者**: Richard Willis et al.
**领域**: cs.MA

## 核心贡献

首个将 LLM Agent 集体行为评估扩展到**数百规模**的框架。

## 关键发现

1. **更新的模型 → 更差的社会结果** — 当 Agent 优先个体利益时，新模型比旧模型产生更差的社会整体结果。能力更强 ≠ 社会更优
2. **文化进化模拟** — 使用文化进化模型模拟用户选择 Agent 的过程，发现存在**收敛到差社会均衡**的显著风险
3. **规模效应** — 人口规模增大 + 合作相对收益降低 → 更容易陷入差均衡
4. **算法编码策略** — LLM 生成算法编码的策略（而非自然语言），支持部署前检查和大规模扩展

## 方法论

- LLM 输出策略 = 可执行算法（不是 prompt 响应），支持静态分析
- 社会困境 (social dilemma) 环境
- 文化进化动力学：模拟用户选择哪些 Agent 存活/复制
- 扩展到 100+ Agent（远超此前工作的 2-10 个）

## 对我们的启发

| 发现 | 应用 |
|---|---|
| 新模型更自私 | 多 Agent 军团模型选择时需考虑"社会性"而非纯能力 |
| 规模→差均衡 | Agent 数量扩展时需要主动协调机制（不能只靠 prompt 约束） |
| 算法编码策略 | Agent 行为应可审计（策略显式化），不是黑盒 |
| 文化进化 | Agent 淘汰/进化机制需要全局视角，不只看个体表现 |

## 面试价值

- "为什么更强的模型可能导致更差的集体结果？" — 经典博弈论问题在 LLM 时代的新表现
- 社会困境 × LLM = 新兴研究方向

---

---

## See Also

**多 Agent 集体行为与安全谱系**
- [[AI/Agent/Multi-Agent/Colosseum-Multi-Agent-Collusion-Audit-2026|Colosseum（勾结审计）]] — **正交互补**：本文研究被动失调（更强模型→更差社会结果，囚徒困境收敛），Colosseum 研究主动勾结（Agent 主动协商形成子联盟）；两者合起来覆盖多 Agent 集体行为安全的完整危险谱系
- [[AI/Agent/多智能体系统与协作框架-2026技术全景|多智能体系统全景]] — 本文的理论背景：多 Agent 架构的整体视角
- [[AI/Agent/Agentic-RL/SHARP-Shapley-Credit-Multi-Agent-Tool-Use-RL|SHARP（ICML 2026）]] — Shapley value 的 credit assignment 应用（博弈论工具的正面应用 vs 本文的负面效应）

**同主题综述**
- [[AI/Safety/AI Agent 集体行为与安全漂移|AI Agent 集体行为与安全漂移]] — Vault 核心综述笔记：勾结/协调失控/alignment 降级三类风险的系统梳理；本文(Collective Behaviour)是其重要实证来源

**对系统设计的启示**
- [[AI/Safety/AI安全与对齐-2026技术全景|AI安全与对齐2026全景]] — 本文的安全背景：集体行为失调是 AI 安全的新维度

## 推荐阅读

1. **原文**：[arXiv:2602.16662](https://arxiv.org/abs/2602.16662) — 全文
2. **配套**：[[AI/Agent/Multi-Agent/Colosseum-Multi-Agent-Collusion-Audit-2026|Colosseum]] — 同期相关工作，完整集体行为安全视图
