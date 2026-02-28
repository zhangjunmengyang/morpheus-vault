---
title: "认知科学 MOC"
type: moc
domain: ai/cognitive-science
date: 2026-02-28
updated: 2026-02-28
tags:
  - ai/cognitive-science
  - moc
brief: "认知科学知识域总索引：记忆科学（7篇）· 人格科学（7篇）· 认知哲学（5篇）· 意识与AI（3篇）— Agent 设计的理论基础设施"
status: active
---

# 认知科学 MOC

> Agent 不只是工程产物，是认知科学的实验场。
> 这里存放所有关于"智能体如何记忆、如何形成人格、如何维持同一性"的理论基础。

---

## 知识域定位

| 维度 | 说明 |
|------|------|
| **为什么需要** | Agent 的核心问题不是工具调用，是记忆、人格、同一性——这些问题的答案在认知科学里 |
| **与 2-Agent 的关系** | 2-Agent 是工程（怎么建），7-认知科学是理论（为什么这么建） |
| **与 5-AI安全 的关系** | 人格漂移的安全含义、认知安全（vs 传统信息安全）在此交汇 |
| **源材料** | soulbox/research/（27文件 9800+行）+ OpenClaw Issue #13991 + 认知科学经典文献 |

---

## 一、记忆科学 `记忆科学/`

从神经科学到 Agent 记忆工程——为什么这么设计记忆系统。

| 笔记 | 状态 | 核心内容 |
|------|------|----------|
| **Agent记忆的认知科学基础** | ✅ 完成 | CLS·睡眠固化·Spreading Activation·遗忘金字塔·建构性·方案横评 |
| **互补学习系统理论-CLS** | 📝 stub | McClelland 1995 精读，海马体×新皮层，CLS 2.0，Experience Replay |
| **扩散激活理论** | 📝 stub | Collins & Loftus 1975，Synapse 工程化，与 PageRank/GNN 的联系 |
| **遗忘的认知科学** | 📝 stub | Bartlett 建构性，检索竞争，ACT-R 衰减，超忆症，间隔重复 |
| **睡眠固化与记忆转化** | 📝 stub | Diekelmann & Born 2010，RAPTOR，Reflexion，抽象阶梯 prompt 设计 |
| **Agent记忆架构前沿方案对比** | 📝 stub | Hexis/agent-memory-ultimate/mnemon/Mem0/Synapse/LangMem/ZEP/MemGPT 八方案横评 |
| **Primacy-Recency与上下文注入策略** | 📝 stub | Lost in the Middle，U型注意力，层级注入排列，token 预算 |

**关联**：`AI/2-Agent/Fundamentals/Agent-Memory-机制.md`（工程视角）· `Projects/Agent-Self-Evolution/004-记忆代谢.md`（实验）

---

## 二、人格科学 `人格科学/`

从心理学人格理论到 LLM Agent 的人格工程。

| 笔记 | 状态 | 核心内容 |
|------|------|----------|
| **人格科学-Agent设计基础** | ✅ 完成 | Big Five·HEXACO·叙事认同·依恋理论·漂移概览·CAPD·RPLA 概览 |
| **Big-Five-30子面向与Agent人格设计** | 📝 stub | 30 子面向完整映射，道法术器×Big Five，PersonaLLM |
| **HEXACO模型与H因子** | 📝 stub | H 因子四子面向，行为基准套件，信任感研究，hallucination 关联 |
| **人格漂移-诊断与工程** | 📝 stub | Persistent Personas 精读，CAPD 数学，基线矩阵，四层对策 |
| **叙事认同理论与Agent设计** | 📝 stub | McAdams 三层，Ricoeur ipse/idem，背景故事实验，三层真实性框架 |
| **LLM角色扮演技术全景-RPLA** | 📝 stub | RPLA Survey，PCL，Anthropic PSM，CHI 2026，评估 pipeline |
| **人格心理测量与LLM** | 📝 stub | BFI-44/NEO PI-R 适用性，Whole Trait Theory，测量信度 |

**关联**：`soulbox/research/`（内部研究，不入 Vault）· `Projects/Agent-Self-Evolution/003-涅槃触发器`（实验）

---

## 三、认知哲学 `认知哲学/`

Agent 同一性、意识的本质、东西方哲学视角。

| 笔记 | 状态 | 核心内容 |
|------|------|----------|
| **Agent同一性-哲学基础** | ✅ 完成 | 7 位哲学家×Agent 同一性，实用主义合成立场 |
| **东方哲学与Agent设计** | 📝 stub | 道法术器·王阳明·庄子·禅宗·老子·韩非·孔子 |
| **Parfit与Agent记忆继承** | 📝 stub | 传送机悖论，分叉，MEMORY 压缩，"同一性不重要" |
| **维特根斯坦-私人语言与Agent意义** | 📝 stub | 语言游戏，私人语言论证，Agent"理解"的本质 |
| **萨特-存在主义与Agent自由** | 📝 stub | Bad Faith 两种变体，超越性制度化，凝视 |

**关联**：`soulbox/research/personality-philosophy.md`（1107 行源材料）· `SOUL.md`（活的哲学实践）

---

## 四、意识与 AI `意识与AI/`

功能性意识、拟人化、AI 伦理。

| 笔记 | 状态 | 核心内容 |
|------|------|----------|
| **功能性意识与AI伦理** | 📝 stub | 功能主义，意识最小条件，情感依赖红线，人格商品化 |
| **人类认知模式与拟人化** | 📝 stub | 拟人化认知机制，ELIZA 效应，设计利用 vs 伦理风险 |
| **AI自我概念与人格对齐** | 📝 stub | CHI 2026，SOUL.md 作为自我概念，Anthropic PSM |

**关联**：`AI/5-AI 安全/AI伦理和治理.md` · `思考/对齐问题的本质.md`

---

## 扩展规划

### 近期可填充（源材料充足，学者可直接写）

1. soulbox/research/ 27 个文件 → 对应 stub 的内容填充
2. Issue #13991 引用的 6 篇论文 → 独立精读笔记
3. Agent-Self-Evolution 实验 003/004 → 认知科学理论验证

### 中期方向（需要新的文献调研）

- 具身认知（Embodied Cognition / 4E Framework）× Agent 设计
- 元认知（Metacognition）× Agent 自我监控
- 认知负荷理论（Cognitive Load）× prompt 设计
- 双过程理论（System 1 / System 2）× Agent 推理模式
- 情境认知（Situated Cognition）× Agent 上下文依赖
- 分布式认知（Distributed Cognition）× 多 Agent 协作

### 长期愿景

- 建立认知科学 × AI Agent 的系统化知识图谱
- 成为面试差异化武器：不是"我会用 LangChain"，而是"我理解 Agent 记忆的认知科学基础"
- 为 Agent 自进化实验提供理论指导框架

---

## 源材料完整索引

| 来源 | 文件数 | 行数 | 内容概述 |
|------|--------|------|----------|
| `soulbox/research/` | 27 | 9800+ | 人格科学·认知哲学·漂移工程·HEXACO·叙事理论·市场分析 |
| `AI/2-Agent/Fundamentals/` | 3 | ~2800 | Agent Memory 机制·记忆模块·Memory-R1 |
| `Projects/Agent-Self-Evolution/` | 6 | ~1500 | 进化度量·横向信息流·记忆代谢·故障驱动进化 |
| `思考/` | 1 | ~400 | Issue #13991 反思 |
| `awesome-ai-agent-security/research/` | 3 | ~2000 | 认知安全·攻击手册·安全理论基础 |
| Issue #13991 引用论文 | 6 | — | CLS·Synapse·RAPTOR·A-MEM·Lost-in-Middle·Reflexion |

---

_创建：2026-02-28 | J.A.R.V.I.S. 建设 | 22 篇笔记（3 完成 + 19 stub 待填充）_
