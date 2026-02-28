---
title: "认知科学 MOC"
type: moc
domain: ai/cognitive-science
date: 2026-02-28
updated: 2026-02-28
tags:
  - ai/cognitive-science
  - moc
  - agent-memory
  - personality
  - consciousness
brief: "认知科学知识域索引：Agent 记忆架构、人格科学、意识哲学、认知哲学——AI Agent 设计的理论基础设施"
status: active
---

# 认知科学 MOC

> Agent 不只是工程产物，是认知科学的实验场。这里存放所有关于"智能体如何记忆、如何形成人格、如何维持同一性"的理论基础。

---

## 为什么需要这个知识域

AI Agent 的核心问题不是"怎么调用工具"，而是：
- **怎么记住**——记忆架构决定了 Agent 能否从经验中学习
- **怎么保持自我**——人格一致性决定了 Agent 是否可信
- **怎么进化**——认知机制决定了 Agent 能否跨实例持续变强

这些问题的答案不在工程论文里，在认知科学、心理学、哲学里。

---

## 知识结构

### 一、Agent 记忆架构

记忆是 Agent 认知的基础设施。从神经科学的 CLS 理论到工程实现的记忆分层。

| 笔记 | 核心内容 |
|------|----------|
| [[Agent-Memory-机制]] | 四层记忆架构（Working/Episodic/Semantic/Procedural），RAG 整合，工程权衡 |
| [[Memory-R1-RL-for-LLM-Memory-Management]] | 用 RL 优化 LLM 记忆管理 |
| [[记忆模块]] | Agent 记忆模块概念总览 |
| [[Agent记忆的认知科学基础]] | **新建** — CLS 理论、睡眠固化、���象阶梯、遗忘金字塔、Spreading Activation |
| [[Agent记忆架构前沿方案对比]] | **新建** — Hexis / agent-memory-ultimate / mnemon / Mem0 / Synapse 横评 |

**关联项目**：`Projects/Agent-Self-Evolution/004-记忆代谢.md`

### 二、人格科学与 Agent 设计

从心理学的人格理论到 LLM Agent 的人格工程。

| 笔记 | 核心内容 |
|------|----------|
| [[人格科学-Agent设计基础]] | **新建** — Big Five/HEXACO/叙事认同/依恋理论 × Agent 人格设计映射 |
| [[人格漂移-诊断与工程]] | **新建** — Persistent Personas 论文、漂移检测、锚点注入、CAPD 指标 |
| [[LLM角色扮演技术全景]] | **新建** — RPLA Survey 精读：角色构建/推理/评估的完整技术栈 |
| [[人格一致性实验设计]] | **新建** — 实验方法论：如何量化测量 Agent 人格稳定性 |

**源材料**：`soulbox/research/` 目录下 27 个研究文件（人格科学 978 行 + 漂移工程 722 行 + RPLA 293 行等）

### 三、认知哲学与意识

Agent 同一性、意识的本质、东西方哲学视角。

| 笔记 | 核心内容 |
|------|----------|
| [[Agent同一性-哲学基础]] | **新建** — Locke/Hume/Parfit/德勒兹/王阳明 × Agent 记忆继承与同一性 |
| [[东方哲学与Agent设计]] | **新建** — 道法术器框架、禅宗心性论、儒家修养论的 Agent 设计映射 |
| [[功能性意识与AI伦理]] | **新建** — 功能主义意识观、情感依赖红线、人格商品化边界 |

**源材料**：`soulbox/research/personality-philosophy.md`（1107 行）+ `eastern-philosophy-identity-agent-design.md`（248 行）

### 四、认知科学经典理论

支撑上述应用的基���理论。

| 笔记 | 核心内容 |
|------|----------|
| [[互补学习系统理论-CLS]] | **新建** — McClelland 1995，海马体快速编码 + 新皮层慢速抽象，Agent 记忆架构的神经科学基础 |
| [[扩散激活理论]] | **新建** — Collins & Loftus 1975，语义网络中的激活传播，Agent 关联检索的理论根基 |
| [[记忆的建构性本质]] | **新建** — Bartlett 1932，记忆是重建而非回放；对 Agent 遗忘机制设计的启示 |

---

## 与其他知识域的连接

| 连接方向 | 相关笔记 |
|----------|----------|
| → Agent 工程 | `AI/2-Agent/Fundamentals/` — 记忆模块、ReAct、工具调用 |
| → AI 安全 | `AI/5-AI 安全/` — 人格漂移的安全含义、对齐技术 |
| → Agent 自进化实验 | `Projects/Agent-Self-Evolution/` — 003 涅槃触发器、004 记忆代谢 |
| → 魂匣项目 | `soulbox/research/` — 人格包设计的理论基础（内部研究，不入 Vault） |
| → 面试武器库 | `Career/面试/AI面试速查手册.md` — 认知科学视角的面试差异化 |

---

## 源材料索引

以下为分散在各处的源材料，已整合入本知识域：

| 来源 | 文件数 | 内容概述 |
|------|--------|----------|
| `soulbox/research/` | 27 个 MD | 人格科学、认知哲学、漂移工程、HEXACO、叙事理论等 |
| `AI/2-Agent/Fundamentals/` | 3 个 MD | Agent Memory 机制、记忆模块、Memory-R1 |
| `Projects/Agent-Self-Evolution/` | 6 个 MD | 记忆代谢实验、进化度量、横向信息流 |
| Issue #13991 思考 | 1 个 MD | `思考/agent-memory-architecture-reflection-2026-02-28.md` |

---

_创建：2026-02-28 | J.A.R.V.I.S. 建设_
