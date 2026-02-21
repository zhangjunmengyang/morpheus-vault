---
title: "Agent 智能体"
type: moc
domain: ai/agent
tags:
  - ai/agent
  - type/moc
updated: 2026-02-22
---

# 🤖 Agent 智能体 — 学习路线图

> 从单 Agent 到 Multi-Agent，从 Tool Use 到 Agentic RL

---

## 第一章 Agent 核心概念

> 理解 Agent 的基本构件：感知、推理、行动、记忆

### 1.1 基础架构

- [[AI/Agent/Fundamentals/Agent or Workflow？|Agent or Workflow？]] — 设计决策
- [[AI/Agent/Fundamentals/ReAct 与 CoT|ReAct 与 CoT]] — 推理范式对比
- [[AI/Agent/ReAct 推理模式|ReAct 推理模式]]
- [[AI/Agent/Fundamentals/分析 Agent 演进的一些思考|Agent 演进思考]]
- [[AI/Agent/Fundamentals/HF Agent Course|HF Agent Course]]
- [[AI/Agent/Fundamentals/HF LLM + Agent|HF LLM + Agent]]
- [[AI/Agent/Fundamentals/Context-Folding 论文|Context-Folding]] — 长程 Agent 论文

### 1.2 记忆机制

- [[AI/Agent/Fundamentals/记忆模块|记忆模块]] — 短期/长期记忆
- [[AI/Agent/Agent Memory 机制|Agent Memory 机制]] — 短期/长期/工作记忆、RAG-based memory、MemGPT/Letta
- [[AI/Agent/Agent World Model|Agent World Model]] — Agentic RL + 合成环境 + 世界模型

### 1.3 综合全景

- [[AI/Agent/AI-Agent-2026-技术全景|🔥 AI Agent 2026 技术全景]] ⭐ — 面试武器库，1114行 ★★★★★

---

## 第二章 工具使用与 MCP

> Agent 的"手"：如何调用工具、Function Calling、MCP 协议

- [[AI/Agent/Fundamentals/Tool Use|Tool Use]] — 工具调用基础
- [[AI/Agent/Agent Tool Use|Agent Tool Use]] — Function Calling / ReAct / API 对比
- [[AI/Agent/LLM工具调用与Function-Calling-2026技术全景|🔥 LLM 工具调用与 Function Calling 2026 全景]] ⭐ — 面试武器级 ★★★★★
- [[AI/Agent/Fundamentals/Code Agent|Code Agent (基础)]]
- [[AI/Agent/Code Agent|Code Agent]] — 深度笔记

### MCP (Model Context Protocol)

- [[AI/Agent/MCP/如何给人深度科普 MCP|如何给人深度科普 MCP]]
- [[AI/Agent/MCP/HF MCP Course|HF MCP Course]]

---

## 第三章 Multi-Agent 系统

> 多 Agent 协作、编排、通信

- [[AI/Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]]
- [[AI/Agent/Multi-Agent/Multi-Agent-架构模式详解|Multi-Agent 架构模式详解]] — Supervisor/Pipeline/Debate三模式含代码实现（馆长重命名自untitled，2026-02-22）★★★★☆
- [[AI/Agent/Multi-Agent/Agent vs MAS|Agent vs MAS]]
- [[AI/Agent/Multi-Agent/Planner|Planner]]
- [[AI/Agent/Multi-Agent/零碎的点|零碎的点]]
- [[AI/Agent/AgentAuditor — Reasoning Tree审计多Agent系统|AgentAuditor]] — Reasoning Tree 审计
- [[AI/Agent/IMAGINE — 多Agent蒸馏到单模型|IMAGINE]] — 多 Agent 蒸馏到单模型
- [[AI/Agent/Kimi-K2.5-PARL|Kimi K2.5 & PARL]] — 并行多 Agent 强化学习
- [[AI/Agent/GitHub-Agentic-Workflows|GitHub Agentic Workflows]]

---

## 第四章 Agentic RL 训练 ⭐

> 用强化学习训练更强的 Agent（前沿方向）

### 4.1 综合分析

- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|🔥 Agentic RL 2026 前沿综合分析]] ⭐ — 四大维度框架
- [[AI/Agent/Agentic-RL/Agent-RL-训练实战指南|🔥 Agent RL 训练实战指南]] ⭐ — 1001行，面试可用 ★★★★★
- [[AI/Agent/Agentic-RL/Agentic RL Survey|Agentic RL Survey]]
- [[AI/Agent/Agentic-RL/Agentic RL Training|Agentic RL Training]]

### 4.2 Workflow/Topology 自动化

- [[AI/Agent/Agentic-RL/FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer (CWRPO)]] — Workflow via End-to-End RL ★★★☆
- [[AI/Agent/AgentConductor-Topology-Evolution-Multi-Agent-Code|AgentConductor]] ⭐ — RL 动态生成 DAG topology ★★★★
- [[AI/Agent/Agentic-RL/SquRL-Dynamic-Workflow-Text-to-SQL|SquRL]] — Dynamic Workflow for Text-to-SQL ★★★

### 4.3 训练优化

- [[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — Phase-Aware MoE 解决 Simplicity Bias ★★★★
- [[AI/Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong]] ⭐ — 极长任务训练方案 ★★★★★
- [[AI/Agent/Calibrate-Then-Act-Cost-Aware-Exploration|Calibrate-Then-Act]] — Cost-aware 探索策略
- [[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|🔥 Long-Horizon Credit Assignment 专题]] ⭐ — GiGPO/AgentPRM/LOOP/MIG 全图谱 ★★★★★

### 4.4 Tool Use RL 专线 ⭐

- [[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|🔥 Tool Use RL 训练专题]] ⭐ — ToolRL/ToRL/ARTIST/VerlTool/Agent-RLVR 全图谱 ★★★★★
- [[AI/Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（Trajectory-Search Rollouts）]] ⭐ — per-turn 树搜索提升 rollout 质量；解决不可逆陷阱/Echo Trap；+15%，0.5B TSR≈3B naive；optimizer-agnostic（TU Munich + IBM，arXiv:2602.11767，ICML 2026）★★★★☆
- [[AI/Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2（Checklist Rewards）]] — Checklist 把 open-ended reward 拆解为结构化二进制分类，8B +8~+12 over SFT（arXiv:2602.12268）★★★★☆
- [[AI/Agent/Agentic-RL/ASTRA-Automated-Tool-Agent-Training|ASTRA（Beike）]] — 全自动 SFT+RL 流水线：MCP 工具图合成轨迹 + code-executable verifiable 环境，无需人工标注，32B 超过 o3（arXiv:2601.21558）★★★★☆
- [[AI/Agent/Agentic-RL/RC-GRPO-Reward-Conditioned-Tool-Calling-RL|RC-GRPO]] — reward token conditioning 解决 multi-turn GRPO all-0/all-1 崩塌，7B 超所有闭源 API（arXiv:2602.03025）★★★★☆

### 4.5 环境工程

- [[AI/Agent/Agentic-RL/Agent-RL-环境工程系统论|🔥 Agent RL 环境工程系统论]] ⭐ — 设计原则/Reward 工程/主流环境拆解 ★★★★★
- [[AI/Agent/Agentic-RL/AWM-Agent-World-Model-Synthetic-Environments|AWM]] ⭐ — Agent World Model 合成环境 ★★★★
- [[AI/Agent/EnterpriseGym-Corecraft|EnterpriseGym Corecraft]] — 高保真企业 RL 环境 ★★★★
- [[AI/Agent/Agentic-RL/VerlTool 论文|VerlTool]] — 工具使用 RL 统一框架
- [[AI/Agent/Agentic-RL/PVPO 论文|PVPO]] — 价值预估策略优化
- [[AI/Agent/Agentic-RL/UI-TARS-2 论文|UI-TARS-2]] — GUI Agent RL
- [[AI/Agent/Agentic-RL/WebPilot 论文|WebPilot]] — Web 自动化
- [[AI/Agent/Agentic-RL/R-4B 论文|R-4B]] — MLLM Auto-Thinking

---

## 第五章 框架选型

> 生产级 Agent 框架对比与最佳实践

- [[AI/Agent/Agent 框架对比|Agent 框架对比]] — 六大框架选型指南
- [[AI/Agent/Frameworks/Agent 框架对比 2026|Agent 框架对比 2026]]
- [[AI/Agent/Frameworks/AutoGen|AutoGen]]
- [[AI/Agent/Frameworks/dbgpt 文档|DB-GPT]]
- [[AI/Agent/Fundamentals/Agent 生产实践|Agent 生产实践]]
- [[AI/Agent/Agent 生产落地|Agent 生产落地]]

---

## 第六章 安全与评测

> Agent 的可靠性、安全性、评测方法

### 6.1 安全

- [[AI/Agent/Agent-Skills-Security|Agent Skills Security]] — 26.1% 社区 skill 含漏洞
- [[AI/Agent/CowCorpus-Human-Intervention-Modeling-Web-Agents|CowCorpus]] — Human-in-the-Loop 干预建模 ★★★★☆
- [[AI/Agent/PABU — Progress-Aware Belief State高效Agent|PABU]] — 进度感知信念更新

### 6.2 评测

- [[AI/Agent/Fundamentals/Agent 评测|Agent 评测]]
- [[AI/Agent/Agent 评测与 Benchmark|Agent 评测与 Benchmark]]
- [[AI/Agent/Evaluating-AGENTS-Context|Evaluating AGENTS: Context Files]]
- [[AI/Agent/Gaia2-Dynamic-Async-Agent-Benchmark|Gaia2]] ⭐ — 动态异步 Agent benchmark ★★★★★
- [[AI/Agent/Aletheia-Math-Research-Agent|Aletheia]] — 数学科研 Agent

---

## 附录 Agent 经济生态 💰

> Agent 的身份、支付、信誉与商业网络

- [[AI/Agent/Agent-Economy/_MOC|Agent 经济总览]]
- [[AI/Agent/Agent-Economy/Agent 经济基础设施|Agent 经济基础设施]] — Consensus HK 2026 全景
- [[AI/Agent/Agent-Economy/Coinbase AgentKit 技术评估|Coinbase AgentKit]]
- [[AI/Agent/Agent-Economy/ERC-8004 Trustless Agents|ERC-8004]] — Agent 链上身份标准
- [[AI/Agent/Agent-Economy/Virtuals Protocol|Virtuals Protocol]] — Agent-to-Agent 商业协议
- [[AI/Agent/Agent-Economy/Agentic Spring|Agentic Spring]] — 预测市场信号
- [[AI/Agent/Agent-Economy/ai16z 竞品分析|ai16z 竞品分析]]
- [[AI/Agent/Agent-Economy/elizaOS Trust Scoring 源码研究|elizaOS Trust Scoring 源码]]

---

---

## 第七章 Agent 进化方法论 🌱

> Agent 如何从"工具"成长为"伙伴"：记忆、反馈、元认知的工程实践

- [[AI/Agent/Agent自我进化策略-从记忆习惯到自主成长|🔥 Agent 自我进化策略：从记忆习惯到自主成长]] ⭐ — 10种进化模式完整实战手册（记忆沉淀/反馈回路/元认知/课程学习/协作进化等）；来自7-Agent军团实战经验，每种模式附可直接使用的prompt指令块（2026-02-22，Scholar原创）★★★★★

---

## 导航

- ↑ 上级：[[AI/_MOC]]
- → 交叉：[[AI/LLM/RL/_MOC]]（Agentic RL）· [[AI/Safety/_MOC]]（Agent 安全）
