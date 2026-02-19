---
title: "Agent 智能体"
type: moc
domain: ai/agent
tags:
  - ai/agent
  - type/reference
---

# 🤖 Agent 智能体

> 从单 Agent 到 Multi-Agent，从 Tool Use 到 Agentic RL

## 基础架构 (Fundamentals)
- [[AI/Agent/Fundamentals/Tool Use|Tool Use]] — 工具调用
- [[AI/Agent/Fundamentals/记忆模块|记忆模块]] — 短期/长期记忆
- [[AI/Agent/Fundamentals/Agent or Workflow？|Agent or Workflow？]] — 设计决策
- [[AI/Agent/Fundamentals/分析 Agent 演进的一些思考|Agent 演进思考]]
- [[AI/Agent/Fundamentals/Context-Folding 论文|Context-Folding]] — 长程 Agent 论文
- [[AI/Agent/Fundamentals/HF Agent Course|HF Agent Course]]
- [[AI/Agent/Fundamentals/HF LLM + Agent|HF LLM + Agent]]
- [[AI/Agent/Fundamentals/ReAct 与 CoT|ReAct 与 CoT]] — 推理范式对比
- [[AI/Agent/Fundamentals/Agent 生产实践|Agent 生产实践]]
- [[AI/Agent/Fundamentals/Agent 评测|Agent 评测]]
- [[AI/Agent/Fundamentals/Code Agent|Code Agent (基础)]]

## Multi-Agent
- [[AI/Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]]
- [[AI/Agent/Multi-Agent/Agent vs MAS|Agent vs MAS]]
- [[AI/Agent/Multi-Agent/Planner|Planner]]
- [[AI/Agent/Multi-Agent/零碎的点|零碎的点]]
- [[AI/Agent/Multi-Agent/untitled_SB2HwKNC|Multi-Agent 草稿]]

## 研究论文 (Recent Papers)
- [[AI/Agent/AgentAuditor — Reasoning Tree审计多Agent系统|AgentAuditor]] — Reasoning Tree 审计多 Agent 系统
- [[AI/Agent/Aletheia-Math-Research-Agent|Aletheia]] — Gemini Deep Think 数学科研 Agent，从 benchmark 到真实科研产出的跨越（arXiv:2602.10177）
- [[AI/Agent/IMAGINE — 多Agent蒸馏到单模型|IMAGINE]] — 多 Agent 蒸馏到单模型
- [[AI/Agent/PABU — Progress-Aware Belief State高效Agent|PABU]] — 进度感知信念更新，高效 Agent
- [[AI/Agent/Agent-Skills-Security|Agent Skills Security]] — arXiv:2602.12430，Skill 架构·获取·安全治理，26.1% 社区 skill 含漏洞

## MCP (Model Context Protocol)
- [[AI/Agent/MCP/如何给人深度科普 MCP|如何给人深度科普 MCP]]
- [[AI/Agent/MCP/HF MCP Course|HF MCP Course]]

## Multi-Agent RL 案例
- [[AI/Agent/Kimi-K2.5-PARL|Kimi K2.5 & PARL]] — 并行多 Agent 强化学习，orchestrator 训练，Agent Swarm 100 subagents（arXiv:2602.02276）

## Agentic RL ⭐（前沿方向）
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] — 三大难题（credit assignment/reward/环境）与对应解法，185行合成笔记（2026-02-20）⭐
- [[AI/Agent/Agentic-RL/Agentic RL Training|Agentic RL Training]] — Agent + RL 融合
- [[AI/Agent/Agentic-RL/Agentic RL Survey|Agentic RL Survey]] — 综述论文
- [[AI/Agent/Agentic-RL/VerlTool 论文|VerlTool]] — 工具使用 RL
- [[AI/Agent/Agentic-RL/PVPO 论文|PVPO]] — 价值预估策略优化
- [[AI/Agent/Agentic-RL/UI-TARS-2 论文|UI-TARS-2]] — GUI Agent RL
- [[AI/Agent/Agentic-RL/WebPilot 论文|WebPilot]] — Web 自动化
- [[AI/Agent/Agentic-RL/R-4B 论文|R-4B]] — MLLM Auto-Thinking

## Agent 经济 (Agent Economy) 💰
- [[AI/Agent/Agent-Economy/_MOC|Agent 经济总览]] — 身份、支付、信誉、商业网络
- [[AI/Agent/Agent-Economy/Agent 经济基础设施|Agent 经济基础设施]] — Consensus HK 2026 全景
- [[AI/Agent/Agent-Economy/Coinbase AgentKit 技术评估|Coinbase AgentKit]] — SDK 评估 + DeFi 策略可行性
- [[AI/Agent/Agent-Economy/ERC-8004 Trustless Agents|ERC-8004]] — Agent 链上身份标准
- [[AI/Agent/Agent-Economy/Virtuals Protocol|Virtuals Protocol]] — Agent-to-Agent 商业协议
- [[AI/Agent/Agent-Economy/Agentic Spring|Agentic Spring]] — 预测市场信号 + 模型能力加速
- [[AI/Agent/Agent-Economy/ai16z 竞品分析|ai16z 竞品分析]]
- [[AI/Agent/Agent-Economy/elizaOS Trust Scoring 源码研究|elizaOS Trust Scoring 源码]]

## 面试深度笔记
- [[AI/Agent/AI-Agent-2026-技术全景|AI Agent 2026 技术全景]] — 面试武器库，1114行，综合 2026 最新 survey + 框架对比 + 生产设计指南
- [[AI/Agent/Agent Memory 机制|Agent Memory 机制]] — 短期/长期/工作记忆、RAG-based memory、MemGPT/Letta
- [[AI/Agent/Agent World Model|Agent World Model]] — Agentic RL + 合成环境 + 世界模型
- [[AI/Agent/Evaluating-AGENTS-Context|Evaluating AGENTS: Context Files 对 Coding Agent 的影响]]
- [[AI/Agent/Agent Tool Use|Agent Tool Use]] — Function Calling、ReAct、工具选择策略、API 对比
- [[AI/Agent/Agent 框架对比|Agent 框架对比]] — 六大框架选型指南
- [[AI/Agent/Agent 生产落地|Agent 生产落地]] — 生产部署实践
- [[AI/Agent/Agent 评测与 Benchmark|Agent 评测与 Benchmark]]
- [[AI/Agent/Code Agent|Code Agent]] — 代码 Agent 深度笔记
- [[AI/Agent/ReAct 推理模式|ReAct 推理模式]]
- [[AI/Agent/GitHub-Agentic-Workflows|GitHub Agentic Workflows]]

## 框架 (Frameworks)
- [[AI/Agent/Frameworks/AutoGen|AutoGen]]
- [[AI/Agent/Frameworks/dbgpt 文档|DB-GPT]]
- [[AI/Agent/Frameworks/Agent 框架对比 2026|Agent 框架对比 2026]]

## 相关 MOC
- ↑ 上级：[[AI/_MOC]]
- → 交叉：[[AI/LLM/RL/_MOC]]（Agentic RL）
