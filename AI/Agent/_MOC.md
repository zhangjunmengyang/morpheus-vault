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
- [[AI/Agent/CowCorpus-Human-Intervention-Modeling-Web-Agents|CowCorpus]] — CMU+Duke：首个 Human-in-the-Loop 干预建模数据集，四种协作风格分类（Hands-off/Hands-on/Collaborative/Takeover），intervention 预测准确率+63%，用户感知有用性+26.5%；PTS 指标设计精妙（时机比结果更重要）（arXiv:2602.17588）★★★★☆
- [[AI/Agent/AgentAuditor — Reasoning Tree审计多Agent系统|AgentAuditor]] — Reasoning Tree 审计多 Agent 系统
- [[AI/Agent/Aletheia-Math-Research-Agent|Aletheia]] — Gemini Deep Think 数学科研 Agent，从 benchmark 到真实科研产出的跨越（arXiv:2602.10177）
- [[AI/Agent/Aletheia (DeepMind 数学研究Agent)|Aletheia 早期概览]] — 2026-02-15 早期版（96行），深度版见上方 Aletheia-Math-Research-Agent
- [[AI/Agent/IMAGINE — 多Agent蒸馏到单模型|IMAGINE]] — 多 Agent 蒸馏到单模型
- [[AI/Agent/PABU — Progress-Aware Belief State高效Agent|PABU]] — 进度感知信念更新，高效 Agent
- [[AI/Agent/Agent-Skills-Security|Agent Skills Security]] — arXiv:2602.12430，Skill 架构·获取·安全治理，26.1% 社区 skill 含漏洞

## MCP (Model Context Protocol)
- [[AI/Agent/MCP/如何给人深度科普 MCP|如何给人深度科普 MCP]]
- [[AI/Agent/MCP/HF MCP Course|HF MCP Course]]

## Multi-Agent RL 案例
- [[AI/Agent/Kimi-K2.5-PARL|Kimi K2.5 & PARL]] — 并行多 Agent 强化学习，orchestrator 训练，Agent Swarm 100 subagents（arXiv:2602.02276）

## Agentic RL ⭐（前沿方向）
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] ⭐ — 四大维度框架（环境×Reward×Workflow/Topology×算法），v2.0 新增"结构即可学习变量"核心命题；FlowSteer/AgentConductor/SquRL/PA-MoE 综合分析（2026-02-21 更新）
- [[AI/Agent/Agentic-RL/FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer (CWRPO)]] — Workflow 结构自动化 via End-to-End RL：policy model 学习构建/调试 DAG workflow，CWRPO 用 conditional release reward 门控消除 shortcut；CUHK+NTU+NUS（arXiv:2602.01664）★★★☆
- [[AI/Agent/AgentConductor-Topology-Evolution-Multi-Agent-Code|AgentConductor]] ⭐ — 上交大 ICML 投稿：RL 训练 3B orchestrator 动态生成 DAG topology，difficulty-aware 密度函数实现准确率+14.6%同时 token cost-68%；固定 topology 的终结者（arXiv:2602.17100）★★★★
- [[AI/Agent/Agentic-RL/SquRL-Dynamic-Workflow-Text-to-SQL|SquRL]] — Theorem 3.1 形式化证明动态 workflow 优于任何静态 pipeline，RL 训练 selector policy 按 query 难度自适应选工具链；Dynamic Actor Masking 防 training collapse；Text-to-SQL 任务（arXiv:2602.15564）★★★
- [[AI/Agent/Agentic-RL/Agent-RL-训练实战指南|Agent RL 训练实战指南]] ⭐ — 现象·坑·解法系统整理：算法层/奖励层/信用分配层三层框架，1001行，面试可用；★★★★★
- [[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — Phase-Aware MoE 解决 Simplicity Bias：复杂任务仅获5%参数容量 → LoRA expert 按行为阶段分配，1.5B 打 7B；★★★★（arXiv:2602.17038）
- [[AI/Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong]] ⭐ — NUS+MIT：极长任务（12h/700+轮）训练方案：轨迹切割 SFT（cold start）+ 渐进式 RL（2h→4h→6h 课程）；106B 在 PaperBench 以 62.59% 超越 Kimi K2 Thinking 1T（51.31%）；专门化训练 > 通用规模（arXiv:2602.17547）★★★★★
- [[AI/Agent/Calibrate-Then-Act-Cost-Aware-Exploration|Calibrate-Then-Act]] — 显式先验注入的 cost-aware 探索策略：94% optimal match（基线 23%），RL 无法自然习得 meta 探索
- [[AI/Agent/Agentic-RL/Agentic RL Training|Agentic RL Training]] — Agent + RL 融合
- [[AI/Agent/Agentic-RL/Agentic RL Survey|Agentic RL Survey]] — 综述论文
- [[AI/Agent/Agentic-RL/VerlTool 论文|VerlTool]] — 工具使用 RL
- [[AI/Agent/Agentic-RL/PVPO 论文|PVPO]] — 价值预估策略优化
- [[AI/Agent/Agentic-RL/UI-TARS-2 论文|UI-TARS-2]] — GUI Agent RL
- [[AI/Agent/Agentic-RL/WebPilot 论文|WebPilot]] — Web 自动化
- [[AI/Agent/Agentic-RL/R-4B 论文|R-4B]] — MLLM Auto-Thinking
- [[AI/Agent/EnterpriseGym-Corecraft|EnterpriseGym Corecraft]] — Surge AI：高保真企业 RL 环境训练可泛化 Agent，OOD 泛化突破，GRPO 框架（arXiv:2602.16179）★★★★

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
- [[AI/Agent/Gaia2-Dynamic-Async-Agent-Benchmark|Gaia2]] ⭐ — 动态异步环境 Agent benchmark：GPT-5(42%) + Claude-4 Sonnet（均未公开发布），Kimi-K2 开源最强(21%)；write-action verifier 可直接用于 Agentic RL 训练；Meta FAIR（arXiv:2602.11964）★★★★★
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
