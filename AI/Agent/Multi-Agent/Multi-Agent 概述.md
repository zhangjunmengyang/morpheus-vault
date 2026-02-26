---
title: "Multi-Agent"
brief: "Multi-Agent System 概述：为何单 Agent 有瓶颈/MAS 的核心优势；协调模式（Hub-and-Spoke/Peer-to-Peer/Hierarchical）；通信协议与共享状态设计；面试速查"
type: reference
domain: ai/agent/multi-agent
created: "2026-02-13"
updated: "2026-02-23"
tags:
  - ai/agent/multi-agent
  - type/reference
---
# Multi-Agent 概述

Multi-Agent System（MAS）是让多个 AI agent 协作完成复杂任务的架构模式。核心思路：**单个 agent 能力有限，多个专精 agent 协作可以突破上限**。

## 为什么要 Multi-Agent

单 agent 的瓶颈：

1. **上下文窗口有限** —— 复杂任务需要的信息量超过单次 prompt 容量
2. **能力不专精** —— 一个 prompt 很难同时让模型既写代码又审代码又做项目管理
3. **错误不可控** —— 没有内部 review 机制，错误会直接输出

Multi-Agent 的核心优势是**分工** + **制衡**。

## 常见架构模式

### 1. 中央调度（Orchestrator Pattern）

一个 orchestrator agent 接收任务，分发给专门的 worker agents，汇总结果。

```
User → Orchestrator → [Coder, Researcher, Reviewer] → Orchestrator → User
```

优点：流程可控，容易 debug。缺点：orchestrator 是单点瓶颈。

### 2. 对话链（Conversation Chain）

Agent 按固定顺序传递消息，类似流水线。

```
User → Planner → Coder → Reviewer → User
```

适合有明确阶段的任务（规划 → 执行 → 审查）。

### 3. 群聊模式（Group Chat）

所有 agent 在同一个「聊天室」中，由一个 selector（通常也是 LLM）决定下一个发言者。

```python
# AutoGen 风格的 Group Chat
group_chat = GroupChat(
    agents=[planner, coder, reviewer, user_proxy],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"  # LLM 动态选择下一个发言者
)
```

灵活性最高，但也最难控制 —— 容易出现无效对话。

### 4. 层级模式（Hierarchical）

类似公司组织架构：manager agent 管理多个 team lead，team lead 管理 worker。

适合大规模、多步骤的复杂任务。CrewAI 的 `hierarchical` 模式就是这种。

## 关键设计问题

### Agent 间通信

- **消息格式** —— 统一的消息 schema 很重要，否则 agent 之间「说不通」
- **同步 vs 异步** —— 同步简单但慢，异步快但需要处理竞态
- **共享状态 vs 消息传递** —— 共享状态（如共享 memory/scratchpad）容易实现但扩展性差

### 任务分配策略

- **静态分配** —— 预定义的工作流，确定性强
- **动态分配** —— orchestrator 根据任务内容动态选择 agent
- **竞争机制** —— 多个 agent 竞标任务，选最合适的

### 冲突处理

当两个 agent 给出矛盾的结论时怎么办？

- **投票** —— 多个 agent 投票，少数服从多数
- **仲裁** —— 有一个专门的仲裁 agent
- **Escalation** —— 交给人类决定

## 常见框架对比

| 框架 | 编排模式 | 特点 |
|------|---------|------|
| AutoGen | 对话驱动 | 代码执行强，agent 间自然对话 |
| CrewAI | 角色扮演 | 上手简单，agent 有角色定义 |
| LangGraph | 状态图 | 最灵活，但学习曲线陡 |
| MetaGPT | SOP 驱动 | 模拟软件公司流程 |
| OpenAI Swarm | 轻量级 handoff | 实验性质，适合学习 |

## 实践建议

1. **先单 agent 跑通，再拆成 multi-agent** —— 过早拆分是万恶之源
2. **Agent 数量越少越好** —— 每多一个 agent，复杂度不是线性增长
3. **明确退出条件** —— 没有退出条件的多 agent 对话会无限循环
4. **Observability** —— agent 间的消息必须可追踪，否则出了 bug 无法排查

## 我的看法

Multi-Agent 目前更偏「研究/探索」阶段，真正生产级的应用还不多。最大的挑战不是框架选型，而是**如何定义好 agent 的边界和交互协议**。这和微服务架构的 bounded context 设计本质上是同一个问题。

## 相关

- [[AI/Agent/Multi-Agent/Planner|Planner]]
- [[AI/Agent/Multi-Agent/AutoGen|AutoGen]]
- [[AI/Agent/Fundamentals/Tool Use|Tool Use]]
- [[AI/Agent/Fundamentals/记忆模块|记忆模块]]
- [[AI/Agent/Multi-Agent/零碎的点|零碎的点]]
