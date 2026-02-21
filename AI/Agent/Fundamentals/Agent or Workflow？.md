---
title: "Agent or Workflow？"
type: thought
domain: ai/agent/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/fundamentals
  - type/thought
---
# Agent or Workflow？

似乎两者已经成为了对立的，尤其是 mt 内对后者非常排斥，仿佛成为了“落后”的代名词。

Agent 和 workflow 我觉得最核心的差异如下：

## 有相对确定的流程

如果有，workflow 没什么问题。甚至考虑两种方式：

- Workflow in agent：典型代表 coze
- Agent in workflow：典型代表 Langraph
## 工具调用能力

fc 是基于 agent 的，如果需要灵活的 fc，则一定是需要 agent，单独 workflow 是无法实现的

## 条件不明确的循环

一个工作应该执行到什么时候结束，是

---

## See Also

- [[AI/Agent/Fundamentals/分析 Agent 演进的一些思考|Agent 演进思考]] — 更深的演化分析
- [[AI/Agent/Agentic-RL/FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer]] — "Agent or Workflow" 的 RL 解法：让 policy 自动选结构
- [[AI/Agent/_MOC|Agent MOC]] — Agent 知识全图谱
- [[AI/Agent/Frameworks/Agent 框架对比 2026|Agent 框架对比 2026]] — 不同框架对 Agent/Workflow 边界的处理
