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
