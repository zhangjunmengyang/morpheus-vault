---
brief: "AutoGen（Microsoft Research）——多 Agent 对话框架；UserProxy/AssistantAgent 双 Agent 对话模式，支持人机混合决策；AutoGen Studio 可视化；适合需要多角色协作（代码生成+审查+执行）的 Agent 任务。"
title: "AutoGen"
type: concept
domain: ai/agent/frameworks
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/frameworks
  - type/concept
---
# AutoGen

AutoGen 是微软推出的 multi-agent 对话框架，核心理念是**通过定义 agent 之间的对话模式来编排复杂任务**。从 0.2 到 0.4 经历了一次大重写（AutoGen 0.4 / AG2），架构变化很大。

## 核心设计

### 对话驱动的 Agent 编排

AutoGen 的核心抽象不是「任务图」或「工作流 DAG」，而是**对话**。每个 agent 是一个可以收发消息的实体，复杂任务通过 agent 之间的多轮对话自然涌现。

```python
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent

# 定义一个 coding assistant
assistant = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4"},
    system_message="你是一个 Python 专家，写出简洁高效的代码。"
)

# UserProxy 可以执行代码并返回结果
user_proxy = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# 发起对话 — agent 之间会自动多轮交互直到任务完成
user_proxy.initiate_chat(
    assistant,
    message="用 matplotlib 画一个正弦波，保存为 sine.png"
)
```

这段代码看起来简单，但背后发生了：executor 发消息 → coder 写代码 → executor 执行代码 → 失败则 coder 修复 → 循环直到成功。

### AutoGen 0.4 的新架构

0.4 版本（AG2）做了彻底重构：

- **Event-driven**：agent 之间通过异步事件通信，不再是同步的 chat loop
- **Runtime 抽象**：引入 `AgentRuntime`，agent 可以分布式部署
- **Team 概念**：`RoundRobinGroupChat`、`SelectorGroupChat` 等预置编排模式
- **Tool 一等公民**：tool 定义更规范，支持类型检查

```python
# AutoGen 0.4 风格
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o")

planner = AssistantAgent("planner", model_client=model,
    system_message="拆解任务为子步骤")
coder = AssistantAgent("coder", model_client=model,
    system_message="根据计划编写代码")
reviewer = AssistantAgent("reviewer", model_client=model,
    system_message="审查代码质量和正确性")

team = RoundRobinGroupChat(
    [planner, coder, reviewer],
    max_turns=6
)
```

## 与其他框架的比较

| 维度 | AutoGen | CrewAI | LangGraph |
|------|---------|--------|-----------|
| 编排模式 | 对话驱动 | 角色扮演 | 图/状态机 |
| 代码执行 | 内置沙箱 | 需外接 | 需外接 |
| 学习曲线 | 中等 | 低 | 高 |
| 灵活性 | 高 | 中 | 最高 |
| 适合场景 | 代码生成、研究 | 业务流程 | 复杂流程控制 |

## 踩坑记录

1. **消息无限循环** —— 两个 agent 互相客气（"你觉得呢？""我觉得你说得对"），需要设好 `max_consecutive_auto_reply`
2. **代码执行安全** —— 默认的 Docker 沙箱有时启动慢，生产环境建议用 E2B 或自建沙箱
3. **0.2 → 0.4 迁移** —— API 完全不兼容，旧代码基本要重写

## 我的看法

AutoGen 的对话驱动模式在**探索性任务**（写代码、做研究）上很自然，但在**确定性流程**上不如 LangGraph 可控。0.4 的重构方向是对的（event-driven + runtime 抽象），但生态还在追赶。选型建议：如果核心场景是 coding agent，AutoGen 很合适；如果是业务流程编排，考虑 LangGraph。

## 相关

- [[Multi-Agent 概述|Multi-Agent 概述]]
- [[AI/2-Agent/Multi-Agent/Planner|Planner]]
- [[Tool Use|Tool Use]]
- [[HF Agent Course|HF Agent Course]]
