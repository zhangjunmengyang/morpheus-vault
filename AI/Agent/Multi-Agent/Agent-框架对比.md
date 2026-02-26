---
title: "Agent 框架对比：LangChain vs LlamaIndex vs AutoGen vs CrewAI vs elizaOS vs Dify"
brief: "六大 Agent 框架横向对比：定位/架构/适用场景/优劣势；LangChain(通用) vs AutoGen(Multi-Agent) vs CrewAI(角色协作) vs elizaOS(链上) vs Dify(低代码)；面试热点——如何选型"
date: 2026-02-13
updated: 2026-02-23
tags:
  - ai/agent
  - ai/agent/framework
  - type/comparison
  - interview/hot
status: active
---

# Agent 框架对比

> 六大主流 Agent 框架的架构差异、适用场景与选型指南

## 1. 总览对比

| 维度 | LangChain/LangGraph | LlamaIndex | AutoGen | CrewAI | elizaOS | Dify |
|------|---------------------|------------|---------|--------|---------|------|
| **语言** | Python/JS | Python/TS | Python/.NET | Python | TypeScript | Python (后端) |
| **核心定位** | 通用链式编排 + 图状态机 | 数据索引 + RAG Agent | 多 Agent 对话 | 角色协作 Multi-Agent | Web3 AI Agent OS | 低代码 LLMOps 平台 |
| **架构模式** | DAG / 状态图 | 查询管线 | 会话协议 | 角色-任务 | 插件系统 + Runtime | 可视化 Workflow |
| **学习曲线** | 中-高 | 中 | 中-高 | 低 | 中 | 极低 |
| **Multi-Agent** | LangGraph 原生支持 | AgentWorkflow | 核心能力 | 核心能力 | 多角色 Character | Workflow 节点编排 |
| **LLM 支持** | 全模型 | 全模型 | OpenAI 为主，可扩展 | 全模型 | 全模型 | 全模型 |
| **部署** | 自托管 / LangSmith | 自托管 / LlamaCloud | 自托管 / Azure | 自托管 | 自托管 | 自托管 / SaaS |
| **GitHub Stars** | ~105k | ~42k | ~40k | ~25k | ~18k | ~60k |

## 2. LangChain / LangGraph

### 架构特点

LangChain 是最早的 LLM 编排框架，v0.2+ 拆分为 `langchain-core`、`langchain-community`、`langgraph` 三层。**LangGraph** 是其核心演进——基于有向图的状态机，每个节点是一个处理步骤，边可以是条件分支。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    next_action: str

graph = StateGraph(AgentState)

def research_node(state):
    # 调用 LLM + 工具
    return {"messages": state["messages"] + [research_result]}

def write_node(state):
    return {"messages": state["messages"] + [draft]}

graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge("research", "write")
graph.add_conditional_edges("write", should_continue, {
    "revise": "research",
    "done": END
})

app = graph.compile()
result = app.invoke({"messages": [user_query], "next_action": ""})
```

### 优劣

- ✅ 生态最大，社区活跃，集成丰富（700+ 集成）
- ✅ LangGraph 支持循环、条件分支、人工介入（human-in-the-loop）
- ✅ LangSmith 提供完整的 observability
- ❌ 抽象层过多，调试困难（"框架税"高）
- ❌ API 变动频繁，升级成本大
- ❌ 简单任务过度工程化

### 适用场景

复杂多步推理、需要精细控制流的生产系统、企业级 Agent 应用。

## 3. LlamaIndex

### 架构特点

LlamaIndex 从 RAG 工具演进为 Agent 框架。核心抽象：**Index → QueryEngine → AgentWorkflow**。v0.11+ 引入 `llama-index-agent` 模块，支持 [[AI/LLM/Application/RAG 工程实践|RAG]] 与 Agent 深度融合。

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.openai import OpenAI

def search_knowledge(query: str) -> str:
    """Search the knowledge base."""
    # 内部调用 VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine().query(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[search_knowledge],
    llm=OpenAI(model="gpt-4o"),
    system_prompt="You are a knowledge assistant."
)
response = await agent.run("Explain transformer attention")
```

### 优劣

- ✅ RAG 能力最强：向量索引、知识图谱、结构化查询
- ✅ 数据连接器丰富（LlamaHub 300+ 数据源）
- ✅ 适合知识密集型应用
- ❌ Agent 编排能力弱于 LangGraph
- ❌ 非 RAG 场景显得笨重

### 适用场景

企业知识库问答、文档分析、数据驱动的 Agent 应用。

## 4. AutoGen

### 架构特点

微软的 [[AI/Agent/Multi-Agent/AutoGen|AutoGen]] 以 **多 Agent 对话** 为核心。v0.4（AutoGen Studio）重构为事件驱动架构，Agent 之间通过消息传递协作。支持 GroupChat 模式。

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o")

researcher = AssistantAgent("researcher",
    model_client=model,
    system_message="You research topics thoroughly.")

writer = AssistantAgent("writer",
    model_client=model,
    system_message="You write clear summaries.")

team = RoundRobinGroupChat(
    participants=[researcher, writer],
    max_turns=4
)
result = await team.run(task="Write a report on quantum computing")
```

### 优劣

- ✅ 多 Agent 对话模式灵活（RoundRobin、Selector、Swarm）
- ✅ 代码执行沙箱内置
- ✅ 微软生态整合（Azure、Semantic Kernel）
- ❌ 学习曲线陡峭，v0.2→v0.4 断代
- ❌ 对话循环可能失控，需要精细的终止条件
- ❌ 文档和示例更新滞后

### 适用场景

代码生成与审查、研究协作、需要多角色讨论决策的复杂任务。

## 5. CrewAI

### 架构特点

CrewAI 用 **角色 (Agent) + 任务 (Task) + 流程 (Process)** 三层抽象来编排 Multi-Agent 工作流。类比现实团队管理。

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Researcher",
    goal="Find the latest AI trends",
    backstory="Expert in AI with 10 years of experience",
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm="gpt-4o"
)

writer = Agent(
    role="Technical Writer",
    goal="Write engaging technical content",
    backstory="Award-winning tech journalist",
    llm="gpt-4o"
)

research_task = Task(
    description="Research the latest developments in {topic}",
    expected_output="Detailed research report with sources",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the research",
    expected_output="1000-word blog post",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  # 或 Process.hierarchical
)
result = crew.kickoff(inputs={"topic": "AI agents"})
```

### 优劣

- ✅ 学习曲线最低，概念直觉化
- ✅ 内置内存管理和任务委托
- ✅ 适合快速原型
- ❌ 复杂流程控制能力有限
- ❌ 底层高度依赖 LangChain
- ❌ 生产级可观测性不足

### 适用场景

内容生产流水线、自动化报告、团队协作模拟。

## 6. elizaOS

### 架构特点

elizaOS 是面向 **Web3 + 社交平台** 的 AI Agent 操作系统。TypeScript 编写，插件化架构（90+ 官方插件），集成 Discord/Twitter/Telegram 等平台，以及 Ethereum/Solana 等链。

```typescript
import { AgentRuntime, ModelClass } from "@elizaos/core";

const runtime = new AgentRuntime({
  character: {
    name: "CryptoAdvisor",
    bio: "A DeFi expert AI agent",
    modelProvider: "openai",
    plugins: ["@elizaos/plugin-solana", "@elizaos/plugin-discord"]
  },
  evaluators: [trustScoreEvaluator],
  providers: [walletProvider, marketDataProvider]
});

// 注册自定义 Action
runtime.registerAction({
  name: "SWAP_TOKEN",
  validate: async (runtime, message) => { /* ... */ },
  handler: async (runtime, message) => {
    // 执行链上 swap
  }
});
```

### 优劣

- ✅ Web3 原生：钱包、链上交互、代币经济
- ✅ 多平台部署（Discord/Twitter/Telegram 一键连接）
- ✅ 记忆系统 + 信任评分（Trust Scoring）
- ❌ Web3 以外场景适配不佳
- ❌ 社区以 crypto 开发者为主，企业采用少
- ❌ 文档质量参差不齐

### 适用场景


## 7. Dify

### 架构特点

Dify 是 **低代码 LLMOps 平台**，提供可视化 Workflow 编辑器。支持 RAG 管线、Agent 编排、API 发布。后端 Python + 前端 React。

核心概念：
- **Chatbot / Completion** — 基础对话/补全应用
- **Workflow** — 可视化 DAG 编排（IF/ELSE、循环、代码节点）
- **Agent** — ReAct/Function Calling 模式
- **Knowledge** — 内置向量数据库 + 文档解析

### 优劣

- ✅ 零代码即可搭建 LLM 应用
- ✅ 内置 RAG、监控、日志、API 发布
- ✅ 部署简单（Docker 一键启动）
- ✅ 开源版功能已非常完善
- ❌ 复杂逻辑表达受限于 UI 编排能力
- ❌ 扩展性弱于代码框架
- ❌ 高并发场景需额外优化

### 适用场景

快速 MVP、非技术团队构建 LLM 应用、企业内部工具。

## 8. 选型决策树

```
需求分析
├── 需要精细控制流 + 生产级？→ LangGraph
├── 核心是知识检索/RAG？→ LlamaIndex
├── 多 Agent 对话/代码执行？→ AutoGen
├── 快速原型 + 团队协作模拟？→ CrewAI
├── Web3 + 社交平台？→ elizaOS
└── 零代码/低代码？→ Dify
```

**混合使用建议**：实践中常见 LlamaIndex（RAG 引擎）+ LangGraph（编排层）的组合，或 Dify（前端）+ 自定义 Agent 后端的架构。

## 9. 面试常见问题

1. **Q: LangChain 和 LangGraph 的区别？**
   A: LangChain 是线性 Chain 编排，LangGraph 是基于有向图的状态机，支持循环、条件分支、持久化状态。LangGraph 是 LangChain 团队对复杂 Agent 编排的演进。

2. **Q: 为什么有人说 LangChain "抽象过度"？**
   A: LangChain 为统一 API 引入了大量中间层（Chain/Agent/Tool/Memory/Callback），简单场景下直接调 LLM API 只需几行代码，但用 LangChain 可能需要理解十几个概念。调试时 stack trace 很深。

3. **Q: AutoGen 和 CrewAI 都做 Multi-Agent，区别在哪？**
   A: AutoGen 以对话协议为核心，Agent 通过消息传递自由交互；CrewAI 以角色-任务映射为核心，更像项目管理。AutoGen 更灵活但更复杂，CrewAI 更直觉但表达力受限。

4. **Q: 生产环境选型怎么考虑？**
   A: 关注四点：可观测性（logging/tracing）、错误恢复（retry/fallback）、成本控制（token 管理）、可测试性。LangGraph + LangSmith 和 Dify 在这方面相对成熟。

5. **Q: elizaOS 的 Trust Scoring 机制是什么？**
   A: 基于 Agent 与用户/其他 Agent 的历史交互打分，用于决定信任级别和权限。核心包括推荐者信任传递、交互衰减、链上验证。

## 相关笔记

- [[AI/Agent/Fundamentals/Agent or Workflow？|Agent or Workflow？]] — 何时用 Agent，何时用固定流程
- [[AI/Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]] — Multi-Agent 设计模式
- [[AI/Agent/Fundamentals/Tool Use|Tool Use]] — Agent 工具调用机制
- [[AI/LLM/Application/RAG 工程实践|RAG 工程实践]] — 检索增强生成
- [[AI/Agent/Multi-Agent/AutoGen|AutoGen]] — AutoGen 详细笔记
- [[AI/Agent/Fundamentals/记忆模块|记忆模块]] — Agent 记忆设计
