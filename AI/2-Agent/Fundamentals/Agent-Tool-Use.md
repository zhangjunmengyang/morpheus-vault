---
title: "Agent Tool Use：Function Calling、ReAct 与工具选择策略"
brief: "Agent 工具调用深度指南：Function Calling 标准格式、ReAct 推理-行动循环、工具选择策略（路由/级联/并行）、错误处理与重试机制；面试热点——工具调用 vs 代码生成"
date: 2026-02-13
updated: 2026-02-23
tags:
  - ai/agent
  - ai/agent/tool-use
  - ai/llm/api
  - type/deep-dive
  - interview/hot
status: active
---

# Agent Tool Use

> Tool Use 是 Agent 从"嘴替"变成"干活的"的关键能力。本文覆盖 Function Calling 原理、ReAct 推理模式、工具选择策略，以及 OpenAI / Anthropic / 开源模型的 Tool Use API 对比。

## 1. 为什么需要 Tool Use

LLM 的固有缺陷 → Tool 来补：

| LLM 缺陷 | 对应工具 |
|----------|---------|
| 训练数据有截止日期 | Web Search、News API |
| 数学计算不精确 | Calculator、Code Interpreter |
| 无法访问私有数据 | Database Query、RAG |
| 无法执行副作用 | API 调用、文件操作、邮件发送 |
| 幻觉 | 事实验证工具、知识库查询 |

## 2. Function Calling 原理

### 2.1 核心流程

```
User: "北京今天天气怎么样？"
         │
         ▼
┌──────────────────────┐
│       LLM            │
│  判断需要调用工具      │
│  生成结构化调用请求    │
└────────┬─────────────┘
         │ {"name": "get_weather", "arguments": {"city": "北京"}}
         ▼
┌──────────────────────┐
│   Tool Execution     │
│  (应用层执行函数)     │
└────────┬─────────────┘
         │ {"temperature": 5, "condition": "晴"}
         ▼
┌──────────────────────┐
│       LLM            │
│  整合结果生成回复      │
└──────────────────────┘
         │
         ▼
"北京今天晴，气温 5°C。"
```

关键点：**LLM 不执行函数**——它只输出结构化的调用意图（JSON），由应用层解析、执行、再将结果送回 LLM。

### 2.2 训练方式

Function Calling 能力通过 SFT 获得：

1. **数据构造**：生成（query, tool_schema, tool_call, tool_result, response）五元组
2. **格式训练**：让模型学会在适当时候输出特殊 token（如 `<tool_call>`）+ JSON 结构
3. **多轮训练**：处理 parallel function calling 和 sequential tool chains

```python
# OpenAI Function Calling 格式
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools,
    tool_choice="auto"  # auto | none | required | {"function": {"name": "..."}}
)

# 模型返回 tool_calls
tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name = "get_weather"
# tool_call.function.arguments = '{"city": "北京"}'
```

## 3. ReAct 模式

[[ReAct 与 CoT|ReAct]]（Reasoning + Acting）是最经典的 Agent 推理框架，由 Yao et al. (2022) 提出：

### 3.1 核心循环

```
Thought: 我需要查找北京的天气信息
Action: get_weather(city="北京")
Observation: {"temperature": 5, "condition": "晴", "humidity": 30}
Thought: 我已获取天气信息，可以回答用户了
Action: finish(answer="北京今天晴，5°C，湿度 30%")
```

**Thought-Action-Observation** 三步循环，直到 Agent 决定输出最终答案。

### 3.2 ReAct vs. 纯 CoT vs. 纯 Action

| 模式 | 特点 | 问题 |
|------|------|------|
| **CoT (Chain-of-Thought)** | 纯推理，不与环境交互 | 会幻觉，无法获取实时信息 |
| **Act-only** | 直接行动，不做推理 | 盲目试错，缺乏规划 |
| **ReAct** | 推理指导行动，行动结果反馈推理 | 最佳平衡，但 token 消耗大 |

### 3.3 现代演进：从 ReAct 到 Tool Calling Agent

```python
# LangChain 中 ReAct Agent vs Tool Calling Agent
from langchain.agents import create_react_agent, create_tool_calling_agent

# 传统 ReAct：依赖 prompt 格式，LLM 输出文本解析
react_agent = create_react_agent(llm, tools, react_prompt)

# 现代 Tool Calling：依赖模型原生 function calling 能力
tc_agent = create_tool_calling_agent(llm, tools, tc_prompt)

# Tool Calling Agent 的优势：
# 1. 结构化输出，不需要文本解析（不会因格式错误而失败）
# 2. 支持 parallel tool calls（一次调用多个工具）
# 3. 模型经过专门训练，tool selection 更准确
```

## 4. 工具选择策略

当 Agent 有大量工具可用时，如何高效选择？

### 4.1 策略分层

```
┌─────────────────────────────────────┐
│   Level 3: 动态工具发现              │
│   (MCP / Tool Registry 按需加载)    │
├─────────────────────────────────────┤
│   Level 2: 工具路由                  │
│   (先分类再选工具，两阶段检索)        │
├─────────────────────────────────────┤
│   Level 1: 全量注入                  │
│   (所有工具 schema 放入 context)     │
└─────────────────────────────────────┘
```

### 4.2 具体方法

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| **全量注入** | 所有工具定义直接放入 system prompt | 工具数 < 20 |
| **语义检索** | 将工具描述 embedding，按 query 语义检索 top-K | 工具数 20-200 |
| **分类路由** | 先用小模型/classifier 判断意图类别，再加载该类工具 | 工具数 > 200 |
| **MCP 动态发现** | 通过 [[AI/2-Agent/Fundamentals/如何给人深度科普-MCP|MCP]] 协议按需从 Tool Server 获取工具 | 分布式系统 |
| **工具推荐模型** | 专门训练一个 tool selection model | 超大规模 |

```python
# 语义检索工具选择示例
import numpy as np
from openai import OpenAI

client = OpenAI()

def select_tools(query: str, all_tools: list, top_k: int = 5):
    """基于语义相似度选择最相关的工具"""
    # 工具描述 embedding（可预计算缓存）
    tool_texts = [f"{t['name']}: {t['description']}" for t in all_tools]
    tool_embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=tool_texts
    ).data

    # query embedding
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # 计算相似度
    scores = [
        np.dot(query_embedding, te.embedding)
        for te in tool_embeddings
    ]

    # 返回 top-K
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [all_tools[i] for i in top_indices]
```

## 5. OpenAI vs Anthropic Tool Use API 对比

| 维度 | OpenAI | Anthropic |
|------|--------|-----------|
| **API 字段** | `tools` + `tool_choice` | `tools` + `tool_choice` |
| **Schema 格式** | JSON Schema（标准） | JSON Schema（标准） |
| **调用位置** | `message.tool_calls[]` | `content` block `type: "tool_use"` |
| **结果回传** | `role: "tool"` + `tool_call_id` | `role: "user"` + `type: "tool_result"` |
| **并行调用** | ✅ 原生支持 | ✅ 原生支持 |
| **强制调用** | `tool_choice: {"function": {"name": "X"}}` | `tool_choice: {"type": "tool", "name": "X"}` |
| **禁止调用** | `tool_choice: "none"` | `tool_choice: {"type": "none"}` |
| **嵌套对象** | ✅ 完整 JSON Schema | ✅ 完整 JSON Schema |
| **Streaming** | ✅ chunk 模式 | ✅ SSE event 模式 |
| **计费** | tool schema tokens 算 input | tool schema tokens 算 input |

### Anthropic 的差异点

```python
# Anthropic Tool Use
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "获取天气",
        "input_schema": {  # 注意：Anthropic 用 input_schema
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }],
    messages=[{"role": "user", "content": "北京天气如何？"}]
)

# 返回格式不同——tool_use 是 content block
for block in response.content:
    if block.type == "tool_use":
        print(block.name)   # "get_weather"
        print(block.input)  # {"city": "北京"}
        print(block.id)     # "toolu_xxx" (用于回传结果)

# 回传结果：放在 user message 中
follow_up = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[...],
    messages=[
        {"role": "user", "content": "北京天气如何？"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": '{"temperature": 5, "condition": "晴"}'
        }]}
    ]
)
```

## 6. MCP：工具生态的标准化

[[如何给人深度科普-MCP|MCP]]（Model Context Protocol）由 Anthropic 提出，旨在标准化 Agent 与工具的交互协议：

- **Server 端**：暴露 tools、resources、prompts
- **Client 端**：LLM 应用通过 MCP Client 连接 Server
- **传输层**：stdio（本地）或 Streamable HTTP（远程）
- **动态发现**：Client 可以运行时查询 Server 有哪些工具
- **OpenAI、Google 已跟进支持 MCP**

## 7. 面试常考题

### Q1: Function Calling 是模型在执行函数吗？原理是什么？
**答**：不是。LLM 只输出结构化的函数调用意图（函数名 + JSON 参数），由应用层代码解析并执行，再将结果返回给 LLM。这个能力通过 SFT 训练获得——构造（query, tools, call, result, response）数据，让模型学会在适当时候输出 `<tool_call>` + JSON 结构而非自然语言。

### Q2: ReAct 和 Function Calling Agent 有什么区别？
**答**：ReAct 是一种 prompting 策略——通过 Thought-Action-Observation 文本模板让模型交替推理和行动，需要解析模型的文本输出来提取 action。Function Calling Agent 利用模型原生的结构化输出能力，直接生成 JSON 格式的工具调用，不需要文本解析。后者更稳定（不会因格式解析失败而出错）、支持并行调用、且模型经过专门训练准确率更高。ReAct 适用于不支持 function calling 的模型。

### Q3: 当工具数量很多时，如何做工具选择？
**答**：分层策略：(1) < 20 个工具：全量注入 context；(2) 20-200 个：工具描述 embedding + 语义检索 Top-K；(3) > 200 个：先用分类器判断意图类别，再加载该类别工具子集；(4) 超大规模：MCP 动态发现 + 专门训练的 tool selection model。也可以给工具分组，通过 "meta-tool"（如 `route_to_category`）做两阶段选择。

### Q4: OpenAI 和 Anthropic 的 Tool Use API 有哪些关键差异？
**答**：核心差异：(1) Schema 字段——OpenAI 用 `parameters`，Anthropic 用 `input_schema`；(2) 返回位置——OpenAI 在 `message.tool_calls` 数组，Anthropic 在 `content` blocks（`type: "tool_use"`）中；(3) 结果回传——OpenAI 用 `role: "tool"` 消息配合 `tool_call_id`，Anthropic 在 `role: "user"` 消息中用 `type: "tool_result"`。功能上两者等价，都支持并行调用、强制/禁止调用、streaming。

### Q5: 什么是 MCP？它解决了什么问题？
**答**：MCP（Model Context Protocol）是 Anthropic 提出的开放标准，解决 M×N 问题——M 个 LLM 应用 × N 个外部工具的集成爆炸。它定义了标准化的 Server/Client 协议：Server 暴露 tools/resources/prompts，Client 通过 stdio 或 HTTP 连接。核心价值：(1) 工具开发者写一次 Server，所有支持 MCP 的 Client 都能用；(2) 支持运行时动态发现工具，不需要硬编码；(3) OpenAI、Google 已跟进支持，正在成为行业标准。
