---
brief: "Tool Use——LLM 工具调用的基础原理：Function Calling 协议/JSON Schema 定义/tool selection 机制；ReAct（推理+行动）和 ToolFormer 的对比；是 Agent 能力的基础组件，RAG/搜索/计算工具的接入方式。"
title: "Tools"
type: concept
domain: ai/agent/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/fundamentals
  - type/concept
---
# Tool Use

> Agent 的核心能力之一：让 LLM 调用外部工具来弥补自身短板（计算、搜索、代码执行等）。

参考：https://mp.weixin.qq.com/s/CbtKUZQAC9YJyoy9r_Hz4Q

## 本质

LLM 本身是一个 text-in-text-out 的函数。Tool Use 是让它学会在合适的时机 **生成结构化的函数调用**，然后把结果注入上下文继续推理。

```
User: 今天北京天气怎么样？
LLM: [思考] 我需要查天气 → [调用] get_weather("北京")
System: {"temp": 25, "condition": "晴"}
LLM: 今天北京 25°C，晴天，适合出门。
```

## 实现方式对比

### 1. Function Calling (主流)

OpenAI 率先推出的方案。模型被训练成在需要时输出特定格式的 JSON：

```python
# 定义 tools
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索互联网信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
}]

# 模型输出
# {"name": "search", "arguments": {"query": "北京天气"}}
```

关键点：
- **模型需要专门训练**才能可靠地输出 function call
- 参数类型验证靠 JSON Schema
- 支持 **parallel function calling**（一次输出多个调用）

### 2. ReAct (Reasoning + Acting)

让模型在文本中交替进行思考和行动：

```
Thought: 用户问天气，我需要用搜索工具
Action: search("北京天气")
Observation: 25°C，晴
Thought: 我已经有了需要的信息
Answer: 今天北京 25°C，晴天。
```

实现更简单（纯 prompt 工程），但格式解析不够鲁棒。

### 3. Code Interpreter

给模型一个 Python 沙箱，让它写代码来解决问题：

```python
# 模型生成的代码
import pandas as pd
df = pd.read_csv("sales.csv")
monthly = df.groupby("month")["revenue"].sum()
print(monthly.to_markdown())
```

这是最灵活的方式 — 任何能用代码表达的操作都变成了 "工具"。

## Tool Selection 的训练

让模型学会选择正确的工具，主要有三种路径：

### SFT 路径
```python
# 用 (query, correct_tool_call, result) 三元组做监督微调
training_data = [
    {
        "messages": [
            {"role": "user", "content": "计算 sqrt(144)"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "calculator", "arguments": '{"expr": "sqrt(144)"}'}}
            ]},
            {"role": "tool", "content": "12.0"},
            {"role": "assistant", "content": "sqrt(144) = 12"}
        ]
    }
]
```

### RL 路径
用 RL 训练模型的 tool use 决策。Reward 设计：

```python
def tool_use_reward(trajectory):
    if correct_tool_selected and correct_params:
        reward = 1.0
    elif correct_tool_selected:
        reward = 0.5  # 工具对但参数错
    elif unnecessary_tool_call:
        reward = -0.3  # 不需要调工具却调了
    else:
        reward = -0.5  # 需要调工具但没调
    
    # 最终还要看 task completion
    if task_completed_correctly:
        reward += 2.0
    return reward
```

### 多轮 Planning 的 RL 训练

这是当前的前沿方向：训练模型在多步 tool use 场景中学会 planning。比如完成一个复杂任务需要：搜索 → 代码执行 → API 调用 → 总结，模型需要学会整个序列的最优策略。

```
# 挑战：
# 1. reward 稀疏（只有最后才知道对不对）
# 2. trajectory 很长（多轮交互）
# 3. action space 巨大（工具选择 × 参数组合）
```

verl 的 agentic RL training 就是解决这个问题的，参见 [[AI/2-Agent/Agentic-RL/Agentic-RL-Training-verl|Agentic RL Training]]。

## 工程实践要点

1. **Tool description 质量决定调用准确率**：描述不清楚，模型就会选错工具
2. **参数验证要做在调用侧**：不能完全信任模型输出的 JSON
3. **错误处理要返回给模型**：让模型看到报错信息并 retry
4. **工具数量有上限**：工具太多（>20）模型选择准确率会显著下降
5. **token 预算**：每个工具定义都会消耗 context window

## 相关

- [[AI/2-Agent/Fundamentals/记忆模块|记忆模块]] — Agent 的另一核心能力
- [[AI/2-Agent/Agentic-RL/Agentic-RL-Training-verl|Agentic RL Training]] — 用 RL 训练 Tool Use
- [[Prompt-Tools|Prompt Engineering Tools]] — 系统 prompt 的工具描述
- [[思考/Multi-Agent 零碎的点|Multi-Agent 零碎的点]] — 多 Agent 工具协作
