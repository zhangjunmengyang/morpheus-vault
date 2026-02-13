---
title: "Agent Course"
type: tutorial
domain: ai/agent/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/fundamentals
  - type/tutorial
---
# HF Agent Course

Hugging Face 出的 [AI Agent 课程](https://huggingface.co/learn/agents-course)，免费开源，质量不错。适合系统性学习 agent 开发的基础知识。

## 课程结构

课程分几个大模块：

1. **Introduction to Agents** —— agent 的定义、与普通 LLM 应用的区别
2. **Frameworks** —— smolagents (HF 自家)、LangGraph、CrewAI 等
3. **Tool Use** —— 如何定义和使用工具
4. **RAG + Agents** —— 把检索增强和 agent 结合
5. **Multi-Agent Systems** —— 多 agent 协作
6. **Vision & Browser Agents** —— 视觉 agent、网页操作 agent

## 核心概念笔记

### Agent Loop（核心循环）

每个 agent 本质上都是一个循环：

```
Observe → Think → Act → Observe → Think → Act → ... → Done
```

在代码层面：

```python
# smolagents 的核心循环简化
class Agent:
    def run(self, task):
        observations = [task]
        while True:
            # Think: LLM 根据历史观察决定下一步
            thought, action = self.llm.plan(observations)
            
            if action == "final_answer":
                return thought
            
            # Act: 执行工具调用
            result = self.execute_tool(action)
            
            # Observe: 把结果加入历史
            observations.append(result)
```

### ReAct Pattern

课程重点讲了 ReAct（Reasoning + Acting），这是目前最主流的 agent 设计模式：

- **Reasoning**：模型先用自然语言「思考」当前状态和下一步
- **Acting**：基于思考结果选择并执行一个 action
- **Observation**：action 执行结果作为新的输入

```
Thought: 用户要查北京天气，我需要调用天气 API
Action: weather_tool(city="北京")
Observation: 北京今天 15°C，晴
Thought: 已经获得天气信息，可以回复用户了
Action: final_answer("北京今天 15°C，晴天")
```

### Tool 定义

smolagents 的 tool 定义很 Pythonic：

```python
from smolagents import tool

@tool
def search_web(query: str) -> str:
    """搜索网页并返回结果摘要。
    
    Args:
        query: 搜索关键词
    """
    # 实现搜索逻辑
    return search_engine.search(query)
```

装饰器会自动从 docstring 和 type hints 提取 tool schema，传给 LLM。这比手写 JSON schema 舒服多了。

### Code Agent vs Tool-calling Agent

课程区分了两种模式：

- **Tool-calling Agent**：LLM 输出结构化的 tool call（JSON），由 runtime 解析执行
- **Code Agent**：LLM 直接写 Python 代码，runtime 执行代码

Code Agent 更灵活（可以写循环、条件判断、变量赋值），但安全风险更大（需要沙箱）。smolagents 默认推荐 Code Agent。

## 实践作业亮点

课程的实践部分做得不错：

- 构建一个能搜索 HuggingFace Hub 的 agent
- 用 agent 操作浏览器完成网页任务
- 多 agent 系统：manager + web_searcher + code_executor

## 我的评价

优点：
- 免费且持续更新
- 代码驱动，不是纯理论
- 覆盖面广，从基础到多 agent 都有

不足：
- smolagents 框架偏小众，生产级项目更多用 LangGraph 或 AutoGen
- 部分章节偏浅，更像入门导览

总体推荐度：⭐⭐⭐⭐（适合系统入门，但深入还需要看论文和框架源码）

## 相关

- [[Tool Use]]
- [[AI/Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]]
- [[AI/Agent/Frameworks/AutoGen|AutoGen]]
- [[记忆模块]]
