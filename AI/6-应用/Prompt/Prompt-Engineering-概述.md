---
brief: "Prompt Engineering 概述——官方/主流资源汇总入口；OpenAI/Anthropic/Google 的 Prompting Guide 导航；快速定位最新 best practices 的参考索引。"
title: "Prompt engineering"
type: reference
domain: ai/llm/prompt-engineering
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/prompt-engineering
  - type/reference
---
# Prompt Engineering 概述

> 参考：https://github.com/dair-ai/Prompt-Engineering-Guide

Prompt Engineering 是跟大模型打交道最基础也最被低估的技能。很多人觉得"不就是写几句话吗"，但实际上从 few-shot 到 chain-of-thought 再到 tool use，这里面有一套完整的方法论。

## 核心原则

**1. 清晰 > 简短**

模型不会读心，含糊的指令只会得到含糊的输出。与其写"帮我总结一下"，不如写"用 3 个 bullet points 总结以下文章的核心论点，每个不超过 30 字"。

**2. 结构化输入输出**

```
## 角色
你是一位资深数据工程师。

## 任务
分析以下 SQL 的性能瓶颈。

## 输出格式
- 问题描述（一句话）
- 根因分析
- 优化建议（含改写后的 SQL）

## 输入
{sql_code}
```

**3. 给模型"思考空间"**

Chain-of-Thought（CoT）是目前最有效的提示技巧之一。加一句"Let's think step by step"就能显著提升推理类任务的准确率。但 CoT 不是万能的——对于简单的分类、提取任务，反而会增加延迟和成本。

## 主要技巧分类

### Few-Shot Prompting

给几个示例让模型学会模式：

```python
prompt = """
将以下中文翻译为英文：

输入：今天天气不错
输出：The weather is nice today.

输入：我在学习 Prompt Engineering
输出：I'm learning Prompt Engineering.

输入：{user_input}
输出：
"""
```

关键是示例的**多样性**和**一致性**。示例太少覆盖不了 edge case，示例格式不一致模型会困惑。

### Chain-of-Thought (CoT)

```
Q: 一个商店有 23 个苹果，卖出 15 个后又进货 8 个，现在有多少？
A: 让我一步步算。初始 23 个，卖出 15 个剩 23-15=8 个，
   再进货 8 个变成 8+8=16 个。答案是 16 个。
```

变体：
- **Zero-shot CoT**: 直接加"Let's think step by step"
- **Self-Consistency**: 多次采样取多数票
- **Tree of Thought**: 让模型探索多条推理路径

### System Prompt 设计

System prompt 定义了模型的"人格"和行为边界。好的 system prompt 应该包含：

1. **角色定义**：你是谁，擅长什么
2. **行为约束**：什么该做什么不该做
3. **输出规范**：格式、语言、风格
4. **工具说明**：可用的工具及调用方式

```python
system_prompt = """你是一位 Python 代码审查助手。

规则：
1. 只关注代码质量问题，不改变业务逻辑
2. 每个问题给出严重等级：🔴 高 🟡 中 🟢 低
3. 必须给出修改建议和修改后的代码片段
4. 如果代码没有问题，直接说"LGTM"
"""
```

## 进阶模式

### ReAct（Reasoning + Acting）

让模型在推理和行动之间交替：

```
Thought: 用户问的是实时股价，我需要调用搜索工具。
Action: search("AAPL stock price today")
Observation: Apple Inc. (AAPL) $198.50 +2.3%
Thought: 拿到数据了，可以回答了。
Answer: 苹果公司 (AAPL) 当前股价为 $198.50，今日上涨 2.3%。
```

### Structured Output

强制模型输出 JSON/XML 等结构化格式。关键技巧是在 prompt 中给出 schema：

```python
output_schema = {
    "sentiment": "positive | negative | neutral",
    "confidence": "0.0 - 1.0",
    "key_phrases": ["phrase1", "phrase2"]
}
```

现在主流 API（OpenAI、Anthropic）都支持原生的 structured output，比 prompt hack 靠谱得多。

## 常见坑

| 坑 | 解决 |
|---|---|
| 模型不遵循格式 | 在 prompt 末尾重申格式要求 |
| 输出太长/太短 | 明确字数或 bullet 数量 |
| 模型编造信息 | 加入"如果不确定请说不知道" |
| 多轮对话丢失上下文 | 将关键信息在每轮开头重复 |
| 复杂任务效果差 | 拆成多个子任务串联 |

## 我的经验

Prompt Engineering 的本质是**把人的意图无损传递给模型**。越是资深的工程师，越能写出好的 prompt——因为他们知道怎么把复杂问题拆解成清晰的步骤。

另外一个反直觉的点：prompt 并不是越长越好。很多时候一个精炼的 prompt 比一个冗长的 prompt 效果更好，因为噪音少了，模型更容易抓住重点。

## 相关

- [[AI/6-应用/Prompt/Prompt-Engineering-基础|Prompt Engineering 实践]]
- [[AI/6-应用/Prompt/Prompt-攻击|Prompt 攻击与防御]]
- [[AI/6-应用/Prompt/Prompt-Tools|Prompt 工具集]]
- [[AI/6-应用/Synthetic-Data/Synthetic-Data|数据合成]]
- [[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek-R1]]
