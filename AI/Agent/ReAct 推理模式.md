---
title: "ReAct 及 LLM 推理模式"
date: 2026-02-14
tags: [agent, reasoning, react, cot, interview]
type: note
---

# ReAct 及 LLM 推理模式

## 1. Chain-of-Thought (CoT)

### 核心思想

CoT 的本质是让模型在输出最终答案之前，先生成中间推理步骤。这模拟了人类"打草稿"的过程——把复杂问题拆解成一系列简单的子步骤，逐步求解。

关键论文：Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*

### Few-shot CoT

在 prompt 中提供若干包含推理过程的示例（exemplars），让模型学会"展示推理步骤"的格式：

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 balls. 5 + 6 = 11. The answer is 11.

Q: <your actual question>
A:
```

**要点**：示例的质量和多样性直接影响效果，一般 3-6 个示例就够。

### Zero-shot CoT

Kojima et al., 2022 发现只需在 prompt 末尾加一句 **"Let's think step by step"**，模型就会自发生成推理链，不需要任何示例。

```
Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?
A: Let's think step by step.
```

**为什么有效**：这个 trigger phrase 激活了模型在预训练阶段学到的"展示推理过程"的模式。Zero-shot CoT 在算术、常识推理、符号推理等任务上都有显著提升。

### CoT 的局限

- **只有推理，没有行动**：无法获取外部信息、调用工具、验证事实
- **幻觉累积**：推理链越长，错误传播越严重，且无法自我纠正
- **对小模型效果差**：一般需要 ≥ 100B 参数的模型才能稳定产生有效 CoT

---

## 2. ReAct (Reason + Act)

### 核心思想

Yao et al., 2023 提出的 ReAct 框架将**推理（Reasoning）** 和**行动（Acting）** 交织在一起。模型不仅思考，还能调用外部工具获取真实信息，然后基于观察结果继续推理。

### 核心循环

```
Thought: 我需要查一下2024年诺贝尔物理学奖得主
Action: Search["2024 Nobel Prize Physics"]
Observation: The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton...
Thought: 找到了，Hinton 因为神经网络的基础工作获奖，让我继续...
Action: ...
Observation: ...
Thought: 现在我有足够信息回答了
Action: Finish[answer]
```

### ReAct vs 纯 CoT

| 维度 | 纯 CoT | ReAct |
|------|--------|-------|
| 信息来源 | 仅依赖模型参数中的知识 | 可以调用外部工具获取实时信息 |
| 幻觉风险 | 高，无法验证 | 低，Observation 提供事实锚点 |
| 可解释性 | Thought 可读 | Thought + Action + Observation 完整可追溯 |
| 适用任务 | 推理密集型（数学、逻辑） | 知识密集型（问答、事实核查、多步检索） |
| 复杂度 | 单次生成 | 多轮交互，需要工具集成 |

### 关键设计细节

- **Action Space 定义**：需要预定义模型可调用的工具集（Search、Lookup、Calculator 等）
- **停止条件**：模型输出 `Finish[answer]` 或达到最大步数
- **解析器**：需要 parser 从模型输出中提取 Action 和参数
- **错误处理**：工具返回空结果时，模型需要学会换个方式查询

---

## 3. Plan-and-Solve

### 核心思想

Wang et al., 2023 — 先让模型制定完整计划，再逐步执行。解决 CoT 中"走一步看一步"容易迷失方向的问题。

### 两阶段流程

```
阶段1 - Planning:
"Let's first understand the problem and devise a plan to solve it."
→ 模型输出: Step 1: ... Step 2: ... Step 3: ...

阶段2 - Execution:
"Let's carry out the plan step by step."
→ 模型按计划逐步执行
```

### PS+ 变体

Plan-and-Solve+ 额外加入约束：
- "extract relevant variables and their corresponding numerals"
- "calculate intermediate results"
- 提高计算精度和变量追踪能力

### 适用场景

- 多步数学应用题
- 需要明确步骤的复杂任务
- 编程问题分解

---

## 4. Tree-of-Thought (ToT)

### 核心思想

Yao et al., 2023 — 把 CoT 从线性链扩展为**树状搜索**。模型在每一步生成多个候选思路，通过评估函数选择最优路径，支持回溯（backtracking）。

### 算法框架

```
                    [Problem]
                   /    |    \
              [T1a]  [T1b]  [T1c]     ← 第1步的多个候选
              / \      |       ✗       ← T1c 被评估为差，剪枝
          [T2a] [T2b] [T2b']          ← 继续展开
            |     ✗     |
          [T3a]       [T3a']          ← 最终选择最优路径
```

### 关键组件

1. **Thought Generator**：每步生成 k 个候选思路（采样 or 独立提示）
2. **State Evaluator**：评估当前状态的好坏（投票 or 打分）
3. **Search Algorithm**：BFS（广度优先）或 DFS（深度优先）
4. **Backtracking**：发现死路时回退到上一个分叉点

### 适用场景

- 24 点游戏、创意写作、填字游戏等需要探索的任务
- 答案空间大、需要试错的问题

### 缺点

- **Token 消耗巨大**：每个节点都需要 LLM 调用
- **延迟高**：需要多轮评估和搜索
- **实现复杂**：需要额外的搜索框架

---

## 5. Reflexion

### 核心思想

Shinn et al., 2023 — 让 Agent 在失败后进行**自我反思**，将反思结果存入记忆，下次尝试时避免相同错误。

### 循环流程

```
尝试1: Actor 执行任务 → 失败
         ↓
反思: "我失败是因为X，下次应该Y"
         ↓
记忆: 将反思存入 episodic memory
         ↓
尝试2: Actor 带着反思记忆再次执行 → 成功
```

### 核心组件

1. **Actor**：执行任务的 Agent（可以是 ReAct Agent）
2. **Evaluator**：判断任务是否成功（可以是规则、LLM 判断、单元测试等）
3. **Self-Reflection**：生成反思文本，分析失败原因
4. **Memory**：存储历次反思，作为后续尝试的上下文

### 与人类学习的类比

Reflexion 模拟了人类的"失败→复盘→再尝试"过程。关键区别在于 LLM 本身没有持久记忆，所以需要显式地把反思写入 prompt。

### 适用场景

- 代码生成（用测试结果作为反馈）
- 需要迭代改进的任务
- 决策类任务

---

## 6. 各模式对比表

| 模式 | 核心机制 | 适用场景 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **CoT** | 线性推理链 | 数学、逻辑推理 | 简单、零成本、可解释 | 无法获取外部信息，幻觉累积 |
| **ReAct** | 推理+工具调用交替 | 知识问答、事实核查 | 接地气、减少幻觉、可追溯 | 需要工具集成，延迟增加 |
| **Plan-and-Solve** | 先规划后执行 | 复杂多步任务 | 全局视角，步骤清晰 | 计划本身可能有误，灵活性差 |
| **ToT** | 树状搜索+回溯 | 创意/探索性任务 | 能发现非显然解，支持回溯 | Token 消耗大，延迟高 |
| **Reflexion** | 失败→反思→重试 | 代码生成、迭代任务 | 能从错误中学习，持续改进 | 需要评估器，多次尝试成本高 |

**选型决策树**：
- 纯推理问题 → CoT
- 需要外部知识 → ReAct
- 复杂多步 + 需要全局规划 → Plan-and-Solve
- 答案空间大需要探索 → ToT
- 允许多次尝试 + 有评估信号 → Reflexion
- 实际生产中 → 往往是多种模式的组合

---

## 7. 与 Agent 框架的关系

### LangChain Agent

LangChain 的 Agent 模块本质上是 ReAct 模式的工程化实现：

```python
from langchain.agents import create_react_agent, AgentExecutor

# 定义工具
tools = [search_tool, calculator_tool, ...]

# 创建 ReAct agent
agent = create_react_agent(llm, tools, prompt)

# AgentExecutor 管理循环
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
```

**LangChain 的角色**：
- 提供 Tool 抽象和注册机制
- 实现 Thought→Action→Observation 的循环控制
- 处理输出解析（OutputParser）
- 管理上下文窗口和记忆

**Agent 类型**：
- `create_react_agent`：标准 ReAct
- `create_structured_chat_agent`：结构化输出
- `create_openai_tools_agent`：利用 OpenAI function calling
- LangGraph：更灵活的图状态机，支持自定义循环和分支

### LlamaIndex Agent

LlamaIndex 更聚焦于 RAG 场景下的 Agent：

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 把 RAG QueryEngine 包装成工具
query_tool = QueryEngineTool.from_defaults(query_engine, name="knowledge_base")

# ReAct Agent
agent = ReActAgent.from_tools([query_tool], llm=llm, verbose=True)
```

**LlamaIndex 的特色**：
- 原生集成 RAG pipeline 作为 Agent 工具
- 支持 Sub-Question Query Engine（自动拆分子问题）
- `RouterQueryEngine`：根据查询类型路由到不同引擎

### 框架 vs 推理模式的关系

```
推理模式（理论层）     框架（工程层）
─────────────────    ──────────────
CoT                  → prompt engineering
ReAct                → LangChain Agent / LlamaIndex ReActAgent
Plan-and-Solve       → LangGraph / CrewAI 的 planning 模块
ToT                  → 需要自定义实现（或 langchain-experimental）
Reflexion            → LangGraph 的循环 + 记忆节点
```

---

## 8. 面试常见问题及回答要点

### Q1: 请解释 ReAct 和 CoT 的区别，什么场景下选择哪个？

**回答要点**：
- CoT 是纯推理，只靠模型内部知识，适合逻辑推理、数学计算
- ReAct 在推理中穿插工具调用，能获取外部信息，适合知识密集型任务
- **关键区别**：ReAct 的 Observation 步骤提供了"事实锚点"，显著降低幻觉
- 举例：问"今天天气"用 CoT 必然幻觉，ReAct 可以调用天气 API
- 实际工程中 ReAct 更常用，因为大多数有价值的任务都需要外部信息

### Q2: ReAct Agent 容易陷入循环怎么办？

**回答要点**：
- **Max iterations 限制**：设置最大步数（一般 5-15 步）
- **重复检测**：检测连续的相同 Action，强制转换策略
- **Early stopping**：检测到 "I don't know" 等模式时提前终止
- **Prompt 优化**：在 system prompt 中明确告知"如果信息足够就直接回答"
- **工具设计**：确保工具返回有用信息，避免空结果导致反复查询
- **LangChain 的处理**：`AgentExecutor` 有 `max_iterations` 和 `early_stopping_method` 参数

### Q3: Tree-of-Thought 在实际生产中能用吗？

**回答要点**：
- **Token 成本**：每个节点都需要 LLM 调用，b=5 d=3 就需要 ~150 次调用
- **延迟**：串行搜索时延迟是 CoT 的几十倍
- **适用场景有限**：适合离线、高价值、探索性任务（代码生成、方案设计）
- **实际替代方案**：
  - Best-of-N sampling（生成 N 个完整答案取最优）更实用
  - Self-consistency（多次 CoT 投票）性价比更高
  - 新一代推理模型（o1/o3/DeepSeek-R1）内置了隐式搜索

### Q4: 如何评估一个 Agent 的推理质量？

**回答要点**：
- **任务完成率**：最终答案是否正确
- **中间步骤质量**：推理链是否合理（人工标注 or LLM-as-judge）
- **工具调用效率**：是否用了最少的步骤完成任务
- **幻觉率**：Thought 中的事实性陈述是否准确
- **鲁棒性**：对问题的不同表述是否给出一致的答案
- **Benchmark**：HotpotQA（多跳推理）、ALFWorld（交互决策）、WebShop（电商导航）、SWE-bench（代码修复）

### Q5: Reflexion 和 RLHF 有什么区别？

**回答要点**：
- **RLHF**：通过梯度更新改变模型权重，是**训练时**优化
- **Reflexion**：不改变模型权重，通过自然语言反思存入 prompt 上下文，是**推理时**优化
- **类比**：RLHF 像长期学习改变认知，Reflexion 像考试时在草稿纸上记错题
- **Reflexion 优点**：不需要训练、即时生效、可解释
- **Reflexion 缺点**：受限于上下文窗口、反思质量依赖模型能力、不能泛化到新任务

### Q6: 实际项目中你会怎么设计一个 Agent 的推理策略？

**回答要点**：
- **分析任务类型**：是否需要外部信息？是否多步？是否需要迭代？
- **从简单开始**：先试 CoT / 直接 function calling，不够再上 ReAct
- **工具设计是关键**：好的工具描述比复杂的推理策略更重要
- **分层架构**：
  - 简单查询 → 直接 RAG，不走 Agent
  - 中等复杂 → ReAct（3-5 步）
  - 高度复杂 → Plan-and-Solve + ReAct 子任务
- **可观测性**：记录每一步的 Thought/Action/Observation，方便调试
- **兜底机制**：设置超时、最大步数、fallback 到人工
- **评估驱动**：建立 eval 数据集，量化比较不同策略的效果
