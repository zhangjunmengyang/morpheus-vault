---
title: "Planner：Multi-Agent 系统的任务分解核心"
brief: "Planner 角色深度解析：高层意图→可执行子任务序列；设计模式（静态规划/动态重规划/Hierarchical）；与 Executor/Critic 协作；好的 Planner 是 MAS 智能上限的天花板"
type: concept
domain: ai/agent/multi-agent
created: "2026-02-13"
updated: "2026-02-23"
tags:
  - ai/agent/multi-agent
  - type/concept
---
# Planner

Planner 是 multi-agent 系统中最核心的角色之一 —— 负责将用户的高层意图分解为可执行的子任务序列，然后分发给专门的 executor agent 执行。好的 planner 是整个系统智能上限的天花板。

## 为什么需要 Planner

直接让一个 LLM 端到端完成复杂任务，会遇到几个问题：

1. **上下文窗口不够** —— 复杂任务需要大量中间状态
2. **单模型能力有限** —— 不可能一个模型又写代码又查数据库又画图
3. **错误传播** —— 一步错步步错，没有纠错机制

Planner 的引入把「思考」和「执行」解耦了。

## Planner 的常见模式

### 1. Static Plan（一次性规划）

最简单的模式：planner 一次性生成完整计划，然后按顺序执行。

```python
PLANNER_PROMPT = """
你是一个任务规划器。给定用户需求，输出一个 JSON 格式的执行计划。
每个步骤包含: step_id, description, agent（选择: coder/researcher/reviewer）, dependencies

用户需求: {user_query}
"""

# 输出示例
{
  "steps": [
    {"step_id": 1, "description": "调研竞品定价策略", "agent": "researcher", "dependencies": []},
    {"step_id": 2, "description": "分析内部销售数据", "agent": "coder", "dependencies": []},
    {"step_id": 3, "description": "生成定价建议报告", "agent": "coder", "dependencies": [1, 2]},
    {"step_id": 4, "description": "审查报告准确性", "agent": "reviewer", "dependencies": [3]}
  ]
}
```

优点是简单可控，缺点是**不够灵活** —— 执行中发现计划有问题没法调整。

### 2. Adaptive Re-planning（动态重规划）

执行每一步后，planner 根据结果决定下一步。这更接近人类的工作方式。

```python
class AdaptivePlanner:
    def __init__(self, llm):
        self.llm = llm
        self.history = []
    
    def plan_next(self, user_goal, completed_steps, current_state):
        prompt = f"""
        目标: {user_goal}
        已完成: {completed_steps}
        当前状态: {current_state}
        
        基于以上信息，决定下一步行动。如果目标已达成，输出 DONE。
        输出格式: {{"action": "...", "agent": "...", "reason": "..."}}
        """
        return self.llm(prompt)
```

### 3. Hierarchical Planning（分层规划）

大任务 → 子目标 → 具体步骤，多层 planner 嵌套。适合特别复杂的场景（比如自动写一个完整项目）。

## 关键设计决策

### 计划的粒度

太粗 —— executor 不知道具体做什么；太细 —— planner 容易出错，且不灵活。经验法则：**每个步骤对应一次 agent 调用**。

### 依赖关系处理

步骤之间的依赖关系决定了并行度。DAG 结构是最常见的表示方式，无依赖的步骤可以并行执行。

### 失败恢复策略

- **Retry** —— 简单重试
- **Re-plan** —— 回到 planner 重新规划
- **Fallback** —— 切换到备选方案
- **Escalate** —— 交给人类处理

实践中 re-plan 最有效，但要注意设置最大重试次数，否则容易陷入无限循环。

## Planner 的 Prompt 技巧

1. **给 planner 明确的 agent 能力描述** —— planner 需要知道每个 executor 能做什么
2. **Few-shot examples** —— 给几个规划示例比纯指令效果好得多
3. **约束输出格式** —— JSON > 自然语言，方便下游解析
4. **Think step by step** —— 让 planner 先分析再输出计划

## 我的看法

Planner 的质量严重依赖底层 LLM 的推理能力。用 GPT-4 级别的模型做 planner，executor 可以用更小的模型 —— 这是一种很好的成本优化策略。另外，**不要过度设计 planner**。很多场景下 static plan + 简单的 retry 就够了，adaptive re-planning 引入了更多复杂度和延迟。

## 相关

- [[Multi-Agent 概述|Multi-Agent 概述]]
- [[AutoGen|AutoGen]]
- [[Tool Use|Tool Use]]
- [[记忆模块|记忆模块]]
