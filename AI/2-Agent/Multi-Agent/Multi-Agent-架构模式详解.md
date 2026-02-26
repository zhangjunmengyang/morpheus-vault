---
title: "Multi-Agent 架构模式详解"
brief: "三种 Multi-Agent 架构深度解析含代码：Supervisor（中央调度）/Pipeline（顺序流水）/Debate（辩论共识）；各自适用场景/优劣权衡/实现模板；是 Multi-Agent 概述的深化版"
type: concept
domain: ai/agent/multi-agent
created: "2026-02-13"
updated: "2026-02-23"
tags:
  - ai/agent/multi-agent
  - type/concept
rating: 4
status: active
---

> [!note] 路径说明
> 原文件名为临时名称（untitled_SB2HwKNC.md），馆长于2026-02-22重命名。内容：Supervisor/Pipeline/Debate三种架构模式含代码实现，是 [[AI/2-Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]] 的深化版。
# Multi-Agent 架构

Multi-Agent 系统是 AI Agent 从单兵作战走向协作的关键一步。核心思想：**一个复杂任务，拆成多个专业 Agent 各司其职，比一个全能 Agent 更可靠**。

## 为什么要 Multi-Agent

单个 Agent 面临的瓶颈：
- **Context 长度限制**：任务越复杂需要的上下文越多
- **专注力不足**：一个 prompt 塞太多角色和工具，模型会困惑
- **可靠性差**：长 chain 的每一步都可能出错，整体成功率指数衰减

Multi-Agent 的核心价值是**分工**和**隔离**——每个 Agent 只关心自己擅长的事。

## 典型架构模式

### Supervisor（监督者）模式

```python
# LangGraph 实现
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

# 定义专业 Agent
researcher = create_react_agent(model, tools=[search_tool])
coder = create_react_agent(model, tools=[code_exec_tool])
writer = create_react_agent(model, tools=[write_tool])

# Supervisor 决定下一步交给谁
def supervisor(state):
    response = model.invoke(
        f"当前任务：{state['task']}\n"
        f"已完成：{state['progress']}\n"
        "下一步交给谁？(researcher/coder/writer/FINISH)"
    )
    return {"next": response.content}

# 构建图
graph = StateGraph()
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("writer", writer)
graph.add_conditional_edges("supervisor", route_to_agent)
```

### Pipeline（流水线）模式

```
需求分析 Agent → 代码生成 Agent → 测试 Agent → Review Agent
```

适合流程固定的场景。每个 Agent 的输出就是下一个 Agent 的输入。

### Debate（辩论）模式

```python
# 两个 Agent 互相 challenge
for round in range(3):
    response_a = agent_a.generate(
        f"问题：{question}\n"
        f"对方的答案：{response_b}\n"
        "请给出你的答案，并指出对方的错误（如果有的话）。"
    )
    response_b = agent_b.generate(
        f"问题：{question}\n"
        f"对方的答案：{response_a}\n"
        "请给出你的答案，并指出对方的错误（如果有的话）。"
    )

# 最终由 Judge Agent 裁定
final = judge.generate(f"A 的答案：{response_a}\nB 的答案：{response_b}\n请选择更好的答案。")
```

论文证明 debate 能显著提升推理准确率，但成本高。

## 设计原则

1. **最小权限**：每个 Agent 只给它需要的工具和信息
2. **明确接口**：Agent 之间的输入输出格式要严格定义
3. **可观测性**：每个 Agent 的输入、输出、决策过程都要记录
4. **优雅降级**：某个 Agent 失败时有 fallback 策略
5. **成本控制**：设置 token budget 和调用次数上限

## 实际应用场景

| 场景 | 架构 | Agent 组成 |
|------|------|-----------|
| 代码开发 | Pipeline | 需求理解 → 编码 → 测试 → Review |
| 数据分析 | Supervisor | SQL Agent + 可视化 Agent + 解读 Agent |
| 客服 | Router | 意图识别 → 路由到专业 Agent（退款/技术/投诉） |
| 研究 | Debate | 多角度分析 → 综合 |

## 相关

- [[AI/2-Agent/Multi-Agent/Multi-Agent 概述|Multi-Agent 概述]] — 本文的上级入口，概述层
- [[AI/2-Agent/Multi-Agent/零碎的点|Multi-Agent 零碎思考]] — 补充细节
- [[AI/5-AI 安全/OMNI-LEAK-Multi-Agent-Orchestrator-Data-Exfiltration|OMNI-LEAK]] — Supervisor模式的安全风险：orchestrator被injection后污染整个网络——本文Supervisor架构的安全反面教材
- [[AI/2-Agent/Multi-Agent/AgentConductor-Topology-Evolution|AgentConductor]] — 动态拓扑Multi-Agent，超越了本文的固定Supervisor/Pipeline模式
- [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026综合分析]] — Multi-Agent RL的整体框架，本文是其中架构模式的具体实现
