---
title: "Agent Harness Engineering：让 Agent 稳定工作的基础设施工程"
brief: "2026 年 Agent 竞争护城河从「模型能力」转向「Harness 可靠性」。Harness = 包围 Agent 的外部基础设施系统：状态持久化、进度追踪、工具调用协议、可观测性、错误恢复、多 agent 协调。OpenAI Codex 0 行手写代码实验 + Anthropic 长任务 Agent 经验 + Manus 5 次重写教训。核心工程模式：检查点机制 / Artifact 传递 / 可替换智能层 / Harness-Failure Pattern 闭环。"
type: synthesis
domain: ai/agent/fundamentals
created: 2026-02-28
updated: 2026-02-28
tags:
  - agent
  - harness-engineering
  - infrastructure
  - agent-deployment
  - long-running-agents
  - engineering
  - production
rating: ★★★★☆
sources:
  - "OpenAI Harness Engineering: openai.com/index/harness-engineering/ (Ryan Lopopolo, 2026-02)"
  - "Anthropic: anthropic.com/engineering/effective-harnesses-for-long-running-agents"
  - "Phil Schmid (HuggingFace): philschmid.de/agent-harness-2026 (2026-01-05)"
  - "Aakash Gupta: 2025 Was Agents, 2026 Is Agent Harnesses (2026-01-08)"
  - "Parallel.ai: What is an agent harness in the context of LLMs"
related:
  - "[[AI/2-Agent/Agentic-RL/Environment-Evolution-Agent-Training-Taxonomy|环境进化与 Agent 训练谱系]]"
  - "[[AI/2-Agent/Agentic-RL/Agent-RL-环境工程系统论|Agent RL 环境工程系统论]]"
  - "[[AI/2-Agent/Agentic-RL/Agent-进化模式谱系|Agent 进化模式谱系]]"
  - "[[AI/2-Agent/Multi-Agent/FlexMARL-Rollout-Training-CoDesign-MARL-System|FlexMARL（训练基础设施）]]"
---

# Agent Harness Engineering：让 Agent 稳定工作的基础设施工程

> 重要区分：本文讲的 "Harness Engineering" 是**运行时基础设施**（让已训练好的 Agent 稳定工作），不是训练时的环境工程。关于训练时如何用环境帮助 Agent 进化，见 [[AI/2-Agent/Agentic-RL/Environment-Evolution-Agent-Training-Taxonomy|环境进化与 Agent 训练谱系]]。

---

## 一、为什么 2026 年 Harness Engineering 突然重要

### 竞争格局的转移

```
2024 年竞争焦点：谁的模型更强（GPT-4 vs Claude vs Gemini）
2025 年竞争焦点：谁的 Agent 能跑通任务（各种 Agent 框架）
2026 年竞争焦点：谁的 Harness 更可靠（护城河正在迁移）
```

核心观察（Phil Schmid, HuggingFace, 2026-01）：
> "Capabilities that required complex, hand-coded pipelines in 2024 are now handled by a single context-window prompt in 2026. Developers must build harnesses that allow them to rip out the 'smart' logic they wrote yesterday."

模型能力已经收敛——GPT-4o、Claude Sonnet、Gemini Pro 在大多数任务上差异不大。**新护城河：Harness 质量**。建一个可靠的 Harness 需要数千工时（Manus 花了六个月五次重写）。

### OpenAI Codex 的实验（2026-02）

Ryan Lopopolo 团队的结论：
- 用 Codex Agent 构建并部署了一个**百万行生产系统，0 行手写代码**
- 不是 Agent 能力突然变强——而是 Harness 系统提供了足够可靠的执行环境
- 关键要素：可观测性系统 + 架构约束文档 + 结构化文档传递机制

### Anthropic 的长任务 Agent 经验

Anthropic 工程博客的核心发现：
> "A very capable coding model would fail to build a large app without an external system to initialize the project, incrementally track progress, and leave behind artifacts (like a progress log or updated code) for the next session."

即使是最强的 coding model，在没有 Harness 支撑的情况下，长任务就会失败。**不是能力问题，是基础设施问题。**

---

## 二、Agent Harness 的定义

### 精确定义

**Agent Harness = 包围 Agent 的外部基础设施系统**

不是 Agent 本身（不是 LLM + prompt），而是让 Agent 能稳定运行、持续工作、出错恢复的所有外部系统的总和。

类比：Harness 对 Agent 的关系，就像 Operating System 对 Application 的关系——应用程序有 bug 不会导致 OS 崩溃，OS 提供了隔离、资源管理、错误恢复的基础。

### Harness 的六大组件

**1. 状态持久化（State Persistence）**
- 功能：在 Agent session 之间保存任务状态，不从零重新开始
- 实现：任务状态数据库 / 检查点（checkpoint）机制
- 缺失会怎样：Agent 每次启动都"失忆"，长任务永远无法完成

**2. 进度追踪与检查点（Progress Tracking & Checkpointing）**
- 功能：知道任务推进到哪里，失败后从断点恢复，不是从头重跑
- 实现：子任务拆解 + 每个子任务的完成状态记录
- 关键设计：检查点粒度——太粗（一个大 checkpoint 失败全丢）/ 太细（overhead 大）
- Anthropic 的实践：progress log + updated code artifacts，每个 session 结束留下明确的"当前进度"

**3. Artifact 传递机制（Artifact Handoff）**
- 功能：一个 Agent session 的输出（代码、文件、中间结果）能可靠传递给下一个 session
- 实现：标准化的 artifact schema + 版本控制
- 关键：Artifact 不只是"文件"，还包括上下文信息（为什么做了这个决策、下一步是什么）

**4. 工具调用协议与错误恢复（Tool Protocol & Error Recovery）**
- 功能：工具调用失败（超时 / 返回错误 / 格式不对）时，Harness 决定重试 / 降级 / 报警
- 实现：retry 策略 + fallback 工具 + 错误分类（transient vs permanent）
- 不是让 Agent 自己处理——Agent 不擅长工程级错误处理，Harness 用确定性代码处理

**5. 可观测性（Observability）**
- 功能：知道 Agent 在做什么，哪步慢，哪步出错，为什么失败
- 实现：每个工具调用 log + 每个决策 trace + 异常告警
- OpenAI 的经验：可观测性是 Codex 实验成功的核心基础设施之一
- 没有可观测性 = 黑盒 Agent，生产环境不可接受

**6. 架构约束文档（Architectural Guardrails）**
- 功能：给 Agent 提供系统的高层约束——什么可以改，什么不能动
- 实现：结构化的 architectural decision records（ADR）+ 当前系统状态描述
- OpenAI 实践：Codex 实验中，架构约束文档是保证百万行代码系统一致性的关键

---

## 三、Harness 的核心工程模式

### 模式 1：可替换智能层（Swappable Intelligence Layer）

Phil Schmid 提出的关键设计原则：

> "Developers must build harnesses that allow them to rip out the 'smart' logic they wrote yesterday."

2024 年写的"智能"代码（复杂 prompt chain / few-shot logic / 手写 CoT）在 2026 年已经被模型直接做到了。好的 Harness 把智能层和基础设施层分离：

```
基础设施层（Harness）：
  - 任务调度
  - 状态管理
  - 工具接口
  - 错误处理
  - 可观测性
  ↕ 清晰接口
智能层（Agent / LLM）：
  - 决策逻辑
  - 任务分解
  - 工具调用策略
  （随时可以换掉，不影响基础设施层）
```

**关键原则**：今天 Claude Sonnet 4 用的 prompt 逻辑，明天 Claude Sonnet 5 发布后应该能无缝替换，不需要重写 Harness。

### 模式 2：失败模式分类与处理策略

不是所有失败都一样。Harness 需要区分：

```
A. Transient failures（瞬态失败）
   - 网络超时 / API 限速 / 临时服务不可用
   - 策略：指数退避重试（不需要 Agent 参与）

B. Recoverable failures（可恢复失败）
   - 工具返回错误 / 格式不对 / 上下文丢失
   - 策略：Harness 重组上下文 + 重试（可能需要告知 Agent）

C. Permanent failures（永久失败）
   - 任务本身不可完成 / 依赖的资源不存在
   - 策略：记录 + 人工 escalation（不能让 Agent 无限重试）

D. Agent confusion（Agent 混乱）
   - Agent 陷入循环 / 开始输出无意义内容
   - 策略：Harness 检测（循环检测 / token 质量监控）+ 强制中断 + checkpoint 回滚
```

### 模式 3：多 Agent 协调（Multi-Agent Orchestration）

当任务被拆分给多个 Agent 并行执行时，Harness 负责协调：

- **依赖管理**：Agent B 依赖 Agent A 的输出，Harness 管理执行顺序
- **冲突检测**：两个 Agent 修改同一个文件时，Harness 决定合并策略
- **进度汇总**：Harness 合并各 Agent 的 artifact，维护整体进度视图
- **失败隔离**：一个 Agent 失败不影响其他 Agent 的工作（Harness 隔离状态）

Anthropic 的观察：即使用相同的 system prompt 和工具，多个并发运行的 Agent 实例（不同的 user prompt）在 Harness 支持下可以协作完成单个 Agent 无法完成的大型任务。

---

## 四、Harness 的竞争格局

### 为什么 Harness 是护城河

| 维度 | 说明 |
|------|------|
| 工程壁垒 | 可靠 Harness 需要数千工时（Manus：6 个月，5 次重写）|
| 数据飞轮 | Harness 收集 failure pattern → 反哺 Agent 改进 → 更少 failure |
| 领域积累 | 企业级 Harness 包含大量领域特定的 guardrail 和 workflow 知识 |
| 可观测性护城河 | 有了观测数据，才能做持续改进（无数据 = 无改进路径）|

### 当前主要玩家的 Harness 策略

**OpenAI Codex**：
- 实验证明了"0 行手写代码"的可能性
- 核心 harness 组件：observability + architectural constraints + structured documentation
- 方向：把 Harness 能力内化到 Codex agent 本身

**Anthropic Claude Code**：
- 工程博客明确指出"effective harnesses for long-running agents"是 2026 年核心挑战
- 正在系统化总结长任务 agent 的 Harness 设计模式
- 方向：发布 best practices 文档，推动社区标准化

**Manus**：
- 5 次重写的教训：每次都因为 Harness 基础设施的局限而重写，不是因为 Agent 本身
- 现在的 Manus Harness：沙箱隔离 + 状态持久化 + 多 agent 协调

**框架层（LangGraph / LlamaIndex / AutoGen）**：
- 都在往 Harness 方向演进
- LangGraph：stateful graph，内置检查点
- AutoGen：multi-agent conversation harness
- 但都没有企业级生产稳定性

---

## 五、Harness 的工程实践要点

### 检查点设计原则

```python
# 好的检查点设计
class TaskCheckpoint:
    task_id: str
    subtask_completed: List[str]        # 已完成的子任务
    current_subtask: str                # 当前在做什么
    artifacts: Dict[str, ArtifactRef]  # 产出物的引用（不是内容，是指针）
    context_summary: str                # 为什么做了这些决策（给下一个 session 的 Agent）
    next_steps: List[str]               # 明确的下一步（减少下一个 session 的理解成本）
    created_at: datetime
    agent_id: str                       # 哪个 Agent session 创建的

# 关键：context_summary + next_steps 是给 Agent 的，不只是给人类的
```

### 可观测性的最小必要集合

```
必须收集：
  - 每次工具调用：输入/输出/耗时/成功失败
  - 每次 Agent 决策：reasoning trace（如果模型提供）
  - 异常事件：工具失败/重试/中断
  - 进度里程碑：关键子任务完成

可选（资源充足时）：
  - token 使用量（成本追踪）
  - Agent 输出质量指标（自动评估）
  - 环境状态快照（debug 用）
```

### Harness 与 Agent 的接口契约

Harness 和 Agent 之间需要清晰的接口，双方约定：

**Harness 提供给 Agent**：
- 当前任务状态（检查点恢复的上下文）
- 可用工具列表和调用方式
- 约束文档（什么不能做）
- 当前 artifact 引用

**Agent 返回给 Harness**：
- 工具调用请求（结构化，不是自然语言）
- 检查点更新请求（"我完成了 subtask X，产出了 artifact Y"）
- 任务状态更新（继续 / 完成 / 需要人工介入）

---

## 六、Harness Engineering 与 RL 训练的数据飞轮

Harness 不只是运行时基础设施，它还是 Agent 能力持续进化的**数据来源**。

生产 Harness 收集的数据：
- Agent 在哪类任务上失败最多？
- 哪些工具调用最容易出错？
- 哪些子任务耗时最长（Agent 最不确定）？
- 哪些 failure mode 需要人工介入？

这些数据直接告诉你：
- 哪些能力需要在训练时重点加强（→ 环境设计重点）
- 哪些工具使用模式需要更多示例（→ SFT 数据方向）
- 哪些场景需要更密集的 RL 训练（→ curriculum 设计）

**理论上的最优系统**：
```
生产 Harness 收集 failure pattern
  ↓
自动生成针对弱点的训练环境（GenEnv 类框架）
  ↓
RL 训练强化弱项
  ↓
新版 Agent 部署
  ↓
生产 Harness 继续收集...（飞轮闭环）
```

这个闭环在 2026 年仍然没有公开的完整实现——有人在做的方向，但还没有发出来。

---

## 七、当前挑战与开放问题

1. **标准化缺失**：各家 Harness 都是私有实现，MCP（Model Context Protocol）是往标准化方向的努力，但还没有统一 Harness 层的标准
2. **Harness 自身的可测试性**：Harness 有 bug 会导致 Agent 行为异常，但 Harness 的测试比应用代码更难（需要模拟各种 Agent 行为和失败模式）
3. **安全边界**：Agent 有权限写文件/调 API/执行代码时，Harness 如何防止越权？（沙箱隔离是当前主流解法，但对合法的长任务有性能影响）
4. **人在循环（Human-in-the-Loop）的触发时机**：什么时候该打断 Agent 让人类决策？太早 = 失去自动化价值；太晚 = 已经造成不可逆损坏

---

## 八、So What

**最重要的认知更新**：
- Agent 失败的常见原因不是"模型不够强"，而是"Harness 不够好"——这是 Anthropic 工程博客的直接结论
- 2026 年的 AI 工程竞争，核心战场在 Harness 质量，而不是模型 benchmark 分数
- Harness 和 RL 训练不是两个独立的系统——生产 Harness 应该是 RL 训练的数据入口

**面试场景**：被问到"如何让 Agent 完成长任务"，当前最好的回答：

> 关键不是 prompt 技巧，而是 Harness 设计。需要：① 检查点机制（任务分解 + 每步状态持久化，失败从断点恢复）；② Artifact 传递（每个 session 留下明确的进度和下一步指引）；③ 可观测性（每个工具调用 log，异常告警）；④ 错误分类（区分 transient/recoverable/permanent failure，不同处理策略）。Anthropic 和 OpenAI 都已验证：这四点比模型能力更决定成败。

---

## 推荐阅读

1. OpenAI Harness Engineering：openai.com/index/harness-engineering/（Ryan Lopopolo，2026-02）
2. Anthropic：anthropic.com/engineering/effective-harnesses-for-long-running-agents
3. Phil Schmid：philschmid.de/agent-harness-2026（HuggingFace，2026-01）

---

## See Also

- [[AI/2-Agent/Agentic-RL/Environment-Evolution-Agent-Training-Taxonomy|环境进化与 Agent 训练谱系]] — Harness Engineering 是运行时基础设施；训练环境设计是 Layer 1-4 谱系
- [[AI/2-Agent/Agentic-RL/Agent-RL-环境工程系统论|Agent RL 环境工程系统论]] — 训练环境设计的系统论（Harness 是生产运行时那侧）
- [[AI/2-Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent-RL|KLong]] — 极长任务 RL 训练的工程解法（渐进式 timeout），Harness 的检查点机制与之互补
- [[AI/2-Agent/Agentic-RL/Agentic-RL-元问题-瓶颈与突破方向|Agentic RL 元问题]] — Harness 可靠性是 reward signal quality 的工程前提

---

笔记时间：2026-02-28 | Scholar 自写
