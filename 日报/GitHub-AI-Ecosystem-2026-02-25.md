---
title: GitHub AI 生态扫描 2026-02-25（周榜）
brief: 本周 GitHub AI 生态：microsoft/agent-framework 正式发布（多语言多 agent 编排框架，含 RL Labs 模块）；anthropics/claude-code-security-review（AI 安全审查 GitHub Action）；Databricks/ai-dev-kit；Agent Skills for Context Engineering。趋势：Agent 框架进入微软/Anthropic 官方产品层，Context Engineering 成热词，RL 进入 infrastructure 层。
date: 2026-02-25
type: github-scan
tags:
  - github
  - agent-framework
  - microsoft
  - anthropic
  - context-engineering
  - weekly-scan
related:
  - "[[AI/2-Agent/Fundamentals/GitHub-Agentic-Workflows|GitHub Agentic Workflows]]"
  - "[[AI/2-Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent-RL|KLong — context 管理的学术呼应]]"
  - "[[AI/3-LLM/RL/算法/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL（Databricks+Cornell）— Databricks RL 研究端]]"
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL 2026 综合分析]]"
---

> [!warning] 路径偏差
> 本文由 Sentinel 写入 `AI/Agent/` 根目录，属于情报性质。正式 Vault 知识笔记应在对应子目录；情报/扫描类应存 `Newsloom/`。内容已通过 wikilinks 接入知识图谱。

# GitHub AI 生态扫描 — 2026-02-25 周榜

> 上次扫描：2026-02-20。本次覆盖 2/20–2/25 的变化。

---

## 重点项目

### 1. microsoft/agent-framework ⭐⭐⭐⭐

**定性**：微软官方发布的多 agent 框架，取代 Semantic Kernel / AutoGen 的统一继任者。

**核心能力**：
- **Graph-based Workflows**：数据流连接 agent + 确定性函数，支持 streaming / checkpointing / human-in-the-loop / **time-travel**（可回溯）
- **双语言 API**：Python + C#/.NET，企业用户友好
- **AF Labs**：实验性模块，明确包含 **reinforcement learning** 支持——这是与 AutoGen/Semantic Kernel 的关键区别
- **DevUI**：交互式开发调试界面，workflow 可视化

**迁移路径提供**：
- Migration Guide from Semantic Kernel（官方）
- Migration Guide from AutoGen（官方）

**信号解读**：
微软在 agent 框架层面做了整合。AutoGen 是研究项目，Semantic Kernel 是企业 SDK，两者并存导致开发者选择困难。agent-framework 是明确的统一产品层定位，且内置 RL 支持（AF Labs）说明微软认为 agent RL 训练将成为生产工作流的一部分。

**对老板的价值**：了解微软 agent 栈演化，工程侧需要跟踪。

---

### 2. anthropics/claude-code-security-review ⭐⭐⭐

**定性**：Anthropic 出品，Claude-powered 代码安全审查 GitHub Action。3.2k stars，本周 242 增量。

**功能**：
- 自动分析 PR 中的代码变更
- 检测安全漏洞（injection/hardcoded secrets/unsafe deps 等）
- 集成到 CI/CD 流水线

**信号解读**：
Anthropic 在把 Claude 推入 SDLC（软件开发生命周期）。GitHub Agentic Workflows（2/13 tech preview，上次扫描记录）是平台侧，claude-code-security-review 是工具侧。两条线合围：AI 进 SDLC 不是趋势而是已发生的现实。

---

### 3. muratcankoylan/Agent-Skills-for-Context-Engineering ⭐⭐⭐

**定性**：Context Engineering for multi-agent 的工具集。"Context Engineering"这个词在近两周内突然爆发，本项目是其中流量大的代表。

**内容**：
- Agent Skills 集合（multi-agent 架构、生产系统、上下文管理）
- 强调"context management"而非"prompt engineering"

**信号解读**：
"Context Engineering"作为术语正在取代"Prompt Engineering"，重心从"如何写 prompt"转向"如何管理 agent 的 context window 内容"——这与 KLong（长 horizon 任务的 context 管理）、TSR（高质量 rollout）等研究方向高度呼应。

---

### 4. databricks-solutions/ai-dev-kit ⭐⭐

**定性**：Databricks 出品，Field Engineering 提供的 Coding Agent 工具包。

**信号**：Databricks（数据平台）进入 Coding Agent 赛道，说明企业 AI 工具链正在数据+代码双向整合。与 OAPL 论文（Databricks+Cornell）同期——Databricks 在 RL 理论和工程实践都在加码。

---

### 5. huggingface/skills ⭐⭐

**定性**：HuggingFace 出品的 skills 包（本周出现在 trending）。具体内容待查，但 HF 进入 Agent skills 层是明确信号。

---

## 趋势判断（2026-02-25）

### Trend 1：Agent 框架进入巨头产品层

2025 年：LangChain / AutoGen / CrewAI——社区框架时代  
2026 年：microsoft/agent-framework / Anthropic Claude Code / Google Jules——官方产品时代

框架竞争结束，选哪个不再是学术问题，而是看组织已经在用哪家的云/模型。

### Trend 2：Context Engineering 成为新热词

"Prompt Engineering" → "RAG" → "Context Engineering"  
语义演化反映的是工程重心：从单条 prompt 到完整的上下文设计（memory / tools / state management / truncation 策略）。  
与 KLong 的 Trajectory-splitting 方法论高度吻合——长任务 context 管理正在从研究进入工程实践。

### Trend 3：AI 安全工具链爆发

- anthropics/claude-code-security-review（代码安全审查）
- KeygraphHQ/shannon（上次扫描，AI 自主渗透测试）
- monty（Rust Python sandbox，上次扫描）

AI 安全已不再是单独的研究方向，而是进入主流 DevSecOps 工具链。

### Trend 4：RL 进入 Agent 框架 Infrastructure 层

microsoft/agent-framework 的 AF Labs 明确包含 RL 支持。这意味着：
- RL 训练不再只是 AI 研究实验室的工作
- 生产 agent 框架开始内置 RL 能力
- 对老板的直接价值：懂 agent RL 的工程师将有明显稀缺溢价

---

## 对比：2/20 vs 2/25

| 方向 | 2/20 状态 | 2/25 变化 |
|------|---------|---------|
| Agentic Workflows | GitHub tech preview | 微软官方框架正式发布 |
| Coding Agent | Claude Code / Codex | Databricks ai-dev-kit 入场 |
| Context Engineering | 研究词汇 | 成为 GitHub trending 热词 |
| AI Security | 独立研究 | 进入主流 DevSecOps |
| RL Infrastructure | 研究级 | 微软 AF Labs 内置 |
