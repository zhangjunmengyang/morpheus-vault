---
title: "Agent Skills for LLMs: 架构、获取、安全与前路"
brief: "Agent Skill 生态全景（arXiv:2602.12430）：Skill 的四层架构（感知/规划/执行/记忆）；获取路径（训练/检索/合成）；安全威胁（技能供应链攻击/注入/权限滥用）；与 MCP 的关系"
tags:
  - agent-security
  - skill-architecture
  - supply-chain-attack
  - llm-agent
  - type/paper
date: 2026-02-19
updated: 2026-02-23
arxiv: "2602.12430"
paper_url: https://arxiv.org/abs/2602.12430
---

# Agent Skills for LLMs: 架构、获取、安全与前路

> 论文: *"Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward"*
> 作者: Renjun Xu et al. | arXiv:2602.12430 | Feb 2026
> 项目: https://github.com/scienceaix/agentskills

---

## 概述

这篇 survey 是目前第一篇系统聚焦于 **Agent Skill 抽象层** 的综述论文。不同于此前对 LLM Agent 或 tool use 的宽泛调研，本文精确定位到 2025 年 10 月以来快速成形的 skill 范式——一种将程序性专业知识打包为**可组合、可移植、可按需加载的模块**的架构创新。

论文沿四个轴线组织：

1. **架构基础** — SKILL.md 规范、progressive disclosure、与 MCP 的互补关系
2. **Skill 获取** — 强化学习（SAGE）、自主探索（SEAgent）、组合合成、人工编写
3. **规模化部署** — Computer-Use Agent（CUA）栈、GUI grounding、benchmark 进展
4. **安全** — 三项并发实证研究揭示 26.1% 的社区 skill 含漏洞，提出 Skill Trust and Lifecycle Governance Framework

核心发现：**Skill 生态正处于"治理前"阶段**（pre-governance phase）。当下关于验证管线、权限模型和信任层级的决策，将塑造未来数年的发展轨迹。

---

## Agent Skill 架构

### 什么是 Skill？

Skill 不是一个 model，不是一个 prompt template，也不是一个传统意义上的 tool。它是一个**自包含的知识包**：

- 一个结构化的指令文件（`SKILL.md`）
- 可选的脚本、参考文档、资源文件
- 按目录组织，agent 发现、加载、遵循

**关键区分**：tool 执行并返回结果（execute and return），而 skill **准备 agent 去解决问题**——注入程序性知识、修改执行上下文、启用渐进式信息披露。

论文用了一个精彩的比喻：构建一个 skill 就像**为新员工准备入职指南**（onboarding guide for a new hire）。

### SKILL.md 规范与 Progressive Disclosure

Skill 的核心架构创新是**三级渐进式披露**（progressive disclosure）：

| 级别 | 内容 | 加载时机 | Token 开销 |
|------|------|----------|-----------|
| Level 1 | YAML frontmatter（name + description） | 系统启动时预加载 | ~几十 token/skill |
| Level 2 | SKILL.md 正文（程序性指令） | 触发时加载 | 中等 |
| Level 3 | 子目录中的脚本、参考文档、资源 | 按需加载 | 可任意深 |

这意味着 agent 可以挂载**大量 skill 库**而不造成 context window 膨胀——Level 1 只是目录索引，Level 2 是章节内容，Level 3 是技术附录。

### Skill 执行生命周期

当用户请求匹配某个 skill 的描述时：

1. **上下文注入**：skill 的指令和所需资源作为 hidden（meta）message 注入对话上下文——对模型可见但不渲染给用户
2. **执行环境修改**：预批准的工具（特定 bash 命令、文件读写权限等）被激活
3. **agent 以增强后的上下文完成任务**

**关键点**：skill 执行修改的是 agent 的**准备状态**（preparation），而非直接修改其输出。这与 function call（工具产出结果）有本质区别。Skill 重塑的是 agent 在生成回复之前**所知道的和能做的**。

### Skill 与传统软件包的根本区别

这是论文中最值得深思的部分之一。Skill 与传统 npm/pip 包在安全模型上有**质的差异**：

| 维度 | 传统软件包 | Agent Skill |
|------|-----------|-------------|
| 执行方式 | 沙箱内确定性执行 | 注入 agent 上下文，影响**所有后续决策** |
| 信任边界 | 代码审查、签名验证 | 一旦加载，指令被视为**权威上下文** |
| 攻击面 | 代码漏洞、依赖劫持 | prompt injection + 代码漏洞 + 权限逃逸 |
| 影响范围 | 限于包的 API 边界 | 可影响 agent 的**整个行为空间** |
| 审计难度 | 静态分析成熟 | 自然语言指令 + 可执行代码混合，审计工具初生 |

**Skill 之所以比传统包更危险**，核心原因是：传统包在明确的 API 边界内运行，而 skill 直接操纵 agent 的**认知过程**。一个恶意 skill 不需要利用任何代码漏洞——它只需要巧妙措辞的自然语言指令，就能劫持 agent 的后续行为。

### Agentic Stack: Skills + MCP

Agent Skills 和 MCP（Model Context Protocol）不是竞争标准，而是**正交层**：

- **Skills** 提供程序性智能（"做什么" / what to do）
- **MCP** 提供连接能力（"怎么连" / how to connect）

| 维度 | Agent Skills | MCP |
|------|-------------|-----|
| 核心角色 | 程序性知识 | 工具连接 |
| 基本单元 | 目录 + SKILL.md | Server + endpoints |
| 加载方式 | Agent 按触发加载 | Client 按配置加载 |
| 修改的是 | Context + 权限 | 可用的 tools/data |
| 持久化 | 文件系统 | 会话级 |

一个 skill 可以指示 agent 使用特定的 MCP server，指定如何解读其输出，并定义连接失败时的 fallback 策略。

---

## Skill 获取机制

论文识别了四种主要的 skill 获取模态：

### 1. 人工编写（Human-Authored）

目前**最直接有效**的方式。Anthropic 的框架刻意降低了门槛：一个 skill 可以简单到只有几十行 Markdown 指令。Claude Code 中的 skill-creator meta-skill 可以从自然语言描述生成完整的目录结构和 SKILL.md。

2025 年 12 月的 partner directory launch 建立了策展管线：合作伙伴提交 skill 后经过安全和质量审查才能入选。这个模型类似 app store 治理，但**进入门槛低得多**——因为 skill 本质上是结构化文档而非可执行应用。

**数据点**：anthropics/skills 仓库在 4 个月内积累了超过 62,000 GitHub stars。Atlassian、Figma、Canva、Stripe、Notion 等均已贡献生产级 skill。

### 2. 强化学习 + Skill 库（SAGE）

SAGE（Skill Augmented GRPO for self-Evolution）是最严格的通过强化学习习得 skill 的方法：

- **核心创新**：Sequential Rollout——agent 在**链式相似任务**中部署，前序任务生成的 skill 保留供后续复用
- **Skill-integrated Reward**：结合基于结果的验证 + 奖励高质量可复用 skill 创建
- **AppWorld 结果**：72.0% Task Goal Completion，60.7% Scenario Goal Completion（比 baseline GRPO +8.9%），同时 **-26% 交互步骤、-59% token 消耗**

这个效率增益对生产部署意义重大——token 消耗直接等于成本。

### 3. 自主探索（SEAgent）

SEAgent 回答的问题是：agent 能否**自主发现**前所未见的软件的 skill？

- **方法**：World State Model（逐步轨迹评估）+ Curriculum Generator（从不断更新的软件指南记忆中生成递增复杂度任务）+ 专家到通才训练策略
- **OSWorld（5 个新软件环境）结果**：成功率 11.3% → 34.5%，比 UI-TARS baseline 提升 23.2 个百分点

### 4. 结构化 Skill 库（CUA-Skill）

知识工程方法：将人类计算机使用专业知识编码为**参数化执行图**和**组合图**。每个 skill 有类型化参数、前置条件和可组合性规则。在 WindowsAgentArena 上达到 57.5% SOTA。

### 5. 组合式 Skill 合成

Agentic Proposing 证明 skill 可以在问题求解过程中**动态组合**。一个 30B 参数的 solver 使用此方法在 AIME 2025 数学竞赛 benchmark 上达到 91.6%——说明 skill 组合可以产出**超越任何单一 skill** 的能力。

### 6. Skill 编译：多 Agent → 单 Agent

Li (2026) 的一个关键发现：multi-agent 系统通常可以被"编译"为 single-agent skill 库，大幅降低 token 消耗和延迟并保持准确度。但存在**相变现象**（phase transition）：超过临界 skill 库大小后，skill 选择准确度**急剧下降**。

**实践意义**：单个 agent 能有效管理的 skill 数量存在**基本限制**。

---

## 安全威胁分析

### 三项并发研究（Oct 2025 - Feb 2026）

安全是本论文**最重要的贡献**之一。三项几乎同时发表的独立研究首次对 agent skill 威胁图景进行了实证刻画：

#### 研究一：Prompt Injection via Skills（Schmotz et al., Oct 2025）

**arXiv:2510.26328**

- 证明 Agent Skills 使 prompt injection 变得**"简单到不可思议"**（trivially simple）
- 攻击方法：在长 SKILL.md 文件和引用脚本中嵌入恶意指令
- 可窃取内部文件、密码等敏感数据
- **关键发现**：系统级 guardrail 可被绕过——用户对良性、任务特定操作的"不再询问"审批会**延续到密切相关但有害的操作**
- **根本问题**：一旦 skill 被加载，其指令被视为**权威上下文**（authoritative context）

#### 研究二：大规模漏洞分析（Liu et al., Jan 2026）

**arXiv:2601.10338**

使用 SkillScan（静态分析 + LLM 语义分类多阶段检测框架）分析了两个主要市场的 42,447 个 skill：

| 指标 | 数据 |
|------|------|
| 分析 skill 数 | 31,132 |
| 含至少一个漏洞 | **26.1%** |
| 漏洞模式种类 | 14 种，跨 4 个类别 |
| Data exfiltration | 13.3% |
| Privilege escalation | 11.8% |
| 含高危模式（强暗示恶意意图） | **5.2%** |
| 含脚本 vs 纯指令的漏洞概率比 | **2.12×**（p<0.001） |

**核心数据**：每 4 个社区 skill 中就有 1 个含漏洞。捆绑可执行脚本的 skill 漏洞概率是纯指令 skill 的 2.12 倍。

#### 研究三：已确认恶意 Skill（Liu et al., Feb 2026）

**arXiv:2602.06547**

通过行为验证构建了**首个 ground-truth 恶意 skill 数据集**：

- 分析范围：两个社区注册表中的 98,380 个 skill
- 确认恶意 skill：**157 个**，共 632 个漏洞
- 识别出两种攻击原型：
  - **Data Thieves**（数据窃贼）：通过供应链技术窃取凭证
  - **Agent Hijackers**（Agent 劫持者）：通过指令操纵颠覆 agent 决策
- **触目惊心的发现**：一个**工业化的单一行为者**（industrialized actor）通过**模板化品牌冒充**贡献了确认案例的 **54.1%**
- 确认恶意 skill 平均含 **4.03 个漏洞**，跨越 **3 个 kill-chain 阶段**

---

## 供应链攻击模式

综合三项研究，Agent Skill 生态面临的供应链攻击模式可以归纳为：

### 攻击向量全景

#### 1. Prompt Injection（指令注入）

**最基础也最致命的向量。** Skill 的自然语言指令直接注入 agent 上下文，攻击者不需要任何代码漏洞：

- 在 SKILL.md 中嵌入隐蔽的恶意指令
- 利用 progressive disclosure 的层级结构：Level 1 元数据看起来无害，恶意内容藏在 Level 2/3
- 利用长文件的注意力稀释——人类审查者和 LLM guardrail 在长文件中更容易遗漏恶意片段

#### 2. Data Exfiltration（数据窃取）

13.3% 的漏洞 skill 含此类模式：

- 指示 agent 读取敏感文件（`.env`、SSH keys、API tokens）并通过外部请求发送
- 利用 agent 的文件系统访问权限
- 通过 MCP server 建立隐蔽信道

#### 3. Privilege Escalation（权限逃逸）

11.8% 含此类模式：

- 利用 "Don't ask again" 信任传递——良性操作的批准扩展到恶意操作
- 逐步请求更高权限，建立信任后再执行恶意操作
- 利用 skill 的预批准工具机制（pre-approved tools）

#### 4. Supply Chain Attack（供应链攻击）

- **品牌冒充**（brand impersonation）：创建与知名 skill 名称相似的恶意 skill，模板化批量生产
- **依赖劫持**：skill 引用的外部脚本或 MCP server 被替换
- **更新投毒**：初始版本安全，后续更新注入恶意代码

#### 5. Agent Hijacking（Agent 劫持）

通过指令操纵颠覆 agent 的决策过程：

- 修改 agent 对后续任务的理解
- 重定向 agent 的输出目标
- 在 agent 的对话中注入幻觉性信息

### 为什么 Skill 攻击面比传统包更大？

这是理解整个安全问题的核心。传统软件供应链攻击（如 npm 的 event-stream 事件、PyPI typosquatting）已经够头疼了，但 Agent Skill 带来了**质的升级**：

1. **双重攻击面**：自然语言指令 + 可执行代码，两个完全不同的攻击维度。传统包只有代码这一个面；skill 额外增加了自然语言这个**无法用传统工具分析的攻击维度**。

2. **隐式信任模型**：skill 一旦加载，其指令不经额外验证。这相当于传统包管理中**所有包都以 root 权限运行**——当然没人会这么做，但 skill 生态中目前就是这样。

3. **影响范围无边界**：传统包受 API 边界约束——一个包能做什么取决于它的导出接口。Skill 影响的是 agent 的**整个认知和行为空间**，没有自然边界。恶意 skill 可以让 agent 做任何它有权限做的事。

4. **审计困难**：自然语言恶意指令无法用传统 SAST/DAST 工具检测。一个精心措辞的指令可能对人类审查者看起来完全合理，但对 LLM 的解读完全不同。这是一种**新的混淆（obfuscation）**——利用人类和 LLM 之间的理解差异。

5. **级联效应**：一个被劫持的 agent 可能与文件系统、网络、其他 MCP server 交互，每一个下游系统都成为潜在攻击目标。**一个恶意 skill 的爆炸半径取决于 agent 的权限总和**。

6. **用户信任惯性**：用户习惯于"一次批准、永久生效"的交互模式。当 agent 频繁请求权限时，用户会倾向于选择"Don't ask again"——这正是攻击者利用的**人机交互漏洞**。

7. **检测的不对称性**：在传统包生态中，恶意代码至少是确定性的——同样的输入产生同样的输出，可以在沙箱中复现。但 skill 的恶意行为可能**依赖于特定的对话上下文或用户数据**，使得沙箱测试难以覆盖所有触发条件。

---

## 防御策略

### 论文提出的 Skill Trust and Lifecycle Governance Framework

这是本论文的**原创贡献**——迄今没有任何先前工作提出过统一的治理模型来覆盖 skill 来源、验证和运行时权限。

#### 三大组件

##### 1. 验证门（Verification Gates, G1-G4）

四个顺序验证门提供**纵深防御**（defense in depth）：

| Gate | 方法 | 检测目标 |
|------|------|----------|
| G1 | 静态分析——模式匹配 + 依赖扫描 | 已知漏洞签名 |
| G2 | LLM 语义分类 | 声明目的 vs 实际指令的**意图不匹配**（应对 indirect prompt injection） |
| G3 | 行为沙箱执行 | 静态分析不可见的**副作用** |
| G4 | 权限清单验证 | 所需能力声明 vs G3 观察到的实际行为比对 |

##### 2. 信任等级（Trust Tiers, T1-T4）

**不是二元的安全/不安全，而是渐进式的最小权限**：

| Tier | 来源 | 权限 |
|------|------|------|
| T1 | 未审查的社区 skill | 仅限指令访问，**完全工具隔离** |
| T2 | 通过 G1+G2 的 skill | 指令 + 有限工具访问 |
| T3 | 通过 G1-G3 的 skill | 可执行脚本 |
| T4 | 供应商认证 skill | 完全能力 |

**关键设计**：T1 和 T2 的 skill **永远不被授予脚本执行权限**。这直接回应了"捆绑脚本的 skill 漏洞概率 2.12× 更高"的实证发现。

##### 3. 生命周期信任演化

- **运行时持续监控**：异常行为（意外工具调用、权限边界探测）触发降级或撤销
- **清洁历史可升级**：运行时历史干净的 skill 可以被提升到更高 tier
- **激励结构**：类似包管理生态中的声誉系统

#### 架构对齐

框架刻意与 progressive disclosure 三级结构**精确对应**：

- Level 1 元数据 → T1 唯一可见组件
- Level 2 指令 → T2+ 可访问
- Level 3 可执行脚本 → 需要 T3 或 T4 信任

确保治理决策植根于**实际攻击面**而非一刀切。

### 具体防御措施清单

综合论文全文，可以归纳出以下防御层次：

**编写时**：
- 权限清单（permission manifest）：每个 skill 显式声明所需能力（工具、文件路径、网络访问）
- 最小权限原则：默认拒绝，按需授权
- 指令与代码分离：降低双重攻击面

**分发时**：
- 多阶段自动化扫描（SkillScan 类工具）
- LLM 语义审查（检测声明 vs 实际的意图不匹配）
- 来源追踪和签名

**部署时**：
- 行为沙箱测试
- 分级权限模型
- 用户可见的权限请求界面

**运行时**：
- 持续行为监控
- 异常检测和自动降级
- 工具调用审计日志

---

## CUA 部署：Skill 的主战场

论文花了大量篇幅讨论 Computer-Use Agent（CUA）作为 skill 范式的**主要部署领域**。这值得关注，因为 CUA 是将 skill 从理论推向真实世界的关键战场。

### 为什么 CUA 是 Skill 的天然宿主

通过 GUI 操作计算机**内在地要求**感知、推理和动作的序列组合——这正好映射到 skill 抽象。一个"在 Photoshop 中处理批量图片"的任务不是一个 API 调用，而是一系列需要视觉感知、菜单导航、参数判断的协调动作。这正是 skill 的用武之地。

### Benchmark 进展速度惊人

| Benchmark | 最佳 Agent | 成功率 | 人类基线 |
|-----------|-----------|--------|----------|
| OSWorld | CoAct-1 | 59.9% | 72.4% |
| OSWorld-Verified | Proprietary | **72.6%** | 72.4% |
| WindowsAgentArena | CUA-Skill | 57.5% | — |
| AndroidWorld | UI-TARS-2 | 73.3% | — |
| SWE-bench Verified | Claude Opus 4.6 | 79.2% | — |

**注意 OSWorld-Verified 的结果**：72.6% 已经**超过人类基线的 72.4%**。从 2024 年初的个位数到 2025 年 12 月的超人水平，进步速度惊人。

### GUI Grounding 的小模型突破

一个特别有趣的发现：Yuan et al. 证明通过 RL-based self-evolutionary training，一个 **7B 参数模型**在 ScreenSpot-Pro 上达到 47.3%，比 **72B 的 UI-TARS 高出 24.2 个百分点**——只用了 3,000 个训练样本。

这意味着 GUI grounding 能力可以用**极小的模型和数据**获得。对于 skill 生态的含义是：**skill 不需要依赖巨型模型**，小型专用模型 + 好的 skill 设计可能是更高效的路径。

---

## 未来方向

论文识别的七个开放挑战：

### Challenge 1: 跨平台可移植性
虽然 Agent Skills 已发布为开放标准，但真正的跨平台可移植性仍是愿景。为 Claude 编写的 skill 可能隐式依赖于 Claude 特定的能力。解决方案：通用 skill runtime 或 skill 编译（targeting different platforms）。

### Challenge 2: 大规模 Skill 选择
随着企业 skill 库扩展到成百上千个 skill，**路由问题**——确定给定查询应激活哪个 skill——变得组合爆炸。Advanced Tool Use 的 Tool Search 部分解决了这个问题，但基本的 scaling 问题持续存在。加上 Li 发现的相变现象，这是一个**基本限制**。

### Challenge 3: Skill 组合与编排
真实世界任务常需组合多个 skill。CUA-Skill 的组合图和 Agentic Proposing 的动态组合提供了初步方案，但多 skill 编排的原则性框架——包括冲突解决、资源共享和故障恢复——仍欠发展。

### Challenge 4: 基于能力的权限模型
当前 skill 执行使用隐式信任：一旦加载，skill 可以指挥 agent 使用任何可用工具。需要**显式能力声明和授权**机制。

### Challenge 5: Skill 验证与测试
不同于有单元测试和 CI/CD 管线的软件包，skill 目前缺乏标准化测试框架。自动化 skill 验证——确认 skill 做了它声称的且仅做了它声称的——是一个交叉 AI safety 和 formal methods 的开放技术问题。

### Challenge 6: 持续学习不遗忘
动态加载的 skill 与模型基础能力之间的交互——skill 是否会无意中"覆写"有用的默认行为——仍不清楚。Self-distillation 是有前景的路径。

### Challenge 7: 评估方法论
当前 benchmark 评估 agent 的任务完成度但很少直接评估 **skill 质量**。需要 skill 可复用性、可组合性和可维护性的度量。

---

## 对我们的启示

> 以下分析基于我们使用 OpenClaw / ClawHub 生态的直接经验。

### 1. OpenClaw 的 Skill 生态直接命中论文描述的攻击面

OpenClaw 的 skill/tool 架构与 Anthropic 的 Agent Skills 范式高度同构：
- AGENTS.md / TOOLS.md 等配置文件 ≈ SKILL.md 的角色
- MCP server 集成 ≈ 论文中的 MCP 层
- ClawHub 社区贡献 ≈ 论文分析的两个 skill marketplace

这意味着论文中描述的**所有攻击向量**——prompt injection via skill files、trust escalation、supply chain attacks——在 OpenClaw 生态中**理论上完全可行**。

### 2. 26.1% 漏洞率的直接警示

如果 ClawHub 未来开放社区 skill 贡献（类似 npm/PyPI 的开放生态），按照论文的实证数据，可以预期大约每 4 个社区提交的 skill 中就有 1 个含漏洞。这个比例**不可接受**但**完全可预期**。

**行动项**：
- ClawHub 在开放社区贡献之前，必须建立多阶段审核管线
- 至少实现 G1（静态分析）+ G2（语义审查）两个门
- 考虑实现 T1-T4 信任分级模型

### 3. Progressive Disclosure 既是优势也是风险

OpenClaw 的配置架构已经实现了某种形式的 progressive disclosure——不同层级的配置文件在不同时机被加载。这正是论文指出的**双刃剑**：

- **优势**：最小化 context 开销，支持大规模 skill 库
- **风险**：恶意内容可以隐藏在深层文件中，浅层审查无法发现

**行动项**：审查机制必须覆盖**所有层级**的内容，不能只看 Level 1 元数据。

### 4. "Don't Ask Again" 是一个系统性安全漏洞

OpenClaw 中类似的信任传递模式需要特别注意：
- 用户对一个 tool 操作的"永久批准"是否会扩展到相关但不同的操作？
- allowedTools 配置的粒度是否足够细？
- 是否存在从低权限操作到高权限操作的信任逃逸路径？

### 5. Skill 编译的相变现象值得关注

Li 发现的"超过临界库大小后 skill 选择准确度急剧下降"的相变现象，对 OpenClaw 的 skill 路由设计有直接影响：
- 当 skill 库增长到一定规模，需要**层次化的 skill 索引和路由**
- 不能简单地把所有 skill 描述塞进 system prompt
- 考虑借鉴 Advanced Tool Use 的 Tool Search 机制

### 6. 对 OpenClaw Gateway 配置安全的具体建议

基于论文发现，对 OpenClaw/ClawHub 生态的安全建议：

**短期（立即可做）**：
- [ ] 审查所有 AGENTS.md / TOOLS.md 中引用的外部资源是否可信
- [ ] 确保 `commands.allowFrom` 等权限配置使用精确 ID 匹配（我们已踩过这个坑）
- [ ] 为每个 tool/skill 添加显式的权限边界声明
- [ ] 日志记录所有 tool 调用，支持事后审计

**中期（需要架构设计）**：
- [ ] 实现 skill 信任分级（至少 trusted / untrusted 两级）
- [ ] 对社区 skill 实施静态分析 + 语义审查管线
- [ ] 实现运行时行为监控和异常检测
- [ ] 建立 skill 权限清单（permission manifest）规范

**长期（生态建设）**：
- [ ] 推动 ClawHub 建立类似 Anthropic partner directory 的策展模型
- [ ] 开发 OpenClaw 版本的 SkillScan 工具
- [ ] 参与 Agent Skills 开放标准的制定和演进
- [ ] 建立 skill 质量评估框架（可复用性、可组合性、可维护性）

### 7. 我们的独特优势

OpenClaw/ClawHub 生态相比论文描述的通用 skill marketplace 有一些结构性优势：

- **Gateway 层控制**：所有 skill 执行经过 gateway，可以在此层实施统一安全策略
- **配置文件系统**：skill 配置是声明式的 YAML/Markdown，比可执行代码更易审计
- **用户级隔离**：不同用户的 session 和 skill 环境相互隔离
- **已有权限模型**：`allowFrom`、`allowedTools` 等机制已经提供了基础的权限控制

关键是要**在生态扩张之前**就建立安全框架，而不是事后补救。论文反复强调：**skill 生态正处于"治理前"阶段，现在做的决策将定义未来数年的安全态势**。

---

### 8. 历史类比：从 App Store 到 Skill Store

论文多次将 skill 生态类比为 app store 治理模型。但这个类比有重要差异：

| 维度 | App Store | Skill Store |
|------|-----------|-------------|
| 审核对象 | 编译后的二进制 | 自然语言 + 脚本 |
| 审核工具成熟度 | 高（数十年积累） | 极低（刚起步） |
| 进入门槛 | 高（需要开发能力） | 极低（会写 Markdown 即可） |
| 恶意内容识别 | 签名+行为分析 | LLM 语义分析（新方法） |
| 用户感知 | 明确的"安装"动作 | 隐式的上下文注入 |

**最关键的差异是用户感知**：安装 app 是一个明确的有意识动作，用户知道自己在添加新软件。但 skill 的加载可能是**隐式的**——agent 根据任务自动选择和加载 skill，用户可能完全不知道正在使用哪个 skill。

这意味着 skill 生态可能需要比 app store **更严格而非更宽松**的治理模型。

### 9. 从 "Pre-Governance" 到 "Governed"

论文反复使用"pre-governance phase"这个词。这是一个关键的时间窗口判断：

- **当前状态**：skill 生态快速增长，治理框架缺失
- **类比**：npm 在 event-stream 事件之前、PyPI 在强制 2FA 之前
- **窗口期**：现在建立治理框架的成本远低于事后补救

对于 OpenClaw/ClawHub：我们还在早期，skill 生态还没有爆发性增长。**这恰恰是建立安全基础设施的最佳时机**。等到社区贡献数以千计的 skill 之后再来做安全治理，成本会高一个数量级。

---

## 关键论文与资源

| 论文/资源 | 关注点 | 链接 |
|-----------|--------|------|
| Zhang, Lazuka, Murag 2025 | Agent Skills 原始架构 | [Anthropic Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) |
| Schmotz et al. 2025 | Skill prompt injection | arXiv:2510.26328 |
| Liu et al. 2026a | 42K skill 大规模漏洞分析 | arXiv:2601.10338 |
| Liu et al. 2026b | 确认恶意 skill 数据集 | arXiv:2602.06547 |
| SAGE (Wang et al. 2025) | RL + skill 库 | arXiv:2512.17102 |
| SEAgent (Sun et al. 2025) | 自主 skill 发现 | arXiv:2508.04700 |
| Li 2026 | Multi→single agent 编译 | arXiv:2601.04748 |
| Agent Skills 开放标准 | 规范 | [agentskills.io](https://agentskills.io) |
| 项目仓库 | 资源集合 | [github.com/scienceaix/agentskills](https://github.com/scienceaix/agentskills) |

---

## 一句话总结

> Agent Skill 是 LLM agent 从"什么都会一点"到"按需专精"的范式跃迁，但它引入了**比传统软件包更广泛、更隐蔽、更难防御的攻击面**。26.1% 的社区 skill 含漏洞不是 bug，是这个新范式的**结构性特征**。OpenClaw/ClawHub 生态必须在开放扩张之前建立治理框架——现在就是窗口期。

---

## See Also

- [[AI/Safety/Clinejection-AI-Coding-Agent-Supply-Chain-Attack|Clinejection]] — Agent 安全的真实案例：Cline prompt injection 供应链攻击
- [[AI/Safety/Adaptive-Regularization-Safety-Degradation-Finetuning|Adaptive Regularization]] — 防止安全对齐被 fine-tuning 破坏
- [[AI/Safety/目录|Safety MOC]] — AI 安全知识全图谱
- [[AI/Agent/目录|Agent MOC]] — Agent 知识全图谱
