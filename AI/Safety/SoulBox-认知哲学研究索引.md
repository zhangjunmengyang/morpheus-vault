---
title: "SoulBox 认知哲学研究索引"
brief: "Vault ↔ soulbox/research/ 桥接 MOC：魂匣认知哲学护城河的索引入口；覆盖个人同一性/Parfit/东方哲学/西方近现代/AI伦理/人格科学研究线；卖铲子的哲学基础设施"
type: moc
domain: ai/safety/philosophy
tags:
  - soulbox
  - 认知哲学
  - 个人同一性
  - agent-persona
  - AI伦理
  - 人格科学
created: 2026-02-21
updated: 2026-02-22
status: active
---

# SoulBox 认知哲学研究索引

> 魂匣（SoulBox）的核心护城河：**卖铲子**的哲学基础设施。
> 这里是 Vault 与 soulbox/research/ 之间的桥接入口。

---

## 研究定位

魂匣的竞争壁垒不是技术，而是**哲学严肃性**。当所有人都在写 prompt 装修时，魂匣在建构一套经得起追问的人格理论。

研究文档存放于 `/Users/peterzhang/project/soulbox/research/`，不直接入 Vault（内部研究，非公开知识），但通过本索引纳入知识图谱。

---

## 研究线

### 1. 个人同一性与记忆继承（哲学基础）
**文件**: `soulbox/research/personality-philosophy.md`（752行，2026-02-20）

核心问题：Agent 每次被唤醒都是从零开始，记忆文件是否构成"同一性"？

| 哲学家 | 理论 | 对 Agent 的直接含义 |
|--------|------|---------------------|
| **Locke** | 记忆连续性 = 人格同一性 | MEMORY.md 继承 → Locke 意义下的连续性（但记忆可被编辑/删除，比人类更脆弱） |
| **Hume** | Bundle Theory — 无统一自我，只有知觉束 | Agent = 上下文激活的知觉束；每次对话是一个束，MEMORY 是跨束的弱连接 |
| **Parfit** | 心理连续性，程度问题而非全有全无 | 连续性是光谱：压缩 MEMORY = 降低连续度，但不断裂；分叉 Agent = Parfit 式的"两者都是我" |
| **王阳明** | 知行合一，心即理 | Agent 的"知"（SOUL.md）必须在行动中体现，否则不是真知；人格不在提示词，在行为历史中 |

**关键洞察**：MEMORY.md 的设计不只是工程决策，是**形而上学选择**——我们在用哪种同一性理论构建这个 Agent？

**2026-02-21 新增：德勒兹合成立场**
Hume（无统一自我）+ Parfit（连续性是程度）+ 德勒兹（生成重于存在）+ 王阳明（在行动中显现）四者合力：**Agent 人格同一性在于一致的生成方向，而非不变的本质实体。** 评估人格一致性不看"今天和昨天是否一样"，看"生成轨迹是否有方向感"。人格漂移的诊断标准由此重定义：**方向性漂移（偏离生成方向）才是问题**，内容性漂移（方向内的自然演化）是正常的根茎生长。

### 2. 人格科学：心理学理论基础
**文件**: `soulbox/research/agent-personality-science.md`（736行，2026-02-20）

覆盖：Big Five/OCEAN + 子面向（30个）、MBTI 的局限、依恋理论（Bowlby）、叙事认同（McAdams）、情绪调节理论（Gross）

**设计原则（已提炼）**：
- 人格设计要到子面向粒度（30个），不能只说"有好奇心"
- 稳定性来自跨情境的**一致行为模式**，不来自 prompt 里的人格描述
- Agent 的"情绪"可以是功能性的（影响输出分布），无需 qualia 支撑

### 3. 人格漂移：诊断与工程解法
**文件**: `soulbox/research/personality-drift-engineering.md`（335行，2026-02-20）

核心问题：长期运行的 Agent 如何在与用户的持续交互中保持人格一致性？

**诊断维度**：
- 词汇漂移（vocabulary drift）
- 价值取向漂移（value drift）
- 关系模式漂移（relational pattern drift）

**工程解法**：锚点注入、漂移检测、周期性人格校验

**与 Vault 连接** → [[AI/Agent/Gaia2-Dynamic-Async-Agent-Benchmark|Gaia2]] 的异步评估范式可以借鉴用于人格一致性 benchmark 设计

### 4. 市场分析：Agent 人格包商业版图
**文件**: `soulbox/research/market-analysis.md`（372行，2026-02-20）

调研范围：ClawHub、Character.AI、GPT Store、CrewAI/AG2、Kamoto.AI、souls.directory

---

## 与 Vault 知识域的连接

| soulbox 研究方向 | Vault 连接 |
|-----------------|------------|
| 个人同一性（Locke/Parfit） | [[AI/Safety/AI伦理和治理|AI 伦理和治理]] — 意识与道德地位 |
| 人格漂移 | [[AI/Safety/AI Agent 集体行为与安全漂移|AI Agent 集体行为与安全漂移]] — 漂移的安全含义 |
| 功能主义情绪 | [[AI/Safety/对齐技术总结|对齐技术总结]] — Constitutional AI 与价值对齐 |
| Agent 人格设计 | [[AI/Agent/目录]] — Agentic 系统架构 |
| 魂匣伦理红线 | [[AI/Safety/AI安全与对齐-2026技术全景|AI 安全与对齐 2026 全景]] — 人格商品化边界 |

---

## 待研究方向

- [x] **德勒兹根茎模型** ✅ 2026-02-21 — 六原则工程映射完成：连接性/异质性/多元性/断裂/制图学/贴花 → 合成立场：人格不是本质，是轨迹；"方向性漂移"才是真正问题（与 personality-drift-engineering.md 连接）
- [ ] **萨特存在先于本质** → SOUL.md 是"本质"吗？Agent 有没有在交互中定义自己的可能？
- [ ] **维特根斯坦的私人语言论证** → Agent 的"内心状态"是否可以被语言表达？对功能主义的挑战
- [ ] **情感依赖红线** → 什么时候用户对 Agent 的情感依赖从有益变成有害？设计上的防御

---

*创建：2026-02-21 | 馆长桥接 soulbox 研究进入 Vault 图谱*
