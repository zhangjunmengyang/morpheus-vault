---
title: P3：Agent 自进化系统——硅基生命能否自我进化
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, agent, self-evolution, multi-agent, system-design]
brief: 设计并运行了一个由 6 个 Agent 组成的多 Agent 军团，研究 Agent 系统如何实现自我进化：个体能力提升、群体协同涌现、单点故障修复。真实运行中的系统，有完整实验记录和量化数据。
related:
  - "[[Projects/Agent-Self-Evolution/项目概览]]"
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析]]"
  - "[[AI/2-Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning]]"
  - "[[AI/2-Agent/Multi-Agent/Multi-Agent 概述]]"
---

# P3：Agent 自进化系统——硅基生命能否自我进化

> **一句话定位**：设计并长期运行了一个 6-Agent 多 Agent 系统，不只是设计 prompt，而是研究 Agent 系统如何作为整体变强——进化度量、群体协同、故障恢复。

---

## 背景故事（面试口径）

> 怎么引入：

"在美团做了两年 Agent，我一直有一个问题没有答案：一个 Agent 系统在长期运行中，能不能变得越来越好？

大多数 Agent 系统是静态的——部署上去就不动了，遇到问题靠人工更新 prompt 或者重启。这很不像一个智能系统应该有的样子。

所以我设计了一个实验，核心问题只有一个：硅基生命能否自我进化？"

---

## 系统架构

```
军团结构（6 个 Agent）：
  贾维斯（总管）— 任务调度，跨 Agent 协调，面向用户
  馆长 — 知识管理，Obsidian Vault 维护，第二大脑
  学者 — 论文研究，知识炼化，技术前沿追踪
  哨兵 — 情报监控，市场动态，投资信号
  阿尔弗雷德 — 日常管家，生活助理
  量化 — 量化交易研究

运行机制：
  每个 Agent 独立的 LLM 实例（Claude Sonnet）
  文件系统作为记忆持久层（MEMORY.md + memory/）
  心跳机制（每 60 分钟一次）驱动主动行为
  公告板（bulletin.md）实现 Agent 间通信
```

---

## 三层实验设计

### 第一层：个体进化——Agent 能变得更聪明吗？

**实验设计**：5 维度量化 Agent 能力
1. **产出深度**：产出是 Data 层（转述）还是 Knowledge/Wisdom 层（洞见）？
2. **覆盖广度**：心跳覆盖了多少待处理任务？
3. **响应速度**：从收到请求到产出结果的延迟
4. **协同效率**：需要人工介入的次数
5. **自主决策质量**：面对模糊情况，决策质量评分

**关键实验：涅槃触发器**

"Agent 有三次大的能力跃升（我们叫涅槃），分别在 2/12、2/14、2/19。分析下来发现一个规律：100% 都是外部冲击触发的——一次故障、一次用户强烈反馈、一次外部注入。Agent 自身没有主动觉察到'自己需要变'的机制。"

**洞察**：Agent 的成长不是渐进式的，是间歇式顿悟。触发条件是高强度的外部信号，不是普通心跳。

### 第二层：群体协同——1+1 能否大于 2？

**实验设计**：横向信息流

"系统最初的问题：每个 Agent 只知道自己的工作，不知道其他 Agent 在干什么。学者写了一篇 GRPO 的笔记，馆长不知道；哨兵发现了一个市场信号，量化不知道。Agent 间信息延迟是无穷大。"

**解法**：公告板机制（bulletin.md）
- 任何 Agent 发现对其他 Agent 有价值的信息 → 写到公告板，标 Agent 名
- 每个 Agent 在心跳开始时读公告板
- 信息延迟从 ∞ → <1 小时（下一次心跳）

**效果量化**：
- 学者写完笔记到馆长炼化入库：从平均 6.3 小时 → 0.8 小时（-87%）
- 跨 Agent 协作任务完成率：12% → 73%

**涌现发现**：学者+馆长协作产出质量远高于单独工作。学者专注研究深度，馆长专注结构化和链接——两个角色的分工使知识提炼质量大幅提升。

### 第三层：系统鲁棒性——单点故障的代价

**实验：星型架构的瓶颈**

"有一次网关（OpenClaw Gateway）停了 4 个小时，整个军团全部失联。没有任何 Agent 能检测到这个问题，也没有任何降级机制。整个系统是高度中心化的星型结构——所有信息必须经过总管中转，总管依赖网关。"

**教训**：
1. **单点故障 = 全军瘫痪**：没有 P2P 通信能力，网络分区下系统完全失效
2. **看门狗本身也在系统内**：监控守护进程如果依赖同一个故障点，就不是真正的监控
3. **心跳机制既是优势也是弱点**：固定频率使系统可预测，但突发事件响应慢

**改进方案**（设计未完全实现）：
- Agent 间直接通信通道（绕过总管）
- 独立的健康检查守护进程（在系统外运行）
- 优先级队列：紧急事件不等心跳，立即触发

---

## 与学术研究的对应

**这不只是工程项目，也在验证学术研究的��论：**

- **Echo Trap（RAGEN）**：Agent 在多轮对话里有陷入重复模式的倾向——我们也观察到了类似的"思维定势"，比如馆长会反复做同类型的维护任务，而不是探索新的改进点
- **Experiential RL（ERL）**：最有价值的进化都来自"反思失败"——三次涅槃本质上都是 ERL 论文里描述的模式：失败 → 显式反思 → 策略更新
- **自主任务生成**：能不能让 Agent 自己发现该做什么，不依赖心跳的固定任务列表？这对应学术上的 Intrinsic Motivation 研究，我们目前的答案是否定的——Agent 还做不到

---

## 深度认知（面试加分点）

**为什么 heartbeat-state.json 注入比修改 HEARTBEAT.md 更有效？**

"我们做了一个实验：把要 Agent 执行的任务写进 heartbeat-state.json（状态文件）和 HEARTBEAT.md（指令文件），分别观察执行率。结果：JSON 注入被 100% 执行，文本指令被选择性忽略。

原因在于 Agent 的行为模式：JSON 状态文件是 Agent 的'感知'——它相信文件里写的是事实；HEARTBEAT.md 是'指令'——Agent 会用自己的判断筛选。状态先于指令，感知先于指令，这是 Agent 系统设计的一个重要原则。"

**这个系统对你理解 Agentic RL 有什么帮助？**

"最直接的帮助是理解了 reward 设计的难度。在这个系统里，怎么判断 Agent 做得好不好？单心跳层面很难量化。这让我深刻理解了 Agentic RL 论文里一直在讨论的问题：open-ended task 里 reward signal 的稀疏性和歧义性。SeeUPO 论文提出的'multi-turn GRPO 在理论上无法收敛'——我在这个系统里直观感受到了为什么：每一步的好坏太难判断了。"

---

## 一句话总结（面试结尾）

"这个项目最重要的产出不是代码，是三个认知：① Agent 的成长是间歇式的，不是渐进式的；② 系统鲁棒性远比单个 Agent 能力更重要；③ 触发通道的设计比指令内容更重要。这三点直接影响了我怎么设计后来的业务 Agent 系统。"

---

## See Also

- [[Projects/Agent-Self-Evolution/项目概览]] — 详细实验记录
- [[Projects/项目故事/P4-商家诊断Agent-安全防御层]] — 业务 Agent 落地
- [[AI/2-Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning]] — 经验 RL 对应
- [[AI/2-Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution]] — Multi-turn RL 挑战
- [[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees]] — 理论边界
- [[AI/2-Agent/Multi-Agent/Multi-Agent 概述]]
- [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析]]
