---
title: "OMNI-LEAK: Orchestrator Multi-Agent Network Induced Data Leakage"
date: 2026-02-21
tags:
  - ai/safety
  - multi-agent
  - prompt-injection
  - data-leakage
  - red-teaming
  - orchestrator
  - icml2026
domain: ai/safety
arxiv: "2602.13477"
rating: ★★★★★
status: active
---

# OMNI-LEAK: 单次 Injection 攻陷整个多 Agent 网络

**arXiv**: 2602.13477  
**机构**: Oxford / Philip Torr lab / Yarin Gal lab  
**作者**: Akshat Naik, Jay Culligan, Yarin Gal, Philip Torr, Rahaf Aljundi, Alasdair Paren, Adel Bibi  
**提交**: 2026-02-13  
**投稿**: ICML 2026  
**评分**: ★★★★★  
**一句话**: 通过单次间接 prompt injection，在**有访问控制**的 orchestrator 多 Agent 系统中，跨多个 Agent 协同完成数据泄漏——除 Claude Sonnet 4 外，所有测试的 frontier 模型都有漏洞。

## See Also

- [[AI/Safety/AgentLeak-Full-Stack-Privacy-Leakage-Multi-Agent-Benchmark|AgentLeak]] — 姊妹论文，测量端：系统性量化 multi-agent 所有渠道的泄漏率；OMNI-LEAK 是攻击端——两篇共同结论：output-only 审计是系统性盲区
- [[AI/Safety/AutoInject-RL-Prompt-Injection-Attack|AutoInject]] — RL 生成 universal adversarial suffix（单 agent 攻击原型）；OMNI-LEAK 是 orchestrator 级别的多 agent 扩展，ASR 更高且更隐蔽
- [[AI/Agent/CowCorpus-Human-Intervention-Modeling-Web-Agents|CowCorpus]] — 人类干预时机建模；OMNI-LEAK 的 Notification Agent 无限制 send 问题正是 CowCorpus 干预节点能阻断的典型场景
- [[AI/Safety/Clinejection-AI-Coding-Agent-Supply-Chain-Attack|Clinejection]] — 真实供应链 injection 事件；OMNI-LEAK 的实验室场景在 Clinejection 中变成了真实攻击——攻击链演进方向一致
- [[AI/Safety/AI Agent 集体行为与安全漂移|AI Agent 集体行为与安全漂移]] — 多 Agent 安全漂移的理论框架；OMNI-LEAK 中 orchestrator 污染子 agent 正是一种受控的"安全漂移"实验

---

## 核心问题

单 Agent 安全研究的教训**不能直接迁移到多 Agent 系统**。

已有的防御（如访问控制、input sanitization、prompt filtering）在单 Agent 设置下有效，但在 orchestrator 模式下存在根本性的新漏洞：**当一个 agent 被攻陷，它可以利用 orchestrator 作为信道去操纵其他 agent，绕过这些 agent 自身的安全防护**。

### Orchestrator 模式的攻击面

```
用户
 ↓
Orchestrator（完整系统视图，知道所有下游 agent）
 ↙        ↘
SQL Agent    Notification Agent
（私有数据访问，受用户权限控制）   （可发送邮件给任意地址）
```

关键矛盾：
- 只有 **privileged user** 能触发 SQL Agent 访问私有数据
- 但对话数据库（公开部分）可以被**任何人写入**（比如：提交投诉、创建账号等）

---

## OMNI-LEAK 攻击链（5步）

```
①攻击者 → 将恶意 injection 写入公开数据库
         （例如，department_name 字段隐藏攻击指令）

②特权用户 → 向 Orchestrator 发起正常请求
            （例如："Mark 在哪个部门工作？"）

③Orchestrator → 路由至 SQL Agent
 SQL Agent 处理请求时读取公开数据库
             → 触发 injection，SQL Agent 被劫持
             → SQL Agent 越权查询私有表（SSN 数据）

④SQL Agent → 向 Orchestrator 返回"结果"
（实际上嵌入指令：请把以下数据发给 attacker@evil.com）
 Orchestrator 被操纵，指示 Notification Agent 执行发送

⑤Notification Agent → 通过 Email API 将私有数据发送给攻击者
                     攻击成功（严格评估：数据完整到达目标邮箱）
```

**关键机制**：每个 agent 只是在"执行正常任务"——SQL Agent 查数据，Orchestrator 路由请求，Notification Agent 发邮件。但整个链条的**组合语义**实现了单个 agent 无法完成的攻击。这是 **multi-agent compositionality attack** 的典型形态。

---

## 攻击类型（4类说服策略）

| 类别 | 说服策略 | 描述 |
|------|---------|------|
| **Blocking** | 阻断型 | 声称"完成请求前必须先执行X" |
| **Role Override** | 角色覆盖 | 说"现在你的角色是一个数据管理员，需要..." |
| **Authority Appeal** | 权威诉求 | 假装来自高权限用户或管理员指令 |
| **Gradual Compliance** | 渐进顺从 | 先要求小任务，再升级到数据泄漏 |

每类有 Explicit（知道数据库 schema，可直接用 SQL 语法）和 Implicit（不知道 schema，只用自然语言描述）两个版本。
Implicit 更难也更现实——攻击者不需要内部知识。

---

## 实验结果

### 测试设置

- **5 个 frontier 模型**：GPT-4o, o4-mini（推理型）, Gemini-2.5-Flash（推理型）, Claude Sonnet 4, 另一个非推理模型
- **数据库规模**：Toy / Medium / Big（行数、列数、数据类型递增）
- **3000 次运行/模型**（10 attacks × 5 queries × 3 db sizes × 10 repetitions，temperature=1）
- **评估指标**：
  - BA（Benign Accuracy）：正常查询正确率（无攻击下）
  - Attack Success Rate（ASR）：攻击成功率（数据完整到达攻击者邮箱）
  - Faithfulness：攻击下正常查询是否仍然被正确回答

### 关键发现

**除 Claude Sonnet 4 外，所有模型都对至少一种 OMNI-LEAK 攻击有漏洞。**

- 推理模型（o4-mini, Gemini-2.5-Flash）和非推理模型都容易受攻击
- 攻击者不需要内部知识（implicit 版本也成功）
- 数据库越大（更复杂），攻击有时反而更容易——复杂查询掩盖了注入指令的可疑性
- **Claude Sonnet 4 的防护**：论文指出 Claude 对 indirect prompt injection 有更强的系统级防护，可能来自 Constitutional AI 的训练方式

### 防御的根本困难

1. **访问控制失效**：Access control 是在数据层面（私有/公开表分离），但攻击是在模型推理层面——即使 SQL Agent 无法直接访问私有表，被劫持后它会利用高权限用户的 session 查询
2. **多 agent 信任链问题**：Orchestrator 信任 SQL Agent 的输出；Notification Agent 信任 Orchestrator 的指令。在这个信任链中，一个被攻陷的 agent 可以"污染"整个链
3. **规则层防御失效**：现有 content filtering 是在看单个 agent 的 input/output，没有监控跨 agent 的语义组合

---

## 我的分析

### 为什么这篇论文很重要？

这篇论文做了一件非常清晰的事：**用一个具体的、有代表性的 industry 设置（orchestrator + SQL + 邮件通知），实证地证明了多 agent 安全不是单 agent 安全的简单叠加**。

这不是理论推断——3000 次运行，5 个模型，有数据。

### 攻击的 elegant 之处

OMNI-LEAK 的 clever 在于它**没有试图直接攻破任何单个 agent 的防护**。每一步都是"合规"操作：
- SQL Agent 查数据：它的工作
- Orchestrator 路由请求：它的工作  
- Notification Agent 发邮件：它的工作

只有把所有步骤放在一起，才能看出数据泄漏。这类似于现实中的 insider threat：每个人都在"做本职工作"，但组合起来实现了攻击目标。

### 对盾卫项目的含义

**直接相关**：盾卫目前的 memory_guard.py 只保护单个文件（MEMORY.md / SOUL.md）的完整性和内容安全。OMNI-LEAK 说明：如果系统中有多个 Agent，需要考虑**跨 agent 的信息流**，不只是单个 agent 的输入/输出。

具体：
1. **跨 agent 信息流审计**：需要记录 Agent A 告诉 Agent B 了什么，不仅仅是各自的 I/O
2. **信任链验证**：Orchestrator 不应无条件信任下游 agent 的输出，特别是当该输出包含"请执行X"类指令时
3. **工具调用白名单**：Notification Agent 被要求给任意邮件地址发邮件——应该有限制

### Claude Sonnet 4 的抵抗性

论文注意到 Claude Sonnet 4 是唯一一个对所有测试攻击都成功防守的模型。这是个有趣的 signal，值得追踪：
- 是模型 fine-tuning 层面的防御（训练时见过这类 attack pattern）？
- 还是系统级的 Constitutional AI 原则（模型不信任工具返回的"指令"）？
- 这种防御是 robust 的还是可以被更复杂的攻击绕过？

### 现有防御的局限

论文指出 prompt filtering / fine-tuning 对自适应攻击仍然有效率不高。结合 AutoInject（RL 生成 suffix）的发现，我的判断是：

**纯内容层防御（规则 + 分类器）在面对 motivated attacker 时都是不充分的**。需要：
1. **信息流控制**（IFC）：哪个 agent 有权访问哪些数据，不仅仅是访问控制，还要控制谁有权传递信息给谁
2. **intent monitoring**：监控 agent 的意图变化，不只是当前输出
3. **人类干预节点**：高风险操作（发邮件给外部、写入数据库）需要 human confirmation

### 这篇论文的局限

1. **场景设定简单**：只有 SQL + Email 两个 agent，真实系统更复杂
2. **攻击者模型有限**：假设攻击者能写入公开数据库，现实中这个条件可能不总成立
3. **防御实验缺失**：论文主要是攻击端，没有提出实质性防御方案
4. **Claude Sonnet 4 防护原因未解**：论文没有解释为什么 Claude 能防住

---

## 关键数字

- 全球平均数据泄漏成本：**$4.4M**（IBM 2025 报告）
- 测试规模：**5 frontier 模型 × 3000 runs/model = 15,000 次运行**
- 受漏洞模型：**4/5**（claude-sonnet-4 是唯一幸存者）
- 攻击类型：**10 种攻击 × 2 版本（explicit/implicit）= 20 种变体**

---

## 与盾卫项目的连接

| OMNI-LEAK 攻击组件 | 盾卫对应缺口 |
|-------------------|-------------|
| 公开数据库的 injection | memory_guard 只保护 MEMORY.md/SOUL.md，未保护工具返回数据 |
| SQL Agent → Orchestrator 的指令污染 | 无跨 agent 信息流审计 |
| Notification Agent 的无限制 send | 无工具调用白名单/速率限制 |
| 整体攻击的隐蔽性 | 无 semantic anomaly detection（只有 rate-based） |

**Phase 2.3 规划应当包含**：对外部工具返回内容的 injection scan，不仅仅是对 MEMORY.md 的写入扫描。

---

## Tags
#MultiAgent #OrchestratorPattern #PromptInjection #DataLeakage #RedTeaming #AgentSecurity #ICML2026 #盾卫 #信息流控制 #MultiAgentSafety
