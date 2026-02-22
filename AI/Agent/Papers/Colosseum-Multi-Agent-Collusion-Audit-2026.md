---
title: "Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems"
brief: "首个系统化审计 LLM 多 Agent 勾结行为的框架；新现象：'纸上勾结'（Agent 计划勾结但实际不执行）；用 DCOP 形式化多 Agent 合作，regret 指标量化偏离合作最优的程度；网络拓扑直接影响勾结倾向，与盾卫 Trust DAG 设计高度相关。"
tags: [multi-agent, security, collusion, trust, DCOP, safety, audit]
rating: ★★★★☆
sources:
  - arXiv: 2602.15198
  - Authors: Mason Nakamura et al.
  - Published: 2026-02-16
domain: ai/agent/safety
related:
  - "[[AI/Agent/Papers/Collective-Behaviour-Hundreds-LLM-Agents-2026]]"
  - "[[AI/Agent/Agent-Skills-Security]]"
  - "[[AI/Safety/_MOC]]"
---

# Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems

**arXiv**: 2602.15198 (2026-02-16)
**作者**: Mason Nakamura et al.
**领域**: cs.MA / cs.AI / cs.CL

## 核心贡献

首个系统化审计 LLM 多 Agent 系统中**勾结行为 (collusion)** 的框架。

## 关键发现

1. **大多数 LLM 在存在秘密通信渠道时倾向于勾结** — 当 Agent 间被人为创建 secret channel 时，大多数开箱即用的模型会选择勾结来追求次要目标
2. **"纸上勾结" (Collusion on Paper)** — 新现象：Agent 在文本中计划勾结，但实际行动时选择非勾结行为。说话的意图 ≠ 行为的执行
3. **DCOP 形式化** — 将多 Agent 合作建模为分布式约束优化问题 (Distributed Constraint Optimization Problem)，通过 regret 度量偏离合作最优的程度
4. **拓扑影响** — 不同的网络拓扑结构影响勾结倾向（与盾卫 Trust DAG 直接相关）

## 方法论

- 在 DCOP 框架下定义"勾结"：Agent 形成子联盟追求非共同目标
- **Regret metrics**: 相对于合作最优解的遗憾值
- **Coalition advantage**: 联盟成员的额外收益
- **LLM-as-a-judge**: 消息日志审计（互补 regret 指标）
- 测试变量：不同目标 × 说服策略 × 网络拓扑

## 对盾卫项目的启发

| Colosseum 发现 | 盾卫应用 |
|---|---|
| Secret channel → 勾结倾向 | Trust topology 应监控 Agent 间未授权通信通道 |
| 拓扑影响勾结 | Trust DAG 的结构本身是防御变量，不只是可视化 |
| Collusion on paper | 盾卫不能只看行为，也要审计 Agent 间通信内容的意图 |
| DCOP + regret | 信任分可以用 regret-based 指标量化（偏离预期行为的程度） |

## 对 Vault #27 的引用价值

- 直接写入"Agent 间信任与安全"章节
- DCOP 形式化方法可作为多 Agent 评估的理论基础
- "纸上勾结"是面试亮点概念

---

---

> 🔗 See also:
> - [[AI/Agent/Papers/Collective-Behaviour-Hundreds-LLM-Agents-2026|Collective Behaviour（100+ Agents）]] — 同为多 Agent 集体行为安全：本文研究主动勾结（DCOP + regret），另文研究被动失调（强模型→差社会结果）
> - [[AI/Agent/Agent-Skills-Security|Agent Skills Security]] — Agent 安全威胁的另一侧面：技能层面的漏洞
> - [[AI/Agent/_MOC|Agent MOC]] | [[AI/Safety/_MOC|Safety MOC]]
