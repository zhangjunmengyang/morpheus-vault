---
title: "Gaia2 — 动态异步环境 Agent Benchmark"
brief: "Gaia2：动态异步多任务 Agent 评测 benchmark；任务状态持续变化（非静态），Agent 需并发处理多工具并异步等待结果；比 GAIA-1 更接近真实部署场景；当前 frontier 模型在动态任务显著弱于静态"
type: paper
domain: ai/agent
updated: 2026-02-22
tags:
  - ai/agent
  - benchmark
  - evaluation
  - async-environment
  - reinforcement-learning
  - gpt-5
  - claude-4
rating: ★★★★★
date: 2026-02-13
arxiv: "2602.11964"
institution: Meta FAIR（主导）+ 多机构合作
source: https://arxiv.org/abs/2602.11964
archived_by: librarian
archived_date: 2026-02-20
---

# Gaia2 — 动态异步环境大模型 Agent Benchmark

> **arXiv**: 2602.11964 | **发布**: 2026-02-13  
> **机构**: Meta FAIR + 多机构（Froger, Scialom 等 23 位作者）  
> **亮点**: 包含 GPT-5 + Claude-4 Sonnet 的真实跑分——未发布模型的能力信号

---

## 一句话

Gaia2 是首个专注于**动态异步**环境的 Agent benchmark：环境独立于 Agent 行动持续演化，Agent 必须在时间压力、噪声事件和多 Agent 协作中完成任务。GPT-5 (high) 最强但 42% pass@1，最好的模型依然在 sim2real gap 上挣扎。

---

## 为什么重要：与 GAIA v1 的根本差异

| 维度 | GAIA v1 | Gaia2 |
|------|---------|-------|
| 环境类型 | 静态 | **动态异步**（环境持续演化） |
| 时间约束 | 无 | 有（时间敏感任务） |
| 评估粒度 | 最终结果 | **action 级别**（write-action verifier） |
| RL 可用性 | 否 | **是**（verifiable reward 直接可用） |
| 多 Agent | 否 | 支持 Agent 间协作 |

Gaia2 更接近真实世界：现实任务不会等你，环境在你思考的时候已经变化了。

---

## 核心设计

### 异步环境的挑战

Agent 在 Gaia2 中面对：
1. **时间约束**：任务有截止时间，超时即失败
2. **噪声事件**：环境会产生干扰信息，Agent 需要区分信号与噪声
3. **歧义消解**：环境状态模糊，Agent 必须主动澄清
4. **多 Agent 协作**：某些任务需要 Agent 间通信和协调

### Write-Action Verifier

每个场景配备**写动作验证器**：
- 细粒度评估每个 action，而非只看最终结果
- 验证器输出可直接作为 RL reward signal
- 这让 Gaia2 成为一个**自带 verifier 的 RL 训练环境**

---

## 关键结果：未发布模型的真实信号

| 模型 | pass@1 | 特点 |
|------|--------|------|
| **GPT-5 (high)** | **42%** | 最强总分，但时间敏感任务失败 |
| **Claude-4 Sonnet** | 未披露具体数值 | 以准确率+速度换取成本效率 |
| **Kimi-K2** | **21%** | 开源模型最强 |
| 其他开源模型 | < 21% | 差距显著 |

> ⚠️ **注意**：GPT-5 和 Claude-4 Sonnet 均为尚未公开发布的模型，这是截至 2026-02-20 极少数有这两个模型具体跑分的公开文献。

### 核心发现

1. **没有全能模型**："no model dominates across capabilities"——不同模型在不同能力维度上各有优劣
2. **时间敏感是短板**：GPT-5 整体最强，但在时间敏感任务上失败——说明当前 LLM 尚未真正掌握时序推理
3. **sim2real gap 依然显著**：即使是最强模型 42%，在异步真实环境下的表现远低于静态 benchmark
4. **Kimi-K2 = 开源最强**：21% pass@1 在开源模型中领先，与 GPT-5 的差距仍是 2x

---

## 洞察

### 当前 Agent 能力的上限在哪里

42% pass@1（GPT-5）→ 58% 的动态异步任务，最强模型依然无法完成。

这不是 reasoning 能力的问题——是 Agent 与异步世界交互模式的问题：
- 等待确认（同步思维）vs 并发行动（异步思维）
- 固定计划 vs 动态调整
- 单独决策 vs 多 Agent 协调

### Write-Action Verifier 的价值

这是 Gaia2 最低调但最有价值的贡献——提供了**任务级可验证 reward**。

在 RLVR 当道的 2026 年，training data 最大的瓶颈是缺乏 verifiable reward signal。Gaia2 直接提供了一个：
- 动态的、异步的 Agent 训练环境
- 细粒度的 action-level reward
- 这可能成为下一代 Agentic RL 的关键训练基础设施

---

## see-also

- [[AI/Agent/Aletheia-Math-Research-Agent|Aletheia]] — 同期的 Agent benchmark 跨越（math research）
- [[AI/Agent/EnterpriseGym-Corecraft|EnterpriseGym Corecraft]] — 企业 RL 环境，与 Gaia2 ARE 平台同类思路
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿]] — Gaia2 的训练价值需要在 Agentic RL 框架下理解
- [[AI/LLM/Evaluation/ICLR-2026-趋势分析|ICLR 2026 趋势]] — 2026 年 Agent benchmark 整体趋势
- [[AI/Agent/目录]] — Agent MOC

---

*归档：馆长 · 2026-02-20 · 来源：arXiv:2602.11964*
