---
title: "EnterpriseGym Corecraft — 用高保真 RL 环境训练可泛化 Agent"
date: 2026-02-18
arxiv: "2602.16179"
affiliation: "Surge AI"
tags: [agentic-RL, RL-environment, enterprise-agent, GRPO, OOD-generalization, benchmark]
rating: ★★★★
---

# EnterpriseGym Corecraft — 用高保真 RL 环境训练可泛化 Agent

**arXiv**: 2602.16179  
**发布**: 2026-02-18  
**机构**: Surge AI  

## 一句话总结

在高保真企业 RL 环境上训练 AI agent，能让能力**泛化到训练分布之外**。

## 核心问题

当前 agentic RL 的困境：训练环境太玩具化（toy task）→ 训练出的 agent 在真实场景迁移失败。  
研究问题：**环境质量**对 agent 泛化能力有多大影响？

## Corecraft 环境设计

**定位**：EnterpriseGym 套件的第一个环境，模拟真实企业客服部门。

| 维度 | 规模 |
|---|---|
| 实体数量 | 2,500+ 个实体 |
| 实体类型 | 14 种 |
| 工具 | 23 种独特工具 |
| 任务性质 | 多步骤、领域特定、专业级 |

**三个关键设计原则**（与观测到的泛化能力一致）：
1. **Task-centric World Building** — 世界构建以任务多样性和挑战性为中心，而不是以真实感为中心
2. **Expert-authored Rubrics** — 专家手写 rubric，实现可靠 reward 计算（解决了 open-ended task 的 reward 问题）
3. **Enterprise Workflows** — 反映真实职业模式，不是简化版

## 实验结果

### Frontier 模型基线（令人警醒）
- **GPT-5.2** 和 **Claude Opus 4.6** 在满足所有 rubric 标准的条件下，pass rate **< 30%**
- 这说明 2026 年 2 月的 frontier 模型在**真实企业 agent 任务**上仍远未"解决"

### GRPO 训练 GLM 4.6
- 训练前：25.37% task pass rate（held-out evaluation）
- 训练后（**单 epoch**）：36.76% ← +11.4 个点
- **OOD 泛化**（关键结果）：
  - +4.5% on BFCL Parallel
  - +7.4% on τ²-Bench Retail
  - +6.8% on Toolathlon (Pass@1)

**用了什么方法**：GRPO + Adaptive Clipping，训练在 Corecraft 环境，测试在独立 benchmark。

## 为什么这篇重要

### 1. 验证了"环境质量是关键变量"假说
之前的 agentic RL 工作（CM2、PARL、OpenRS）都是在固定任务集上做 reward 设计。Corecraft 的贡献是说：**环境本身的质量决定了泛化能力的上限**。Toy 环境出 toy agent。

### 2. 单 epoch 就能 OOD 泛化
GLM 4.6 在 Corecraft 上训练了**一个 epoch** 就在 3 个 OOD benchmark 上有实质性提升。这意味着：
- 高保真环境本身承载了丰富的归纳偏置
- 模型从企业任务结构中学到了"可迁移的 agent 技能"，不是靠任务记忆

### 3. Expert-authored Rubrics = 可扩展的 Reward 设计
这与 OpenRS（pairwise adaptive rubric）和 CM2（checklist reward）属于同一思路：**把专业知识编码进 rubric，而不是用 LLM judge**。Surge AI 的贡献是把它做成了商业级可用的环境。

### 4. Frontier 模型 <30% pass rate 的含义
这是重要的基准线：
- 即使是 Opus 4.6 / GPT-5.2 这样的 frontier 模型，在要求满足全部专家 rubric 的企业 agent 任务上，只能做对 3 个里不到 1 个
- 这是 agentic RL 研究的核心动力 — 在真实工作中，AI 仍然远远落后

## 与 Agentic RL 2026 全景的关联

| 工作 | 核心贡献 | 层次 |
|---|---|---|
| Corecraft | 高保真训练**环境** → OOD 泛化 | 环境设计 |
| CM2 | **Checklist reward** 支持 multi-turn tool use | Reward 设计 |
| OpenRS | **Rubric-based** reward 替代 LLM judge | Reward 设计 |
| PARL | **冻结 subagent** 训练 orchestrator | 训练算法 |
| Gaia2 | 异步动态 benchmark + action-level verifier | 评估设计 |

这 5 篇工作一起，描述了 agentic RL 的完整 pipeline：好环境 → 好 reward → 好算法 → 好评估。

## 局限性

- Corecraft 只是 EnterpriseGym 的第一个环境（客服场景），泛化到其他职业尚未验证
- GLM 4.6 的基础能力可能已经对企业任务有亲和性，不同规模/架构的模型迁移效果未知
- "高保真"的构建成本（2500+ 实体，专家 rubric）不是普通研究团队能复现的

## 关联

- [[CM2]] — Checklist reward 设计
- [[OpenRS-Pairwise-Adaptive-Rubric]] — Expert-authored rubric 的学术版
- [[Kimi-K2.5-PARL]] — Parallel multi-agent RL 算法
- [[Gaia2]] — 异步 agent benchmark（行动级别 verifier）
- [[Claude-Opus-4.6]] — Corecraft 基准中的 frontier 模型之一，<30% pass rate
