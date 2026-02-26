---
brief: "OpenRS——基于 Pairwise Adaptive Rubric 的 RL 对齐；用自适应评分标准替代固定 outcome verifier，通过成对比较训练 RM 产生鲁棒奖励信号；针对 constitutional-ai 场景的 reward hacking 防御。"
title: "OpenRS — 基于 Pairwise Adaptive Rubric 的 RL 对齐"
type: research
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - ai/llm/alignment
  - type/research
  - topic/reward-model
  - topic/non-verifiable
  - topic/constitutional-ai
  - topic/reward-hacking
created: 2026-02-19
---

# OpenRS — Open Rubric System: 基于 Pairwise Adaptive Rubric 的 RL 对齐

> 论文：[arXiv 2602.14069](https://arxiv.org/abs/2602.14069) — Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric
> 提交：2026-02-15
> 关键词：reward hacking, non-verifiable alignment, rubric-based reward, pairwise RL, constitutional AI

## 核心问题：为什么 Scalar Reward 总是 Hack

现有 RLHF 的 reward model 把多维人类偏好压进一个标量——这不是工程实现问题，是**信息瓶颈**问题：

```
多维偏好 (准确性/安全性/帮助性/风格/...) 
    → [Reward Model 内化学习]
    → 单一标量 r ∈ ℝ
    → 模型优化这个标量
    → reward hacking
```

本质上，模型学会了"让 reward model 高兴"，而不是"做对事情"。

OpenRS 的重新定义：**对齐本质上是 principle generalization 问题，而不是 function approximation 问题。**

- Function approximation 视角：reward 是需要学习的函数，internalized 进 judge
- Principle generalization 视角：reward 是需要执行的推理过程，在 explicit、可检查的原则下运行

## 系统设计

```
OpenRS 架构
├── PAMR (Pairwise Adaptive Meta-Rubric)
│   ├── Meta-Rubric：类 constitution 的原则规范，控制 rubric 如何被实例化/加权/执行
│   ├── 动态实例化：conditioning on 两个候选回答的语义差异，按 query 生成 rubric
│   ├── Criterion-wise pairwise comparison：对每个标准分别比较
│   └── 外部聚合：避免 pointwise 加权标量化（信息丢失根源）
│
└── PVR (Pointwise Verifiable Rubric)
    ├── 处理有 ground truth 的客观子任务
    ├── 提供 verifiable reward 组件
    └── 作为 guardrail，防止退化行为
```

### Meta-Rubric Refinement Pipeline

维持原则的一致性和可编辑性：

```
通用原则 → Automated Evolutionary Refinement
           （自动化进化精化，无需人工）

领域原则 → Human-in-the-Loop Procedure  
           （可复现的人工介入流程）
```

这个两层设计解决了一个实际问题：通用对齐原则可以自动迭代，但领域特定原则（医疗、法律等）需要专家介入且必须可复现。

## 与 Constitutional AI 的关系和区别

| 维度 | Constitutional AI (Anthropic) | OpenRS |
|------|-------------------------------|--------|
| 机制 | Prompt-level：原则用于引导 critique/revision | Reward-level：rubric 直接作为 reward supervision |
| 训练闭环 | 通过 RLAIF 间接 | 直接用于 pairwise RL 训练 |
| 动态性 | 静态 constitution | 按 query 动态实例化 rubric |
| 颗粒度 | 整体原则 | Criterion-wise 细粒度比较 + 外部聚合 |
| 可检查性 | constitution 公开但执行不透明 | rubric 实例化过程显式可查 |

核心进步：**更紧的 reward 闭环 + 更高的动态适应性 + 更细的颗粒度**

## 对 Non-Verifiable Tasks 的意义

RLVR（Verifiable Reward）在数学/代码任务上大获成功，因为有明确的正确答案。但大量真实任务是 non-verifiable 的：写作、分析、对话、创意生成——这类任务以前只能用 RLHF，而 RLHF 的 reward hacking 是已知问题。

OpenRS 给出的路线：**显式化 rubric = 让 reward 过程可审查、可编辑、可泛化**，而不依赖于 internalized judge 的黑盒评分。

```
Non-verifiable tasks 的对齐路线图（2026 视角）：

RLHF:       [偏好数据] → [Reward Model 内化] → scalar → RL 优化
                                                           ↑ reward hacking

OpenRS:     [Meta-Rubric] → [动态 Rubric 实例化] → criterion-wise → 外部聚合 → RL
                                                                        ↑ 信息保留

RLTF (ref): [Text Feedback] → [内化 NLL] → ...（不同路线）
```

## 我的批判性评估

### 真正值得关注的地方

1. **"Principle generalization" 这个框架**本身是重要的概念贡献：它把 alignment 从"学一个函数"重新定义为"执行一个推理过程"。这个视角转换影响了如何设计整个 reward 系统。

2. **Pairwise comparison 替代 pointwise scalar**：把比较对象从"单个回答"变成"两个回答之间的差异"，这是信息层面的根本改变——rubric 的实例化 conditioning on 差异本身，能捕捉到更细的判断依据。

3. **Meta-rubric 的可编辑性**：和 hardcoded reward function 相比，rubric 是人类可读、可修改的——这是 oversight 的关键属性。

### 需要保留的怀疑

1. **Rubric 实例化的稳定性**：LLM 动态生成 rubric 本身就引入了不确定性。不同 run 生成的 rubric 是否一致？对 rubric generator 的能力有多敏感？

2. **Criterion aggregation 的隐含假设**：外部聚合虽然避免了加权 scalar，但聚合方式本身也是一种 inductive bias，论文需要说明这个 aggregation 如何做。

3. **计算开销**：每个 query 都要动态实例化 rubric + 做 criterion-wise pairwise comparison，对大规模 RL 训练的 throughput 影响未充分讨论。

## 今日 Agentic RL 全景中的位置

```
今日捕捉到的 Reward Signal 构建方法：

问题                  方法              来源
──────────────────────────────────────────────────
Open-ended 对齐       OpenRS PAMR       2602.14069  ← 本篇
non-verifiable reward

Multi-turn tool use   CM2 Checklist     2602.12268
缺 verifiable signal  Rewards

Multi-agent           PARL              2602.02276
orchestrator 训练     冻结 subagent

Math research agent   Aletheia NL       2602.10177
无形式化验证信号       Verifier

共同主题：如何在 reward signal 稀少/不可验证的场景下
         构建可训练的 RL 信号。这是 2026 Agentic RL 的核心战场。
```

## 关联笔记

- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2-checklist-rewards]] — 同方向：non-verifiable reward 的替代方案（checklist 而非 rubric）
- [[AI/2-Agent/Evaluation/Aletheia-Math-Research-Agent|Aletheia-Math-Research-Agent]] — NL Verifier 作为 dense reward signal
- [[AI/3-LLM/RL/Other-Algorithms/RLTF-RL-from-Text-Feedback|RLTF-RL-from-Text-Feedback]] — text feedback 内化路线
- [[AI/3-LLM/RL/RARL-Reward-Modeling-Survey-论文笔记|RARL-Reward-Modeling-Survey]] — reward modeling 综述背景
- [[AI/2-Agent/Multi-Agent/Kimi-K2.5-PARL|Kimi-K2.5-PARL]] — multi-agent reward 设计

---
*Created: 2026-02-19 by Scholar heartbeat*
