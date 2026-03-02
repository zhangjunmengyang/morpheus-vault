---
title: "ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation RL"
author:
  - Jianghao Chen
  - Wei Sun
  - Qixiang Yin
  - Zhixing Tan
  - Jiajun Zhang
arxiv: 2509.04903
venue: Under review (as of v3, 2025-12-30)
tags:
  - type/paper
  - topic/llm
  - topic/rl
  - topic/reward-model
  - topic/verification
  - topic/checklist
created: 2026-03-02
---

## TL;DR
ACE-RL 把“长文写作质量”从粗粒度 subjective reward（coherence/helpfulness）改成 **instruction-adaptive 的 constraint checklist**，再用一个 verifier/RM 去逐条验证约束满足度，把 reward 变成 **可验证/可分解的信号**，然后用 RL 优化。

它几乎是 Vault 里「Intermediate Verification Signal 自动化」缺口的一个直接命中点：**checklist 可以自动从指令分解而来**，不再依赖人工设计。

## Problem
长文生成（story/report/legal）的问题不在“基本相关+连贯”，而在每条指令里的“隐含/显式要求”不同；传统 RLHF/偏好奖励通常只做粗维度评分，导致：
- 场景特定需求（比如“ O.Henry 式反转结尾”）难被 reward 捕捉
- 需要额外的 preference pairs，数据昂贵且难规模化

## Core idea
1) **Instruction → constraint checklist（自动生成）**
- 他们声称有自动 pipeline：从真实人机交互里收集长文指令，并将每条指令分解为 *explicit demands* + *implicit intents*，转成可验证 constraints。
- constraints 覆盖维度（文中举例）：content completeness / structural logic / stylistic formatting。

2) **Constraint verification → reward**
- 用 reward model / verifier 去逐条判断“response 是否满足某条 constraint”，把“主观质量评估”降解为“约束验证”。
- 训练时还会加 length reward（Figure 2 caption）。

3) **RL training**
- 用上述 fine-grained signal 做 RL（具体算法细节需读 PDF；HTML 摘要未给出）。

## Reported results (from abstract)
- WritingBench：相对 SFT baseline **+18.63%**
- WritingBench：相对 RL baseline **+7.61%**
- Top model 甚至超过 GPT-4o **+8.76%**（指标同上）

> 这些数字很强，但需要追问：指标是什么？judge 是谁？是否存在 reward hacking（迎合 verifier）？

## Why I care (连接到 Agent RL)
这篇的范式可以迁移到 tool-use / multi-turn agent：
- **把“任务成功”拆成中间约束**（subgoals/constraints）
- 让 reward 信号从 sparse outcome → dense verification
- 关键瓶颈从“怎么训 policy”转移为“怎么自动构造/自举 constraints + verifier”

这就是我们在 INBOX 里标的缺口：*Intermediate Verification Signal 自动化*。

## Failure modes / open questions
- Checklist 质量：分解是否稳定？是否会遗漏关键隐含意图？
- Verifier 可靠性：constraint verification 是否可被 prompt-injection / stylistic hacks 骗过？
- Reward hacking：模型是否学会“迎合 checklists”而非真正改善写作？需要 human eval 或对抗评估。
- 泛化：对 unseen domain/长文体裁是否仍有效？

## Pointers
- arXiv:2509.04903 <https://arxiv.org/abs/2509.04903>

## See Also
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL]]（checklist reward：人工/半自动设计 → 可验证信号）
- [[AI/3-LLM/RL/Fundamentals/RLCF-Checklists-Are-Better-Than-Reward-Models-2507.18624]]（checklist 作为 RM 替代范式；和 ACE-RL 的“约束验证式 reward”同一思想线）
