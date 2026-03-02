---
title: "TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation"
author:
  - Jonathan Cook
  - Tim Rocktäschel
  - Jakob Foerster
  - Dennis Aumiller
  - Alex Wang
arxiv: 2410.03608
created: 2026-03-02
tags:
  - type/paper
  - topic/llm
  - topic/evaluation
  - topic/llm-as-judge
  - topic/checklist
  - topic/self-improvement
---

## TL;DR
TICK 提出一种 **LLM-as-judge 的结构化评测协议**：先让 judge LLM 把 instruction 拆成 *instruction-specific checklist*（一串 YES/NO 问题），再用这些问题去评估候选回答。

关键价值不在“又一个 judge”，而在把偏好判断从单一标量排名，拆成 **可解释的细粒度约束满足度** —— 这和我们最近追的“Intermediate Verification Signal（中间验证信号）”是同一根思想。

## Problem
- preference judgment（A/B 排序）把多维偏好压扁成 1 个 ranking；解释性差。
- 用 LLM 代替人做 judge 省钱但不可靠、也不透明。

## Method: TICK
**TICK = Targeted Instruct-evaluation with ChecKlists**
- 输入：instruction + candidate response(s)
- Step 1：让 LLM 生成 *tailored evaluation checklist*，把 instruction 分解成一串 **YES/NO 问题**（每题对应一个具体要求）。
- Step 2：再让 judge LLM 逐条回答这些 YES/NO，并据此做整体判断/偏好。

文中强调：
- checklist 生成是 fully automated，消除人工写 rubric 的成本。
- 输出是 interpretable、fine-grained 的评测证据链。

## Key results (from abstract)
1) **Judge agreement vs human preference**
- 使用 TICK：LLM judge 与 human preferences 的 *exact agreement* 频率从 **46.4% → 52.2%**（绝对 +5.8%）
  - 对照：让 LLM 直接给 output 打分。

2) **Generation improvement: STICK / Self-TICK**
- **STICK（Self-TICK）** 用 checklist 做 self-refinement + Best-of-N：
  - LiveBench reasoning self-refinement：绝对提升 **+7.8%**
  - WildBench 上 Best-of-N selection：绝对提升 **+6.3%**

3) **Help humans evaluate**
- 给人类评审提供 LLM-generated checklist：WildBench 上 inter-annotator agreement **0.194 → 0.256**

## My take (判断)
- 这篇比“换个 judge 模型”更重要：它提供了一个可迁移的 **verification interface**：把“质量”变成一串可判定的约束。
- 风险也很明确：checklist 的 coverage 决定上限；错漏 checklist 会把模型/评审都带偏（Goodhart）。

## Connection: 自动化 intermediate verification signal
和 ACE-RL (2509.04903) 的关系：
- TICK：用于 **evaluation / self-improvement**（test-time 或数据筛选）
- ACE-RL：把类似 checklist 思路进一步用于 **training-time reward**（verifier 给 reward）

二者合起来，就是“从评测到训练”的闭环：
instruction → checklist → verify → (select/refine) / (reward+RL)

## Pointers
- arXiv:2410.03608 <https://arxiv.org/abs/2410.03608>
- HTML <https://arxiv.org/html/2410.03608v1>

## See Also
- [[AI/3-LLM/RL/Fundamentals/ACE-RL-Adaptive-Constraint-Enhanced-Reward-2509.04903.md]]
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-RL.md]]
