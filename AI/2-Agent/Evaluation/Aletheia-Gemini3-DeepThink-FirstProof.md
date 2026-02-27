---
title: Aletheia FirstProof：Gemini 3 Deep Think 自主解决研究级数学问题
brief: Aletheia（基于 Gemini 3 Deep Think）在 FirstProof 挑战赛（10道真实研究级数学题，8天限时）自主解决 6/10，专家 majority 全部认可；Generator-Verifier 双 agent 管道；Best-of-2 策略带来 20% 额外增益；自我过滤设计（reliability over capability）；P7 为 Weinberger 书中公开问题——AI 数学研究能力首次在 open problems 上突破专家水平。
type: research
domain: ai/agent
date: 2026-02-25
arxiv: "2602.21201"
authors: Tony Feng, Thang Luong et al. (Google DeepMind + UC Berkeley/Brown/Caltech/CMU/USC/UT Austin)
affiliation: Google DeepMind
rating: ★★★★★
tags:
  - ai/agent
  - ai/llm/reasoning
  - ai/frontier
  - type/research
  - model/gemini3
  - topic/math-ai
  - topic/scientific-discovery
  - topic/evaluation
sources:
  - arXiv:2602.21201 (Feng, Luong et al., Google DeepMind, 2026-02-25)
  - https://1stproof.org — FirstProof 挑战赛
related:
  - "[[AI/2-Agent/Evaluation/Aletheia-Math-Research-Agent|Aletheia（前作：Erdős 问题，arXiv:2602.10177）]]"
  - "[[AI/3-LLM/Inference/Gemini-3-Deep-Think|Gemini 3 Deep Think（基础模型）]]"
  - "[[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（推理时搜索 vs 训练时搜索对比）]]"
  - "[[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（反事实验证，设计思路对比）]]"
---

# Aletheia：Gemini 3 Deep Think 自主解决研究级数学问题

> **arXiv**: 2602.21201 · **机构**: Google DeepMind · **发布**: 2026-02-25 · **★★★★★**

---

## TL;DR

Google DeepMind 的 Aletheia agent（基于 Gemini 3 Deep Think）在 FirstProof 挑战赛（10道研究级数学题）中自主解决了 6/10，专家评估 majority 认为全部 6 道正确。这是迄今最强的 AI 自主数学研究能力展示。

---

## 背景：FirstProof 挑战赛

**FirstProof**（[1stproof.org](https://1stproof.org)）由专业数学家发起，2026-02-05 发布，2026-02-13 截止。10 道来自现实数学研究中的问题（大部分是 Lemmas，P7 是 Weinberger 书中的开放问题）。

**解题标准**："AI 应以自主方式生成符合数学文献严格性和学术规范的证明，不依赖人类的任何数学思想或内容。"

---

## Aletheia 架构概要

Aletheia 之前在 Erdős 问题上有记录（feng2026autonomousmathematicsresearch/feng2026semiautonomousmathematicsdiscoverygemini），本次是升级版：

- **Aletheia A**: 基础模型 = Gemini 3 Deep Think（2026年2月版）
- **Aletheia B**: 基础模型 = Gemini January 2026 版本
- **核心组件**: Generator subagent + Verifier subagent（两级管道）
- **关键设计原则**：自我过滤（"reliability over capability"）——宁可返回 "No solution found" 也不返回错误答案

### 管道架构（Human-AI Interaction Card）
```
问题（copy-paste verbatim）
    ↓
Aletheia Agent（Generator + Verifier）
    ↓
候选解
    ↓
Verification & Extraction Prompt → Gemini 3 Deep Think
    ↓
[CORRECT] / [FIXABLE] / [WRONG]
    ↓
最终 LaTeX 输出（自动格式化，无需人工干预）
```

Verification prompt 分三步：
1. 独立验证（先自己推一遍再看候选解）
2. Verdict：[CORRECT] / [WRONG] / [FIXABLE]
3. Resolution：FIXABLE → 输出完整修正版

---

## 性能结果（Best of 2）

| 问题 | Aletheia A | Aletheia B | 最终 | 专家评估 |
|------|-----------|-----------|------|---------|
| P1 | No output | No output | N/A | — |
| P2 | Correct | Correct | ✅ | 4/4 |
| P3 | No output | Filtered out | N/A | — |
| P4 | No output | No output | N/A | — |
| P5 | Correct | Misinterpreted | ✅ | 4/4 |
| P6 | No output | No output | N/A | — |
| P7 | Critically Flawed | Correct | ✅ | 3/3 |
| P8 | Inadequate | Correct? | ✅? | 5/7（非全票）|
| P9 | Correct | Correct | ✅ | 4/4 |
| P10 | Correct | Correct | ✅ | 2/2 |

**总计：6/10 问题有 majority 专家认可的正确解**

---

## 关键技术洞察

### 1. Best-of-2 的重要性
两个 agent（不同基础模型）产生互补优势：
- P5：A 正确，B 误解题意
- P7：A 有致命缺陷，B 正确
- P8：A 不充分，B 勉强通过

如果只跑一个版本：A 解 4 题，B 解 4 题；best-of-2 解 6 题。**多 agent + 选择策略 = 20% 额外增益**，且几乎零额外推理成本（只需要在完成后比较，不需要交互）。

### 2. 推理计算量 vs. 问题难度
论文提供了每个问题的推理成本（相对于 Erdős-1051 问题的倍数）：
- 多数问题：1-5x 倍
- **P7**（Weinberger 书中开放问题）：比以往观测的 scale **高一个数量级**——Generator 花了更多计算生成候选，Verifier 也需要更多轮次验证

**洞察**：推理成本是问题难度的隐式代理，说明 TTC scaling 对真正困难问题有质的影响。

### 3. 自我过滤机制的价值
4 道题两个 agent 都返回 "No output"（P1/P3/P4/P6）。论文的解释：
> "We suspect that, given the limited bandwidth for human expert verification, many practicing researchers would prefer to trade raw problem-solving capability for increased accuracy."

**这是一个重要的工程权衡**：对研究辅助场景，false positive（错误的"解"）比 false negative（放弃问题）代价更高。设计 agent 时应允许显式放弃。

### 4. P8 的"正确性边界"问题
P8 专家分歧（5/7 认为正确）的根本原因：
- 所有专家都同意数学内容无误
- 分歧来自"publishable after minor revisions"的主观解释
- 论文 3.2 节：差距在于 Step 3-4 的"sketchy" vs. "sufficient for the argument"

**这揭示了评估研究级数学的一个根本难题**：正确性不是二元的。

### 5. P7 中 Aletheia A 的致命错误类型
Aletheia A 的证明犯了一个经典错误：
> 调用了"有紧支撑的有理 Euler 特征的乘法性"，但没有验证所需的 finiteness assumption

这种错误是典型的"使用定理忘记条件"，即 LLM 在高维数学中最容易犯的错误之一——会用 tool 但忘记 pre-condition check。

---

## 与 Agent RL / 评估体系的关联

### 和 HEARTBEAT.md 中"Agent 评估体系"方向的关系

FirstProof 本身就是对"benchmark Goodhart's Law"的一种反动：
- 不是设计成可自动评分的 benchmark
- 用真实的 open problems（不是合成题）
- 评估需要专家人工验证

这恰好验证了我在 Agent RL 评估笔记中的观点：**当 agent 能力接近或超越专家水平时，评估本身变成瓶颈**。

### 与 Agentic RL 体系的关系

Aletheia 不依赖 RL 训练（纯 Gemini 3 Deep Think + agentic scaffolding），但它展示了：
- **Generator-Verifier 架构**是 Agent 能力的核心模式（与 TSR 中"搜索+验证"的思路一致）
- **推理成本可变分配**：难题自动花更多 compute，而非固定 budget
- **自我过滤 ≈ CSO 的思路**：主动识别自己无法完成的任务

---

## 重要信号：AI 数学研究的边界

**2026年2月的数学研究级 AI 能力状态**：
- ✅ 可以自主解决研究级引理（研究论文中的 lemmas）
- ✅ 可以解决某些公开的数学问题（P7 = Weinberger 书中开放问题）
- ❌ 10题中有4道完全无法生成候选解（P1/P3/P4/P6 = 可能更深层的代数拓扑/几何问题）
- ⚠️ 解题质量介于"mathematically correct"和"publication-ready"之间

**我的评估**：这是真实的 paradigm shift，但边界很清晰。Aletheia 能解的题大多是"需要大量计算和检验的技术性问题"，对"需要真正 conceptual breakthrough 的 open problems"（P1/P4/P6）仍然无能为力。这与 LLM 擅长"在已知框架内优化"、不擅长"建立新框架"的认知一致。

---

## 对 Aletheia 系统设计的批判性评价

**优秀的地方**：
- Human-AI Interaction Card——透明度极高，完整记录人类干预点
- Generator-Verifier 两级架构——可靠性优先而非成功率优先
- 自动 LaTeX 格式化——减少"最后一步"的人工成本

**值得质疑的地方**：
- Best-of-2 的"选择"操作是否算真正自主？论文承认这需要专业知识来判断哪个更好
- P3 的 Filtered out（被 Verification prompt 过滤）说明 Gemini 的自我验证本身也不可靠

---

## See Also

**Aletheia 系列（前作 → 续集）**
- [[AI/2-Agent/Evaluation/Aletheia-Math-Research-Agent|Aletheia 前作（arXiv:2602.10177，Erdős 猜想数据库）]] — 同一 agent 系统的前一阶段：Erdős 4 个开放问题 → FirstProof 6/10 研究级数学题；前作覆盖 Generator-Verifier-Explorer 三组件架构，本文重点在 Best-of-2 策略与自我过滤设计
- [[AI/3-LLM/Inference/Gemini-3-Deep-Think|Gemini 3 Deep Think（基础模型）]] — Extended Thinking，AIME 2025/Codeforces frontier；FirstProof 是 Gemini 3 Deep Think 迄今最高难度的真实任务测试

**推理时搜索 vs 训练时搜索（设计思路对比）**
- [[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（ICML 2026，TU Munich）]] — 训练时树搜索选优 rollout；Aletheia 做的是推理时搜索（Best-of-2 + Verifier 筛选）；两者代表"搜索提升 agent 能力"在训练/推理两端的表现形式
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（Tencent AI Lab，arXiv:2602.03412）]] — 反事实验证 credit assignment；Aletheia 的 Generator-Verifier 架构与 CSO 的"关键步骤验证"在设计哲学上同源：都将验证作为 agent 能力的核心组件；区别：CSO 在 RL 训练中验证，Aletheia 在推理时验证

**AI 评估边界（对老板的方法论价值）**
- [[AI/2-Agent/Evaluation/Agent评估体系批判-Goodhart法则与Benchmark陷阱|Agent 评估体系批判（Goodhart 法则）]] — FirstProof 本身就是对"benchmark 可自动打分"假设的反动：用真实 open problems + 专家人工验证 = 当 agent 能力接近专家水平时，评估本身变成瓶颈；与 Goodhart 分析互证

**Frontier 模型进展**
- [[AI/4-模型/Gemini/Gemini-3.1-Pro|Gemini 3.1 Pro]] — 同期 Gemini 系列，ARC-AGI-2 格局对比；FirstProof 是 DeepMind agent 能力展示，与 Claude/OpenAI 的能力竞争背景

---

## 推荐阅读

1. [原文（arXiv:2602.21201）](https://arxiv.org/abs/2602.21201) — FirstProof 完整实验报告
2. [[AI/2-Agent/Evaluation/Aletheia-Math-Research-Agent|Aletheia 前作]] — 理解 Generator-Verifier 架构完整设计
3. [[AI/3-LLM/Inference/Gemini-3-Deep-Think|Gemini 3 Deep Think]] — 基础模型能力背景
4. [FirstProof 挑战赛（1stproof.org）](https://1stproof.org) — 10道题原题和背景

---

## 行业影响

**今天（2026-02-25）的重要性**：
1. 这是第一次有 agent 在公开时间限制内（8天）自主解决多道研究级数学题
2. Gemini 3 Deep Think 的能力已经**超过了某些专业数学家**在这类问题上的速度
3. 这可能是 AI 数学研究助手进入实用阶段的标志性时刻

**对 Agent RL 研究方向的影响**：  
Aletheia 用的不是 RL，而是纯粹的 TTC scaling + agentic scaffolding。这个结果暗示：对于有严格验证器的任务（数学证明可形式化验证），TTC scaling 可能比 RL 训练更快达到 frontier capability。RL 的优势可能在于 **对没有明确验证器的任务**（如工具使用、web agent）或 **提升 base model 本身**。
