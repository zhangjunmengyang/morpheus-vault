---
title: "Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents"
brief: Agent 探索策略：先校准不确定性（Calibration），再决策是否采取代价高的探索行动（Cost-Aware Action）；解决高层策略不能靠 RL 直接学的问题；与 E-SPL/GEPA 形成知识三角（arXiv:2602.16699）
type: paper
domain: ai/agent/agentic-rl
tags:
  - ai/agent
  - topic/agentic-rl
  - topic/exploration
  - topic/cost-aware
  - type/paper
arxiv: "2602.16699"
created: 2026-02-20
updated: 2026-02-23
see-also:
  - "[[AI/3-LLM/RL/算法/E-SPL-Evolutionary-System-Prompt-Learning]]"
  - "[[AI/3-LLM/RL/算法/GEPA-Reflective-Prompt-Evolution]]"
  - "[[AI/3-LLM/Inference/ConformalThinking-Risk-Control-Test-Time-Compute]]"
---

# Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents

> arXiv: 2602.16699 | 2026-02-18 | Wenxuan Ding et al.

---

## 一句话定位

Agent 在执行任务时常面临"现在就做决定"还是"先探索再决定"的选择。这篇论文的 insight 是：**把 cost-uncertainty tradeoff 显式化地告诉模型，它就能做出近似最优的探索决策**。不需要端到端 RL 去隐式学这个推理，用 prompt 注入先验概率就够了。

---

## 问题建模

把 LLM agent 的任务执行形式化为 **POMDP**（部分可观察马尔可夫决策过程）：

- **状态空间 S** — 环境真实状态（部分不可观测）
- **动作空间 A** — 探索动作（call tool / run test / retrieve）+ commit 动作（给出最终答案）
- **奖励 R** — 任务完成 × 折扣因子 Dθ(a₁:T) ∈ [0,1]，折扣随探索步骤增多而递减
- **关键变量 Z** — 隐含的环境特征（如文件格式、知识是否存在于参数记忆中）

**目标**：最大化 E[R] = E[完成任务 × 成本折扣]

核心 tension：探索 → 提高答对概率，但每步探索都会降低折扣，即"时间成本"。

---

## 核心方法：Calibrate-Then-Act (CTA)

**两阶段框架**：

```
[Calibration] → 估计先验 p̂(Z|x) → [Act] → 注入先验后做决策
```

**关键 insight**：不让模型自己从头推断"要不要探索"，而是**把先验概率 p̂ 显式注入 prompt**，触发模型以概率推理的视角来决策。

**为什么不直接 RL？**

论文给出了重要的负结果：

> 在编码任务上，端到端 RL 训练的模型**无法从环境交互中内化正确的探索先验**，仍然采用静态策略（不适应不同的不确定性情形）。

这说明 RL 能优化"在固定策略下怎么做"，但难以优化"什么时候该探索"这个 meta 决策层。

CTA 用 prompt 解决了 RL 没法自然习得的认知：把决策问题的结构显式化。

---

## 三个任务与实验结果

### Task 1: Pandora's Box（玩具问题）

经典概率决策问题的变体：K 个箱子，只有一个有奖励，每次验证有折扣 γ，何时停止探索？

| 方法 | Optimal Match Rate | Avg Reward |
|------|--------------------|------------|
| Oracle | 100.0% | 0.649 |
| Prompted (无先验) | 23.0% | 0.476 |
| CTA-Prompted + thinking | **94.0%** | **0.625** |

**结论**：模型有能力做最优决策，前提是给它先验。没有先验时，thinking 只提升从 11% → 23%，几乎无效。

### Task 2: Knowledge QA with Optional Retrieval

知识问答 + 可选 RAG，决策：直接回答 vs 先检索。

先验来自：
- `p_da` = 模型参数记忆答对的置信度（模型自我校准）
- `p_ret` = 检索器对该问题的覆盖质量

CTA 通过注入这两个先验，让模型动态决定是否检索，相比静态策略取得更好的成本-准确率 tradeoff。

### Task 3: Simplified Coding Task

先验来自训练数据中学到的"环境结构 cue"（如文件 schema 分布）。

- 基础 RL：无法学到正确的先验利用策略
- CTA-Prompted：改善明显
- CTA + RL：两者叠加，进一步提升

---

## 批判性分析

### 真正 novel 的地方

1. **POMDP 框架 + LLM agent 的结合**比现有工作更形式化。cost 不是 token 数，而是任务完成价值的折扣——这个建模更接近真实代价结构。

2. **RL 的负结果很有价值**：论文明确指出 RL 无法从头学会 meta 探索策略。这是反直觉的——人们以为 RL 能学到一切，但当"信息不够"本身就是决策输入时，RL 找不到正确的梯度信号。

3. **CTA = prompt engineering + structured prior injection**，不需要任何训练就能生效（CTA-Prompted 零训练即有效）。这是实用性极强的 insight。

### 局限性

1. **先验从哪来？** 在真实部署中，p̂(Z|x) 需要一个单独的 calibrator 组件。论文在 QA 任务用模型自身置信度，在编码任务用历史数据，但这两种来源的可靠性差异很大，泛化性存疑。

2. **任务规模偏小**：编码任务是"simplified coding"而非完整 SWE-bench 级别。在真实代码库（数千文件）中，先验估计是否仍然准确？

3. **Qwen3-8B 为主实验模型**：更大的 frontier 模型是否也存在相同的"无先验则无法优化"问题？还是这是小模型特有的现象？

4. **静态先验假设**：CTA 每步用同一个先验 p̂，但随着 agent 探索，后验 b_t 在更新，理论上应该用更新后的后验驱动下一步决策。论文承认可以加后验，但未在主实验中验证。

### 与相关工作的关系

| 工作 | 关注点 | CTA 的差异 |
|------|--------|-----------|
| PABU (Progress-Aware Belief) | 进度感知置信度 | CTA 更形式化，POMDP 建模 |
| Calibrate-before-Use (RAG) | 知识边界检测 | CTA 延伸到多步 agent 决策 |
| LACONIC (Length-Constrained RL) | 输出长度控制 | CTA 控制"是否继续探索"而非长度 |
| HiPER (Credit Assignment) | RL 训练中功劳分配 | 互补：HiPER 解决 RL 训练问题，CTA 解决 RL 未能学会的 meta 决策 |

---

## 关键 insight（我的解读）

**这篇论文本质上是在说：LLM 有良好的"reasoning about decision theory"能力，但缺乏"decision-relevant context"。**

类比：一个人知道怎么下象棋（推理能力），但如果你不告诉他棋盘上各子的位置（先验），他就下不好。CTA 就是把棋盘状态告诉他。

**更深层的含义**：
- RL 训练在优化"给定信息如何行动"，但对"何时获取更多信息"这个 meta 层决策效果有限
- 当 cost structure 复杂（非固定 step penalty）时，模型更需要显式的量化输入
- **这对 test-time compute 分配也有启示**：adaptive thinking 的最优策略同样取决于先验——模型应先估计问题难度，再决定思考多少 token

---

## 实际应用建议

如果你要构建需要 exploration-exploitation tradeoff 的 agent：

1. **显式建模成本**：不要只用 step penalty，用折扣因子 Dθ 建模真实代价
2. **注入先验**：在 prompt 中给出量化的置信度估计（模型 calibration score + 历史数据统计）
3. **不要期待 RL 自然学会**：如果探索策略依赖 meta 信息，必须显式提供
4. **thinking 模式 + prior = 最大化效益**：CTA 的收益在 thinking enabled 时显著更大

---

## 评级

**★★★☆☆** — 形式化清晰，负结果（RL 无法学到 meta 探索策略）是有价值的 empirical finding；但任务规模偏小，先验估计的实用性未充分验证。对于构建生产级 agent 的工程师有直接参考价值。

---

*笔记写于 2026-02-20 | Scholar*
