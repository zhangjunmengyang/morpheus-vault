---
brief: "RLVR at the Edge of Competence——揭示 RLVR 只在模型能力边界附近有效（可以解但不稳定的任务）；太难 or 太简单均无法产生学习信号；对课程学习和任务难度调度有直接指导意义。"
title: "RLVR at the Edge of Competence"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - type/paper
  - rl/rlvr
  - rl/theory
created: 2026-02-16
arxiv: "2602.14872"
---

# RLVR at the Edge of Competence — 训练动力学理论

## 基本信息

- **论文**: On the Learning Dynamics of RLVR at the Edge of Competence
- **arXiv**: [2602.14872](https://arxiv.org/abs/2602.14872)
- **提交**: 2026-02-16
- **方向**: RLVR 理论、训练动力学

---

## 核心问题

RLVR（Reinforcement Learning with Verifiable Rewards）为什么能让模型在 long-horizon 推理上突破？只靠最终结果的 binary reward，怎么可能 overcome long-horizon barrier？这是一个真正的 open question。

---

## 主要贡献

### 理论框架

用 **Fourier analysis on finite groups** 分析 transformer 在 compositional reasoning tasks 上的 RL 训练动力学。

这是技术上的真正贡献：把群论的 Fourier 分析工具迁移到 RL for LLM 的理论分析中。

### 核心发现：难度谱平滑性决定学习轨迹

**Case 1: 难度谱不连续（abrupt discontinuities）**
- 学习出现 **grokking-type phase transitions**
- 长时间 plateau → 突然进步 → 再 plateau
- 原因：中间难度缺失，梯度信号不连续，模型无法 "接力"

**Case 2: 难度谱平滑（smooth difficulty spectrum）**
- 出现 **relay effect**：easier problems 上持续的梯度信号逐步提升模型能力，直到 harder problems 变得可解
- 结果：steady and continuous improvement，无明显 plateau

### 对 Data Mixture 的启示

RLVR 的有效性与数据课程设计强相关。**设计一个覆盖连续难度谱的数据集** 比单纯堆数据量更重要。

---

## 我的评价

### 真正 Novel 的地方

- 首次用严格的数学框架（Fourier analysis on groups）解释 RLVR 为何能突破 long-horizon barrier
- **relay effect** 机制的形式化是新的，之前只是经验观察
- 对 grokking 现象在 RL 语境下的重新解释

### Boundary 和局限

- 理论建立在 compositional reasoning tasks 上，泛化性待观察
- 合成实验为主，真实大模型上的验证不够充分
- Fourier 分析框架对 MoE 等复杂架构的适用性未讨论

### 对工程实践的启示

1. **数据课程设计**：构造 smooth difficulty spectrum，避免难度断层
2. **RLVR data mixture**：不要只用 hard problems 或只用 easy problems，要覆盖完整难度梯度
3. **诊断 plateau**：如果训练出现长期停滞，检查数据难度分布是否有 gap

---

## 连接到已有知识

- Grokking 现象之前在 mechanistic interpretability 文献中讨论（Power et al. 2022），这篇把它连接到 RLVR
- 与 GRPO/MEL 等方法的关系：这些方法的数据采样策略可以通过本文理论来理解
- ICLR 2026 的 test-time compute scaling 热点：更好地理解 RLVR 训练动力学是 scaling 的基础

---

## Tags

`#RLVR` `#training-dynamics` `#theory` `#grokking` `#data-curriculum` `#fourier-analysis`

---

## See Also

- [[AI/3-LLM/RL/算法/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] — 能力边界上的 curriculum：Goldilocks 选"刚好难"的任务，RLVR Edge 研究有效训练区间
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 2026 全景]] — Exploration 维度：能力边界与探索策略的关系
- [[AI/3-LLM/RL/Theory/MARS-Margin-Aware-Reward-Modeling-Self-Refinement|MARS]] — 同为 difficulty-aware：MARS 在 reward model 层，RLVR Edge 在 policy 层
-  — LLM 强化学习全图谱
