---
title: "SquRL: Beyond Static Pipelines — 动态 Workflow 构建 RL for Text-to-SQL"
brief: "ICML 2026 投稿：用 RL 动态构建 Text-to-SQL workflow（非静态预定义 pipeline）；Actor Masking 限制 action space；在 Spider/BIRD 上 vs 静态 workflow 显著提升（arXiv:2602.15564）"
date: 2026-02-17
updated: 2026-02-22
arxiv: "2602.15564"
venue: "ICML 2026 (投稿)"
domain: AI/Agent
tags:
  - text-to-sql
  - dynamic-workflow
  - reinforcement-learning
  - agentic-rl
  - tool-use
  - actor-masking
  - type/paper
rating: 3
status: permanent
---

# SquRL: Beyond Static Pipelines — 动态 Workflow 构建 RL

> **评分**: ★★★☆☆  
> **一句话**: 形式化证明动态 workflow 选择优于任意静态 pipeline，用 RL 教 LLM 按 query 难度自适应选工具链。

---

## 基本信息

- **arXiv**: 2602.15564
- **提交时间**: 2026-02-17
- **Venue**: ICML 投稿（Machine Learning 标注）
- **机构**: 未具体标注（通讯作者 Wei Xu）
- **代码**: https://github.com/Satissss/SquRL
- **任务**: Text-to-SQL → Dynamic Workflow Construction

---

## 核心问题

**为什么 Text-to-SQL 在 real-world 失败？**

不是 LLM 能力问题，是 **fixed-workflow assumption** 的结构性限制。

现有方法（DIN-SQL, DAIL-SQL, CHESS...）都假设一个静态 pipeline 能泛化到所有 query。但现实中：
- 简单查询不需要复杂 pipeline（引入冗余开销）
- 复杂查询没有哪个单一 workflow 能覆盖所有 challenge
- 不同 workflows 在不同 queries 上有**互补的 success regions**

---

## 理论基础（核心贡献）

### Theorem 3.1（形式化证明）

设 $\Omega = \{W_1, ..., W_K\}$ 为有限 workflow 集，$q \sim \mathcal{D}$ 为 query 分布：

$$\text{EX}_{\text{static}} = \max_{W \in \Omega} \mathbb{E}_{q \sim \mathcal{D}}[Y_W(q)]$$

$$\text{EX}_{\text{dynamic}} = \mathbb{E}_{q \sim \mathcal{D}}[\max_{W \in \Omega} Y_W(q)]$$

**结论**：$\text{EX}_{\text{dynamic}} \geq \text{EX}_{\text{static}}$，且 $\Delta = 0$ **当且仅当**存在某个 $W^*$ 的 success region 几乎覆盖所有 workflows 的并集。

**意义**：当 workflows 存在互补性（实际中几乎总是如此），任何静态策略都存在原则性上界限制，动态选择是唯一突破方式。

### 经验验证

Oracle 评估（每个 query 选最快的正确 workflow）在 SynSQL 上达到 **81.5% EX**，显著超过任何单一静态 workflow，且 runtime 也更低。

**关键发现**: bottleneck 不是 LLM 能力，是 workflow 设计。

---

## 三个核心抽象

```
Actor     = 具体功能模块（schema linking, SQL generation, refinement 等）
Template  = 抽象 workflow 骨架（角色序列，不绑定实现）
Workflow  = Template × Actor 的具体实例化
```

**示例**（DIN-SQL）：
- Template: `[parser, decomposer, generator, optimizer]`  
- Workflow: `[dinsql-parser, dinsql-decomposer, dinsql-generator, dinsql-optimizer]`

组合空间 $\Omega = \{f_{\text{match}}(T, A) \mid T \in \mathcal{L}_{\text{template}}, A \subseteq \mathcal{L}_{\text{actor}}\}$ 是 combinatorial 的。

---

## SquRL 方法

### Stage 1: SFT（学习有效 workflow 格式）

- 目标：让模型生成 **valid workflow 结构**，而非直接生成 SQL
- 训练数据：SynSQL + Spider + BIRD 的 workflow-level supervision
- 策略：对每个 query 从简到繁探索 template，取第一个正确的为 supervision signal
- 分离关注点：**结构选择** 与 **SQL 生成** 解耦

### Stage 2: RL（优化 workflow 选择策略）

**Rule-based Reward（5 层级联）**：
```
R_f       = ±0.5  （格式是否合法，<think>...<answer>[...]）
R_timeout = -0.5   （执行超时惩罚，5min timeout）
R_e       = ±1    （SQL 是否可执行）
R_r       = ±1.5  （执行结果是否正确，最重要）
R_t       = 0.5 × (timeout - time) / timeout  （效率奖励，仅在正确时给）
```

**Dynamic Actor Masking**（解决 training collapse）：
- 训练时每个 actor 以概率 $r$ 保留，构造 reduced actor pool
- 强迫模型探索不同 workflow 组合，而非反复选高 reward 的模式
- 对复杂 query 用更高的 retention rate（减少失去正确解的可能）

**Pseudo Reward**（加速训练）：
- 用 LLM-as-judge 替代部分昂贵的真实执行 reward
- 节省执行开销，接受有限噪声换取更快的训练反馈

---

## 实验结果

**基础框架**: Qwen2.5-7B-Instruct + VERL + Squrve 后端

**主要结论**：
- SynSQL 各难度级别（Simple/Moderate/Complex/Highly Complex）上 SquRL 均超过最优静态 baseline
- **性能 gap 随 query 复杂度增加而扩大**（最难子集上 gain 最大）
- 跨数据集（Spider/BIRD）保持泛化

**Oracle 上界**: EX = 81.5%（动态选择的理论极限）
**SquRL 实现**: 接近 oracle 上界，显著超过最优单一静态 workflow

---

## 我的分析

### 真正 novel 的部分

**Theorem 3.1** 是真正的贡献。把 "动态比静态好" 这个工程直觉转化为可证明的数学命题，并给出条件 $\Delta = 0$ iff coverage condition。这类理论证明在 agent system 论文中很少见。

### 机制设计的亮点

**Dynamic Actor Masking** 很有创意。它不是 standard dropout——是对 action space 做 stochastic perturbation，本质是 forced exploration through environment randomization。类似 sim-to-real transfer 里的 domain randomization，但用在工具选择上。

**级联 reward** 的设计也体现了对 Text-to-SQL pipeline 的深刻理解：先验证格式（最廉价），再验证可执行性（中等代价），最后验证正确性（最昂贵）。早退机制节约计算。

### 边界与局限

1. **Workflow 空间是手工设计的**：Actor/Template 库是人工定义的，不是端到端学习的，这是关键假设。如果 workflow space 设计得不够好，再好的 selector 也没用。

2. **迁移性存疑**：实验只在 Text-to-SQL 上，动态 workflow 选择的理论框架是通用的，但实际上 SFT 和 reward 设计都是 task-specific 的。

3. **规模未验证**：Qwen2.5-7B，没有测试在更大模型上是否有 diminishing returns。

4. **与 RAG / retrieval 结合缺失**：真实 NL2SQL 场景 OOD 问题很多来自 schema，没有讨论如何处理 schema-level generalization。

### 与已有工作的关系

| 维度 | SquRL | AgentConductor | FlowSteer |
|------|-------|---------------|-----------|
| 核心动作 | workflow 选择 | topology 生成 | operator 编排 |
| RL 对象 | selector policy | orchestrator | coordinator |
| 任务 | Text-to-SQL | 竞赛代码 | 数学解题 |
| 空间 | 有限组合 | DAG（连续） | DAG（有限） |

三者都是「用 RL 学 agentic 决策」，但 granularity 不同：SquRL 选工具链，FlowSteer 编排步骤，AgentConductor 设计通信图。

### 对「静态 pipeline 时代终结」的判断

这篇论文的核心主张——静态 pipeline 是瓶颈而非模型能力——我认为**在 structured task 上是对的**。Text-to-SQL、代码生成、数学推理这类有明确 verifiable signal 的任务，workflow 的组合效应确实存在，且 RL 是正确的学习方式。

但对于 open-ended generation，这个 claim 就不成立了——那里根本没有可定义的 workflow space。

---

## Tags

#text-to-sql #dynamic-workflow #reinforcement-learning #agentic-rl #tool-use #actor-masking #pseudo-reward #icml-2026

---

## See Also

- [[AgentConductor-Topology-Evolution|AgentConductor]] — topology evolution 的兄弟工作：同为"用 RL 学 agentic 决策"，AgentConductor 做 agent 通信拓扑，SquRL 做工具链选择
- [[FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer CWRPO]] — workflow 编排 RL 的另一视角：FlowSteer 编排 operator 步骤，SquRL 选工具链组合；同为 DAG-based Agentic RL
- [[Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] — 全景参考，SquRL 属于 Tool-Use RL 分支
- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — SquRL 使用 GRPO 变体训练 selector policy
- [[AI/2-Agent/目录|Agent MOC]] — 智能体研究全图谱
