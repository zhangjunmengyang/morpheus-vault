---
brief: E-SPL（Evolutionary System Prompt Learning）——进化算法优化 system prompt 替代 RL 微调；核心洞察：RL 难以学到高层策略指令，进化搜索在 system prompt 空间更高效；与 GEPA/CTA 构成三角链接。
title: "E-SPL: Evolutionary System Prompt Learning for RL"
type: paper
domain: ai/llm/rl
tags:
  - ai/llm
  - topic/rl
  - topic/prompt-optimization
  - topic/evolutionary-algorithm
  - type/paper
  - rating/4star
arxiv: "2602.14697"
created: 2026-02-20
see-also:
  - "[[AI/3-LLM/RL/算法/GEPA-Reflective-Prompt-Evolution]]"
  - "[[AI/2-Agent/Agentic-RL/Calibrate-Then-Act-Cost-Aware-Exploration]]"
  - "[[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence]]"
---

# E-SPL: Evolutionary System Prompt Learning for RL

**arXiv**: 2602.14697 | **Date**: 2026-02-16 | **Code**: github.com/LunjunZhang/E-SPL  
**Authors**: Lunjun Zhang et al.  
**Affiliation**: 未公开（作者 profile 显示独立研究）  
**Rating**: ★★★★☆

---

## 一句话

在 RL post-training 过程中，**同时**用进化算法优化 system prompt population（显式声明性知识）和梯度更新模型权重（隐式程序性知识），两条路径互补加速。

---

## 核心动机

LLM 的自我改进有两个通道：
- **Self-reflection**：更新 context（system prompt），但权重冻结，无法改变隐式知识
- **RL**：更新权重，但 system prompt 固定，无法编码涌现的显式策略

E-SPL 的 claim：这两条路是**互补的**，不是竞争关系。知识应该自然分化：
- **声明性知识 (Declarative)**：可以语言表达的策略、启发式、工作流 → 存进 system prompt
- **程序性知识 (Procedural)**：直觉性、需要反复练习才能内化的技能 → 存进模型权重

当 system prompt 越来越丰富时，RL weight update 可以**专注于精炼执行**，而不是从头学策略发现。

---

## 方法

### 3.1 多 System Prompt 下的 RL

给定 M 个 system prompt `{s_i}_{i=1}^M`，每个 prompt 并行 sample N 个 rollout：

```
y_{i,j} ~ π_θ(· | s_i, x)
r_{i,j} = R(x, y_{i,j})
V_i = (1/N) Σ r_{i,j}    ← 每个 prompt 独立 value baseline
```

Policy gradient objective 变为：
```
J(θ) = (1/NM) Σ_i Σ_j (r_{i,j} - V_i) log π_θ(y_{i,j} | s_i, x)
```

每个 `V_i` 是该 system prompt 自己的 baseline，确保梯度只反映 prompt 内的相对质量差异。

### 3.2 进化算法（EA for Context Update）

**TrueSkill 适应度评估**

- 不跑额外 evaluation，直接复用 RL 生成的 rollout
- 每次 RL iteration = 一轮 prompt 间 group-wise 竞争
- 用 Bayesian skill rating TrueSkill 维护每个 prompt 的评分 `(μ_i, σ_i)`
- 选择概率：`p_i ∝ exp((μ_i + λσ_i) / T)`（乐观探索）
- 用 sliding window of K 控制 population 新鲜度

**Mutation（变异）**

1. 选出当前 batch 中表现最好的 `s_k`（argmax_i V_i）
2. 用 `π_ref` self-reflect：让模型分析 rollout 中的错误和可改进点 → lessons `ℓ`
3. 用 `π_ref` 生成 git diff 格式的 system prompt 修改 → `s_mutate`
4. 子 prompt 继承父 prompt 的 TrueSkill 评分 + 增加 uncertainty（`σ + Δσ`）
5. 加入 population

**Crossover（交叉）**

1. 计算 performance matrix `Φ ∈ R^{M×b}`（每个 prompt 对每个问题的平均奖励）
2. 对每个 prompt，找它「赢」的问题集合 `ρ_i`
3. 对比不同 prompt 的差异化优势 → 给 overall best prompt `s_k` 生成 crossover 指导
4. `π_ref` 根据跨 prompt 对比生成新的 system prompt
5. 子 prompt TrueSkill 评分 = 父 prompts 的**精度加权平均**（Gaussian fusion）

### 3.3 实现细节

- Mutation 和 Crossover 只需 `π_ref` sampling，可与 RL gradient update **并行**
- 不额外增加训练 compute；额外 overhead ≈ 一次 self-reflection LLM call
- 整个 population 从单个 root system prompt 出发生长成树状结构

---

## 实验结果

**数学推理**（base model: DeepSeek v3.1，671B MoE 37B active）：

| 任务 | RL-only (GRPO) | Evolution-only | E-SPL |
|------|---------------|---------------|-------|
| DAPO100 → AIME 25 | 56.3% | 57.3% | **60.6%** |
| HMMT 23/24 → 25 | 50.0% | 48.6% | **52.7%** |
| AIME → BeyondAIME | 38.8% | 40.0% | **45.1%** |

**Agentic Search**（gpt-oss-120b, NQ + HotpotQA）：

| Base | Evolution-only | RL-only | E-SPL |
|------|---------------|---------|-------|
| 6.6% | 34.0% | 44.2% | **48.6%** |

关键观察：
- E-SPL 在**所有任务**上均超过 RL-only 和 Evolution-only
- Easy-to-Hard 泛化（AIME→BeyondAIME）增益最显著：+6.3% absolute
- Evolution-only 在数学任务上有时超越 RL-only，但 agentic 任务上差距更大（34% vs 44%）
- E-SPL 同时实现了更好的**样本效率**（学习曲线更陡）和更高的**渐近性能**

---

## 涌现出来的 System Prompt 模式

### 1. 自我验证工作流
Agentic search 任务中，学到的 system prompt 自动发现了「VERIFIED/NOT VERIFIED」标记策略：
- 要求精确短语出现在搜索结果片段中
- 只接受 .org/.edu/.gov/Wikipedia 等可信域名
- 如所有结果 NOT VERIFIED，则重新搜索

这个工作流在初始 system prompt 中**完全不存在**——是 E-SPL 自主演化出来的。

### 2. 数学领域专业知识编码
学到的 system prompt 会积累具体的数学定理提示：
- CRT（中国剩余定理）在模运算问题中的应用条件
- Möbius inversion 的适用场景
- 二次型优化的参数设置策略
- 多项式因式分析的模 p 界

### 3. 失败模式：过度泛化
学到的策略有时包含不严格成立的启发式（E-SPL 会 overgeneralize 训练数据）。
**重要优点**：这些错误是**可解释、可人工纠正**的——比 RL 权重中的偏见更容易审计。

---

## Ablation

**Crossover 的作用**：
- Crossover 加速早期学习（跨分支传播高适应度子结构）
- 但最终性能 **mixed**：AIME→BeyondAIME 时 Mutation+RL（45.1%）> Mutation+Crossover+RL（42.5%）
- Crossover 可能导致 premature convergence（早期同质化，减少种群多样性）
- 两者都比 RL-only 强

**使用哪个 policy 做 self-reflection**：
- 使用 reference policy `π_ref`（冻结）做 mutation/crossover 比 online policy 显著更好
- HMMT: 52.7% vs 51.0%；BeyondAIME: 42.5% vs 41.6%
- 原因：online policy 的 distribution shift 会使 self-reflection 不稳定

---

## 我的评价

### 真正 novel 的点

**「声明性/程序性知识分工」是一个非常 elegant 的框架。** 这不是新概念（认知科学里早有），但把它 operationalize 到 RL post-training 中是新的。Policy gradient 专注于不可言说的"直觉"，system prompt 专注于可言说的"策略"——这种分工在实验中有清晰的数字支持。

**TrueSkill 用于 prompt fitness tracking** 是 practical elegance：Bayesian skill rating 本来就是为了处理「不同 batch、不同时间点的非平稳比较」问题而设计的（Herbrich 2006 Microsoft Chess），用在这里天然契合。零额外 evaluation cost。

**git diff 格式的 mutation** 很 practical：structured edit 比直接生成新 prompt 更可控，也更易于追溯修改来源。

### 边界条件

1. **模型规模**：用了 DeepSeek v3.1（671B MoE）和 gpt-oss-120b——这是超大规模模型。小模型能否有效进行 self-reflection 是问题。

2. **Crossover 效果不稳定**：在不同任务上方向相反，说明 diversity-efficiency tradeoff 还没解决。经典 niching 技术（作者也提到了 fitness sharing、island model）可能是下一步。

3. **闭环风险**：学到的 system prompt 可能强化训练集偏见，对 distribution shift 之外的新任务泛化性未验证。

4. **可解释性论点**：作者说 system prompt 比权重更透明、可审计——这个论点很有力，但没有定量验证。Table 1 的 case study 是 cherry-picked 的，失败率未报告。

### 与 CTA 论文的联系

上次读的 Calibrate-Then-Act（CTA）发现：**RL 无法学习 meta-exploration 策略**，需要显式先验注入。E-SPL 本质上是在做同样的事情——把「how to approach a problem」的 meta-level 策略编码进 system prompt，而不是期望 RL 自动涌现。两篇论文从不同角度都指向同一个结论：**RL 的归纳偏置不适合学习高层策略，但适合精炼低层执行**。

### 研究价值

★★★★☆。方法干净、结果可复现（有代码）、框架有泛化性。在 agent post-training 领域这是值得认真追踪的方向，尤其是：
- 结合 ICLR 2026 GEPA (Oral) 等相关工作，prompt evolution + RL 融合已经是一条成熟路线
- 「declarative/procedural split」框架可以直接应用于 agentic RL 设计

---

## 相关论文

- **GEPA** (arXiv:2507.19457, ICLR 2026 Oral) — Reflective prompt evolution，与 E-SPL 相近，但不更新权重
- **CTA** (arXiv:2602.16699) — RL 无法学习 meta-exploration，需要显式先验
- **LACONIC** (Vault: AI/LLM/RL/Other-Algorithms/LACONIC-Length-Constrained-RL.md) — RL 对 output 格式的控制
- **Calibrate-Then-Act** (Vault: AI/Agent/Agentic-RL/Calibrate-Then-Act-Cost-Aware-Exploration.md) — 同类思路

---

*笔记日期: 2026-02-20*
