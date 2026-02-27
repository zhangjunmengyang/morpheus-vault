---
title: "Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training"
tags: [AgentRL, ToolUseRL, RAG, RewardShaping, CreditAssignment, SearchAgent, GRPO]
created: 2026-02-27
status: permanent
rating: ★★★★
arxiv: "2602.22576"
authors: "Tianle Xia et al. (Tencent)"
related:
  - "Search-R1-Retrieval-Augmented-Reasoning"
  - "[[AI/2-Agent/Agentic-RL/Search-R1plus-Tool-Use-RL-Ablation]]"
  - "[[AI/2-Agent/Agentic-RL/ASTRA-Automated-Tool-Agent-Training]]"
---

# Search-P1: Path-Centric Reward Shaping for Agentic RAG Training

> arXiv:2602.22576 | Tencent | 2026-02-26 | ★★★★

---

## 一句话定位

Search-R1 的三个问题：稀疏奖励、失败样本零梯度、收敛慢。Search-P1 用**路径级密集奖励**全部解决——Avg. ACC +7.7 over Search-R1（7B），工业 AD-QA +20.6。

---

## 问题：Search-R1 的三大缺陷

标准 Agentic RAG RL 训练（如 Search-R1）：

```
R_outcome = 𝟙[match(â, a*)]   — 二值奖励
```

这带来三个系统性问题：
1. **奖励稀疏**：中间推理质量被完全忽视，梯度信号极弱
2. **样本效率低**：部分正确的轨迹（搜索路径合理但最终答案错）贡献 zero reward
3. **收敛慢**：大多数样本得到相同二值奖励 → 梯度信息量低

---

## 方法：Path-Centric Reward Framework

### 核心思路

不只评价最终答案，而是**评价整个推理路径的结构质量**：

```
R_total = λ_p · R_path + λ_a · R_outcome + λ_f · R_format
                 ↓               ↓               ↓
          路径质量(密集)   结果(软打分)   格式遵循
```

最优超参：λ_p = 0.3，λ_a = 0.6，λ_f = 0.1

### 创新 1：显式 Planner（轨迹重构）

把隐式计划变显式：

```
标准：𝒯 = (r₁, a₁, o₁, ..., rₙ, aₙ, oₙ, r_final, â)
Search-P1：𝒯 = (p, r₁, a₁, o₁, ..., rₙ, aₙ, oₙ, r_final, â)
                ↑
         显式 planner，声明推理策略
```

两个目的：① 提供自我声明计划供执行一致性评估；② 让推理结构可观测

### 创新 2：参考 Planner 生成（离线）

```
1. 对每个训练样本 (q, a*)，用强模型（HY 2.0-Instruct）生成 K 条轨迹
2. 过滤出正确轨迹
3. LLM Voting：提炼出最优参考计划 P_ref = Vote({T_i | correct(T_i)})
4. 得到参考推理路径 ℛ_ref = {s₁, s₂, ..., sₘ}
```

这是**一次性离线成本**（90K 训练样本均值 1.91 次 LLM 调用），训练时缓存使用。

### 创新 3：Dual-Track Path Scoring（双轨评分）

**Track A — Self-Consistency（自我一致性）**：
```
S_self = r_planner × (n_exec_self / n_plan) × (n_exec_self / n_actions)
```
模型执行了多少自己计划的步骤？效率比防止 reward hacking（堆砌冗余步骤）。

**Track B — Reference-Alignment（参考对齐）**：
```
S_ref = (n_covered / |ℛ_ref|) × (n_covered / n_actions)
```
覆盖了多少参考路径的关键步骤？**order-agnostic**（顺序无关）匹配——尊重推理多样性。

**最终路径奖励取最大值**（而非加权平均）：
```
R_path = max(S_self, S_ref)
```
**关键设计**：当参考计划次优或模型发现更好策略时，self-consistency track 可以主导，不被低参考对齐分拉低。

### 创新 4：Soft Outcome Scoring（软结果打分）

不再是二值奖励，失败轨迹也有学习信号：

```python
R_outcome = {
    1.0                              if correct(â)
    0.8 × r_acc + 0.2 × r_reason   otherwise  # α = 0.8
}
```

- `r_acc`：部分答案正确性（0/0.5/1.0）
- `r_reason`：推理质量（独立于最终答案，0/0.5/0.8/1.0）

**效果分层**：
- General QA（单跳）：+1.1~1.5%
- Multi-Hop QA：+3.3~3.7%
- 工业 AD-QA：**+8.8~11.0%**（最复杂场景受益最大）

---

## 实验结果

### 主实验（7 个公开 QA 基准 + 工业 AD-QA）

| 方法 | 7B Avg. | 3B Avg. | AD-QA(7B) |
|------|---------|---------|-----------|
| Search-R1 | 39.6 | 33.6 | 65.6 |
| HiPRAG | 42.9 | 36.6 | 75.6 |
| **Search-P1** | **47.3** | **41.5** | **86.2** |

**+7.7 over Search-R1（7B），+7.9（3B）**——跨模型规模一致。

### 消融：双轨评分各自贡献

| 配置 | 7B Avg. |
|------|---------|
| Search-P1 Full | 47.3 |
| w/o Reference-Alignment | 42.0（-5.3）|
| w/o Self-Consistency | 44.2（-3.1）|
| Search-R1 Baseline | 39.6 |

Reference-Alignment 贡献更大（-5.3 vs -3.1），因为参考路径提供了外部指导信号，在 multi-hop 场景尤其关键。

### 效率提升

- **训练收敛**：Search-P1 在 60 步内达到 Search-R1 最终精度（~40%），Search-R1 需 150+ 步
- **推理效率**：Search-P1 在成功/失败案例间 turn 数一致；Search-R1 在 multi-hop 上失败案例多用 60% 的 turn

### RL 算法比较

GRPO vs PPO：GRPO 略高精度，PPO 训练稳定性更好（方差更低）。
**Path-centric rewards 在所有 model+RL 组合下均有一致提升** → 正交于算法选择。

---

## 关键洞察

### 1. "失败不等于零价值"

Search-P1 最核心的认识：一条最终答案错误的轨迹，如果推理路径是正确的（覆盖了正确的搜索步骤），它依然应该得到正奖励。这个思想可以推广：**在任何 long-horizon agent RL 中，把"部分成功"量化为奖励信号是提升样本效率的关键**。

### 2. max(S_self, S_ref) 而非 weighted average

这不是工程细节而是哲学选择：承认推理的多样性。当模型发现了一条不同于参考路径但同样正确的路径时，不惩罚它。这与 GiGPO 的 "order-agnostic step coverage" 精神一致——都在对抗 reward 对特定执行顺序的偏好。

### 3. 离线 Planner 生成的成本控制

参考 planner 是在训练前一次性生成的（90K 样本，均值 1.91 LLM 调用），训练时缓存使用。这解决了训练时调用 LLM 的延迟问题，同时保留了强模型的规划质量。与 FineWeb-Edu 的"强模型标注 → 缓存推理"模式相同。

### 4. 与 Search-R1++ 的定位差异

| 维度 | Search-R1++ (2602.19526) | Search-P1 (2602.22576) |
|------|-------------------------|------------------------|
| 核心问题 | Fast Thinking 崩溃 / F1 answer avoidance | 稀疏奖励 / 失败样本浪费 |
| 解法 | 修复 reward 信号 (F1+) + 算法选择 | 增加路径级密集信号 + 软打分 |
| 互补性 | ✅ 正交，可组合 | |

两篇论文可以叠加：Search-R1++ 解决"奖励质量"问题，Search-P1 解决"奖励密度"问题。

---

## 局限与批判

**LLM 评估器依赖**：Dual-Track 评分需要在训练时调用 LLM 评估器（推理时不需要）。实验显示 Qwen3-8B 评估器使精度下降 3.2 点——说明路径质量评估的可靠性依赖于足够强的评估模型。

**离线参考 planner 的质量上限**：参考路径由强模型生成，其质量受限于 teacher model。在专业领域（如 AD-QA），可能需要领域特定的参考路径。

**"正确参考路径"的假设**：LLM Voting 假设多条成功轨迹的公约数就是最优策略，但在有多种等效解法的问题上，这个假设可能过于保守。

---

## 在 Credit Assignment 谱系中的位置

```
Search-P1 是 Tool Use RL 中 credit assignment 的"路径级"解法：

轨迹级（Search-R1 binary）→ 路径级（Search-P1 path-centric）→ 步骤级（AgentPRM, GiGPO）

特殊点：Search-P1 同时做了 credit redestribution（失败轨迹→软奖励）
      和 dense signal（路径覆盖→中间信号）
      但没有做 step-level granularity（仍然是路径聚合）
```

与 SELAUR（2602.21158）对比：两者都在"从不确定/失败中提取信号"，但 SELAUR 用 token 级不确定性，Search-P1 用路径结构覆盖。

---

## See Also

- Search-R1-Retrieval-Augmented-Reasoning — 直接被超越的 baseline，arXiv:2503.09516
- [[AI/2-Agent/Agentic-RL/Search-R1plus-Tool-Use-RL-Ablation]] — 并行工作，解决 reward 质量问题（F1+/REINFORCE稳定性）
- [[AI/2-Agent/Agentic-RL/ASTRA-Automated-Tool-Agent-Training]] — MCP 工具图 + verifiable RL，不同场景的 tool use RL
- [[AI/2-Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards]] — token 不确定性 reward shaping，orthogonal 角度
- [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization]] — 步骤级 credit，order-agnostic 共同设计原则

