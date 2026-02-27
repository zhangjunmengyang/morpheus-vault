---
title: "GiGPO: Group-in-Group Policy Optimization for LLM Agent Training"
brief: "NTU + Skywork AI，NeurIPS 2025。解决 group-based RL 在多步 Agent 中的 credit assignment 问题：通过 Anchor State Grouping 机制，在无需额外 rollout 和 critic 的前提下实现 step-level 细粒度 advantage 估计。两级 advantage（episode-level macro + step-level micro）相加，ALFWorld +13.3%、WebShop +10.6% over GRPO（1.5B），QA 任务 42.1%（3B）和 47.2%（7B）。内存开销与 GRPO 完全相同。"
sources:
  - "arXiv:2505.10978v3 (NeurIPS 2025)"
  - "作者: Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An（NTU + Skywork AI）"
  - "代码: https://github.com/langfengQ/verl-agent"
tags: [agent, RL, credit-assignment, GRPO, multi-turn, agentic-RL, NeurIPS-2025]
rating: 5
status: permanent
date: 2026-02-22
---

# GiGPO: Group-in-Group Policy Optimization

> 核心问题一句话：GRPO 等 group-based RL 在单轮任务（数学/代码）很有效，但直接用到多步 Agent 时，整条轨迹只有一个 advantage 值——无法区分"哪一步走错了"。GiGPO 用"找重复状态"这一巧思，零成本地引入了 step-level credit signal。

## 动机：为什么 GRPO 在多步 Agent 中失效？

$$\text{GRPO advantage: } A(\tau_i) = \frac{R(\tau_i) - \mu}{\sigma}$$

**问题**：整条轨迹的所有 token 共享同一个 $A(\tau_i)$。在 50 步的 ALFWorld episode 中，第 3 步的关键错误和第 47 步的完美操作，得到的梯度方向完全相同。

**更根本的问题**：

```
选项A：为每个状态额外采样新 action → 计算量 ×N（N是 group size），不可行
选项B：用 Critic 估计 value → 引入额外网络，破坏 critic-free 优势
选项C（GiGPO）：利用多条轨迹中天然出现的重复状态 → 零额外成本
```

**关键观察**：在 Agent 环境中，多条轨迹从同一初始状态出发，自然会多次经历相同的中间状态（revisit 同一房间、同一网页、同一游戏场景）。这些重复的状态就是免费的"对照组"——在同一状态下，不同轨迹选择了不同的 action，自然可以比较。

---

## 算法核心：两级 Advantage 结构

### Level 1 — Episode Relative Advantage（宏观，从 GRPO 继承）

采集 $N$ 条完整轨迹，计算 episode 级 advantage：

$$A^E(\tau_i) = \frac{R(\tau_i) - \text{mean}(\{R(\tau_j)\}_{j=1}^N)}{F_{\text{norm}}(\{R(\tau_j)\}_{j=1}^N)}$$

- $F_{\text{norm}} = \text{std}$：标准 GRPO 归一化
- $F_{\text{norm}} = 1$：Leave-One-Out（LOO）无偏估计，对难任务更稳定

**功能**：捕捉整条轨迹的全局质量，提供稳定的宏观训练信号。

### Level 2 — Step Relative Advantage（微观，GiGPO 的核心创新）

#### Anchor State Grouping 机制

1. 在 $N$ 条轨迹中识别所有**重复出现的环境状态** $\mathcal{U} = \{\tilde{s}_1, \tilde{s}_2, ..., \tilde{s}_U\}$
2. 把每个重复状态作为"锚点"，收集所有从该状态出发的 (action, return) 对：

$$G^S(\tilde{s}) = \{(a_t^{(i)}, R_t^{(i)}) \mid s_t^{(i)} = \tilde{s}, 1 \leq i \leq N, 1 \leq t \leq T\}$$

3. 在每个 step-level group 内计算相对 advantage：

$$A^S(a_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}(\{R_t^{(j)} \mid \text{in } G^S(\tilde{s})\})}{F_{\text{norm}}}$$

其中 $R_t^{(i)} = \sum_{k=t}^T \gamma^{k-t} r_k^{(i)}$ 是从 step $t$ 开始的折扣回报（捕获长期影响，而非 immediate reward）。

**实现细节**：用 hashmap 做状态匹配，时间复杂度 $O(NT)$，额外时间开销 < 0.002%。

#### 一个直觉性例子（WebShop 购物）

```
搜索结果页（anchor state）
├── 轨迹 τ₁: 先点"第2个商品"(错)→ 返回 → 点"第1个商品"(对) → 成功
│   折扣回报: R(点2nd) = 低（经过了错误步骤）; R(点1st) = 高
├── 轨迹 τ₂: 点"下一页" → 迷失 → 失败
│   折扣回报: R(点NextPage) = 0

step-level 排序: A^S(1st Item) > A^S(2nd Item) > A^S(Next Page)
```

GRPO 在这里只能看到 τ₁ 成功、τ₂ 失败——但无法知道τ₁的第一步（选错商品）也是负的贡献。GiGPO 可以。

### 两级合并

$$A(a_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(a_t^{(i)})$$

默认 $\omega = 1$，无需调参。最终 PPO-clip 目标：

$$\mathcal{J}_{\text{GiGPO}}(\theta) = \mathbb{E}\left[\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T \min\left(\rho_\theta A, \text{clip}(\rho_\theta, 1\pm\epsilon)A\right)\right] - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 实验结果

### ALFWorld（具身任务规划，最多 50 步）

| 方法 | 1.5B All | 7B All | 备注 |
|------|---------|--------|------|
| GPT-4o（prompting）| — | 48.0% | 闭源大模型 |
| Gemini-2.5-Pro（prompting）| — | 60.3% | 闭源最强 prompting baseline |
| GRPO | 72.8% | 77.6% | 无 step-level credit |
| PPO（有 critic）| 54.4% | 80.4% | 需要额外 critic 网络 |
| **GiGPO** | **86.1%** | **90.2%** | ✅ critic-free，step-level credit |

**GiGPO vs GRPO**：1.5B +13.3%，7B +12.6%
**GiGPO vs PPO**：1.5B 超越 PPO 31.7%，7B 超越 PPO 9.8%——还无需 critic！

GiGPO (7B) 的 90.2% 甚至**超过 Gemini-2.5-Pro 30%**。这是 RL 训练的力量体现。

### WebShop（网页购物，真实 HTML 环境）

| 方法 | 1.5B Score / Succ | 7B Score / Succ |
|------|------------------|-----------------|
| GRPO | 75.8 / 56.8% | 79.3 / 66.1% |
| **GiGPO** | **83.5 / 67.4%** | **86.2 / 75.2%** |

GiGPO vs GRPO：Succ +10.6%（1.5B）/ +9.1%（7B）

### Search-augmented QA（多步工具调用）

| 模型 | Search-R1 | StepSearch | GiGPO |
|-----|-----------|-----------|-------|
| 3B avg | 32.5 | — | **42.1** |
| 7B avg | 38.5 | — | **47.2** |

**工具效率**：7B 模型单跳 QA 平均只需 ~0.9 次工具调用，比其他方法更简洁。

### Normalization Factor 的任务依赖性

- **困难任务**（Look、Pick2、WebShop）：$F_{\text{norm}} = 1$（LOO）更好——std 归一化会对高方差组过度放大梯度
- **一般任务**：两种 norm 差别不大
- **实践建议**：默认用 $F_{\text{norm}} = 1$，观察到训练不稳定再换 std

---

## GiGPO 的工程优势

| 属性 | PPO | GRPO | GiGPO |
|------|-----|------|-------|
| Critic 网络 | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| 额外 rollout | ❌ | ❌ | ❌ |
| 显存开销 | +critic 参数 | baseline | **= GRPO** |
| 时间开销 | 正常 | 正常 | **+<0.002%** |
| Step-level credit | ✅（通过 value net）| ❌ | **✅（anchor state grouping）** |

**GiGPO = GRPO 的 drop-in 升级**：同样的 rollout，同样的显存，只增加了一个 hashmap 分组步骤。正交于现有 GRPO 改进（DAPO、DEEP-GRPO 等），可以组合使用。

---

## 批判性评价

**真正 novel 的地方**：

1. **Anchor State Grouping 是一个 elegant insight**：把"多条轨迹的 state reuse"这个看似是低效的现象（agent 转圈、重复状态），转变为免费的 credit signal 来源。这是对 Agent 训练环境特性的深刻理解。

2. **两级 advantage 的加法组合理论上合理**：$A^E$ 提供全局稳定信号，$A^S$ 提供局部细粒度信号，不互相干扰。

3. **结果扎实**：3 个 random seed 平均，多个 model size，多个 benchmark，比较对象包括 PPO 和多个 group-based baselines。

**我的质疑**：

- **状态重复率的假设是否成立？** 在 ALFWorld 这类结构化家庭环境里，agent 经常 revisit 同一房间，anchor state 很丰富。但在 open-ended web 环境（如真实 SWE-bench 代码任务）里，两条轨迹经历的状态可能几乎从不重叠——这时 step-level group 几乎是空的，GiGPO 退化为 GRPO。论文没有讨论这个边界条件。

- **State 相等的定义**：论文对"相同状态"的定义依赖环境。ALFWorld 是离散状态，WebShop 是网页 HTML，QA 任务用 "similarity ≥ 0.9（最长公共子序列）"——这个相似度阈值如何影响结果？没有充分消融。

- **ω=1 是否需要调参？** 论文说"no further tuning"，但两个 advantage 的量级是否天然匹配？$A^E$ 是跨 episode 的，$A^S$ 是跨 step 的，它们的方差可能相差很大。

**总体评价**：方法简洁有力，结果令人信服。Anchor State Grouping 是真正的创新，而不是边际改进。作为 NeurIPS 2025 工作完全实至名归。

---

## 与 Credit Assignment 专题的关系

GiGPO 在 credit assignment 谱系中的位置：

```
方案                   类型              依赖
─────────────────────────────────────────────────────────
GRPO baseline         trajectory-level  无额外依赖
PPO with critic       step-level        critic 网络
AgentPRM (2502.10325) step-level        MC rollout 估计
LOOP (2502.01600)     hybrid            value-free PPO
GiGPO (本文)         step-level        anchor state grouping（零额外成本）
iStar (2509.19199)    step-level        trajectory DPO → implicit PRM
MIG (2602.01034)      step-level        信息论（Marginal Information Gain）
```

**GiGPO 的独特地位**：是所有 step-level credit assignment 方案中**工程成本最低的**——不需要额外 rollout、不需要 critic、不需要标注，只要环境里会出现状态重复就能工作。

---

## 推荐阅读

**原始论文**：
- [arXiv:2505.10978](https://arxiv.org/abs/2505.10978) — GiGPO（NeurIPS 2025）
- [GitHub: verl-agent](https://github.com/langfengQ/verl-agent) — 代码（基于 verl 框架）

**相关 Vault 笔记**：
- [[AI/2-Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — Credit assignment 全景：GiGPO/AgentPRM/LOOP/MIG/MCTS/iStar
- [[AI/2-Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL|iStar（2509.19199）]] — 另一种 step-level credit assignment，基于 DPO 等价，支持 unverifiable reward
- [[AI/2-Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — 在 GiGPO 基础上加 phase-aware MoE，ALFWorld 再 +7.7%
- [[AI/3-LLM/RL/Theory/RL-Signal-Granularity-Causal-Structure-Principle|RL 信号粒度与因果结构匹配原则]] ⭐ — W 层元命题：GiGPO 是四路印证之一（anchor state = Agent 任务的最小因果单元）
- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]] — 理论基础：GiGPO 是 GRPO 在 Agent 场景的根本性改造（episode-level advantage → step-level advantage via anchor state grouping）

---

## 落地应用

**工程实践**：
- GiGPO 可以作为 GRPO 的直接替换：只需在 advantage 计算阶段加一步 hashmap 聚合
- 最适合的场景：结构化环境、有状态复现的长 horizon 任务（ALFWorld、WebShop、工具调用链）
- 最不适合：开放式、状态几乎不重复的任务（纯生成、代码补全单步）

**面试高频问法**：
- "如何在不引入 critic 的情况下实现 step-level credit assignment？" → GiGPO: anchor state grouping，利用轨迹中的自然状态重复
- "GRPO 在 Agent 训练中的主要问题是什么？" → 所有 step 共享 episode-level advantage，无法区分好坏 step
- "GiGPO 和 PPO 相比有什么优势？" → 同等或更好的性能，无需 critic，显存和 GRPO 一样

**实用配方**（基于论文）：
```python
# 训练 ALFWorld/WebShop 时
# Group size N = 8
# ω = 1 (step weight)
# F_norm = 1 (LOO，适合困难任务)
# γ = 1.0 (sparse reward 时折扣因子设 1)
# anchor state 匹配：完全相同（离散状态环境）
#                   或 similarity ≥ 0.9（连续/文本状态环境）
```


---

## See Also

- [[AI/2-Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER（ICML 2026）]] — **正交升维**：GiGPO 做 step-level（anchor grouping within segment），HiPER 做 segment-level（HAE between segments）；两者理论可组合，GiGPO+HiPER 或许是 Credit Assignment 的最强组合
- [[AI/2-Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — 完整谱系（trajectory/segment/step 三层）
- [[AI/2-Agent/Agentic-RL/MIG-Step-Marginal-Information-Gain-Credit-Assignment|MIG]] — 信息论视角 step-level CA（与 GiGPO 正交：anchor grouping vs 信息增益）
- [[AI/2-Agent/Agentic-RL/Tree-GRPO-Tree-Search-LLM-Agent-RL|Tree-GRPO（ICLR 2026）]] — 树搜索 + 双层 advantage（与 GiGPO 同问题，不同粒度，可组合）
- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]] — GiGPO 的理论基础
- [[AI/2-Agent/Agentic-RL/SHARP-Shapley-Credit-Multi-Agent-Tool-Use-RL|SHARP（ICML 2026，arXiv:2602.08335）]] — **正交互补（横向维度）**：GiGPO 解决纵向（单 agent 不同时间步）credit assignment，SHARP 解决横向（多 agent 之间）credit assignment；两者从不同维度完善 multi-agent 系统的 CA 图景
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（Tencent AI Lab+HKU，arXiv:2602.03412）]] — **互补信号来源**：GiGPO 从成功轨迹的 anchor state 归因（在线 RL），CSO 从失败轨迹反事实验证归因（离线 DPO）；两种信号互补，联合使用可覆盖成功和失败轨迹的 credit 信息
- [[AI/2-Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards|SELAUR（arXiv:2602.21158）]] — **失败信号激活**：GiGPO 精化成功轨迹的 step-level credit，SELAUR 用 token 不确定性激活失败轨迹的 reward 信号（原本 =0）；两者**正交互补**：联合使用覆盖成功+失败轨迹的完整 credit 图景，ALFWorld/WebShop 超越单独 GiGPO
