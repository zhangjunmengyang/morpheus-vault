---
title: "CSO: Verified Critical Step Optimization for LLM Agents"
brief: "CSO 从失败轨迹出发做精准 credit assignment：用 PRM 定位 policy 弱点步骤，用 expert model 生成候选替代动作，用 policy 自身继续 rollout 验证成功——只在 16% 的关键步骤做 DPO。GAIA-Text-103 +37% over SFT，8B 超 GPT-4.1。核心洞察：高熵步骤原则推广到 Agent——只有工具选择分叉点等关键决策步骤决定成败，大多数执行步骤不需要监督。"
arxiv: "2602.03412"
date: 2026-02-25
rating: ★★★★☆
authors: "Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, Haitao Mi, Dong Yu"
affiliation: "Tencent AI Lab + University of Hong Kong + Northwestern University"
tags:
  - ai/agent/agentic-rl
  - credit-assignment
  - dpo
  - counterfactual
  - post-training
  - type/paper
sources:
  - "arXiv:2602.03412 — Tencent AI Lab / HKU / Northwestern University"
  - "代码：https://github.com/kiaia/CSO"
related:
  - "[[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]]"
  - "[[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]]"
  - "[[AI/Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL|iStar]]"
  - "[[AI/Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER]]"
  - "[[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|Tool-Use-RL 训练专题]]"
---

# CSO: Verified Critical Step Optimization for LLM Agents

---

## TL;DR

CSO 解决了 Agent 后训练中 credit assignment 的核心难题：从失败轨迹出发，用 PRM 定位候选关键步骤，用 expert model 生成替代动作，用 policy 自身继续执行到验证成功，最终只在 **16% 的步骤**上做 DPO 训练。GAIA-Text-103 +37%（相对 SFT 基线），8B 模型达到 GPT-4.1 水平。核心洞察：**不是所有步骤都重要，关键在于找到"能翻转结果的分叉点"**。

---

## 一、问题定位：三种失败的 credit assignment 方法

```
问题：Agent 长 horizon 任务的后训练信号如何精确到步骤级？

现有方法的三种缺陷：

1. 轨迹级优化（ETO / MiroThinker）
   → 整条轨迹统一打分
   → 失败轨迹中的合理步骤被不公平惩罚
   → 成功轨迹中的次优步骤被错误强化

2. 步骤级估计（AgentRPM / Step-DPO）
   → 中间步骤的 PRM 分数有系统性噪声
   → 每步都监督 = 大量计算浪费在不重要的步骤上

3. 混合方法（IPR — Monte Carlo rollout）
   → 每步采样完整 MC rollout 估计价值
   → 复杂任务计算成本"prohibitive"
```

**CSO 的核心洞察**：受 RLVR 高熵 token 研究启发——**只有少数高熵 token 驱动了 RL 的真正有效学习**。类比到 Agent：只有少数关键决策点（工具选择、查询构建）决定任务成败，其余步骤是"执行"而非"决策"。

---

## 二、方法：六步流水线

### 整体框架

```
[失败轨迹收集]
     ↓
[Expert 生成候选替代动作]  ← Claude-3.7-Sonnet
     ↓
[PRM 评分：找 policy 低分 + expert 高分 的步骤]  ← 候选关键步骤
     ↓
[分支 rollout 验证：替换步骤 t，用 policy 继续到完成]
     ↓
[只保留验证成功的分支]  ← 避免 reward 噪声，直接验证结果
     ↓
[DPO 训练：(st, a+_t, a-_t) 对]
```

### 2.1 问题形式化

ReAct 轨迹：$\tau = (s_1, a_1, o_1, s_2, a_2, o_2, \ldots, s_T, a_T, o_T)$

$s_t = (q, a_1, o_1, \ldots, a_{t-1}, o_{t-1})$（状态 = 历史）

$a_t \sim \pi_\theta(\cdot | s_t)$（策略采样动作）

$o_t = \mathcal{E}(s_t, a_t)$（环境返回观察）

### 2.2 候选关键步骤选取

**判定条件**（双阈值）：
$$t \in \mathcal{C}_\tau \iff r_t^{\text{policy}} < \gamma_{\text{low}} \;\text{and}\; \max_j r_{t,j}^{\text{expert}} > \gamma_{\text{high}}$$

- $\gamma_{\text{low}} = 0.45$，$\gamma_{\text{high}} = 0.65$
- PRM 用 Claude-3.7-Sonnet rubric 打分（非专门训练的小模型）
- 每步生成 $k=5$ 个 expert 替代动作

### 2.3 分支 rollout 验证

关键设计：**替换步骤 $t$ 的动作后，后续步骤仍由 policy 自身执行**（不由 expert 接管）。

原因：确保训练数据处于 policy 的可达分布内——如果后续步骤由 expert 执行，训练目标可能超出 policy 能力边界。

$$\tau_j' = (s_1, a_1, \ldots, s_t, a_{t,j}', o_t', s_{t+1}', a_{t+1}'\sim\pi_\theta, \ldots)$$

只有 $y_j' = 1$（验证成功）的分支才进入训练集。

### 2.4 DPO 训练目标

$$\mathcal{L}_{\text{CSO}}(\theta) = -\mathbb{E}_{(s_t, a_t^+, a_t^-) \sim \mathcal{D}_{\text{pref}}} \left[\log\sigma\left(\beta\log\frac{\pi_\theta(a_t^+|s_t)}{\pi_{\text{ref}}(a_t^+|s_t)} - \beta\log\frac{\pi_\theta(a_t^-|s_t)}{\pi_{\text{ref}}(a_t^-|s_t)}\right)\right]$$

标准 DPO，但数据质量极高：
- $a_t^+$ = 经验证能翻转结果的替代动作
- $a_t^-$ = 原始失败动作
- 只在关键步骤上做，不是每步都做

### 2.5 迭代在线精炼

$$\pi_{\theta_0} \xrightarrow{\mathcal{D}_{\text{pref}}^{(0)}} \pi_{\theta_1} \xrightarrow{\mathcal{D}_{\text{pref}}^{(1)}} \pi_{\theta_2} \xrightarrow{\mathcal{D}_{\text{pref}}^{(2)}} \cdots$$

每轮：部署新 policy → 收集新失败轨迹 → 新关键步骤 → 新 DPO 数据

随着 policy 提升，关键步骤自然移向**更难的决策点**（类似课程学习的自动化版本）。

---

## 三、实验结果

### 3.1 主要对比（GAIA-Text-103 + XBench-DeepSearch）

| 方法 | GAIA-All (%) | XBench Score | 特征 |
|------|-------------|--------------|------|
| GPT-4.1 | 45.6 | 27.0 | 闭源参考 |
| Claude-3.7-Sonnet | 62.1 | 41.0 | 闭源参考 |
| Qwen3-8B（base）| 20.4 | 7.0 | — |
| CK-Pro-8B (SFT) | 35.9 | 23.0 | 基线 |
| + ETO | 38.9 | 22.0 | 轨迹级 DPO |
| + RFT | 34.9 | 20.0 | 拒绝采样 FT |
| + Step-DPO | 38.9 | 25.0 | 步骤级（有噪声）|
| + IPR | 44.6 | 24.0 | MC rollout（贵）|
| **+ CSO (Ours)** | **49.5** | **29.0** | 验证关键步骤 |

**关键数字**：
- +37% over SFT（相对提升）
- 8B 模型 (49.5%) vs GPT-4.1 (45.6%) — 超越
- 只监督 **16%** 的轨迹步骤

### 3.2 消融实验关键发现

- **无 PRM 筛选**（随机选步骤）→ 大幅下降：证明关键步骤选取不可替代
- **无验证步骤**（PRM 筛选后直接用，不做 rollout 验证）→ 性能下降：证明 PRM 噪声真实存在，验证是必要的
- **无 expert model**（用 policy 自己生成替代动作）→ 性能下降：证明 expert-guided exploration 的价值
- 迭代 2 轮 > 1 轮：迭代有效

---

## 四、理论位置：credit assignment 地图新分支

CSO 引入了一个新的信号来源维度——**"失败轨迹中的反事实验证"**：

```
Credit Assignment 方法谱系：

来自成功轨迹：
├── 轨迹级: GRPO, RLOO（整条打分）
├── MC rollout: AgentPRM, IPR（每步前向采样）
└── 层级段: HiPER（subgoal 聚合）

来自失败轨迹（CSO 的新维度）：
└── 反事实验证: CSO
    ├── 定位：PRM 找 policy 弱点
    ├── 替代：expert 生成候选
    └── 验证：policy 自身 rollout 确认
```

**CSO 的独特性**：它不问"什么步骤做对了"，而是问"什么步骤换一个动作能让整件事成功"——这是反事实推理视角（counterfactual causality）在 credit assignment 中的应用。

---

## 五、方法论评价

### 优势

1. **验证而非估计**：PRM 只用于筛选候选，真正的质量判断来自 ground-truth rollout 结果——这消除了 PRM 噪声对训练质量的污染

2. **从 policy 弱点出发**：从失败轨迹挖掘，直接靶向模型的薄弱点，比从 expert 演示出发更有针对性

3. **Policy 可达性保证**：后续步骤由 policy 执行的设计避免了"训练目标超出能力范围"的分布偏移问题

4. **计算效率**：只在 16% 的步骤上监督，比 IPR（每步 MC rollout）便宜得多

### 局限性

1. **Expert model 依赖**：需要 Claude-3.7-Sonnet 生成替代动作——这在成本上不可忽视，且要求任务域内 expert model 确实更强

2. **PRM 质量瓶颈**：候选筛选仍依赖 PRM（这里用 LLM 打分），如果 PRM 漏掉了真正的关键步骤，验证阶段也救不回来

3. **验证计算成本**：K=5 个候选 × 每个候选完整 rollout——对长 horizon 任务成本不低（虽比 IPR 便宜）

4. **离线 DPO 的固有限制**：虽然迭代训练让其接近 on-policy，但本质仍是 DPO，不具备 PPO 的在线策略梯度优势

### 与其他方法的关系

| 比较对象 | CSO 的优势 | CSO 的劣势 |
|---------|-----------|-----------|
| GiGPO | 无需平行轨迹的 anchor state，适用更通用场景 | GiGPO 是在线 RL，CSO 是离线 DPO |
| AgentPRM | 验证而非估计，无 PRM 噪声累积 | AgentPRM 可直接集成到在线 RL |
| iStar | 无需双 LLM，计算效率高 | iStar 不依赖 expert model |
| HiPER | 无需层级结构假设 | HiPER 在层级结构任务中更系统 |

---

## 六、对工程实践的启示

### 6.1 什么时候用 CSO

**最适合场景**：
- 有大量失败轨迹可利用（在线部署的 agent 系统）
- 有更强的 expert model 可用于替代动作生成
- 任务有可验证的二元结果（成功/失败明确）
- 需要 offline/semi-offline 训练（计算资源限制）

**不适合场景**：
- expert model 不比 policy 强（alternative generation 无意义）
- 任务结果难以自动验证
- 需要在线 RL 的实时反馈（推荐 GiGPO 或 HiPER）

### 6.2 高熵 token 原则的推广

CSO 把 RLVR 中"只有高熵 token 驱动有效学习"的原则推广到了 Agent 领域：

> 在 Agent 任务中，"高熵步骤"= 工具选择分叉点、查询策略选择点、关键推理跳跃点

这给 agent 训练提供了一个启发式原则：**优先在策略不确定（高熵）且结果关键（能翻转成败）的步骤上投入监督资源**。

---

## 七、See Also

**Credit Assignment 谱系（CSO 的定位）：**
- [[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — CSO 是这个专题的新成员，填充"反事实验证"视角（来自失败轨迹的 counterfactual credit）
- [[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]] — 在线 RL 版的 step-level credit assignment；CSO 是离线 DPO 版，两者互补（在线 vs 离线）
- [[AI/Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL|iStar]] — 隐式 PRM 方案（DPO ≡ step-wise BT model）；iStar 不依赖 expert model，CSO 依赖但提供 ground-truth 验证
- [[AI/Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER（ICML 2026）]] — subgoal-segment 层级方案；HiPER 侧重层级结构，CSO 侧重关键分叉点
- [[AI/Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents|AgentPRM]] — MC rollout 估计方案（CSO 的主要对比对象之一）；CSO 验证胜过估计
- [[AI/Agent/Agentic-RL/MIG-Step-Marginal-Information-Gain-Credit-Assignment|MIG]] — 信息论视角的 step credit；与 CSO 正交（信息量 vs 反事实成功率）

**失败信号利用（三种深度）：**
- [[AI/Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning|ERL（arXiv:2602.13949）]] — 失败 → 反思循环 → SFT 蒸馏，中等成本；CSO 是 offline 数据工程，ERL 是 online 训练循环——两者覆盖失败轨迹利用的不同阶段
- [[AI/Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards|SELAUR（arXiv:2602.21158）]] — 失败 → token-level 不确定性 → reward reshaping，零额外成本；信息最浅但工程成本最低；三种方法构成"失败信号利用深度谱系"（SELAUR 浅 → ERL 中 → CSO 深）

**跨域稀疏原则验证：**
- [[AI/LLM/Inference/SIA-Sparse-Inference-time-Alignment|SIA（ICML 2026，NTU）]] — **同一稀疏哲学的跨域实证**：CSO 发现 16% 关键步骤决定 Agent 成败，SIA 发现 20% Junction token 承担 100% 对齐负担——两个独立工作从不同领域（Agent RL credit ↔ 推理时对齐）证明了"关键决策天然稀疏"这一原则

**工程应用：**
- [[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|Tool-Use-RL 训练专题]] — CSO 适用于 tool-use agent 后训练；关键步骤通常是工具选择分叉点

## 推荐阅读

1. **原文**：[arXiv:2602.03412](https://arxiv.org/abs/2602.03412) — CSO: Verified Critical Step Optimization
2. **代码**：[github.com/kiaia/CSO](https://github.com/kiaia/CSO)
3. **理论上位**：[[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon CA 专题]] — CSO 在谱系中的位置（反事实验证分支）
4. **对比阅读**：[[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]] — 同为 step-level CA，在线 vs 离线的设计权衡
