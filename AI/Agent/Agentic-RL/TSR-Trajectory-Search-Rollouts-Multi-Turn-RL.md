---
title: "TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents"
brief: "TSR 把 test-time scaling 思想搬到训练时 rollout 生成：每轮采样多候选动作后用树搜索（Best-of-N/Beam/Lookahead）选优质轨迹，解决稀疏 reward/不可逆陷阱/Echo Trap 三大 multi-turn RL 痛点。Optimizer-agnostic，与 PPO/GRPO 正交可叠加。Sokoban/WebShop +15% absolute，0.5B+TSR≈3B naive，训练稳定性显著提升。"
arxiv: "2602.11767"
date: 2026-02-25
venue: "ICML 2026"
rating: ★★★★☆
authors: "Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Holger Boche"
affiliation: "TU Munich + IBM Research"
tags:
  - ai/agent/agentic-rl
  - rollout-quality
  - multi-turn-rl
  - tree-search
  - training-infrastructure
  - type/paper
sources:
  - "arXiv:2602.11767 (v2, 2026-02-21) — ICML 2026"
related:
  - "[[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]]"
  - "[[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]]"
  - "[[AI/Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER]]"
  - "[[AI/Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution|RAGEN & StarPO]]"
---

# TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents

---

## TL;DR

TSR 把 test-time scaling（推理时搜索）的思想搬到了**训练时的 rollout 生成阶段**：在每轮 turn 采样多个候选动作，用轻量级树搜索（Best-of-N / Beam / Lookahead）选出高质量动作，构造更好的训练轨迹。**不改变 policy 优化目标**，optimizer-agnostic，与 PPO/GRPO 正交可叠加。Sokoban/FrozenLake/WebShop 最高 +15% absolute 提升，训练更稳定，小模型可以打败大模型。

核心洞察：**训练阶段的 rollout 质量是 multi-turn RL 成功的关键因素，而这一点被严重忽视。**

---

## 一、问题定位

### 1.1 Multi-Turn RL 的三个核心痛点

```
痛点 1：稀疏/延迟 Reward
  典型案例：Web Agent 购物任务
  → 需要精确执行：搜索→点击商品→选颜色→点购买
  → 前三步完美，第四步失败 = reward 0
  → 所有轨迹前缀都得到 0 reward
  → Advantage 估计高方差，训练不稳定

痛点 2：不可逆陷阱（Irreversible Traps）
  典型案例：Sokoban 箱子推入墙角
  → 一步失误 → 剩余 episode 全部无效
  → Naive 采样可能生成大量"dead"前缀
  → 浪费计算资源，提供零学习信号

痛点 3：多样性不足 → Mode Collapse（Echo Trap）
  → Policy 重复采样同一成功轨迹
  → 该状态下所有 action 的相对 advantage ≈ 0
  → PPO/GRPO 梯度更新停滞
  → 策略卡在局部最优，性能突然崩溃
```

### 1.2 根本问题：Rollout 质量被忽视

现有 multi-turn RL 研究大量关注：
- 优化算法（PPO vs GRPO vs REINFORCE）
- Credit assignment（轨迹级 vs 步骤级）
- Reward 设计

但**几乎没有研究 rollout 生成本身的质量问题**。TSR 问：

> "如果允许在 rollout 阶段多花一点计算，能得到多少回报？"

---

## 二、TSR 方法

### 2.1 框架形式化

问题设定：多轮 POMDP $\mathcal{M} = \langle \mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, P, \mathcal{R} \rangle$

轨迹：$\tau = (a_0, o_0, r_0, \ldots, a_K, o_K, r_K)$

累积 reward：$R(\tau) = \sum_{t=0}^K r_t$

策略梯度目标：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{K-1} \nabla_\theta \log\pi_\theta(a_t|\tau_{<t})\,\hat{A}_t\right]$$

**TSR 的核心想法**：不改变上式的优化目标，**只改变如何生成轨迹 $\tau$**。

### 2.2 TSR 框架

每轮 turn $t$，不是 sample 一个动作，而是 sample 候选集：

$$\mathcal{A}_t = \{a_t^{(1)}, \ldots, a_t^{(M)}\}, \quad a_t^{(j)} \sim \pi_\theta(\cdot|\tau_{<t})$$

用打分函数评估每个候选：

$$S(\tau_{<t}, a_t, o_t) \in \mathbb{R}$$

$S(\cdot)$ 的设计依任务类型：
- **确定性任务**（Sokoban）：直接用 per-turn environment reward
- **随机任务**（FrozenLake）：risk-aware heuristic 分数
- **长 horizon 任务**（WebShop）：progress-based 或语义相关分数

TSR 形式化：

$$\text{TSR}: (\pi_\theta, u, S, K, \mathcal{F}_\phi) \longrightarrow \{\tau_1, \ldots, \tau_L\}$$

其中 $\mathcal{F}_\phi$ 是搜索策略（参数 $\phi$），$L$ 是生成的轨迹数量。

### 2.3 三种搜索策略

#### 策略 A：Trajectory-Level Best-of-N

```
生成 N 条完整轨迹，用累积 reward 选最好的：

τ* = argmax_{τ ∈ {τ(1),...,τ(N)}} R(τ)

特点：简单，只在轨迹末尾做选择，是"naive baseline"
缺陷：浪费计算在坏的中间前缀上（已经走进死胡同的轨迹也要走完）
```

#### 策略 B：Per-Turn Beam Search

```
每轮维护 B 条 active beams，在每步扩展：

B_{t+1} = Top_B {(τ_{<t} ∘ a_t) | τ_{<t} ∈ B_t, a_t ∈ A_t}

参数：branching factor M = 4, beam width B = 2
特点：主动引导 rollout 向高价值区域，能从局部次优中恢复
```

#### 策略 C：Shallow Lookahead Search

```
每步额外展开短 horizon D ≪ K 的子树再评估动作：

类比：下棋时多看 D 步再决定下一步

参数：B = 2, M = 4, depth D = 2
特点：比 greedy 打分质量更高，比完整 beam 计算更轻
```

三种策略对比：

| 搜索策略 | 选择时机 | 前缀质量 | 计算量 |
|---------|---------|---------|------|
| Best-of-N | 轨迹完成后 | 低（不影响中间步骤）| 最低 |
| Beam Search | 每步 | 高（主动引导）| 最高 |
| Lookahead | 每步（+D步预看）| 中高 | 中 |

### 2.4 Instance Filtering（多样性保障）

TSR 提高了**质量（exploitation）**，但单独使用可能降低**多样性（exploration）**。

配合 uncertainty-based instance filtering：

$$U(u; \pi_\theta) = \text{Std}_{\tau \sim \pi_\theta(\cdot|u)}[R(\tau)]$$

**只保留高不确定性的任务组**（reward 方差大 = 模型还在学习 = 有梯度信号），丢弃低方差任务（要么太简单，要么已经失败固化）。

**完整训练循环（Algorithm 1）**：
```
for each training step:
  采样 P 个任务组
  for each 任务组 u_i:
    用 TSR 生成 L 条轨迹 G_i
    计算不确定性 U_i = Std(R(τ) for τ in G_i)
  按 U_i 排序，保留 top-p% 的任务组
  用保留的轨迹做 PPO/GRPO 更新
```

---

## 三、实验结果

### 3.1 主要结果

**环境**：Sokoban（确定性逻辑）/ FrozenLake（随机导航）/ WebShop（长 horizon web agent）  
**模型**：Qwen2.5-0.5B 和 Qwen2.5-3B  
**基线**：Instance Filtering + naive rollout sampling（RAGEN 默认设置）

**关键数字**（最高 +15% absolute improvement）：

| 任务 | 模型 | 基线 | TSR-Best | 最佳 TSR 变体 |
|------|------|------|---------|------------|
| Sokoban | 0.5B | ~40% | ~50% | Beam/Lookahead (+10-15%) |
| Sokoban | 3B | ~55% | ~65% | Beam/Lookahead |
| FrozenLake | 0.5B | ~30% | ~40% | Lookahead |
| FrozenLake | 3B | ~50% | ~60% | Beam |
| WebShop | 3B | ~45% | ~52% | Beam |

**重要 bonus**：TSR 使 Qwen2.5-0.5B 在 Sokoban 上超过了 naive 训练的 Qwen2.5-3B——**小模型 + 好 rollout > 大模型 + 差 rollout**。

### 3.2 稳定性改进

TSR 显著减少了训练曲线中的 variance spike（Echo Trap 现象），训练更稳定。原因：
- 高质量 rollout → advantage 估计 signal-to-noise ratio 提升
- Instance filtering 保持任务多样性 → 防止 mode collapse

### 3.3 计算-性能权衡

TSR 的代价：**一次性增加 rollout 阶段的计算量**（M × K × L 而不是 K × L）。

关键：这是 **training-time** 成本，不是 inference-time 成本——部署时不需要搜索。

实验参数：M=4 branching factor，性能提升 10-15%，计算增加 3-4×。

---

## 四、理论分析

### 4.1 TSR 的本质：Test-Time Scaling 的对称操作

```
Test-Time Scaling（推理时搜索）：
  给定固定 policy → 在推理时搜索更好的 response
  → 不改变 policy，只在推理时多花计算

TSR（训练时搜索）：
  给定当前 policy → 在 rollout 时搜索更好的训练轨迹
  → 不改变 optimization 目标，只在训练时多花计算
  → 更好的训练轨迹 → 更好的 policy
```

TSR 回答了一个对称问题：如果 test-time 可以用搜索换质量，**training-time rollout 阶段也应该可以**。

### 4.2 Optimizer-Agnostic 的价值

TSR 的设计保证：$\hat{A}_t$ 的计算不受影响，只改变轨迹 $\tau$ 的生成方式。

这意味着：
- **与任何 policy gradient 方法兼容**（PPO, GRPO, REINFORCE++, OAPL...）
- **可以叠加任何 credit assignment 技巧**（GiGPO, HiPER, iStar 都兼容）
- **即插即用**：现有 RL 训练框架只需替换 rollout generator

### 4.3 为什么 Beam Search > Best-of-N？

Best-of-N 在轨迹末尾做选择，但**已经浪费了走进死胡同的计算**。

Beam Search 在每步做选择，**主动避免进入不可逆陷阱**，就是 Sokoban 案例所展示的：

```
Naive 采样：
  → 80% 概率推箱子进墙角 → 无效轨迹
  → 只有 20% 的轨迹能提供学习信号

Beam Search：
  → 每步保留最优 B 个前缀
  → 主动绕过墙角
  → 几乎所有轨迹都能完成任务
```

---

## 五、关联与位置

### 5.1 在 Agent RL 知识体系中的位置

TSR 填补了一个特殊的位置：

```
Agent RL 的两个轴：

轴 1：Rollout 质量（数据侧）
  ← TSR 在这里！
  naive sampling → Best-of-N → Beam → Lookahead
  （每步计算量递增，质量递增）

轴 2：Optimization 算法（优化侧）
  GRPO → PPO → OAPL → ...
  （TSR 与这个轴完全正交）
```

与其他方法的关系：

| 方法 | 解决什么 | TSR 的关系 |
|------|---------|-----------|
| GiGPO | Credit assignment（步骤级优势估计）| 正交，可叠加 |
| CSO | Credit assignment（反事实验证）| 正交，但 TSR 是在线训练，CSO 是离线 DPO |
| CM2 | Reward 设计（unverifiable checklist）| 正交，TSR 不关心 reward 类型 |
| OAPL | Off-policy 训练稳定性 | 正交，TSR 改善 rollout 质量，OAPL 改善 policy lag |
| SeeUPO | Multi-turn 更新顺序 | TSR 是其"rollout 质量"支柱之一 |
| StarPO/GFPO | Rejection-sampling rollout 选择 | TSR 是这类方法的树搜索升级版 |

### 5.2 HEARTBEAT.md 主线定位

HEARTBEAT.md 描述 TSR 为"多轮 RL 双支柱"之一。我的理解：

**双支柱框架**：
1. **TSR（rollout 质量支柱）**：训练时搜索，确保每次更新用的轨迹质量高
2. **Credit Assignment 方法（梯度质量支柱）**：GiGPO/HiPER/iStar，确保从高质量轨迹中提取精确的步骤级信号

两者缺一不可：高质量轨迹 + 精确信号 = 真正稳定的 multi-turn agent RL。

---

## 六、评价

### 优点

1. **洞察深刻**：把"rollout 质量"独立出来作为研究对象，是一个被忽视但重要的视角
2. **Optimizer-agnostic 设计**：普适性极强，没有绑定特定算法
3. **小模型超大模型**：用计算换能力的路线在 training-time 也成立，有 practical 价值
4. **三种搜索策略清晰**：提供了不同计算预算下的选择空间

### 局限性

1. **与 Curriculum Learning 的隐性关系**：Instance Filtering 本质上是隐性的 curriculum learning（按难度/不确定性筛选训练任务），但论文没有与 PACED-RL 等显性 curriculum 方法对比，无法判断两种 curriculum 策略的互补性或替代性
2. **Episode-end-only reward 的退化问题**：TSR 的搜索依赖 per-turn reward 或 proxy 作为评分函数 $S(\cdot)$。如果环境只有 episode-end reward（无中间信号），beam search 的评分函数退化为 0，失去 per-turn 剪枝能力——此时 TSR ≈ Best-of-N，搜索优势消失
3. **Scoring function 设计依赖 domain knowledge**：FrozenLake/WebShop 需要 proxy 分数，实际部署中如何设计 $S(\cdot)$ 是非平凡工程问题
2. **Training-time 计算成本**：M=4 branching 意味着训练时 3-4× 的 rollout 计算，大规模训练代价不小
3. **有 verifiable reward 前提**：TSR 的搜索依赖能评估 action 质量的分数函数——如果任务完全 unverifiable（无 proxy），搜索效果退化为 random
4. **短 horizon 限制**：实验用 K=5 turn horizon，对于真正长 horizon（如 SWE-bench 50+ 步），beam search 的分支爆炸问题需要更大的搜索预算

### 深层 insight

TSR 的最大价值不是 "15% 提升" 本身，而是它证明了：

> **训练数据质量 > 训练数据数量**

在 rollout-based RL 中，与其生成 100 条 naive 轨迹，不如生成 25 条 beam-search 精选轨迹。这个原则与 SFT 时代的"质量比数量重要"一脉相承，但在 RL 的语境下需要主动设计来实现。

---

## See Also

**Multi-Turn RL 训练基础设施（TSR 的生态位）：**
- [[AI/Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution|RAGEN & StarPO（arXiv:2504.20073）]] — **TSR 的出发点**：RAGEN 发现 Echo Trap（rollout 多样性崩溃），TSR 用树搜索 + Instance Filtering 直接解决 Echo Trap；RAGEN 是症状诊断，TSR 是训练侧工程修复

**Credit Assignment 协同（Rollout 质量 × 梯度质量双支柱）：**
- [[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — TSR 提升 rollout 质量（输入端），CA 方法提升梯度归因精度（输出端）；两者正交可组合
- [[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO（NeurIPS 2025）]] — step-level credit assignment；TSR+GiGPO = 高质量 rollout + 精确 step credit，潜在最优组合
- [[AI/Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER（ICML 2026）]] — segment-level CA；TSR 兼容所有 CA 方法（optimizer-agnostic 特性）

**Rollout 质量 vs 其他训练优化视角：**
- [[AI/Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（arXiv:2602.03412）]] — 同样关注轨迹质量，但策略不同：TSR 是 online RL 阶段选优质轨迹，CSO 是 offline DPO 从失败轨迹反事实验证；两者维度互补（在线生成 vs 离线挖掘）
- [[AI/Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2（arXiv:2602.12268）]] — multi-turn RL 的 reward 构造维度；CM2 解决"reward 信号怎么设计"，TSR 解决"rollout 怎么生成"；两者覆盖 multi-turn RL 训练基础设施的不同层

**反思内化协同（训练动态层）：**
- [[AI/Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning|ERL（arXiv:2602.13949）]] — 与 TSR 正交可叠加：TSR 解决"哪条 rollout 有训练价值"（树搜索选优）；ERL 解决"失败后如何显式提取纠错信号"（反思循环 + 蒸馏内化）；理论上 TSR+ERL 组合 = 高质量 rollout + 结构化纠错，Sokoban +81%

**综述导航：**
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL-2026前沿综合分析]] — TSR 在五大维度框架中的定位（Environment/Rollout 工程维度）

## 推荐阅读

1. **原文**：[arXiv:2602.11767](https://arxiv.org/abs/2602.11767) — TSR: Trajectory-Search Rollouts, ICML 2026
2. **前驱工作**：[[AI/Agent/Agentic-RL/RAGEN-StarPO-Multi-Turn-RL-Self-Evolution|RAGEN & StarPO]] — Echo Trap 发现，TSR 的出发点
3. **协同方案**：[[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]] — 与 TSR 正交可组合（rollout 质量 × step credit）
4. **全景导航**：[[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 综合分析]] — 训练基础设施完整框架

<!-- 2026-02-26 dedup: 删除了TSR副本（TSR-Trajectory-Search-Rollouts.md），合并了Curriculum Learning关系批判、episode-end-only reward退化分析 -->
