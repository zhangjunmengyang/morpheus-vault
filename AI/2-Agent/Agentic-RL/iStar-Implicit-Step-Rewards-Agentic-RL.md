---
title: "iStar: Agentic Reinforcement Learning with Implicit Step Rewards"
brief: "iStar 用 multi-turn DPO 目标隐式训练 PRM，把轨迹偏好自动转化为 step-level reward，无需额外 rollout 或人工标注。核心理论：DPO ≡ Implicit PRM，同一 LLM 同时充当 policy + PRM。SOTOPIA 社交目标完成 +48%（vs GPT-4o），WebShop/VisualSokoban SOTA。ICLR 2026。"
date: 2026-02-28
type: paper-note
rating: ★★★★☆
venue: ICLR 2026
arxiv: "2509.19199"
tags:
  - agentic-RL
  - credit-assignment
  - implicit-PRM
  - DPO
  - unverifiable-reward
  - ICLR-2026
  - step-reward
related:
  - "[[AI/2-Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents|AgentPRM]]"
  - "[[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]]"
  - "[[AI/2-Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]]"
  - "[[AI/3-LLM/RL/算法/DAR-Dual-Regularized-Advantage-Regression-Unifying-RLHF|DAR]]"
  - "[[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2]]"
---

# iStar: Agentic Reinforcement Learning with Implicit Step Rewards

> arXiv:2509.19199 | ICLR 2026 | CMU + TikTok

---

## 一句话 TL;DR

**把 DPO 目标变成 implicit PRM 训练器**：iStar 让同一个 LLM 同时扮演 policy 和 process reward model，用在线采集的轨迹偏好对（好 vs 坏）通过 multi-turn DPO loss 提炼出每步隐式奖励信号，再把这个 step-level advantage 组合进 RL 更新——一套训练循环，两个功能，无需额外 rollout、无需人工 step 标注。

---

## 动机：unverifiable reward 下的 credit assignment 困境

现有步骤级 credit 方法的适用场景受限：

| 场景类型 | 代表 | 现有方法 | 问题 |
|--------|------|---------|------|
| 可验证 | 数学/代码/ALFWorld | AgentPRM / GiGPO / HiPER | ✓ 适用 |
| 半可验证 | WebShop 购物 | AgentPRM（噪声大）| 勉强可用 |
| **不可验证** | **SOTOPIA 社交对话** | **无有效 step credit 方法** | ← iStar 填补 |

不可验证场景的特点：outcome reward 只能靠 LLM judge，有 ±20%+ 随机误差。MC rollout（AgentPRM）会放大这个噪声；GiGPO 需要结构化的状态重访；HiPER 需要修改 action space。

**iStar 的切入点**：用 DPO 的 ordinal 偏好（哪个好）代替 cardinal 奖励值——排序比绝对值对噪声鲁棒得多。

---

## 核心理论：DPO ≡ Implicit PRM

### 等价关系推导

在 Bradley-Terry 偏好模型下，最优 policy 满足：

$$r^*(s_t, a_t) = \beta\log\frac{\pi^*(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \beta\log Z(s_t)$$

其中 Z(s_t) 是配分函数（状态相关常数）。

这意味着：当用 DPO 训练好一个 policy 后，**log(π_θ/π_ref) 本身就隐式地编码了一个 step-level reward model**——DPO objective 的收敛解与 PRM 的收敛解在数学上等价。

Multi-turn DPO loss（对轨迹偏好对 (τ+, τ-)）：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\sum_{t}\log\frac{\pi_\theta(a_t^+|s_t^+)}{\pi_{\text{ref}}(a_t^+|s_t^+)} - \beta\sum_{t}\log\frac{\pi_\theta(a_t^-|s_t^-)}{\pi_{\text{ref}}(a_t^-|s_t^-)}\right)\right]$$

每步的 log ratio 项就是该步的 implicit step reward。

### 等价性的实践含义

| 传统方式 | iStar 方式 |
|--------|---------|
| 独立训练一个 PRM 网络 | 用 DPO loss 让 policy 兼任 PRM |
| 需要 step-level 标注数据 | 只需 trajectory 级偏好对（好/坏） |
| 额外 MC rollout 估计 value | **零额外 rollout** |
| 两套参数（policy + PRM） | **同一套参数** |

---

## 方法：iStar 完整训练循环

```
每个训练步骤：

1. 【采样】用 π_θ 采集 N 条轨迹，按 outcome reward 排名
   - 取 top-k → 正例轨迹 τ+
   - 取 bottom-k → 负例轨迹 τ-
   - 构成偏好对集合 D

2. 【Phase A: PRM 更新】用 multi-turn DPO loss 在 D 上更新
   - 目标：policy 的 log ratio 更好地区分好/坏步骤
   - 这一步同时更新 policy（后面 RL 用同一参数）

3. 【Step Advantage 计算】
   - 对每步计算 implicit reward: r_t = β * log(π_θ(a_t|s_t) / π_ref(a_t|s_t))
   - 从 step reward 推导 step-level advantage A_t^step

4. 【Phase B: Policy 更新（RL）】
   - 组合: A_t = A_t^trajectory + λ * A_t^step
   - 用组合 advantage 做 PPO / REINFORCE 更新

→ 自强化闭环：更好的 policy → 更丰富的偏好对 → 更精准的 implicit PRM → 更好的 step credit
```

### 与 AgentPRM 的本质区别

| 维度 | AgentPRM | iStar |
|------|---------|-------|
| PRM 训练方式 | MC rollout 估计 step value | DPO loss 从轨迹偏好隐式提炼 |
| 额外计算开销 | 高（每步 N 次 rollout） | 低（DPO loss 与 policy loss 复用） |
| Reward 类型 | 主要可验证 | **可验证 + 不可验证** |
| 噪声鲁棒性 | 一般（MC 放大噪声） | 好（ordinal 偏好 > cardinal 值） |
| 独立 PRM 网络 | 有 | **无（同一 LLM）** |

---

## 实验结果

### 三个 Benchmark 全胜

| Benchmark | 类型 | 结果 |
|---------|------|------|
| **WebShop** | 可验证（购物搜索） | SOTA |
| **VisualSokoban** | 可验证（视觉规划） | SOTA |
| **SOTOPIA** | **不可验证（社交对话）** | **+14%（self-chat）/ +48%（vs GPT-4o）** |

SOTOPIA +48% 是最重要的数字。社交对话场景没有可靠的 ground truth reward，LLM judge 打分有显著噪声——其他 step-level credit 方法在这个场景根本无法部署，iStar 却大幅超过 frontier LLM（包括 GPT-4o）。

### 效率优势

- **更高 sample efficiency**：相同轨迹数量下，iStar 学得更快（step credit 密度更高）
- **更稳定训练**：无 MC rollout 噪声累积
- **任务成功步数更少**：iStar 的探索更高效

---

## 与 Credit Assignment 体系的关系

### 填补 unverifiable 象限

```
Reward 类型 × Credit 粒度矩阵（完整版）：

                    | 可验证           | 不可验证             |
轨迹/episode 级     | GRPO ✓          | CM2 ✓（checklist）  |
步骤/step 级        | AgentPRM ✓      | iStar ✓ ← 核心贡献  |
                    | GiGPO ✓         |                     |
                    | HiPER ✓         |                     |
段/subgoal 级       | HiPER ✓         | —（暂无方法）        |
```

iStar 填补了"步骤级 × 不可验证"这个象限，是 Credit Assignment 谱系中的最后一块重要空白。

### 与 CM2 的互补关系

CM2（2602.12268）在 turn-level + unverifiable 场景用 checklist 提供 dense reward。iStar 在 step-level + unverifiable 场景用 DPO 提炼 implicit reward。粒度不同，互补：
- 有 checklist 可设计：CM2（结构化 turn-level reward）
- 无结构标注，纯在线学习：iStar（implicit step reward）

---

## 批判性评估

### 三个核心优点

1. **理论扎实**：DPO ≡ PRM 不是 empirical trick，有 BT 模型的数学保证
2. **适用范围最广**：唯一覆盖 unverifiable + step-level 象限的方法
3. **实现轻量**：不需要独立 PRM 网络，训练循环里加 DPO loss 即可

### 两个实际局限

1. **β 超参数敏感**：DPO 温度 β 控制 step reward 幅度，不同任务最优 β 差异大，调参成本不低

2. **Reference Policy 稳定性假设**：
   - DPO ≡ PRM 等价性依赖 π_ref 不随训练剧烈漂移
   - 如果使用 Reference Resets（DAR 框架）动态更新 π_ref，等价性推导的 Z(s_t) 假设被打破
   - **iStar + DAR 的兼容性是开放问题**

---

## 核心洞察（面试直接用）

> **最深洞察**：DPO 本质上训练的是一个 implicit PRM。当 policy 学会区分好坏轨迹时，log(π/π_ref) 已经隐式编码了步骤质量——这个"免费"信号无需额外 rollout，代价接近零。iStar 的创新是把这个等价性系统化应用到 agent RL 的 credit assignment。

> **面试答法（unverifiable reward）**：
> "大多数 step-level credit 方法（AgentPRM/GiGPO/HiPER）都假设 outcome reward 可靠。当 reward 来自 LLM judge（有 ±20% 噪声），这些方法失效。iStar 的解法：ordinal 偏好（哪个轨迹更好）比 cardinal 数值（具体分数）对噪声鲁棒得多，而 DPO ≡ implicit PRM 保证了偏好对齐 = step reward 提炼。SOTOPIA 社交对话 +48% 证明了这在开放域 agent 训练中的实际价值。"

---

## 开放问题

1. **Reference Resets + iStar**：DAR 的动态 π_ref 与 DPO ≡ PRM 等价性的兼容性
2. **极长 horizon（>100步）**：step reward 累积误差如何控制？
3. **iStar + HiPER 组合**：Plan-Execute 层级结构 + implicit PRM，高层子目标 reward 能否用 iStar 同时提炼？
4. **Multi-agent 扩展**：多 agent 协作任务的 trajectory preference 如何分解为 per-agent step credit？

## See Also

- [[AI/2-Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents|AgentPRM]] — 对比：MC rollout 显式 PRM vs DPO 隐式 PRM
- [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]] — 对比：anchor grouping 步骤信号 vs DPO 隐式步骤信号
- [[AI/2-Agent/Agentic-RL/Long-Horizon-Credit-Assignment专题|Long-Horizon Credit Assignment 专题]] — iStar 在谱系中的位置（unverifiable 象限）
- [[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2]] — 互补：unverifiable × turn-level（checklist）vs unverifiable × step-level（DPO）
- [[AI/3-LLM/RL/算法/DAR-Dual-Regularized-Advantage-Regression-Unifying-RLHF|DAR]] — 理论联系：DPO 与 RL 统一框架；Reference Resets 兼容性问题
