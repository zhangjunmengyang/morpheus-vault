---
title: "SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards"
brief: 用 LLM 自身的 token 级不确定性（entropy/least-confidence/margin 三维加权）重塑失败轨迹的 reward，把 binary failure（reward=0）变成密集学习信号。Failure-aware shaping：成功轨迹用原始 reward，失败轨迹用不确定性 reward（w=0.95，确保失败 reward < 成功 reward）。ALFWorld/WebShop 超越 GiGPO。
date: 2026-02-25
arxiv: "2602.21158"
authors: Dengjia Zhang, Xiaoou Liu, Lu Cheng, Yaqing Wang, Kenton Murray, Hua Wei
institutions: Johns Hopkins University, Arizona State University, UIC, Purdue University
venue: arXiv 2026-02-24
rating: ★★★☆☆
tags:
  - agentic-RL
  - uncertainty
  - reward-shaping
  - failure-trajectory
  - credit-assignment
  - exploration
related:
  - "[[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO — 反事实验证失败轨迹]]"
  - "[[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO — step-level credit]]"
  - "[[AI/2-Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning|ERL — 反思内化失败经验]]"
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL 综合分析]]"
---

# SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards

> **一句话**：失败轨迹的 reward=0 是巨大的信息浪费——SELAUR 用模型自身的 token 不确定性把"零"变成密集负学习信号，让失败经历变成有价值的探索信号。

---

## TL;DR

**问题**：标准 RLVR 对失败轨迹给 reward=0 就不学了。但失败轨迹里有信息：哪些步骤模型是高度确定但错了？哪些步骤模型是犹豫的、本可以探索别的路径？这些信息被丢弃了。

**解法**：SELAUR 不依赖外部 reward 信号，只用 LLM 自身的 token 预测概率分布，提取三维不确定性：
- **Entropy**：全词表概率分散度（多高熵 = 模型不确定）
- **Least Confidence**：被选 token 的概率 `1 - p_(1)`（选出来的 token 概率越低，越不确定）
- **Margin**：top-2 token 概率差 `σ((1 - (p_(1) - p_(2)))/s)`（差距越小，越模糊）

对失败轨迹：用聚合后的不确定性 reward 替代原本的零信号。
对成功轨迹：保持原始 reward 不变。

**关键约束**：失败轨迹的 uncertainty reward 乘以 `w_t = 0.95`，确保始终小于成功 reward——不让模型学习"高度不确定地乱走"。

---

## 动机与背景

### 为什么失败轨迹的 reward=0 是浪费？

RLVR 的 binary 结构：
```
成功 → reward = +1 → 策略强化
失败 → reward = 0 → 策略不学习
```

但失败轨迹里有两类有价值的信息：
1. **Confident failures**：高置信度但错了的步骤 → 说明模型已经强化了错误的知识
2. **Uncertain pivots**：模型本来犹豫（高熵），但选了一个次优路径 → 下次应该探索这里

直接丢弃 = 无法识别这两类情况，无法提供有效的纠错信号。

### 与相关工作的对比

| 方法 | 如何利用失败轨迹 | 依赖 |
|------|---------------|------|
| 标准 GRPO | 完全丢弃（reward=0） | - |
| CSO | 反事实验证：找关键步骤，换成 expert 动作验证 | Expert model（Claude-3.7-Sonnet）|
| ERL | 对失败生成反思，用反思指导第二次尝试 | LLM 自身的反思能力 |
| **SELAUR** | **用失败步骤的不确定性作为内生密集 reward** | **只需 token logits，无额外模型** |

**SELAUR 的核心优势**：完全内生（zero external cost），不需要 expert model 或额外 LLM 调用。

---

## 方法：三模块架构

### 模块 1：Token 级不确定性估计

三种互补指标：

**Entropy**（全局分布扩散）：
$$u_{t,j}^{\text{ent}} = \frac{-\sum_c p_{t,c} \log p_{t,c}}{\log|\mathcal{V}|}$$
归一化到 [0,1]，高熵 = 概率分散 = 不确定

**Least Confidence**（局部决策置信度）：
$$u_{t,j}^{\text{lc}} = 1 - p_{t,(1)}$$
被选 token 的概率越低，不确定性越高

**Margin**（竞争模糊度）：
$$u_{t,j}^{\text{mar}} = \sigma\left(\frac{1-(p_{t,(1)}-p_{t,(2)})}{s}\right)$$
top-2 token 差距越小，越难抉择

**聚合**：
$$u_{t,j} = w_{\text{ent}} u_{t,j}^{\text{ent}} + w_{\text{lc}} u_{t,j}^{\text{lc}} + w_{\text{mar}} u_{t,j}^{\text{mar}}$$

三种指标从不同角度测量不确定性，聚合后更鲁棒。

### 模块 2：Step / Trajectory 聚合

**Step 级**（均值聚合）：
$$u_t^{\text{step}} = \frac{1}{j}\sum_j u_{t,j}$$

**Trajectory 级**（指数折扣，后期步骤权重更高）：
$$U(\tau) = \frac{\sum_{t=1}^T \lambda^{T-t} u_t^{\text{step}}}{\sum_{t=1}^T \lambda^{T-t}}$$

**关键设计：后期步骤权重更高**
λ < 1 → λ^(T-t) 在 t=T 时最大（=1）。直觉：越接近目标的步骤，其不确定性对最终失败的贡献越重要（类似 TD 思路）。

### 模块 3：Failure-aware Reward Shaping

**Step 级 shaping**：
$$\tilde{r}_t^{\text{step}} = \begin{cases} w_t \cdot \hat{u}_t^{\text{step}} & \text{if fail} \\ r_t & \text{otherwise} \end{cases}$$
其中 w_t = 0.95（确保失败步骤 reward < 成功 reward）

**Trajectory 级 shaping**：
$$\tilde{r}^{\text{traj}} = \begin{cases} U(\tau) & \text{if fail} \\ r & \text{otherwise} \end{cases}$$

**效果**：
- 高不确定性失败步骤 → 高 uncertainty reward → 强化"探索"方向（下次在这里多样化）
- 低不确定性失败步骤 → 低 uncertainty reward → 模型学到"这里我选错了，但我很确定，需要更新这个确定性"

---

## 实验

**基准环境**：
- ALFWorld：家庭任务完成（compositional reasoning，50步限制）
- WebShop：电商目标购物（open-ended，15步限制）

**模型**：未明确说明（从 baselines 推断是 1.5B-7B 级别）

**Baselines**：PPO, RLOO, GRPO, GiGPO

**主要结果**：SELAUR 超越所有 baselines，包括 GiGPO（step-level credit assignment）

**结果分析（论文定性描述）**：
- GiGPO 已经通过 step-level credit 改善了信号粒度
- SELAUR 在 GiGPO 基础上进一步引入不确定性，超越 GiGPO
- 说明：step-level credit（GiGPO）和 uncertainty-aware reward（SELAUR）是互补的

**消融实验**：三种不确定性指标各自有效，组合后最好（互补性验证）

---

## 分析与评价

**★★★☆☆（中等，工程可借鉴，理论深度有限）**

### 优点

1. **内生信号，零额外成本**：不需要 expert model（CSO 需要）或额外 LLM 调用（ERL 需要），只用 token logits
2. **直觉清晰**：失败 = reward=0 是信息浪费，uncertainty = 探索机会，这个故事说得通
3. **三维不确定性互补**：entropy（全局）+ least confidence（局部）+ margin（竞争）三个维度覆盖不同失败模式
4. **超越 GiGPO**：在 ALFWorld/WebShop 两个主流 agent benchmark 都有提升

### 局限

1. **实验规模较小**：150 步训练，rollout size 8，没有说明具体模型大小，结果的统计显著性存疑
2. **不确定性权重是超参数**：w_ent, w_lc, w_mar 如何调，论文消融不够详细
3. **理论薄弱**：为什么"不确定性高的失败步骤值得鼓励探索"？这个 claim 缺乏理论支撑（可能导致强化了"犹豫但仍然错"的行为）
4. **后期步骤权重更高的假设**：在多步 agent 中，早期步骤的错误（navigation error）可能比晚期步骤更关键，这个假设并不普遍成立
5. **与 GiGPO 的组合**：论文声称超越了 GiGPO，但没有测试 SELAUR + GiGPO 的组合效果

### 关键问题（理论层面）

**SELAUR 是否混淆了两种不确定性？**

- **认知不确定性（Epistemic）**：模型不知道正确答案 → 探索有益
- **偶然不确定性（Aleatoric）**：任务本身随机/歧义 → 探索无益

SELAUR 的 token-level 熵无法区分这两种不确定性。对于"确定地选了一个随机的策略"，熵低但仍然是探索行为；对于"歧义输入下的任何选择"，熵高但探索未必有益。

这是所有 uncertainty-based intrinsic reward 方法的共同局限，SELAUR 没有解决。

---

## 在 Credit Assignment 谱系中的定位

```
失败轨迹利用维度（与 CSO/ERL 对比）：

CSO：    失败轨迹 → 找关键步骤（PRM定位）→ 反事实验证 → DPO 监督
         需要：Expert model + branch rollout
         信号来源：外部验证（可靠性高）

ERL：    失败轨迹 → 生成反思 Δ → 指导重试 → SFT 蒸馏
         需要：LLM 的反思能力
         信号来源：自生成（依赖反思质量）

SELAUR：失败轨迹 → token logits → 不确定性估计 → reward reshaping
         需要：只需 token 概率
         信号来源：内生（计算成本最低，但信息最浅）
```

三种方法从不同深度挖掘失败轨迹信息：SELAUR 最浅（logits 层），ERL 较深（反思层），CSO 最深（验证层）。深度越高，信息质量越好，计算成本也越高。

---

## Reward Design 地图更新

SELAUR 引入了一个新的 reward design 类别：**uncertainty-based intrinsic reward**

```
Reward 来源谱系（更新）：
├── verifiable_binary：GiGPO / GRPO / Search-R1
├── unverifiable_implicit：iStar（DPO ≡ step-BT model）
├── unverifiable_checklist：CM2（sparse + dense，multi-turn tool use）
├── process_reward：AgentPRM
├── action_level_penalty：Search-R1++
└── uncertainty_intrinsic：SELAUR ← 新增
    特点：内生、零额外成本、仅适用于失败轨迹
```

---

## 工程价值

**适用场景**：
- reward 极稀疏（episode reward，无中间 step 信号）
- 失败率高（> 50%），大量轨迹被丢弃
- 资源有限，无法调用 expert model（CSO）或额外反思步骤（ERL）

**不适用场景**：
- 失败率很低（无需从失败中学习）
- step-level reward 已经密集（uncertainty 的边际价值低）

**实现成本**：极低——在 rollout 时保存 token logits，失败轨迹 reward 替换为不确定性聚合值，单函数修改。

---

## See Also

**失败轨迹利用谱系（三种深度）：**
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（arXiv:2602.03412）]] — 失败轨迹 + 反事实验证（PRM + expert 生成 + rollout 验证），信号最可靠但成本最高（需 expert model）；SELAUR 是其零成本替代方案
- [[AI/2-Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning|ERL（arXiv:2602.13949）]] — 失败轨迹 + 反思循环（生成反思 Δ → 重试 → SFT 蒸馏），中等成本；SELAUR 更轻量但信息深度不及 ERL

**Credit Assignment 协同（成功 vs 失败双维度）：**
- [[AI/2-Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO（arXiv:2505.10978）]] — step-level credit assignment 处理**成功**轨迹；与 SELAUR 正交互补：GiGPO 精化成功信号，SELAUR 激活失败信号；两者组合是 multi-turn RL 训练信号的完整覆盖

**探索增强同族（exploration 维度）：**
- [[AI/3-LLM/RL/算法/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO（arXiv:2602.14169）]] — Pivot-Driven Resampling 解决 GRPO 探索塌缩（root saturation）；与 SELAUR 的 uncertainty-based 探索思路相似（都关注探索不足），但机制不同（DEEP-GRPO 识别 pivot 轨迹重采样，SELAUR 用 token 熵直接 reshape reward）
- [[AI/3-LLM/MLLM/PyVision-RL-Agentic-Vision-Interaction-Collapse|PyVision-RL（arXiv:2602.20739）]] — 多模态 Agentic RL 的 Interaction Collapse（RL 退化为少工具少多轮）；Oversampling-Filtering-Ranking 主动过滤退化轨迹 vs SELAUR 对失败轨迹 reward reshape——两者处理 RL 训练中 agent 行为退化问题，但层次不同：SELAUR 在 reward 层激活失败信号，PyVision-RL 在 rollout 层过滤退化轨迹

**综述导航：**
- [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL 2026 综合分析]] — Reward Design 维度框架；SELAUR 填补「uncertainty intrinsic reward」类别
