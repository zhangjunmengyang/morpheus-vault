---
title: "Long-Horizon Credit Assignment 专题 — Agent RL 的核心难题"
brief: "多步 Agent RL 中最本质的技术难题：如何把稀疏末端 reward 分配给数百步的中间步骤。整理从 GRPO baseline 到 GiGPO/AgentPRM/LOOP/MIG 的完整解法图谱，含算法原理、数学推导、效果对比。GiGPO 是当前最优的无 critic 方案（ALFWorld +12% vs GRPO，无额外内存开销），通过 anchor state grouping 实现 step-level credit 而不需要额外 rollout。"
date: 2026-02-21
type: synthesis
domain: ai/agent/agentic-rl
tags:
  - agentic-rl
  - credit-assignment
  - long-horizon
  - process-reward-model
  - GiGPO
  - MCTS
  - turn-level-reward
  - synthesis
  - interview-prep
status: v2
sources:
  - "GiGPO: Feng et al., arXiv 2505.10978"
  - "AgentPRM / InversePRM: Choudhury et al., arXiv 2502.10325"
  - "LOOP: arXiv 2502.01600"
  - "Turn-Level Reward: Wei et al., arXiv 2505.11821"
  - "MIG: arXiv 2602.01034"
  - "PA-MoE: arXiv 2602.17038"
  - "KLong: Liu et al., arXiv 2602.17547"
  - "HiPER: Vault 笔记（层级 advantage 估计）"
related:
  - "[[AI/LLM/RL/Other-Algorithms/HiPER-Hierarchical-RL-Credit-Assignment|HiPER]]"
  - "[[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL-2026前沿综合分析]]"
  - "[[AI/Agent/Agentic-RL/Agent-RL-训练实战指南|Agent-RL-训练实战指南]]"
  - "[[AI/Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong]]"
  - "[[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]]"
  - "[[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|Tool-Use-RL-训练专题]]"
  - "[[AI/Agent/Agentic-RL/Agent-RL-环境工程系统论|Agent-RL-环境工程系统论]]"
---

# Long-Horizon Credit Assignment 专题 — Agent RL 的核心难题

> 核心问题：一条 700 步的交互轨迹，最终 reward = 0——**每一步的贡献是多少？** 这是 Agentic RL 最本质的技术挑战。这篇笔记系统整理问题定义和全解法图谱。

---

## 一、问题的精确定义

### 1.1 LLM Agent 的 credit assignment 为什么比传统 RL 更难

传统 RL（Atari/棋类）：
- 每步 action 是离散原子动作（"向右移动"）
- 状态精确可比（像素 hash / 棋盘状态）
- Horizon 10~100 步

LLM Agent RL 的三重困难：

```mermaid
graph LR
    A["双重 CA 问题\nToken-level:\n同一 response 哪些 token 有贡献\nStep-level:\n哪步是关键决策"] --> D[CA 极难]
    B["高维连续状态\n对话历史无法精确比较\n不同轨迹的'相同状态'难以识别"] --> D
    C["极长 Horizon\nPaperBench 700+ 轮\n折扣累积 variance 爆炸"] --> D
    
    style D fill:#f96
```

### 1.2 Reward 稀疏性谱系

| 任务 | Horizon | 稀疏程度 | 典型场景 |
|------|---------|---------|---------|
| 数学推理 (CoT) | 30 步 | 低 | 步骤间有逻辑连贯性 |
| Web 任务 | 20 步 | 中 | 操作间关联性弱 |
| SWE-bench | 50 步 | 高 | 探索步骤多，关键步骤少 |
| PaperBench | 700+ 步 | **极高** | 早期选择影响全程 |

---

## 二、解法全图谱

```mermaid
graph TD
    A[Credit Assignment 解法] --> B[算法层面]
    A --> C[数据层面]
    A --> D[任务分解层面]

    B --> B1["Trajectory-level\nGRPO baseline\n全轨迹共享 reward"]
    B --> B2["Step-level Group\nGiGPO 2505.10978\nanchor state grouping"]
    B --> B3["Turn-level Reward\n2505.11821\n逐步折扣累积"]
    B --> B4["Actor-Critic w/ PRM\nAgentPRM 2502.10325\nMC rollout 估值\nLOOP 2502.01600\nvalue-free PPO"]

    C --> C1["MCTS-guided Collection\n离线搜索 step-level Q 值"]
    C --> C2["PRM Training\n单独训练 step verifier"]
    C --> C3["MIG 2602.01034\n信息论视角 step 贡献"]

    D --> D1["Trajectory Splitting\nKLong 2602.17547"]
    D --> D2["Hierarchical RL\nHiPER / Agent-as-Tool"]
```

---

## 三、GRPO baseline 的失效模式

标准 GRPO 的 advantage 估计：

$$A_i = \frac{R_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^G R_j$$

**关键缺陷**：$A_i$ 是 trajectory-level，轨迹内所有 token 共享同一个值。

```python
# 具体后果：
# 轨迹1（成功）: [step1正确, step2出错修正了, step3正确] → A = +1
# step2 的错误尝试也被 reinforce（因为整条轨迹成功了）

# 轨迹2（失败）: [step1正确, step2正确, step3出了问题] → A = -1
# step1 和 step2 明明对的，但被 penalize
```

结论：好步骤在失败轨迹里被惩罚，坏步骤在成功轨迹里被强化 → 训练信号嘈杂。

---

## 四、GiGPO — 当前最优无 Critic 方案

**arXiv 2505.10978 | Feng et al. 2025**

### 核心 Insight：Anchor State Grouping

在大量并行 rollout 中，**不同轨迹会多次经过相同的环境状态**（anchor state）。

```mermaid
graph LR
    S0 --> S1
    S1 --> S2["s₂ (anchor state)"]
    S2 -->|"动作 a₃ᵃ"| T1["→ 最终成功 ✓"]
    S2 -->|"动作 a₃ᵇ"| T2["→ 最终失败 ✗"]
    S2 -->|"动作 a₃ᵃ"| T3["→ 最终成功 ✓"]
    
    style T1 fill:#6f6
    style T3 fill:#6f6
    style T2 fill:#f66
```

在 anchor state $s_2$，可以计算 step-level micro advantage：

$$A_{\text{micro}}(s_2, a_3^a) = \frac{R(a_3^a) - \mu_{\text{anchor}}}{\sigma_{\text{anchor}}}$$

即：同一 state 下不同 action 的相对 reward，精确反映该步决策的价值。

### 两层嵌套结构

**Episode-level（macro advantage）**：

$$A_{\text{macro},i} = \frac{R_i - \mu_G}{\sigma_G}$$

**Step-level（micro advantage，via anchor grouping）**：

$$A_{\text{micro},t} = \frac{R_{\tau(t)} - \mu_{\text{anchor}(s_t)}}{\sigma_{\text{anchor}(s_t)}}$$

**最终 advantage**：

$$A_t = \alpha \cdot A_{\text{macro},i} + (1-\alpha) \cdot A_{\text{micro},t}$$

其中 $\alpha$ 是超参，平衡全局 trajectory 质量和局部 step 贡献。

### 核心优势

- **无 critic**：无需 value network，内存 = GRPO
- **无额外 rollout**：复用已有轨迹中的重合状态（anchor state 天然出现）
- **精确 step 归因**：相同 state 下不同 action 的结果直接比较，因果明确

### 效果（arXiv 2505.10978, Table 2）
- ALFWorld: vs GRPO **+12%**
- WebShop: vs GRPO **+9%**
- PA-MoE（arXiv 2602.17038）在 GiGPO 基础上加 phase-aware MoE：ALFWorld +7.7%，WebShop +14.9%

---

## 五、AgentPRM — MC Rollout 估 Step Value

**arXiv 2502.10325 | Choudhury et al.**

### 问题：LLM-scale Critic 太贵

传统 actor-critic 需要一个与 policy 同规模的 value network（即两倍 LLM）。

**AgentPRM 解法**：用 Monte Carlo rollout 替代 learned critic：

$$V_{\text{MC}}(s_t) = \frac{1}{K} \sum_{k=1}^K R(\tau_k), \quad \tau_k \sim \pi(\cdot | s_t)$$

即从 $s_t$ 出发做 $K$ 次 rollout，取平均末端 reward 作为 step value 估计。

### Step Advantage 计算

$$A_t = V_{\text{MC}}(s_t) - b, \quad b = \frac{1}{T} \sum_{t'=1}^T V_{\text{MC}}(s_{t'})$$

### InversePRM 变体

从演示数据反向学习 process reward，无需 outcome 标注：

$$\text{InversePRM: } \hat{r}(s_t, a_t) = f_\theta(s_t, a_t) \approx \text{边际贡献}(s_t, a_t \to \text{outcome})$$

### 效果（arXiv 2502.10325, Table 1）
- ALFWorld benchmark: **3B 模型 + AgentPRM > GPT-4o baseline**

---

## 六、LOOP — Value-free PPO for Long-Horizon IDA

**arXiv 2502.01600 | Interactive Digital Agents**

### 核心设计：单 LLM，无 Value Network

标准 PPO 需要 policy + value（两个 LLM），内存翻倍。LOOP 的方案：

- 用 **折扣 return** 替代 learned critic 提供 baseline：

$$\hat{V}_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

- GAE（Generalized Advantage Estimation）估计 advantage：

$$A_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma \hat{V}_{t+1} - \hat{V}_t$$

- 内存效率 = SFT（只有一个 LLM）

### 任务场景（AppWorld）

多域 API 协同任务：地图 + 日历 + 联系人 + 音乐 + 文件

LOOP 训练后模型学到的行为（arXiv 2502.01600, Section 5）：
1. 主动查 API 文档（不假设参数）
2. 先验证再行动（避免不必要假设）
3. 从错误中恢复（retry 策略）

### 效果（arXiv 2502.01600, Table 2）
- 32B 模型 + LOOP: 超过 OpenAI o1 agent **+9%**（绝对），+15%（相对）

---

## 七、Turn-Level Reward

**arXiv 2505.11821 | Wei et al.**（详见 [[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|Tool-Use-RL 训练专题]]）

Turn-level advantage 公式：

$$A_t = r_t^{\text{turn}} + \gamma r_{t+1}^{\text{turn}} + \gamma^2 r_{t+2}^{\text{turn}} + \cdots + \gamma^k R_{\text{final}}$$

通过对每步单独定义 $r_t^{\text{turn}}$（verifiable 或 LLM-as-judge），把 trajectory-level 粒度降到 turn-level。

---

## 八、MIG — Marginal Information Gain（信息论视角）

**arXiv 2602.01034**

### 核心思路

**步骤的贡献 = 该步减少了多少对最终答案的不确定性**：

$$\text{MIG}_t = H(Y \mid C_{1:t-1}) - H(Y \mid C_{1:t})$$

其中 $C_{1:t}$ 是到第 $t$ 步为止的 CoT prefix，$Y$ 是最终答案。

- $\text{MIG}_t$ 越大 → 该步越有信息量，越关键
- $\text{MIG}_t \approx 0$ → 该步是废话，不应给 credit

### Monotonic Historical Watermark

防止后续步骤通过重复前面信息来 inflate credit：

$$r_t^{\text{process}} = \begin{cases} \text{MIG}_t & \text{if } \text{MIG}_t > \max_{t' < t} \text{MIG}_{t'} \\ 0 & \text{otherwise} \end{cases}$$

只有真正带来新信息的步骤才获得 reward。

### Decoupled Masking

$$\mathcal{L} = \mathcal{L}_{\text{CoT}}(\text{process reward}) + \mathcal{L}_{\text{output}}(\text{outcome reward})$$

CoT 部分用 MIG，final output 用 answer correctness——避免两种 reward 的混淆。

---

## 九、MCTS 在 Credit Assignment 中的角色

### 为什么 MCTS 天然解决 credit assignment

MCTS 通过 backup 显式计算每个节点的 $Q$ 值：

$$Q(s_t, a_t) \approx \frac{1}{N(s_t, a_t)} \sum_{\tau \ni (s_t,a_t)} R(\tau)$$

即经过 $(s_t, a_t)$ 的轨迹的平均末端 reward——这直接就是 step-level value。

```mermaid
graph TD
    S0 --> |"a₁, Q=0.7"| S1
    S0 --> |"a₂, Q=0.3"| S2
    S1 --> |"a₁₁, Q=0.8"| T1
    S1 --> |"a₁₂, Q=0.6"| T2
    S2 --> |"a₂₁, Q=0.4"| T3
    
    style T1 fill:#6f6
    style T2 fill:#9f9
    style T3 fill:#f96
```

### 实际应用模式

**离线数据增强**（推荐）：
1. 用 MCTS 在训练环境搜索，收集带 step-level $Q$ 值的轨迹
2. 用 MCTS $Q$ 值训练 PRM
3. 用 PRM 提供 dense reward 给 RL 训练

**Agent Q**（Putta et al. 2024）：MCTS + DPO，在每个 state 下用 MCTS 找好/坏 action，自动构建 pairwise 数据。比纯 RL 数据效率更高。

**MCTS 的局限**：
- 计算代价高（每步搜索树）
- 不可回溯的真实环境（web browser/OS）无法使用——需要模拟器

---

## 十、HiPER — 层级 Advantage 估计

**Vault 已有单独笔记**

显式三层结构：

$$A_{\text{final}} = w_1 \cdot A_{\text{task}} + w_2 \cdot A_{\text{subtask}} + w_3 \cdot A_{\text{action}}$$

需要**预定义 subtask 边界**（人工标注或启发式规则）。

与 GiGPO 的对比：

| | HiPER | GiGPO |
|-|-------|-------|
| Subtask 边界 | 预定义（需人工） | 自动发现（anchor state）|
| 额外标注成本 | 高 | 无 |
| 适用前提 | 任务有自然分解结构 | 并行 rollout 中存在重合状态 |

---

## 十一、方法综合对比

| 方法 | 需要 Critic | 额外 Rollout | 精度 | 内存 | 适用场景 |
|------|-----------|------------|------|------|---------|
| GRPO（baseline） | ❌ | ❌ | 低（traj-level） | 低 | 短任务/简单工具 |
| GiGPO | ❌ | ❌ | 高（step-level） | 低（同 GRPO） | 多步 agent，需并行 rollout |
| AgentPRM | ❌ | ✅（MC rollout） | 高 | 中 | 中等 horizon，有计算预算 |
| Turn-level GRPO | ❌ | ❌ | 中高（turn-level） | 低 | 多轮工具调用 |
| LOOP | ❌ | ❌ | 中（GAE+discount） | 最低 | 长 horizon API agent |
| MIG | ❌ | ❌ | 高（信息论） | 低 | CoT 推理 |
| MCTS + PRM | ❌（PRM） | ✅（搜索树） | 最高 | 高 | reversible env，数据效率优先 |

---

## 十二、启发思考

### So What

Credit assignment 是 Agentic RL 能否 scale 到真实任务的关键瓶颈。GiGPO 的贡献不只是性能提升，更重要的是**证明了 step-level credit 可以无代价获得**——只需要并行 rollout 中的 anchor state，不需要额外的模型或数据。这是一个 elegant 的解法：用并行 rollout 中本就存在的信息（重合状态），解决了之前需要 critic 才能解决的问题。

### 局限与未解问题

1. **GiGPO 的 anchor state 密度**：如果并行 rollout 数量少，anchor state 很少，micro advantage 估计方差大。需要多少并行 rollout 才够？论文没有充分讨论
2. **连续状态空间的 anchor state 识别**：ALFWorld/WebShop 有离散状态，容易找 anchor；对于 OS/browser 这种连续状态，"相同状态"如何定义？
3. **PRM 的 scalability**：MC rollout 估 step value 计算贵，K 次 rollout × T 步 horizon = K×T 条轨迹。实践中 K 多大合适？
4. **稀疏 credit 的组合**：哪些方法可以组合？GiGPO + Turn-level？有没有 interference？

### 脑暴拓展

- **Implicit anchor state**：能不能用语义相似度（embedding distance）来识别"近似相同"的状态，扩展 GiGPO 到连续状态空间？
- **Credit assignment 用于 curriculum**：能不能根据哪些 step 的 credit variance 大（即"难以决策"的步骤），自动生成针对性训练数据？
- **Process reward 自举**：用已训练的 policy 给出 PRM 标注，再用这个 PRM 训练更好的 policy——类似 self-improvement loop，是否可行？

---

## 十三、落地应用

### 选型指南

```mermaid
flowchart TD
    A{有多少并行 rollout?} -->|足够多，同一任务能重复经过相同状态| B[GiGPO]
    A -->|不多| C{能做 MC rollout?}
    C -->|有计算预算| D[AgentPRM]
    C -->|算力紧张| E{任务 horizon 有多长?}
    E -->|中等，有自然 turn 边界| F[Turn-level GRPO]
    E -->|极长，API 调用为主| G[LOOP]
```

### 面试高频问法

**Q: 什么是 credit assignment 问题？LLM Agent 为什么特别难？**
> Credit assignment 是多步决策中 reward 归因的问题。LLM Agent 有三重困难：(1) 双重粒度——每步 action 是长文本，token-level 和 step-level 都需要归因；(2) 极长 horizon——PaperBench 700 轮，折扣累积 variance 爆炸；(3) 连续高维状态——对话历史无法精确比较，不同轨迹的"相同状态"难以识别。传统 RL 的 Bellman 方程在这里不再适用，需要新的解法。

**Q: GiGPO 怎么在不增加内存的情况下实现 step-level credit assignment？**
> GiGPO 的 insight：并行 rollout 中不同轨迹会多次经过相同 anchor state。在这些 state 上，把"从该 state 出发的不同 action"组成 step-level group，用各 action 对应轨迹的最终 reward 对比计算 micro advantage $A_{\text{micro}} = (R - \mu_{\text{anchor}})/\sigma_{\text{anchor}}$。不需要额外 rollout（用已有轨迹的重合状态），不需要 critic（无参数化 value model），内存 = GRPO。ALFWorld 上 +12% vs GRPO（arXiv 2505.10978）。

**Q: MCTS 在 Agent RL 里起什么作用？为什么不是主流？**
> MCTS 通过 backup 显式计算每个 (state, action) 的 Q 值，天然提供 step-level value，是最直接的 credit assignment 解法（Agent Q：MCTS + DPO）。不是主流的原因：(1) 计算贵，每步需要 rollout 树搜索；(2) 不可回溯的真实环境（web browser/OS）无法使用，必须有模拟器或 virtual env。只在有 reversible env 且数据效率要求高时才推荐。

**Q: PRM 和 ORM 的区别，什么时候用哪个？**
> ORM 对整条轨迹末端打分，简单但稀疏；PRM 对每步打分，dense 但需要 step 标注。实践原则：短任务 / 有中间检查点 → ORM 够用；长 horizon / reward 极稀疏 → 需要某种形式的 PRM；标注充足 → 离线训练 PRM；标注不足 → GiGPO（无标注）或 AgentPRM（MC rollout 自动估）。

---

## 推荐阅读

### 原始论文
- **GiGPO**: https://arxiv.org/abs/2505.10978
- **AgentPRM**: https://arxiv.org/abs/2502.10325 | 代码: https://github.com/sanjibanc/agent_prm
- **LOOP**: https://arxiv.org/abs/2502.01600
- **Turn-Level Reward**: https://arxiv.org/abs/2505.11821
- **MIG**: https://arxiv.org/abs/2602.01034
- **PA-MoE**: https://arxiv.org/abs/2602.17038

### 相关 Vault 笔记
- [[AI/LLM/RL/Other-Algorithms/HiPER-Hierarchical-RL-Credit-Assignment|HiPER]] — 层级 advantage 方法
- [[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — GiGPO 基础上加 phase-aware MoE
- [[AI/Agent/Agentic-RL/Tool-Use-RL-训练专题|Tool-Use-RL 训练专题]] — Turn-level reward 详细展开
- [[AI/Agent/Agentic-RL/KLong-Extremely-Long-Horizon-Agent|KLong]] — 任务分解层面的解法
- [[AI/Agent/Agentic-RL/Agent-RL-训练实战指南|Agent-RL 训练实战指南]] — 训练实战中的坑
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 综合分析]] — 宏观框架
