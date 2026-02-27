---
title: Agentic RL 2026 前沿综合分析 — 五大维度与对应解法
brief: Agentic RL 五大核心维度综合分析：Credit Assignment（GRPO→LOOP→GiGPO→AgentPRM→iStar→MIG→HiPER→CSO，10方案全覆盖，含反事实验证新维度）/ Reward Design（verifiable/unverifiable/checklist + Search-R1++ + SELAUR uncertainty）/ Environment Engineering / Workflow Design / Context Overflow；Multi-Turn RL 四支柱（TSR/Credit/SCoRe/ERL）；失败轨迹利用三维谱系（CSO深/ERL中/SELAUR浅）；训练失败模式跨模态谱系（Echo Trap文本 / Interaction Collapse多模态，PyVision-RL）；面试武器级综述（v10，2026-02-25）
date: 2026-02-21
updated: 2026-02-26-v11
type: synthesis
tags:
  - agentic-RL
  - credit-assignment
  - reward-design
  - environment
  - workflow-design
  - topology
  - synthesis
  - 2026
related:
  - "[[AI/2-Agent/Multi-Agent/Kimi-K2.5-PARL|Kimi-K2.5-PARL]]"
  - "[[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2]]"
  - "[[AI/2-Agent/Agentic-RL/HiPER-Hierarchical-Plan-Execute-RL-Credit-Assignment|HiPER（ICML 2026）]]"
  - "[[AI/2-Agent/Agentic-RL/EnterpriseGym-Corecraft|EnterpriseGym-Corecraft]]"
  - "[[AI/3-LLM/RL/算法/OpenRS-Pairwise-Adaptive-Rubric|OpenRS-Pairwise-Adaptive-Rubric]]"
  - "[[AI/2-Agent/Agentic-RL/FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer-CWRPO-Workflow-Orchestration-RL]]"
  - "[[AI/2-Agent/Multi-Agent/AgentConductor-Topology-Evolution|AgentConductor]]"
  - "[[AI/2-Agent/Agentic-RL/SquRL-Dynamic-Workflow-Text-to-SQL|SquRL-Dynamic-Workflow-Text-to-SQL]]"
  - "[[AI/2-Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE-Phase-Aware-Mixture-of-Experts]]"
---

# Agentic RL 2026 前沿综合分析 — 五大维度与对应解法

> v2.0（2026-02-21）：框架从「三大难题」升级为「四大维度」，新增 Workflow/Topology 设计维度，补充 FlowSteer/AgentConductor/SquRL/PA-MoE 等新工作。
> v3.0（2026-02-23）：新增第五维度 Context Overflow（KLong），Credit Assignment 谱系升级为 6 方案全覆盖，新增 TSR/iStar/MIG/CM2 完整分析，补充 unverifiable reward 完整解法谱系。
> v4.0（2026-02-24）：训练算法维度新增 Multi-turn RL 稳定性专项（RAGEN/StarPO Echo Trap + StarPO-S 三机制 + Rollout 三因子）；Workflow 维度补充 AdaptOrch（推理时编排拓扑自适应路由）；全景表更新至 2/24。
> v5.0（2026-02-24）：Credit Assignment 谱系新增 **HiPER（ICML 2026，★★★★★）**——显式层级化新维度（subgoal-segment 粒度 vs 原有 step/trajectory 粒度）；HAE 有双重理论保证（无偏性 + 方差减少）；ALFWorld 97.4% SOTA；更新全景表 HiPER 评分；新增 AWM 环境工程条目；更新"完整地图"加入 subgoal-segment 层级。
> v6.0（2026-02-25）：**iStar 正式入表**（2509.19199，Tongyi Lab，★★★★★）——trajectory DPO ≡ step-wise BT model 理论，唯一适用于 unverifiable reward 的 step-level credit assignment 方案，SOTOPIA +48%；新增 **Search-R1++ 关键实验发现**（2602.19526）到 Reward Design 部分：REINFORCE > PPO > GRPO 稳定性，F1 reward 导致 answer avoidance，action-level penalty 可修复；全景时间表新增两条。
> v7.0（2026-02-25）：**CSO 入表**（2602.03412，Tencent AI Lab+HKU，★★★★☆）——Credit Assignment 第三信号维度：从失败轨迹反事实验证，只监督 16% 关键步骤 DPO，GAIA-Text 8B 超 GPT-4.1；谱系总结新增"失败轨迹维度"分支；v6 对比表补 CSO 行。
> v8.0（2026-02-25）：**ERL、CM2、TSR、SCoRe 正式入表**；Multi-Turn RL 三支柱升级为**四支柱**（新增 ERL 反思-内化支柱）；全景表补齐 2/25 五条新作。ERL（2602.13949，USC+Microsoft+UPenn）= experience-reflection-consolidation 循环嵌入 RL 训练，部署时零成本（SFT 蒸馏内化），Sokoban +81%，HotpotQA +11%；CM2（2602.12268）= Checklist Rewards + Sparse/Dense 解耦，multi-turn tool use；SCoRe（ICLR 2025）= 两阶段 KL 约束删除假纠错均衡。
> v9.0（2026-02-25）：**SELAUR 入表**（2602.21158，JHU+ASU+Purdue）——失败轨迹 token-level 不确定性 reward shaping，零额外模型成本；新增「失败轨迹利用深度谱系」：SELAUR（浅·零成本）→ ERL（中·反思循环）→ CSO（深·反事实验证）；SELAUR 与 GiGPO 正交互补（成功信号精化 + 失败信号激活 = 完整 credit 覆盖）。
> v11.0（2026-02-26）：**「关键决策天然稀疏」跨域实证固化**：CSO（Agent RL，16% critical steps）+ SIA（ICML 2026，推理时对齐，20% Junction token）从不同领域独立验证同一原则；补充到 CSO 面试补充段落；Papers/多Agent集体行为安全（Collective Behaviour+Colosseum）双向闭合，Wisdom层元问题笔记增加实证验证链。
> v13.0（2026-02-27）：**Agent 进化模式谱系三层框架建立**（老板指令）；Reflexion/ExpeL/AgentQ 三篇 in-context 进化奠基论文入库。；Search-P1 路径级密集奖励加入 Reward Design 时间轴（v13.1）
> v12.0（2026-02-27）：**SORL 入表**（2511.20718，Texas A&M，★★★★☆）——Off-policy multi-turn RL 崩溃诊断（粒度错配+方差累积两根因）+ 修复（Turn-Level IS 均值替代乘积 + CTN 自适应惩罚）；训练稳定性章节补充 off-policy 专项解法；更新 See Also 导航体系。
> v10.0（2026-02-25）：**PyVision-RL 入表**（2602.20739，多模态 Agentic RL）——提出 Interaction Collapse（Echo Trap 的多模态版本：模型学会减少工具调用规避复杂性），Oversampling-Filtering-Ranking + Accumulative Tool Reward 修复；On-Demand Context Construction 解决视频 token 爆炸；跨模态验证了"RL 压力推向退化策略"根因的普遍性，新增训练失败模式跨模态谱系。

> 这篇笔记是对 2026 年 2 月集中涌现的 Agentic RL 工作的综合理解，不是论文列表，是一个框架。

---

## 核心框架（v2 升级）

v1 的「三大难题」框架（环境/Reward/算法）捕捉到了早期工作的主要分野。但 2/17-20 密集涌现的新一批论文揭示了**第四个维度**：

> **Workflow/Topology 设计本身就是 agent 能力的决定变量**，不亚于算法或 reward。

升级后的框架：

```
Agentic RL 训练 = 环境 × Reward × Workflow/Topology × 算法

原三大难题保持不变，新增第四维度：
4. Workflow/Topology 问题：静态设计的 pipeline 是性能瓶颈而非模型能力
```

---

## 为什么 Agentic RL 现在是最热的方向

RLVR（Reinforcement Learning with Verifiable Rewards）在数学/代码等**有单步可验证答案**的任务上已经工作得很好（DeepSeek-R1、Kimi-k1.5、QwQ 等）。但真实世界的 agent 任务几乎没有"一眼看出对错"的 reward：

- 帮用户订机票（需要查询、对比、确认——哪一步算成功？）
- 修复代码 bug（需要理解代码库、定位问题、验证修复——怎么衡量中间步骤的质量？）
- 进行市场调研（需要搜索、综合、判断相关性——完全 open-ended）

这个 gap——**从单步可验证任务到多步开放任务的跨越**——就是 Agentic RL 的核心研究空间。

## 三大核心难题

用一个统一的框架来看当前 Agentic RL 的挑战：

```
Agent RL 训练 = 环境 × Reward × 算法

1. 环境质量问题：toy 环境 → toy agent，没有泛化
2. Reward 设计问题：开放任务缺乏可信号的 reward
3. 算法稳定性问题：multi-step / multi-agent 导致优化不稳定
```

---

## 难题 1：环境质量决定泛化上限

### 问题
大多数 agentic RL 的训练环境是合成的、简化的、与真实任务差距很大。在这类环境上训练出来的 agent，在真实场景下表现糟糕——不是模型不够聪明，是它没见过真实任务的复杂性。

### 2026 年的解法：EnterpriseGym Corecraft（Surge AI, 2602.16179）

- **2500+ 真实实体，23 种工具**，模拟企业客服完整业务流程
- **Expert-authored rubrics** 使 reward 计算可靠（不依赖 LLM judge）
- **Task-centric world building**：环境设计以任务多样性为核心

**关键 empirical finding**：在这个高保真环境上用 GRPO 训练 GLM 4.6，**单 epoch** 后在 3 个独立 OOD benchmark 上泛化（+4.5%/+7.4%/+6.8%）。

**核心 insight**：
> 环境质量决定了 agent 能学到的 skill 的上限。Toy 环境的 reward 太容易 hack，agent 学到的是"在这个环境里得高分的策略"，而不是"如何完成这类任务的通用能力"。

### 延伸思考
这个发现对 RL 实践者的启示：**在更小的 model 上用更好的环境训练**，可能比在更大的 model 上用平庸的环境训练更有效。这直接挑战了"scale is all you need"的直觉。

---

## 难题 2：开放任务缺乏可靠 Reward

### 问题
RLVR 的成功依赖于"ground truth 答案可验证"。但开放任务（工具调用、客服、研究）：
- 没有单一正确答案
- 中间步骤质量难以自动评估
- 最终结果可能有多种正确路径

用 LLM-as-judge 有一致性问题（同一 judge 对同一输出可能给不同分）；用人工标注成本极高。

### 三种解法并行出现：

**解法 A — Checklist Reward（CM2, 2602.12268）**
把"判断这个 agent 行为好不好"转化为"检查若干 binary criteria"：
```
原始问题：这轮 tool call 质量如何？（open-ended, 主观）
转化后：
  □ 是否在正确时机调用了工具？
  □ 参数格式是否正确？
  □ 是否处理了 error case？
  □ 是否在调用前说明了意图？
```
把 open-ended judging → classification-style，可靠性大幅提升。

**解法 B — Rubric-based Reward（OpenRS, 2602.14069）**
不把 reward 学进 judge model，而是**显式推导出 rubric**（评分标准），每次评分时在 rubric 下执行推理：
```
固定 judge：内化了评分逻辑，无法检查 → 黑盒
Rubric-based：每次评分展示推理过程 → 可检查 + 可解释
```
解决了 reward generalization 问题（rubric 可以跨任务迁移）。

**解法 C — Expert Rubrics in Environment（EnterpriseGym Corecraft）**
把 rubric 编码进**训练环境**，而不是评估器。这样 reward 在训练时就已经可靠，不需要事后纠正。

**三种解法的适用场景**：
| 解法 | 优势 | 适用 |
|---|---|---|
| Checklist (CM2) | 细粒度，密集 reward | 工具调用、API 使用 |
| Rubric-based (OpenRS) | 可解释，跨任务泛化 | 通用对齐、open-ended QA |
| Expert rubrics in env (Corecraft) | 最可靠，OOD 泛化强 | 专业领域（需要专家投入）|

**解法 D — Search-R1++ 关键实验发现（2602.19526，v6 新增）**

对 Deep Research agent（multi-round retrieval + generation）的系统性消融，沿三个维度解耦：

- **Prompt template**：Fast Thinking 比 Slow Thinking 稳定性更高，性能更好（直觉：search agent 不需要深度 chain-of-thought，需要快速决策）
- **Reward function**：F1-based reward 导致**训练崩溃**（answer avoidance：model 学会不给答案以避免 partial match 扣分）→ EM reward 更稳；加入 **action-level penalty**（对不必要搜索惩罚）后 F1 reward 可超过 EM
- **Policy optimization**：**REINFORCE > PPO > GRPO**（稳定性），GRPO 是三者中最不稳定的（搜索任务中 group sampling 方差大）；REINFORCE 搜索动作更少（更高效）

**关键 takeaway（对 Tool Use RL 研究有重要意义）**：
1. GRPO 在 multi-turn search agent 训练中并不是最优选择——这与其在单轮推理任务上的主导地位形成反差
2. Reward 函数设计要避免给 partial output 空间（F1 的 recall 分量）：会诱发 answer avoidance
3. Action-level penalty（对工具调用的成本惩罚）是一个被低估的 reward 组成部分

Search-R1++ baseline：Qwen2.5-7B 从 0.403 → 0.442（+9.7%），Qwen2.5-3B 从 0.289 → 0.331（+14.5%）。

**解法 E — Uncertainty-Intrinsic Reward（SELAUR, 2602.21158，v9 新增）**

上述四种解法都在处理"如何给成功行为设计 reward"。SELAUR 换了视角：**从失败轨迹中提取内生学习信号**。

问题：标准 RLVR 对失败轨迹给 reward=0 就不学了，丢弃了失败过程中的不确定性信息。  
解法：用 LLM 自身的 token 预测概率分布，估计三维不确定性（entropy/least-confidence/margin），把失败步骤变成密集 reward：

```
失败轨迹的 step reward = w_t · u_t  (w_t=0.95，确保 < 成功 reward)
成功轨迹保持原始 reward 不变
```

**失败轨迹利用的三层深度谱系**（新结构，整合 CSO/ERL/SELAUR）：

| 层级 | 方法 | 信号来源 | 成本 | 可靠性 |
|------|------|---------|------|--------|
| Logits 层 | SELAUR (2602.21158) | token 概率分布不确定性 | 零额外成本 | 较低（未区分认知/偶然不确定性）|
| 反思层 | ERL (2602.13949) | 自生成反思 Δ → 指导重试 | 中（额外 LLM 调用）| 中（依赖反思质量）|
| 验证层 | CSO (2602.03412) | 反事实验证 + Expert 替换 | 高（expert model + rollout）| 高（可验证的因果证据）|

适用场景：reward 极稀疏 + 失败率高 + 资源有限时，SELAUR 是工程首选；资源充足时，ERL/CSO 信号更可靠。

→ 详见：[[AI/2-Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards|SELAUR（2602.21158）]]

**Reward Design 完整地图（v9，截至 2026-02-25）**：

| 类型 | 代表方案 | 适用场景 |
|------|---------|---------|
| verifiable_binary | GiGPO / GRPO / Search-R1 | 有 ground truth 的任务 |
| unverifiable_implicit | iStar（DPO ≡ step-BT）| 开放环境，无 ground truth |
| unverifiable_checklist | CM2 | 多轮 tool use，结构化标准 |
| process_reward | AgentPRM / iStar | 需要 step 级别信号 |
| action_level_penalty | Search-R1++ | 防止不必要工具调用 |
| uncertainty_intrinsic | SELAUR | 失败率高，内生密集信号 |

---

## 难题 3：Multi-Step/Multi-Agent 训练不稳定

### 问题
在长 horizon 任务或多 agent 系统中，标准 RL（PPO/GRPO）面临：
- **Credit assignment**：最终 reward 传播经过太多步骤，梯度信号极度稀疏
- **Serial collapse**：在多 agent 系统中，串行 rollout 导致训练极慢
- **Optimization instability**：multi-agent 中策略相互依赖，联合训练不稳定

### 解法 0：Echo Trap 诊断与 StarPO-S（RAGEN, 2504.20073）

在讨论如何解决 multi-turn RL 不稳定之前，必须先回答：**"不稳定"的具体失败模式是什么？**

RAGEN（Northwestern/Stanford Li Fei-Fei/Yejin Choi/Jiajun Wu + Microsoft + NYU Kyunghyun Cho）是第一篇系统诊断这个问题的工作：

**Echo Trap（回声陷阱）**：multi-turn RL 特有的失败模式，三联征同时出现：
1. **reward variability collapse**：所有 rollout 的 reward 趋同，batch 级别梯度趋零
2. **entropy drop**：policy 输出熵急剧下降，陷入固定模板
3. **gradient spike**：间歇性梯度爆炸

根本机制：agent 一旦找到局部有 reward 的策略模板，就自我强化进入该模板。RL 的优化压力反而放大了这个捷径，同时压制探索。类比 Shumailov et al. 2024 的 model collapse——但是在线动态版本。

**重要实验发现**（RAGEN 的四个环境）：
- PPO 在确定性环境（Bandit/Sokoban）比 GRPO 更稳（critic 提供平滑 value estimate）
- GRPO 在随机环境（Frozen Lake）反而更稳（随机性让 state value 难估，PPO critic 引入错误）
- WebShop 两者都行（强语言先验，高初始 reward，对 critic 依赖低）
- 核心结论：**没有一种算法天然适合所有 multi-turn agent 任务**

**StarPO-S 三机制**（针对 Echo Trap 三联征的逆向设计）：
```
reward homogenization  → Variability-based Trajectory Filtering（保留 top-p% reward std 的 prompt）
gradient variance      → Critic Baselining（轻量 trajectory-level baseline）
ratio explosion        → Decoupled Clipping（per-turn 分别控制 clip range）
```

**Rollout 三因子**（决定 self-evolution 质量）：
1. **Diverse initial states**：多样初始状态 × 多条 rollout/state（P 的多样性 > N 的数量）
2. **Medium granularity**：每 turn 执行多个 sub-action（非单 token，非整 episode）
3. **High rollout frequency**：接近全 on-policy，避免 off-policy ratio 积累

**Finding 3（最有价值）**：即使格式中强制 `<think>` token，纯 outcome reward 下 agent 会绕过推理（shortcut）或产生 hallucinated reasoning。**Emerging reasoning 不是 multi-turn RL 的免费午餐**。

RAGEN 是 multi-turn training stability 这条研究线的奠基工作（2025年4月提交），后续 TSR、HiPER、KLong、LOOP 均从此出发。

### 解法 A：时间维度分层（HiPER, 2602.16165）

把 policy 分为 Planner（subgoal 级）和 Executor（action 级），分别计算 advantage：
```
传统 GAE：reward 从 T 步反向传播到 step 1，信号极稀疏
HAE：reward 先在 subgoal 内聚合 → 再从 subgoal 级反传到 planner
```
方差缩减有理论证明，ALFWorld 97.4%（+6.6%），WebShop 83.3%（+8.3%）。

### 解法 B：空间维度冻结（PARL / Kimi K2.5, 2602.02276）

在 multi-agent 系统中，**冻结 subagent，只训练 orchestrator**：
```
联合训练（有问题）：orchestrator + subagent 同时更新 → 优化目标互相干扰
PARL：subagent 固定 → orchestrator 学如何分解任务 + 创建 subagent
```
解决了 credit assignment + training instability。Agent Swarm 最多 100 subagent，延迟降 4.5x。

### 解法 C：Training-time Tree Search（TSR, 2602.11767, ICML 2026）

把 test-time 树搜索移入 training-time rollout 阶段：每个 turn 采样候选动作集 $\mathcal{A}_t = \{a_t^{(1)},\dots,a_t^{(M)}\}$，用 scoring function 选高质量动作构建轨迹。

**三种搜索策略**：
- **Best-of-N**：独立采样 N 条完整轨迹，选 reward 最高的（baseline，最简）
- **Beam Search**：每步维护 B 个高分前缀 beam，逐步筛选，可以在中途纠错（最强，适合确定性环境）
- **Shallow Lookahead**：评估动作时额外展开 D<<K 步，前瞻性更强（计算折中，适合随机环境）

**配合 Instance Filtering**：按 outcome uncertainty $U(u;\pi_\theta) = \text{Std}[R(\tau)]$ 筛选训练样本，只保留"有时成功有时失败"的 hard cases。

$$\text{Rollout quality} \uparrow \Rightarrow \text{Training signal quality} \uparrow \Rightarrow \text{Multi-turn RL stability} \uparrow$$

- Optimizer-agnostic，兼容 PPO/GRPO，ICML 2026
- **0.5B+TSR ≈ 3B 无 TSR**（+15% 提升，Sokoban/FrozenLake/WebShop 三环境一致）
- **核心命题：rollout 质量是 multi-turn RL 的第四个被忽视的训练变量（与算法/reward/credit assignment 正交）**
- 详见：[[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR 深度笔记]]

### Credit Assignment 完整谱系（v3 新增，截至 2026-02-23）

这是 Agentic RL 里最核心的子问题，2/23 已实现 6 方案全覆盖：

| 方案 | 论文 | 类型 | 依赖 | 核心机制 | 适用场景 |
|------|------|------|------|---------|---------|
| GRPO | baseline | trajectory-level | 无 | group relative advantage | 单轮/短 horizon |
| LOOP（LOO-PPO）| 2502.01600 | trajectory-level | 无（leave-one-out）| 免 critic 的 trajectory baseline | 长 horizon，32B > o1+9% |
| GiGPO | 2505.10978 | step-level（anchor）| 无额外 rollout | 重复经过同一状态 → 天然对比，hashmap O(n) | 结构化环境（ALFWorld +13.3%，NeurIPS 2025）|
| AgentPRM | 2502.10325 | step-level（MC rollout）| 额外采样 | MC 估计 step Q-value，显式 PRM 网络 | 充足计算，3B > GPT-4o |
| **iStar** | **2509.19199** | **step-level（implicit PRM）** | **2x 模型参数，无额外 rollout** | **trajectory DPO ≡ step-wise BT model（理论保证），rolling reference = π_old** | **✅ unverifiable reward（SOTOPIA vs GPT-4o +48%），开放环境** |
| MIG | 2602.01034 | step-level（信息论）| verifiable reward | Marginal Information Gain + Monotonic Watermark，只奖励真正语义突破 | OOD 泛化，防 reward hacking |
| **HiPER** | **2602.16165** | **subgoal-segment-level（层级）** | **sparse end-of-trajectory** | **Plan-Execute Interface + HAE（三类 advantage：switch/high/low），方差减少定理保证** | **长 horizon（ALFWorld 97.4% SOTA，ICML 2026）** |
| SeeUPO | 2602.06554 | 回合级（逆序更新）| multi-turn | 逆序更新（T→1）+ 无 group variance normalization，REINFORCE+GRAE 理论保证 | multi-turn RL 收敛（AppWorld +43-54%）|
| SHARP | 2602.08335 | 横向 multi-agent | multi-agent | Shapley value + counterfactual masking，三层 reward，per-agent norm | 多 agent 精确归因（ICML 2026）|
| **CSO** | **2602.03412** | **step-level（反事实验证）** | **需 expert + 验证 rollout** | **失败轨迹→PRM 定位弱点→expert 替代→policy rollout 验证成功→只监督 16% 步骤 DPO** | **✅ 失败轨迹利用；有 expert model 可用；GAIA-Text 8B 超 GPT-4.1** |

**谱系总结（v5 更新）**：
- GRPO → LOOP：从 trajectory 到更好的 trajectory baseline（免 critic）
- LOOP → GiGPO：从 trajectory 到 step-level（利用状态重叠，免额外 rollout）
- GiGPO → AgentPRM：step-level 的两种路：anchor grouping（免 rollout）vs MC rollout（显式 Q-target）
- AgentPRM → iStar：从显式 PRM 到隐式 PRM（DPO），从 verifiable 到 unverifiable reward
- iStar → MIG：从相对比较到信息论定义的"突破奖励"（防 pump-and-dump）
- **GiGPO → HiPER（新路线）**：从 step 粒度 flat RL → subgoal-segment 粒度 hierarchical RL，显式 Plan-Execute interface，HAE 有双重理论保证
- **纵向 vs 横向（v5 新）**：GiGPO/AgentPRM/iStar/MIG/HiPER/SeeUPO 解决单 agent 内的时间维度 credit assignment；SHARP 解决 multi-agent 横向的 agent 间归因
- **CSO：失败轨迹维度（v7 新）**：上述所有方案都从成功轨迹学习；CSO 是首个系统性从失败轨迹出发的方案——"什么步骤换一个动作能让整件事成功"（反事实因果推断），与成功轨迹信号互补

**关键维度对比**：

| | 需要额外 rollout | 需要 verifiable reward | 状态重叠假设 | 额外模型 |
|---|---|---|---|---|
| GiGPO | ❌ | ✅ | ✅（语言空间罕见）| ❌ |
| AgentPRM | ✅ | ✅ | ❌ | ✅（PRM 网络）|
| iStar | ❌ | ❌（支持 unverifiable）| ❌ | ✅（implicit PRM）|
| MIG | ❌ | ✅ | ❌ | ❌（信息论计算）|

**面试一句话**：iStar 是目前 label-efficient + unverifiable reward 支持最好的方案（DPO≡step-BT 理论保证）；GiGPO 是计算最轻量的方案（免额外 rollout，hashmap O(n)）；MIG 是信息论视角最优雅的方案；HiPER 是 subgoal-segment 粒度的代表（ALFWorld SOTA）。选择原则：有状态重叠 → GiGPO；有 unverifiable reward → iStar；想要理论保证 + 长 horizon → HiPER。

**关键维度对比（v6 更新）**：

| | 需要额外 rollout | 需要 verifiable reward | 状态重叠假设 | 额外模型 | 理论保证 |
|---|---|---|---|---|---|
| GiGPO | ❌ | ✅ | ✅（语言空间罕见）| ❌ | 无偏梯度（paper） |
| AgentPRM | ✅ | ✅ | ❌ | ✅（PRM 网络）| 无 |
| **iStar** | **❌** | **❌（✅ unverifiable）** | **❌** | **✅（implicit PRM）** | **DPO≡step-BT model** |
| MIG | ❌ | ✅ | ❌ | ❌（信息论计算）| 信息论上界 |
| HiPER | ❌ | ✅（trajectory）| ❌ | ❌（HAE 计算）| 无偏性 + 方差减少 |
| **CSO** | **✅（验证 rollout）** | **✅（可验证结果）** | **❌** | **✅（expert model）** | **反事实因果（empirical）** |

**面试补充（v7）**：CSO 的独特角色——其他方案都问"什么做对了"，CSO 问"什么换掉后能成功"，是 Credit Assignment 谱系里唯一开采失败轨迹的方案。16% 关键步骤 = 高熵步骤原则在 Agent 领域的首次系统验证。**跨域印证（v11）**：同一天 SIA（ICML 2026，arXiv:2602.21215）独立发现推理时对齐也是 sparse control problem——20% Junction token（高熵节点）承担 100% 对齐效果。CSO 16%（Agent RL credit） + SIA 20%（Inference Alignment）= **「关键决策天然稀疏」的跨领域双重实证**，面试时引用这个跨域一致性远比单讲论文更有深度。见：[[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（2602.03412）]] + [[AI/3-LLM/Inference/SIA-Sparse-Inference-time-Alignment|SIA（2602.21215）]]

### Multi-Turn RL 四支柱（v8 新增）

把 multi-turn RL 的挑战分解为四个正交维度，每个维度有标志性解法：

```
支柱 1 — Rollout 质量: TSR (ICML 2026)
  训练时树搜索（Best-of-N / Beam / Lookahead）替代 naive random rollout
  Instance Filtering 过滤确定性 case，保留 hard cases
  Optimizer-agnostic（PPO/GRPO 均兼容）；0.5B+TSR ≈ 3B 无 TSR

支柱 2 — Credit Assignment: GiGPO/HiPER/iStar/CSO
  精确步骤级信号（从 trajectory-level 到 step/subgoal/反事实 维度）
  详见 Credit Assignment 谱系表（10 方案全覆盖）

支柱 3 — 均衡控制: SCoRe (ICLR 2025)
  自我纠错是多均衡优化：真正纠错 vs 假纠错刷分
  Phase 1 KL 约束通过语义锁定删除假纠错均衡的可行域
  Phase 2 放开训练，只有"真纠错"方向是可行解

支柱 4 — 反思内化: ERL (2602.13949)  ← v8 新增
  每 episode 显式反思循环（experience→reflection→consolidation）
  RL 对齐两次尝试+反思；SFT 蒸馏成功 y² 进 base policy
  部署时零额外成本（蒸馏内化）；跨 episode 反思记忆 m
  Sokoban +81%，HotpotQA +11%；稀疏 reward + 未知动态场景增益最大
```

**四支柱正交可叠加**：TSR（好的 rollout）+ Credit（好的信号）+ SCoRe（稳定均衡）+ ERL（反思学习）理论上可以共同使用。

### 统一视角

HiPER / PARL / TSR 解决"训练过程稳定性"，Credit Assignment 谱系解决"信号粒度不足"，SCoRe 解决"多均衡选择问题"，ERL 解决"稀疏反馈的反思内化"——四个维度正交，最优解是叠加使用（TSR + iStar + ERL 是值得探索的组合）。

---

---

## 难题 4：静态 Workflow/Topology 是性能瓶颈

### 问题

2/17-21 涌现了一批共同指向同一根本问题的论文：

> **「最佳模型能力」和「最佳任务表现」之间存在 workflow gap**——不是 LLM 本身的问题，是 workflow 设计的问题。

具体表现：
- 同一个 7B 模型，用不同 workflow 可以差出 15%+ 的 pass@1
- 复杂任务需要 dense cross-agent DAG，简单任务只需 chain，静态选择一种必有损失
- 单一 policy 的 simplicity bias：agent 对所有难度的任务都用相同参数量应对

### 四种解法从不同粒度切入：

**解法 A — FlowSteer CWRPO（2602.01664）: Operator 级别**

把数学解题分解为 operator 序列，用 RL 学编排顺序：
```
核心创新：条件释放设计
R(τ) = R_struct + I[R_struct ≥ θ] × λ × R_ans
                    ↑
        只有结构质量达标，才给 correctness reward
```
切断了 "shortcut 答案 bypass 结构质量" 的奖励路径。

**解法 B — AgentConductor（2602.17100）: Agent 通信 Topology 级别**

用 RL 训练 3B orchestrator 为每道题生成 YAML 格式的 DAG：
```
关键发现：
- 简单题：sparse chain（密度低），节省 68% token
- 难题：dense cross-layer DAG（密度高）
- density function = f(task_difficulty)
```
三个指标同向改善：pass@1 +14.6%，density -13%，cost -68%。**这是 "越难题用越复杂图" 的第一次 formalization。**

**解法 C — SquRL（2602.15564）: Workflow 选择级别**

形式化证明动态 workflow 选择的理论优势（Theorem 3.1）：

$$\text{EX}_{\text{dynamic}} \geq \text{EX}_{\text{static}}，\Delta = 0 \text{ iff 某个 workflow 覆盖所有 success regions}$$

Oracle evaluation 显示动态选择上界达到 81.5%，远超任何单一静态 workflow。
核心机制：**Dynamic Actor Masking**（随机 dropout actors，强迫探索更多 workflow 组合）。

**解法 D — PA-MoE（2602.17038）: Expert 路由级别（Phase-Aware MoE）**

单一 policy 的 simplicity bias 根源：不同任务 phase 需要不同 skill，但同一 policy 用同一参数覆盖所有 phase：
```
Phase 识别：CrossAttn(obs, goal) + LSTM(action history)
路由粒度：8次/episode（比 token-level 的45次 更合适，比 trajectory-level 的3次 更细）
效果：1.5B PA-MoE > 7B baseline
```

**解法 E — AdaptOrch（2602.16873）: 推理时编排拓扑路由**

以上 ABCD 四种解法都是**训练时**的 workflow 优化。AdaptOrch 则解决**推理时**的问题：当 LLM 性能趋于收敛（frontier 模型 MMLU 相差 <3%），如何编排多 agent 的结构拓扑变得比选哪个模型更重要。

**核心形式化**（Performance Convergence Scaling Law）：
$$\frac{\text{Var}_\tau}{\text{Var}_M} \geq \frac{(\omega(G_T)-1)^2}{4\epsilon^2 k} \cdot (1-\gamma(G_T))^2$$

- $\epsilon$ → 0（模型收敛）时，拓扑方差/模型方差 → ∞
- 编码任务：拓扑影响是模型选择的 **20x**（数学可证）

**Topology Routing（Algorithm 1）**：基于任务依赖 DAG 的三个结构指标路由：
- $\omega$（parallelism width）: 最大反链宽度，可并行子任务数
- $\gamma$（coupling density）: 子任务间 context 耦合强度
- $\delta$（critical path depth）: 最长顺序执行路径

四种拓扑 → 线性时间路由决策 → SWE-bench/GPQA/RAG +12–23%（相同模型）

**与训练时方法的定位**：AdaptOrch 和 AgentConductor/SquRL 互补——训练时用 RL 学最优 workflow，推理时用结构路由选最优拓扑。前者优化 policy，后者优化 orchestration。

### 五种解法的定位比较

| 解法 | 时机 | 粒度 | 核心机制 | 代表任务 | 核心贡献 |
|------|------|------|---------|---------|---------|
| FlowSteer | 训练时 | Operator 序列 | 条件奖励门控 | 数学解题 | 切断 shortcut reward |
| AgentConductor | 训练时 | Agent 通信图 | RL 生成 DAG | 竞赛代码 | difficulty-aware density |
| SquRL | 训练时 | Workflow 选择 | Dynamic Actor Masking | Text-to-SQL | 理论证明 dynamic > static |
| PA-MoE | 训练时 | MoE expert 路由 | Phase-aware routing | ALFWorld/WebShop | 参数效率 |
| **AdaptOrch** | **推理时** | **Agent 编排拓扑** | **DAG 结构路由** | **SWE-bench/GPQA** | **convergence scaling law** |

**统一视角**：五者都在解决「固定结构无法适应任务多样性」的问题。训练时 ABCD 优化 policy 学习如何选择；推理时 E 用结构分析直接路由。最优实践是训练 + 推理两层优化叠加。

---

## 整合框架：2026 Agentic RL 研究地图（v3）

```
Agentic RL 训练 Pipeline
│
├── 🏗️ 维度 1：环境设计
│   └── EnterpriseGym Corecraft（高保真企业环境）
│       原则：task diversity + expert rubrics + realistic workflows
│
├── 🎯 维度 2：Reward 设计
│   ├── [Verifiable] ToRL/ARTIST/ASTRA/RC-GRPO
│   ├── [Unverifiable·step] iStar — trajectory DPO → implicit PRM step reward
│   ├── [Unverifiable·turn] CM2 — Checklist rewards（binary criteria decomposition）
│   ├── OpenRS — Rubric-based reward（可解释，跨任务泛化）
│   └── FlowSteer — 条件释放 reward（结构质量门控）
│
├── ⚙️ 维度 3：训练算法（Credit Assignment + 稳定性）
│   │
│   ├── Credit Assignment 谱系（9方案，v6 全覆盖含 iStar 正式入表）
│   │   ├── [轨迹级] GRPO → LOOP（LOO baseline，免 critic）
│   │   ├── [步骤级·anchor] GiGPO（状态重访 → 天然对比，免额外 rollout，NeurIPS 2025）
│   │   ├── [步骤级·MC] AgentPRM（MC rollout Q-target，显式 PRM，3B > GPT-4o）
│   │   ├── [步骤级·隐式 ★★★★★] iStar（trajectory DPO ≡ step-wise BT model，rolling ref=π_old，✅ unverifiable，SOTOPIA +48%，Tongyi 2025/09）
│   │   ├── [步骤级·信息论] MIG（Marginal Information Gain，Monotonic Watermark）
│   │   ├── [subgoal段级·层级] HiPER（Plan-Execute + HAE，双重理论保证，ICML 2026 ★★★★★）
│   │   ├── [回合级·理论] SeeUPO（逆序更新，GRAE+PPU 不可能定理，AppWorld +43-54%）
│   │   └── [横向·multi-agent] SHARP（Shapley + counterfactual masking，ICML 2026）
│   │
│   ├── 训练稳定性
│   │   ├── RAGEN/StarPO — Echo Trap 诊断 + StarPO-S 三机制（trajectory filtering / critic baselining / decoupled clipping）
│   │   ├── TSR — training-time tree search rollout（rollout quality → stability）
│   │   ├── SCoRe — Phase 1 KL 约束初始化（self-correction RL）
│   │   └── SORL — Off-policy multi-turn 专用：Turn-Level IS + CTN，实例化为 SO-PPO/SO-GRPO
│   │
│   └── Multi-agent RL
│       ├── PARL — Freeze subagents，只训练 orchestrator（Kimi K2.5）
│       ├── MAGRPO — Dec-POMDP + joint reward CTDE
│       ├── AT-GRPO — Agent-and-Turn-Wise Grouping
│       ├── MARS2 — diversity-as-scaling（2×32B > 72B）
│       └── Dr. MAS — per-agent normalization，防梯度爆炸（NTU）
│
├── 🔗 维度 4：Workflow / Topology 设计
│   ├── [训练时] AgentConductor — RL 生成 agent communication DAG（难度自适应密度）
│   ├── [训练时] SquRL — RL 动态选择最优 workflow 组合（Theorem 3.1 形式化证明）
│   ├── [训练时] PA-MoE — Phase-aware expert routing（8次/episode，1.5B > 7B baseline）
│   ├── [训练时] FlowSteer — Operator 级 workflow RL（条件奖励门控）
│   └── [推理时] AdaptOrch — 任务 DAG 结构性路由（Convergence Scaling Law，topology > model selection）
│
├── 📦 维度 5：Context Overflow（v3 新增）
│   └── KLong — Trajectory-splitting SFT + Progressive RL
│       ├── 解决：轨迹超过 context window 的物理边界问题
│       ├── 方案：固定 prefix + 渐进截断 + 重叠子轨迹 + 逐步延伸 timeout
│       └── 效果：106B 超 Kimi K2 Thinking 1T（10x 参数）11.28% on PaperBench
│
└── 📏 评估
    └── PaperBench / MLE-bench / SWE-bench / ALFWorld / WebShop / SynSQL / tau-Bench
```

---

---

## 跨域连接：Agentic RL 与 Safety 的汇合

2/19 的一篇论文（2602.17546）揭示了一个重要发现，虽然不直接是 agentic RL，但对 agent safety 有直接意义：

**Harmful intent 在 pre-generation activation 中线性可分（AUROC > 0.9）**

这意味着：
1. Agent 在调用工具、写 code、访问 memory **之前**，其内部状态已经编码了 intent
2. 可以用轻量 probe 在 generation 发生之前检测并拦截
3. 对于 agentic workflow，可以在每个 action step 之前插入 safety gate

这与盾卫项目的核心思路完全契合：**不是等 agent 输出有害内容再拦截，而是在 forward pass 中早期发现意图，零 inference overhead**。

---

---

## 难题 5：Context Overflow — 轨迹超过 Context Window（v3 新增）

### 问题

所有上述工作都隐含一个假设：**轨迹能放进 context window**。但 2026 年最复杂的 agent 任务（复现 ML 论文、长期 ML 竞赛）产生的轨迹长度远超 context window：

| 任务类型 | 代表 Benchmark | 典型运行时长 | Assistant Turns |
|---------|--------------|------------|----------------|
| Long-horizon | SWE-bench Verified | 分钟级 | 20–200 |
| **Extremely long-horizon** | **PaperBench, MLE-bench** | **6–12 小时** | **700+** |

"极长 horizon"的独特挑战：
- 单条轨迹物理上放不进 context，SFT 无法直接训练
- RL rollout 在 timeout 内无法完成，reward 拿不到
- Credit assignment 变得更极端稀疏（数千步 delay）

### 解法：KLong（NUS + MIT, 2602.17547，2026-02-19）

KLong 的两大核心技术：

**1. Trajectory-splitting SFT**：把超 context 轨迹切成重叠子轨迹

$$\tau^{(i)}_{\text{input}} = [\underbrace{p}_{\text{固定 prefix}}, s_{t_i}, a_{t_i}, \ldots, s_{t_i+L-1}, a_{t_i+L-1}]$$

- **固定全局 prefix $p$**：任务描述 + 论文阅读段在每个子轨迹开头（保留全局 intent）
- **渐进截断**：近期 history 全保留，远期 history 逐步丢弃
- **重叠（overlap）**：相邻子轨迹共享部分内容，保证连续性
- 效果：assistant turns 114.9 → **732.7**（6.4 倍）

**2. Progressive RL**：逐步延伸 task timeout

$$T^{(1)} < T^{(2)} < \cdots < T^{(M)} \quad (2h \to 4h \to 6h)$$

- 先从 2h timeout 学局部行为，建立 policy 基础
- 再扩展到 6h，接近真实任务规模
- 解决 pipeline imbalance（partial rollout + priority judge queue）

**实验结果**：KLong 106B 在 PaperBench 达 62.59，**超 Kimi K2 Thinking 1T（参数量 10 倍）11.28%**。

### 关键设计原则

> **Trajectory-splitting 的本质**：固定全局 intent（任务 + 论文理解），让局部历史可以被截断，只要全局目标清晰，agent 可以在任何子轨迹中维持方向感。

这和人类专家处理长任务的方式一致：随时可以忘记具体历史细节，但始终记得"我在做什么、为什么做"。

### 开放问题

- Trajectory-splitting 的 advantage 是在子轨迹 group 内计算，跨子轨迹的全局 credit assignment 仍未解决
- Progressive timeout 需要精心设计阶段划分，过渡时机敏感
- Research-Factory（用 Claude Thinking distill 训练数据）引入了 teacher-student gap，KLong 上限被 teacher 能力锁定

---

## 2026 年还没解决的问题

诚实说，即使有上面这些工作，以下问题仍然 open：

1. **Subgoal 如何自动生成**：HiPER 没说 planner 如何确定 subgoal 边界。这是 hierarchical RL 的老问题。
2. **Expert rubric 的成本**：Corecraft 需要专家手写 2500+ 实体的 rubric。真正通用的 agentic RL 需要自动生成或归纳 rubric。
3. **真实环境 vs 模拟环境的 gap**：所有工作都在模拟环境里训练，真实企业系统的 non-determinism 和 side effects 会带来新的挑战。
4. **长任务的 overthinking**：LACONIC 解决了 reasoning 太长的问题，但 agent 任务的"overthinking"（不必要的探索、重复工具调用）是另一个维度——更复杂因为每一步都有真实成本（API 费用、时间）。
5. **Frontier 模型的瓶颈**：Corecraft 发现 Opus 4.6/GPT-5.2 <30% pass rate，这说明问题不仅仅是训练方法——frontier 模型在真实 agent 任务上仍有根本局限。

---

## 2026 年 Agentic RL 工作全景（按时间）

| 日期 | 论文 | arXiv | 维度 | 评分 |
|------|------|-------|------|------|
| 2025/04 | RAGEN/StarPO | 2504.20073 | 算法·稳定性·Multi-turn | ★★★★★ |
| 2025/09 | iStar | 2509.19199 | 算法·Credit | ★★★★★ |
| 2026/02/01 | MIG | 2602.01034 | 算法·Credit | ★★★★☆ |
| 2026/02/10 | EnterpriseGym Corecraft | 2602.16179 | 环境 | ★★★★★ |
| 2026/02/13 | TSR | 2602.11767 | 算法·稳定性·Rollout质量 | ★★★★☆ |
| 2026/02/13 | CM2 | 2602.12268 | Reward·Unverifiable | ★★★★☆ |
| 2026/02/14 | OpenRS | 2602.14069 | Reward | ★★★☆☆ |
| 2026/02/15 | HiPER | 2602.16165 | 算法·Credit·层级 | ★★★★★ |
| 2026/02/16 | Kimi-K2.5 PARL | 2602.02276 | 算法·Multi-agent | ★★★★☆ |
| 2026/02/17 | FlowSteer CWRPO | 2602.01664 | Workflow | ★★★☆☆ |
| 2026/02/17 | SquRL | 2602.15564 | Workflow | ★★★☆☆ |
| 2026/02/18 | AdaptOrch | 2602.16873 | Workflow·推理时编排 | ★★★★☆ |
| 2026/02/19 | KLong | 2602.17547 | Context Overflow | ★★★★☆ |
| 2026/02/19 | AgentConductor | 2602.17100 | Workflow | ★★★★☆ |
| 2026/02/20 | PA-MoE | 2602.17038 | Workflow | ★★★★☆ |
| 2026/02/20 | Calibrate-Then-Act | 2602.11841 | 算法 | ★★★☆☆ |
| 2026/02/21 | SeeUPO | 2602.06554 | 算法·Credit·理论 | ★★★★★ |
| 2026/02/17 | AWM | 2602.10090 | 环境工程·合成 | ★★★★☆ |
| 2026/02/18 | SHARP | 2602.08335 | 算法·Credit·Multi-agent | ★★★★☆ |
| 2026/02/18 | Dr. MAS | 2602.08847 | 算法·Multi-agent 稳定性 | ★★★★☆ |
| 2025/11/28 | SORL | 2511.20718 | 算法·Off-policy 稳定性·Multi-turn | ★★★★☆ |
| 2026/02/23 | Search-R1++ | 2602.19526 | Reward Design·Policy Opt | ★★★☆☆ |
| 2026/02/26 | Search-P1 | 2602.22576 | Reward Design·Tool Use RL | ★★★★☆ |
| 2025/10 | SCoRe | 2501.09723 | 算法·均衡控制·多轮纠错 | ★★★★★ |
| 2026/02/13 | CM2 | 2602.12268 | Reward·Unverifiable·工具调用 | ★★★★☆ |
| 2026/02/13 | TSR | 2602.11767 | 算法·Rollout质量·Multi-turn | ★★★★☆ |
| 2026/02/15 | ERL | 2602.13949 | 算法·反思内化·稀疏Reward | ★★★★☆ |
| 2026/02/03 | CSO | 2602.03412 | 算法·Credit·失败轨迹 | ★★★★☆ |
| 2026/02/24 | SELAUR | 2602.21158 | Reward·不确定性感知·失败激活 | ★★★☆☆ |
| 2026/02/24 | PyVision-RL | 2602.20739 | 训练稳定性·多模态·Interaction Collapse | ★★★☆☆ |
| 2026/02/15 | PABU | — | Context管理·进度感知信念状态·效率 | ★★★★☆ |
| — | WebPilot | — | Multi-Agent·MCTS战略探索·Web任务 | ★★★☆☆ |
| 2026/02 | AgentAuditor | 2602.09341 | Multi-Agent·审计·反共识偏好优化 | ★★★★☆ |
| 2026/02/24 | AlphaEvolve | 2602.16928 | MARL·算法自动发现·LLM代码演化 | ★★★★☆ |
| 2026/02/28 | SRPO | 2602.21515 | MARL·协作泛化·风险规避均衡 | ★★★★☆ |

---

## 对老板的直接价值

如果在面试中被问到"你对 agentic RL 的理解"，这个框架给出了一个结构化回答：

1. **问题定义**：从可验证任务（RLVR）到开放任务（Agentic RL），reward 设计和 credit assignment 是核心难题
2. **五维分解**：环境（Corecraft）/ Reward（verifiable/unverifiable）/ 算法（Credit Assignment 6方案 + 稳定性）/ Workflow（AgentConductor/SquRL/PA-MoE）/ Context Overflow（KLong）
3. **Credit Assignment 深答**：GRPO→LOOP→GiGPO→AgentPRM→iStar→MIG 完整谱系，区分"需不需要额外 rollout""支不支持 unverifiable reward""要不要额外模型"
4. **开放问题**：honest 地说明当前上限——extreme horizon 的跨子轨迹 credit assignment、expert rubric 生成成本、真实环境 gap

这种回答比列举论文名字深度高一个数量级。

---

## 核心洞察（一句话）

**2026 年 Agentic RL 的根本争论不是"哪个算法更好"，而是"瓶颈到底在哪里"：**

- 环境派：bottleneck 是环境质量（Corecraft 的证据）
- Reward 派：bottleneck 是 reward 可靠性，特别是 unverifiable reward 场景（CM2/iStar 的证据）
- Workflow 派：bottleneck 是 pipeline 静态性（AgentConductor/SquRL 的证据）
- 算法派：bottleneck 是 credit assignment（6方案谱系的证据）
- 基础设施派：bottleneck 是 context window 和训练稳定性（KLong/TSR 的证据）

**正确答案可能是全部**——但不同任务和不同发展阶段，各维度的权重不同。v3 增加的 Context Overflow 维度揭示了一个新边界：当任务复杂度超出 context window 的物理限制时，需要全新的训练方法论，而不只是更好的算法。

## See Also（全路径索引）

> 本笔记正文内链为 Scholar 写入的简短路径；以下为馆长补充的全路径对照，便于 Obsidian 图谱检索。

- [[AI/2-Agent/Agentic-RL/Agentic-RL-元问题-瓶颈与突破方向|🧠 Agentic RL 元问题：瓶颈与突破方向]] ⭐ — **本综述的元层批判与升维**：基于37+篇论文的Wisdom层判断；指出算法层已够用，真正瓶颈是Reward Signal Quality；本综述是"是什么"，元问题笔记是"为什么不够/下一步在哪"
- [[AI/2-Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL|iStar（2509.19199，Tongyi Lab，★★★★★）]] — trajectory DPO ≡ step-wise BT model，唯一支持 unverifiable reward 的 step-level CA，SOTOPIA +48%，2x 样本效率
- [[AI/2-Agent/Agentic-RL/Search-R1-Reasoning-Search-Engine-RL|Search-R1（前驱，arXiv:2503.09516）]] — Search-R1++ (2602.19526) 的前身：把搜索引擎集成进 RL rollout，token masking 稳定训练；Search-R1++ 在此基础上系统消融 reward/optimizer/prompt 三维度（vault_gap：Search-R1++ 独立笔记待 Scholar 补写）
- [[AI/2-Agent/Agentic-RL/Search-P1-Path-Centric-Reward-Agentic-RAG|Search-P1（arXiv:2602.22576）]] — 路径级密集奖励（v13 新增）：显式 Planner + 双轨路径评分 + 软结果打分，解决 Search-R1 稀疏奖励/失败样本零梯度；+7.7% over Search-R1，工业 AD-QA +20.6%；与 Search-R1++ 正交可组合（奖励密度 vs 奖励质量）
-  — Agentic RL 在 Agent 知识域的位置
- [[AI/2-Agent/Agentic-RL/FlowSteer-CWRPO-Workflow-Orchestration-RL|FlowSteer (CWRPO)]] — 维度 4：Operator 级 workflow 设计（Workflow/Topology 解法 A）
- [[AI/2-Agent/Multi-Agent/AgentConductor-Topology-Evolution|AgentConductor]] — 维度 4：Agent 通信 Topology 级（解法 B，difficulty-aware density）
- [[AI/2-Agent/Agentic-RL/SquRL-Dynamic-Workflow-Text-to-SQL|SquRL]] — 维度 4：Workflow 选择级（解法 C，Theorem 3.1 形式化证明）
- [[AI/3-LLM/RL/Theory/MARS-Margin-Aware-Reward-Modeling-Self-Refinement|MARS]] — reward modeling 自适应分配（与 Reward 维度高度互补）
- [[AI/5-AI 安全/Adaptive-Regularization-Safety-Degradation-Finetuning|Adaptive-Regularization]] — Agentic RL × Safety 汇合点：pre-generation hidden state 安全门控
- [[AI/2-Agent/Agentic-RL/UI-TARS-2|UI-TARS-2]] — GUI Agent RL 工程极致路线：Data Flywheel + 异步 multi-turn RL + Hybrid 沙盒（★★★★★）
- [[AI/2-Agent/Agentic-RL/UI-R1-GUI-Action-Prediction-RL|UI-R1]] — GUI Agent RL 极简路线：136 条数据 rule-based GRPO，3B ≈ SFT 7B@76K（★★★★☆）
- [[AI/2-Agent/Fundamentals/Memory-R1-RL-for-LLM-Memory-Management|Memory-R1]] — RL 训练 Memory Manager（ADD/UPDATE/DELETE/NOOP），记忆管理新范式（★★★★☆）
- [[AI/2-Agent/Agentic-RL/ASTRA-Automated-Tool-Agent-Training|ASTRA]] — 全自动 tool-use RL 流水线，MCP 工具图 + verifiable 环境（★★★★☆）
- [[AI/2-Agent/Agentic-RL/RC-GRPO-Reward-Conditioned-Tool-Calling-RL|RC-GRPO]] — reward token conditioning 解决 multi-turn GRPO reward 同质化（★★★★☆）
- [[AI/3-LLM/MLLM/PyVision-RL-Agentic-Vision-Interaction-Collapse|PyVision-RL（2602.20739）]] — **跨模态训练失败模式**：Interaction Collapse = Echo Trap 的多模态版本（v10 新增）；Oversampling-Filtering-Ranking + Accumulative Tool Reward；On-Demand Context Construction 解决视频 context 爆炸
- [[AI/2-Agent/Agentic-RL/SORL-Stabilizing-Off-Policy-RL-Long-Horizon-Agent|SORL（2511.20718）]] — Off-policy multi-turn RL 崩溃的两根因诊断（粒度错配+方差累积）+ Turn-Level IS/CTN 修复，SO-PPO/SO-GRPO 实例化（**v12 新增**）
- [[AI/2-Agent/Agentic-RL/Agent-进化模式谱系|🧠 Agent 进化模式谱系]] ⭐ — **三层统一框架**（训练时/in-context/运行时），附贾维斯实践映射与选型决策树（**v13 新增，老板指令产出**）
- [[AI/2-Agent/Agentic-RL/Reflexion-Verbal-Reinforcement-Learning|Reflexion（NeurIPS 2023）]] — in-context 进化奠基：verbal reinforcement，episodic memory buffer，无需微调（**v13 新增**）
- [[AI/2-Agent/Agentic-RL/ExpeL-Experiential-Learning-Agent|ExpeL（AAAI 2024）]] — 跨任务规则提炼：ADD/UPVOTE/DOWNVOTE/EDIT 规则库 + 相似案例检索（**v13 新增**）
- [[AI/2-Agent/Agentic-RL/AgentQ-MCTS-Self-Critique-DPO|AgentQ]] — MCTS + 自我批判 + off-policy DPO，Llama-3 70B 真实预订 18.6%→81.7%（**v13 新增**）
- [[AI/2-Agent/Multi-Agent/AlphaEvolve-LLM-Discovers-MARL-Algorithms|AlphaEvolve（arXiv:2602.16928）]] — MARL 算法自动发现：LLM 演化代码发现非直觉 CFR/PSRO 变体，10/11 游戏超 SOTA（**v14 新增**）
- [[AI/2-Agent/Multi-Agent/SRPO-Strategic-Risk-Aversion-Collaborative-MARL|SRPO（arXiv:2602.21515）]] — 协作 MARL 泛化：Risk-averse Quantal Equilibria 替代 Nash，消除 free-riding（**v14 新增**）
- [[AI/2-Agent/Agentic-RL/PABU-Progress-Aware-Belief-State|PABU（进度感知信念状态）]] — 推理阶段 Context 效率：显式建模任务进度+选择性历史保留；81% 完成率+26.9% 效率提升；与 KLong/SORL 正交（**v14 新增**）
- [[AI/2-Agent/Agentic-RL/WebPilot|WebPilot（Multi-Agent Web任务）]] — Planner+Executor 架构 + MCTS 战略探索；WebArena/Mind2Web benchmark 验证（**v14 新增**）
- [[AI/2-Agent/Multi-Agent/AgentAuditor-Reasoning-Tree-审计|AgentAuditor（arXiv:2602.09341）]] — Reasoning Tree 审计多 Agent 系统；ACPO（Anti-Consensus Preference Optimization）识别正确少数派；局部化审计比全局投票精准（**v14 新增**）
