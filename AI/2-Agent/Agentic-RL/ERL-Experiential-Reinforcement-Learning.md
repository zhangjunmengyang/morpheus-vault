---
title: "ERL: Experiential Reinforcement Learning"
brief: 在 RL 训练循环内嵌入 experience-reflection-consolidation 三循环。每轮两次尝试：y¹ → 反思 Δ → y²，RL 对齐两次尝试+反思，SFT 蒸馏成功 y² 进 base policy（部署时零额外成本）。Sokoban +81%，HotpotQA +11%。
date: 2026-02-25
arxiv: "2602.13949"
authors: Taiwei Shi, Sihao Chen, Bowen Jiang, Linxin Song, Longqi Yang, Jieyu Zhao
institutions: USC, Microsoft, UPenn
venue: arXiv 2026-02-15
rating: ★★★★☆
tags:
  - agentic-RL
  - experiential-learning
  - reflection
  - self-distillation
  - multi-turn
  - sparse-reward
  - credit-assignment
related:
  - "[[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO]]"
  - "[[AI/2-Agent/Agentic-RL/CM2-Checklist-Rewards-Multi-Turn-Tool-Use-RL|CM2]]"
  - "[[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR]]"
  - "[[AI/2-Agent/Agentic-RL/SCoRe-Self-Correction-via-Reinforcement-Learning|SCoRe (ICLR 2025)]]"
  - "[[AI/2-Agent/Agentic-RL/iStar-Implicit-Step-Rewards-Agentic-RL|iStar]]"
  - "[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL 综合分析]]"
---

# ERL: Experiential Reinforcement Learning

> **一句话**：RL 不只学结果，还学"如何从失败中反思"——把反思循环编进训练过程，成功的纠错行为蒸馏进 base policy，部署时自动更好。

---

## TL;DR

标准 RLVR 的问题：环境反馈（sparse + delayed）要求 policy 从纯标量信号中**隐式推断**如何改正行为，导致探索效率低、训练曲线振荡。

ERL 的解法：在每个训练 episode 里加一个**显式的 experience–reflection–consolidation 循环**：
1. **Experience**：第一次尝试 y¹ + 环境反馈 f¹
2. **Reflection**：基于 (y¹, f¹) 生成反思 Δ，描述如何改进
3. **Refinement**：以 Δ 为条件生成第二次尝试 y²
4. **RL**：对齐 y¹、Δ、y² 的奖励
5. **Consolidation（内化）**：把成功的 y² 蒸馏进 base policy（输入只有 x，不含 Δ）

**部署时零成本**：蒸馏确保改进行为已经内化，inference 不需要反思步骤。

---

## 动机：隐式推断的根本局限

人类的实验性学习（Kolb, 2014）：体验 → 反思 → 概念化 → 实验，循环驱动行为改进。

LLM RL 流水线的现状：
- **SFT**：模仿固定示例，无法适应部署后的新失败
- **RLVR**：trial-and-error + 标量信号，纠错结构必须隐式涌现
- **RLVR 的问题**：失败轨迹如何转化为行为改变？没有显式机制，只能靠重复采样碰运气

ERL 把这个"隐式推断"变成"显式反思"，然后通过蒸馏让反思结果永久写入 base policy。

---

## 核心方法

### 训练循环（Algorithm 1）

```
初始化: 反思记忆 m ← ∅

对每个训练 episode:
  1. 第一次尝试: y¹ ~ π_θ(·|x)
     获得反馈 (f¹, r¹)

  2. [条件触发] 若 r¹ < τ:
     生成反思: Δ ~ π_θ(·|x, y¹, f¹, r¹, m)
     第二次尝试: y² ~ π_θ(·|x, Δ)
     获得反馈 (f², r²)
     若 r² > τ: m ← Δ  (存入跨episode记忆)

  3. RL 更新:
     ℒ_policy(θ) = -E[A · log π_θ(y|x,·)]
     覆盖 y¹, Δ, y²；A 来自对应奖励

  4. 内化（蒸馏）:
     ℒ_distill(θ) = -E[𝕀(r²>0) · log π_θ(y²|x)]
     条件: 只有 x（不含 Δ），只蒸馏成功的 y²
```

**关键设计细节：**

**① 反思作为中间信号**
- 反思 Δ 的奖励 = y² 的奖励（r̃ ← r²）
- 这训练 π_θ 生成"能够导向成功的反思"，而不是空洞的自我评价

**② 跨 episode 反思记忆 m**
- 成功的反思（r² > τ）存入 m
- 下一次生成反思时可以参考，稳定反思质量，复用有效纠错策略

**③ 内化（selective distillation）**
$$\mathcal{L}_{\text{distill}}(\theta) = -\mathbb{E}\left[\mathbb{I}(r^{(2)}>0) \cdot \log\pi_\theta\left(y^{(2)}\mid x\right)\right]$$
- 输入只有原始 x（无反思 Δ），目标是成功的 y²
- 作用：让 π_θ 学会"直接产生纠正后的行为"，无需推理时的反思步骤
- 只蒸馏成功（r² > 0）的情况，防止错误行为扩散

**④ 计算均衡**
- ERL 每个任务 2 次尝试 + 1 次反思，RLVR 每个任务 10 次 rollout
- 实验中：ERL 5 rollout/attempt（2×5=10 total），与 RLVR 的 10 相当
- 结论：ERL 用同等计算预算取得显著更好的结果

### 与 RLVR 的对比

| 维度 | RLVR | ERL |
|------|------|-----|
| 反馈利用 | 标量信号 → 隐式优化 | 标量 + 文本反馈 → 显式反思 → 第二次尝试 |
| 纠错机制 | 无；靠重复探索 | 显式：失败→反思→改进→蒸馏 |
| 记忆 | 无跨 episode 记忆 | 跨 episode 反思记忆 m |
| 部署成本 | 零 | 零（内化后不需要 Δ） |
| 适用场景 | 通用 | 未知动态 + 稀疏 reward 场景增益最大 |

---

## 实验结果

### 任务设置
- **FrozenLake**：稀疏奖励网格导航，只有 observations + action 接口，无规则提示
- **Sokoban**：长 horizon 规划，复合错误回滚，稀疏终末奖励
- **HotpotQA**：多跳问答，工具辅助检索，F1 reward（密集）
- 模型：Qwen3-4B-Instruct-2507 + Olmo-3-7B-Instruct（两个 backbone 都测）

### 结果

| 任务 | 模型 | RLVR | ERL | 提升 |
|------|------|------|-----|------|
| Sokoban | Qwen3-4B | 0.06 | 0.87 | **+81%** |
| Sokoban | Olmo-3-7B | 0.04 | 0.20 | +400% |
| FrozenLake | Qwen3-4B | - | - | +27% |
| HotpotQA | 平均 | - | - | **+11%** |

**Sokoban 的巨大提升**：长 horizon 规划 + 未知动态——恰好是 ERL 最擅长的。反思循环帮助 agent 分析失败（"方块卡死了"）并生成具体的修正策略（"下一步先腾出通路"）。

**HotpotQA 的较小提升**：reward 相对密集，交互模式均匀（搜索→综合），RLVR 已有较好的梯度信号，ERL 的额外增益有限。

**学习效率**：Figure 4 显示 ERL 在训练早期就显著领先——wall-clock time 曲线更陡，说明反思加速了探索收敛。

---

## 与相关工作的定位

### vs SCoRe (ICLR 2025)
- **SCoRe**：两阶段训练解决自我纠错的"假纠错均衡"问题（Phase 1 KL 约束删除假均衡）
- **ERL**：每 episode 都有反思循环，不限于自我纠错，适用于任意环境反馈
- **共同点**：都认为 trial-and-error RLVR 不足以学习纠错，需要显式机制
- **关系**：SCoRe 是"架构层修正"，ERL 是"训练动态修正"

### vs TSR (ICML 2026)
- **TSR**：训练时树搜索，解决 naive rollout 的高方差/Echo Trap 问题
- **ERL**：训练时反思循环，解决 sparse reward → 隐式推断问题
- **可组合**：TSR（搜索 rollout 策略）+ ERL（反思纠错）可以叠加

### vs iStar
- **iStar**：step-level implicit credit（DPO ≡ step-wise BT），不需要显式纠错，适用于 unverifiable reward
- **ERL**：episode-level 反思，需要可读取的环境反馈（textual feedback f），适用于 verifiable/semi-verifiable reward

### vs CSO
- **CSO**：从失败轨迹中反事实验证关键步骤，用 DPO 监督
- **ERL**：从失败尝试中生成反思，用 SFT 蒸馏成功行为
- **关系**：都利用失败信号，但 CSO 是 offline 数据挖掘，ERL 是 online 训练循环

---

## 理论分析

### 为什么 ERL 的内化有效？

内化 loss ℒ_distill 是条件 MLE：
$$\mathcal{L}_{\text{distill}} = -\mathbb{E}\left[\mathbb{I}(r^{(2)}>0)\log\pi_\theta(y^{(2)}\mid x)\right]$$

直觉：y² 是"在反思 Δ 的引导下的改进行为"，如果把 Δ 从条件中去掉，π_θ 被强制学习"不依赖反思就能产生改进行为"——即把纠错逻辑内化进参数。

这与知识蒸馏（teacher-student）类似，但这里 teacher = 带反思的自己，student = 不带反思的自己。

### 选择性触发（τ 阈值）的重要性

条件：r¹ < τ 才触发反思-重试。
- 避免对已经好的结果做不必要的"修正"（可能引入噪音）
- 把计算集中在最能从反思中获益的 failure case
- 本质是 hard negative mining 的反思版本

---

## 我的评价

**★★★★☆（非常扎实，工程可用）**

**优点：**
1. **理念清晰**：Kolb 实验性学习理论 → LLM RL 的直接实现，故事完整
2. **内化机制精妙**：部署时零成本是硬性工程要求，ℒ_distill 的设计干净解决了这个问题
3. **跨episode记忆**：不是简单的 per-episode 反思，而是有记忆的累积改进，像真正的"学习者"
4. **实验强劲**：Sokoban +81% 是真实的大增量，两个 backbone 都验证

**局限：**
1. **环境需要可读 textual feedback**：如果环境只返回 binary reward，反思质量会退化
2. **两次尝试的计算开销**：虽然实验控制了计算量，但在真实 agent 任务中（长 horizon + 工具调用），两次完整尝试的成本不容忽视
3. **HotpotQA 增益有限（+11%）**：说明在 reward 已经足够密集的环境中，反思的边际价值下降——这是适用场景的边界
4. **反思质量依赖 base model 能力**：弱模型可能生成无意义反思，导致 y² 不比 y¹ 好

**工程价值**：高。对于 reward 稀疏 + 环境动态未知的 agent 场景（机器人控制、开放 web agent），ERL 是目前最干净的解法之一。反思记忆的设计可以单独借鉴。

**研究价值**：中高。ERL 是 SCoRe 范式的泛化，但还缺少严格的理论分析（为什么内化收敛？反思记忆的稳定性？）。

---

## 与 Multi-Turn RL 三支柱的定位

```
Multi-Turn RL 挑战解法谱系（v5，2026-02-25）：

1. Rollout 质量: TSR (ICML 2026) — 训练时树搜索，optimizer-agnostic
2. Credit Assignment: GiGPO/HiPER/iStar/CSO — 精确步骤级信号
3. 均衡控制: SCoRe (ICLR 2025) — 删除假纠错均衡
4. 反思-内化: ERL (2026) — 显式反思循环 + 蒸馏，无部署成本 ← 新支柱
```

ERL 引入了第四个支柱：**训练时的显式反思内化**。这与前三个支柱正交，可以与 TSR/SCoRe 叠加。

---

## 附：反思记忆机制细节

```python
# 伪代码：ERL 的反思记忆
reflection_memory = []

for episode in training:
    y1, (f1, r1) = model.attempt(x), env.step(y1)
    
    if r1 < tau:  # 触发条件
        delta = model.reflect(x, y1, f1, r1, reflection_memory)
        y2, (f2, r2) = model.attempt(x, delta), env.step(y2)
        
        if r2 > tau:
            reflection_memory.append(delta)  # 存入成功反思
        
        # RL：对齐 y1, delta, y2
        optimizer.step(policy_loss(y1, delta, y2, rewards))
        
        # 蒸馏：内化 y2 → base policy
        if r2 > 0:
            optimizer.step(distill_loss(y2, x))  # 无 delta 条件
```

---

## Connections

- Credit Assignment 视角：ERL 的 ℒ_distill 是一种软性 offline 监督，类似 CSO 的"从失败轨迹提取关键步骤"——都在用结构化的失败信息强化正确行为
- RLHF 视角：反思 Δ 作为中间 reward signal，比纯标量更信息丰富，类似 Process Reward Model 的"步骤级信号"
- Meta-Learning 视角：跨 episode 反思记忆 m 是一种隐式的 meta-learning，存储有效的纠错策略供后续复用

---

## See Also

**失败信号利用谱系（三种深度，ERL 居中）：**
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization|CSO（arXiv:2602.03412）]] — 失败 → 反事实验证（最深，需 expert model）；ERL 是其轻量替代——无需外部 expert，用 LLM 自身反思代替
- [[AI/2-Agent/Agentic-RL/SELAUR-Self-Evolving-LLM-Agent-Uncertainty-Rewards|SELAUR（arXiv:2602.21158）]] — 失败 → token-level 不确定性 reward（最浅，零成本）；ERL 信号质量更高（反思 Δ 比熵值信息更丰富）；三者共同构成失败信号利用深度谱系

**Multi-Turn RL 四支柱中的位置（ERL = 第四支柱）：**
- [[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（ICML 2026）]] — Rollout 质量支柱；与 ERL 正交可叠加（好 rollout + 好反思）
- [[AI/2-Agent/Agentic-RL/SCoRe-Self-Correction-via-Reinforcement-Learning|SCoRe（NeurIPS 2024）]] — 均衡控制支柱；ERL 是"训练动态修正"，SCoRe 是"架构层修正"

**综述导航：**
- [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL 2026 综合分析]] — ERL 在 Multi-Turn RL 四支柱框架中的定位
- [[AI/2-Agent/Agentic-RL/Agent-进化模式谱系|Agent 进化模式谱系]] — 三层框架：ERL 属于第①层训练时进化
