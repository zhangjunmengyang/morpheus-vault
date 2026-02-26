---
brief: "LACONIC（arXiv:2602.14468，ICML）——用 Primal-Dual RL 做 LLM 输出长度约束；将长度限制建模为约束优化问题，Lagrangian 对偶自适应调整惩罚系数；在保持 accuracy 的前提下压缩 CoT 长度 40%+。"
title: "LACONIC — 用 Primal-Dual RL 控制 LLM 输出长度"
date: 2026-02-16
arxiv: "2602.14468"
venue: "ICML (submitted)"
authors: "Chang Liu, Yiran Zhao, Lawrence Liu, Yaoqi Ye, Csaba Szepesvári, Lin F. Yang"
tags: [RL, CoT-compression, length-control, primal-dual, RLVR, inference-efficiency]
rating: ★★★
---

# LACONIC — 用 Primal-Dual RL 控制 LLM 输出长度

**arXiv**: 2602.14468  
**投稿至**: ICML  
**发布**: 2026-02-16  

## 问题

RL 训练 reasoning LLM 会导致输出越来越长（overthinking）。现有 length-control 方法直接在 reward 里加固定 length penalty——这是一个**根本性的错误**：

> 优化 (task_reward - fixed_penalty) ≠ 优化 true task_reward

固定 penalty 将训练目标变成了与真实目标错位的 surrogate，同时超参数调起来脆：penalty 太强 → 牺牲准确率；penalty 太弱 → 长度控制失效。

## 核心方法：Primal-Dual 约束 RL

LACONIC 将 length-control 构造成**约束优化问题**：

```
maximize   E[task_reward(response)]
subject to E[length(response)] ≤ B   (B = target token budget)
```

用 Lagrangian 松弛转化，引入对偶变量 λ：

**目标函数**：`task_reward(r) - λ · length_cost(r)`

**两步交替迭代**：
1. **Primal Update**：用当前 λ 更新 policy（正常 RL-tuning 步骤，GRPO/PPO 均兼容）
2. **Dual Update**：根据当前 batch 的平均长度更新 λ
   - 当前 batch 平均长度 > B → 提高 λ（加强长度惩罚）
   - 当前 batch 平均长度 < B → 降低 λ（放宽惩罚）

**自动退化属性**：当 model 持续满足 budget 时，λ 自动收敛到 0，训练退化为标准 RL-tuning，允许 model 在不超 budget 的前提下继续最大化 task reward。

```
λ_new = max(0, λ_old + η · (avg_length - B))
```

## 为什么这比 Fixed Penalty 好

| 维度 | Fixed Penalty | LACONIC |
|---|---|---|
| 目标对齐 | 优化 surrogate（错位）| 直接约束，真实目标 |
| 超参调节 | 每个任务/设置需要重调 | B（budget）是直觉易懂的参数 |
| 收敛行为 | 可能在 length vs accuracy 上来回振荡 | 理论保证收敛到 budget 约束下的最优策略 |
| 极端情况 | 固定 penalty 无法自适应 | λ → 0 自动退化为无约束 RL |

## 实验结果

**数学推理**（in-distribution）：
- pass@1 **保持或提升**，同时输出长度减少 **>50%**

**通用知识 & 多语言**（out-of-distribution）：
- 性能维持，token 数减少 **44%**

**部署开销**：
- 与标准 RL-tuning 兼容（GRPO/PPO）
- 推理时无任何改变（不需要特殊解码策略）

## 理论贡献

论文提供了方法的 theoretical guarantee（投 ICML，具体 convergence bound 在正文中）。核心结论：在适当步长设置下，LACONIC 收敛到约束优化问题的 KKT 点，对应 budget 约束下的 local optima。

这使 LACONIC 成为这个方向**第一个有理论保证**的方法（对比同类 engineering 工作）。

## 与同类工作的对比

### 同类 length-control 方法
- **Fixed penalty RL**（Aggarwal & Welleck 等）：直接在 reward 里减去 length，brittle
- **CoT compression via distillation**（Constraint-Rectified Training 等）：蒸馏方式，不是端到端 RL
- **Prompt engineering**（"be concise"）：inference-time only，效果弱

### 相关但不同的工作
- **LACONIC vs LACONIC (concurrent)**：该方向在 2/14-2/18 提交了多篇类似工作，LACONIC 的差异化是 theoretical guarantee + adaptive dual variable
- **Long CoT Compression via FGGPO**（2602.10xxx）：用 fine-grained group policy optimization 做 CoT 压缩，也是 RL 路线但方法不同

## 我的判断

★★★ — 实用价值明确，理论基础扎实，但技术 novelty 中等。

**为什么值 ★★★ 而不是更高**：
- 核心 insight（primal-dual 约束优化 for budget constraint）是清晰且 elegant 的
- ICML 级工作：有理论保证，实验扎实
- 但问题本身（让 reasoning model 说话短一点）不是 breakthrough 级研究问题——是工程需求驱动的优化工作

**实用价值**：
- 对于需要控制推理模型输出长度的工程师（推理成本、延迟、context window）直接可用
- Budget B 是个直觉参数，比 fixed penalty 的 coefficient 容易设置
- 无推理时改动：plug-and-play 进现有 RL-tuning pipeline

**理论 vs 实际的 gap**：
论文在数学推理 benchmark 上验证了 50% 长度削减 + pass@1 保持。真实 agent 场景（多步骤，需要详细 reasoning trace）的 budget 设置策略未给出——过激的 budget 会不会在复杂任务上导致 reasoning collapse 还需要更多分析。

## 关联

- [[AI/3-LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR-Edge-of-Competence]] — 难度分布对 RLVR 有效性的理论分析，LACONIC 的 primal-dual 框架与其互补
- [[AI/3-LLM/RL/GRPO/Blockwise-Advantage-Estimation|Blockwise-Advantage-Estimation]] — GRPO credit assignment 改进，LACONIC 可以在 GRPO 上应用
- CoT Compression 方向的其他工作（concurrent）：Constraint-Rectified Training (2602.1xxxx), Extreme-Ratio CoT Compression (2602.xxx)

## See Also

- [[AI/3-LLM/RL/Other-Algorithms/IntroLLM-Introspective-Temperature-Policy-Hierarchical-RL|IntroLLM（分层RL自适应温度控制）]] — 约束 RL 的互补维度：LACONIC 约束输出**长度**（primal-dual 硬约束），IntroLLM 约束采样**温度**（分层 RL 软约束）——两者合用可同时控制探索强度和输出成本
- [[AI/3-LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO（虚假Token感知策略优化）]] — token 级干预的不同方向：STAPO 屏蔽低质量 spurious token（质量约束），LACONIC 约束总长度（效率约束）——都是在 RL 训练中对 token 流施加结构性干预
- [[AI/3-LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性统一分析]] — 宏观框架：LACONIC 属于"输出质量约束"层，与 STAPO/DEEP-GRPO 一起构成 token 级稳定化方法族
