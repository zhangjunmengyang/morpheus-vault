---
title: "强化学习 for LLM"
type: moc
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - type/reference
---

# 🎯 强化学习 for LLM

> LLM Post-Training 的核心方向 — 从 RLHF 到 GRPO 再到 Agentic RL

## 基础理论 (Fundamentals)
- [[AI/LLM/RL/Fundamentals/马尔科夫|马尔科夫]] — MDP 基础
- [[AI/LLM/RL/Fundamentals/贝尔曼方程|贝尔曼方程]] — 价值函数
- [[AI/LLM/RL/Fundamentals/策略梯度方法|策略梯度方法]] — PG 族算法基础
- [[AI/LLM/RL/Fundamentals/On-Policy vs Off-Policy|On-Policy vs Off-Policy]]
- [[AI/LLM/RL/Fundamentals/KL散度|KL散度]] — 正则化核心概念
- [[AI/LLM/RL/Fundamentals/MCTS|MCTS]] — 蒙特卡洛树搜索
- [[AI/LLM/RL/Fundamentals/为什么 PPO 优于 PG|为什么 PPO 优于 PG]]
- [[AI/LLM/RL/Fundamentals/PPL 计算 交叉熵损失与 ignore_index|PPL 计算]]
- [[AI/LLM/RL/Fundamentals/RL 概览|RL 概览]]
- [[AI/LLM/RL/Fundamentals/RL & LLMs 入门|RL & LLMs 入门]] — HF Course
- [[AI/LLM/RL/Fundamentals/HF Deep RL Course|HF Deep RL Course]]
- [[AI/LLM/RL/Fundamentals/强化学习的数学原理|强化学习的数学原理]]
- [[AI/LLM/RL/Theory/RLVR-Edge-of-Competence|RLVR at the Edge of Competence]] — 能力边界上的 RLVR，研究训练信号有效区间

## 核心算法

### PPO
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]] — 最经典的 RLHF 算法
- [[AI/LLM/RL/PPO/PPO-TRL实践|PPO-TRL实践]]
- [[AI/LLM/RL/PPO/PPO-verl实践|PPO-verl实践]]

### GRPO ⭐（重点方向）
- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — 核心原理
- [[AI/LLM/RL/GRPO/DeepSeek R1 学习笔记|DeepSeek R1 学习笔记]]
- [[AI/LLM/RL/GRPO/DeepSeek-Math|DeepSeek-Math]] — 数学推理论文
- [[AI/LLM/RL/GRPO/Blockwise-Advantage-Estimation|Blockwise Advantage Estimation]] — GRPO credit assignment 改进
- [[AI/LLM/RL/GRPO/TRL 中实现 GRPO|TRL 中实现 GRPO]]
- [[AI/LLM/RL/GRPO/GRPO-TRL实践|GRPO-TRL实践]]
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO-verl实践]]
- [[AI/LLM/RL/GRPO/GRPO-Unsloth实践|GRPO-Unsloth实践]]
- [[AI/LLM/RL/GRPO/GRPO-demo|GRPO-demo]]
- [[AI/LLM/RL/GRPO/OpenR1|OpenR1]]
- [[AI/RL/iGRPO|iGRPO]] — 迭代式自反馈 GRPO (arXiv:2602.09000)

### DPO
- [[AI/LLM/RL/DPO/DPO-TRL实践|DPO-TRL实践]]
- [[AI/LLM/RL/DPO/DPO-Unsloth实践|DPO-Unsloth实践]]

### DAPO
- [[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO-verl实践]]

### KTO
- [[AI/LLM/RL/KTO/KTO-TRL实践|KTO-TRL实践]]

### RLOO
- [[AI/LLM/RL/RLOO/RLOO-TRL实践|RLOO-TRL实践]]

### 其他算法 (Other-Algorithms)
- [[AI/LLM/RL/Other-Algorithms/DCPO 论文|DCPO]] — Dynamic Clipping
- [[AI/LLM/RL/Other-Algorithms/Beyond Correctness 论文|Beyond Correctness]] — Process + Outcome Rewards
- [[AI/LLM/RL/Other-Algorithms/GPG-verl实践|GPG]]
- [[AI/LLM/RL/Other-Algorithms/OPO-verl实践|OPO]]
- [[AI/LLM/RL/Other-Algorithms/SPIN-verl实践|SPIN]]
- [[AI/LLM/RL/Other-Algorithms/SPPO-verl实践|SPPO]]
- [[AI/LLM/RL/Other-Algorithms/CollabLLM-verl实践|CollabLLM]]
- [[AI/LLM/RL/Other-Algorithms/OpenRS-Pairwise-Adaptive-Rubric|OpenRS]] — Pairwise Adaptive Rubric，non-verifiable reward 对齐，解决 reward hacking（arXiv:2602.14069）
- [[AI/LLM/RL/Other-Algorithms/GSPO-Unsloth实践|GSPO]]
- [[AI/LLM/RL/Other-Algorithms/MEL-Meta-Experience-Learning|MEL]] — Meta-Experience Learning
- [[AI/LLM/RL/Other-Algorithms/CM2 — Checklist Rewards多轮Tool Use RL|CM2]] — Checklist Rewards 多轮 Tool Use RL
- [[AI/LLM/RL/Other-Algorithms/SkillRL — 递归技能增强的Agent演化|SkillRL]] — 递归技能增强 Agent 演化
- [[AI/LLM/RL/Other-Algorithms/RLTF-RL-from-Text-Feedback|RLTF]] — RL from Text Feedback，文本反馈奖励设计（arXiv:2602.02482）
- [[AI/LLM/RL/Other-Algorithms/HiPER-Hierarchical-RL-Credit-Assignment|HiPER]] — 分层 RL + 显式 Credit Assignment，多步 Agent 长 horizon（arXiv:2602.16165）★★★★
- [[AI/LLM/RL/Other-Algorithms/LACONIC-Length-Constrained-RL|LACONIC]] — Primal-Dual RL 控制 CoT 输出长度，推理效率（arXiv:2602.14468）★★★
- [[AI/LLM/RL/Other-Algorithms/E-SPL-Evolutionary-System-Prompt-Learning|E-SPL]] — RL 权重更新（程序性知识）+ 进化算法 system prompt 优化（声明性知识）联合训练；AIME25 56.3→60.6%（arXiv:2602.14697）★★★★
- [[AI/LLM/RL/Other-Algorithms/GEPA-Reflective-Prompt-Evolution|GEPA]] ⭐ — 纯 prompt 进化超越 GRPO（5/6任务），rollout 减少 35x；E-SPL=GEPA+RL；ICLR 2026 Oral，UCB+Stanford+MIT（arXiv:2507.19457）★★★★★
- [[AI/LLM/RL/Other-Algorithms/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] — Teacher 模型在线预测题目难度，选 p≈0.5 的样本训练，逃离 sparse reward 低效陷阱；Apple+EPFL（arXiv:2602.14868）★★★★
- [[AI/LLM/RL/Other-Algorithms/STAPO-Spurious-Token-Aware-Policy-Optimization|STAPO]] — 0.01% spurious tokens 携带虚假梯度是 RL 训练崩溃根源；mask 掉即可稳定训练；清华+滴滴（arXiv:2602.15620）★★★★
- [[AI/LLM/RL/Other-Algorithms/Stable-Asynchrony-VCPO-Off-Policy-RL|Stable Asynchrony (VCPO)]] — 异步 off-policy RL 的方差爆炸根因与修复：Variance-Controlled Policy Optimization，解决 generation/training 解耦后的 staleness 问题；MIT HAN Lab（Song Han）★★★★
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 统一梯度利用+概率质量+信号可靠性的 GRPO 三维改进：软裁剪替代硬裁剪 + 概率质量校正 + reward 信号可靠性加权；微软亚研（arXiv:2602.17xxx）★★★★
- [[AI/LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — Root Saturation 问题根治：Pivot-Driven Resampling 专攻深层 error-prone states；对比 TreeRL/AttnRL 探索启发式的缺陷；ICML 投稿，2602.14169（★★★★☆）
- [[AI/LLM/RL/Other-Algorithms/VESPO-Variational-Sequence-Policy-Optimization|VESPO]] ⭐ — 变分推导闭合形式 soft kernel `ϕ(W)=W^α·exp(-λW)`，理论严格超越所有 heuristic clip（GRPO/GSPO/SAPO），staleness ratio 64× 异步训练稳定；★★★★★（arXiv:2602.10693）
- [[AI/LLM/RL/Other-Algorithms/AT-RL-Anchor-Token-Reinforcement-Learning-Multimodal|AT-RL]] — 多模态 RLVR：仅 15% token 有强视觉-文本耦合（"视觉锚点"），图聚类识别并选择性强化；32B 模型 MathVista 80.2 超越 72B-Instruct；仅 1.2% 开销（arXiv:2602.11455）★★★★

## 训练框架 (Frameworks)
- [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — NVIDIA+MIT HAN Lab：统一 FP8 on-policy RL 训练精度流，解决 BF16-train/FP8-rollout 在长 rollout(>8K) 时精度崩溃和训练发散问题（arXiv:2601.14243）★★★★
- [[AI/LLM/RL/Frameworks/QeRL-Quantization-Enhanced-RL|QeRL]] ⭐ — ICLR 2026（NVIDIA+MIT+HKU+THU+Song Han）：量化噪声是有益的——4-bit 量化+LoRA 的 RL 训练不仅 1.5× 加速，在多项基准上还**超越** 16-bit LoRA；see-also: [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]]（arXiv:2510.11696）★★★★
- [[AI/RL/Slime RL Framework|Slime RL Framework]] — GLM-5 的异步 RL 基础设施：解决 generation bottleneck >90%，APRIL 框架（see-also 指向深度版）

## 综述与深度笔记
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 改进全景 2026]] ⭐ — 六维框架元分析：2025-10~2026-02 全部 GRPO 改进工作按问题层次分类（算法层/样本层/探索层/系统层/多模态层），面试级（2026-02-20）★★★★★
- [[AI/LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性 2026 统一分析]] ⭐ — Scholar 综合笔记 v3：STAPO/Goldilocks/VCPO/DEEP-GRPO/MASPO/DAPO/LACONIC 四维拓扑（Token/样本/探索/系统），持续更新中（2026-02-20）★★★★★
- [[AI/LLM/RL/RLHF 全链路|RLHF 全链路]] — 完整 RLHF 三阶段
- [[AI/LLM/RL/RLHF-DPO-2026-技术全景|RLHF/DPO 2026 技术全景]] — 面试武器版，1147行，RLHF→RLAIF→DPO 全链路（2026-02-20）
- [[AI/LLM/RL/对齐技术综述|对齐技术综述]] — RLHF → DPO → ORPO → KTO → SteerLM → Constitutional AI
- [[AI/LLM/RL/RARL-Reward-Modeling-Survey|RARL Reward Modeling Survey]] — RL reasoning alignment 综述

## 相关 MOC
- ↑ 上级：[[AI/LLM/_MOC]]
- → 交叉：[[AI/Agent/_MOC]]（Agentic RL）
