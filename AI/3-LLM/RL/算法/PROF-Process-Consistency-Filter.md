---
brief: "Beyond Correctness——将过程奖励（process reward）与结果奖励（outcome reward）统一到同一 RL 框架；实验证明两者互补而非对立，过程 reward 提升中间步骤质量，outcome reward 维持最终准确率。"
title: "Beyond Correctness: Harmonizing Process and Outcome Rewards through RL   Training"
type: paper
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/paper
---
# Beyond Correctness: Harmonizing Process and Outcome Rewards through RL   Training

by：亚马逊

论文链接：https://hf.co/papers/2509.03403

PaperScope.ai 解读：https://paperscope.ai/hf/2509.03403

由亚马逊和伊利诺伊大学厄巴纳-香槟分校提出了PROF（Process Consistency Filter）框架，该工作**通过一致性驱动的样本选择策略**，有效整合了细粒度但存在噪声的过程奖励模型（PRM）与准确但粗粒度的结果奖励模型（ORM），解决了强化学习中推理任务的奖励信号矛盾问题。

PROF通过评估PRM与ORM的一致性，过滤掉 **结果正确但推理错误** 或 **结果错误但推理合理 **的样本，在保持正负样本平衡的同时消除冲突梯度。

实验表明，PROF与GRPO结合的PROF-GRPO方法相比传统奖励混合方案，在数学推理基准测试中平均准确率提升超4%，且显著改善了中间推理步骤的质量。通过蒙特卡洛估计和LLM评判验证，PROF-GRPO生成的推理链更符合逻辑、步骤更完整。

该方法通过分离正确/错误样本分组过滤、动态平衡样本比例等设计，在Qwen和LLaMA模型上均展现出鲁棒性，同时**避免了奖励劫持问题**，为构建可解释的推理系统提供了新思路。

## 相关

- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]]
- [[AI/3-LLM/RL/算法/PPO 原理|PPO 原理]]

## See Also

- [[AI/3-LLM/RL/算法/PRIME-Process-Reward-Implicit-MLE|PRIME]] — 同维度：过程奖励，PRIME 用隐式 MLE，PROF 用过程一致性过滤
- [[AI/3-LLM/RL/Theory/Reward-Design-三维框架|Reward Design 三维框架]] — PROF 属于"过程奖励质量"维度的解法
- [[AI/3-LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL Training Stability 专题]] — 过程一致性过滤对训练稳定性的影响
