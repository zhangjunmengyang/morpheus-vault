---
title: "PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic   Reasoning"
category: "论文精读"
tags: [Agent, CEP, LLM, 强化学习, 预训练]
created: "2026-02-13"
updated: "2026-02-13"
---

# PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic   Reasoning

by：阿里云

论文链接：https://hf.co/papers/2508.21104

PaperScope.ai 解读：https://paperscope.ai/hf/2508.21104

提出了PVPO（Pre-Estimated Value-Based Policy Optimization），该工作针对传统强化学习方法在复杂任务中依赖**多次采样和组内比较导致的局部最优及高计算成本问题**，提出了一种基于预估价值的策略优化框架。

PVPO通过引入参考模型作为优势参考锚点，并结合数据预采样策略，有效解决了组策略方法中累积偏差和样本效率低下的核心痛点。

![image](assets/B0CwdVqWDo0ueixSDumcEp5Cn2b.png)

核心创新包括：

- 静态V值估计机制，通过预训练参考模型生成任务奖励锚点，替代传统动态组内平均值，显著降低策略更新方差；
- 组采样策略，利用参考模型离线评估样本难度，过滤低价值数据并生成零准确率样本的高质量轨迹，提升训练效率。