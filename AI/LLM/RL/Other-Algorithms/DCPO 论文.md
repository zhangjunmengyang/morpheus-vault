---
title: "DCPO: Dynamic Clipping Policy Optimization"
type: paper
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/paper
---
# DCPO: Dynamic Clipping Policy Optimization

by：百川 Baichuan.inc

论文链接：https://hf.co/papers/2509.02333

PaperScope.ai 解读：https://paperscope.ai/hf/2509.02333

提出的Dynamic Clipping Policy Optimization(DCPO)通过动态调整剪辑边界和优势标准化技术，有效解决了强化学习中**零梯度问题**。该方法引入动态剪辑策略，**根据token先验概率自适应调整剪辑边界**，**增强低概率token的探索空间**；同时采用平滑优势标准化技术，通过累积训练步骤的奖励分布优化响应级利用效率。

】实验显示，DCPO在四个数学推理基准测试中均取得最优表现，在Qwen2.5-Math-7B模型上AIME24基准的Avg@32指标达到38.8，显著优于GRPO(32.1)和DAPO(31.6)。在Qwen2.5-14B模型上AIME25基准的Avg@32达到19.0，较GRPO(10.5)和DAPO(15.3)有大幅提升。

DCPO将非零优势比例平均提升28%，训练效率较DAPO提高一倍，token剪辑比例降低一个数量级。该方法通过动态适应token概率分布特性，在保持高置信度token更新稳定性的同时，显著增强低概率token的探索能力，同时通过累积优势标准化**有效缓解高熵采样导致的训练波动问题**，为大语言模型的强化学习提供了更高效的数据利用和更稳定的优化路径。
