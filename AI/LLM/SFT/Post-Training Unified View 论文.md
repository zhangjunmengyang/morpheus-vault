---
title: "Towards a Unified View of Large Language Model Post-Training"
type: paper
domain: ai/llm/sft
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/sft
  - type/paper
---
# Towards a Unified View of Large Language Model Post-Training

by：清华大学、上海人工智能实验室和微信AI

论文链接：https://hf.co/papers/2509.04419

PaperScope.ai 解读：https://paperscope.ai/hf/2509.04419

提出了Unified Policy Gradient Estimator和Hybrid Post-Training（HPT）算法，该工作通过理论推导揭示了**监督微调（SFT）与强化学习（RL）在大语言模型后训练中的统一性**，提出将两者梯度计算纳入统一框架的策略梯度估计器，并基于此设计了动态切换训练信号的HPT算法。

研究发现SFT与RL的梯度可视为同一目标函数在不同数据分布下的特例，**其差异源于稳定性掩码、参考策略分母、优势估计和似然梯度四个组件的偏差-方差权衡**。HPT通过实时评估模型在单个问题上的多轨迹验证准确率，动态调整SFT与RL损失的权重比例：当模型表现弱时采用SFT进行知识注入，表现强时切换到RL促进探索。

在数学推理任务上的实验表明，HPT在Qwen2.5-Math-7B上相比SFT→GRPO和LUFFY基线平均提升7.2和6.2分，在AIME 2024数据集上取得33.0的Pass@1成绩，较最强基线提升6.9分。消融实验显示动态混合策略显著优于固定比例混合，且模型响应长度在切换到RL后保持稳定增长，证明HPT能有效平衡探索与利用，提升模型推理能力。

非常有意思的思想，把 RL 和 SFT 从数学角度解释差异，切换训练方法。

**就好像一个铁匠有的时候横着敲，有的时候竖着敲。**
