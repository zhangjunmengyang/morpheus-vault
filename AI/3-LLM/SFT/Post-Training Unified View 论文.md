---
title: Towards a Unified View of Large Language Model Post-Training
brief: Towards a Unified View of LLM Post-Training——将 SFT/RLHF/DPO 等 post-training 方法统一到同一理论框架；发现不同方法本质上都是在优化相同的目标函数的不同近似，差异来自稳定性掩码/参考策略分母/优势估计/似然梯度四组件的偏差-方差权衡；提出 HPT 动态切换 SFT↔RL，AIME 2024 Pass@1=33.0，超最强基线+6.9分。
type: paper
domain: ai/llm/sft
created: 2026-02-13
updated: 2026-02-25
rating: ★★★★☆
tags:
  - ai/llm/sft
  - type/paper
  - post-training
  - unified-view
  - hybrid-training
sources:
  - arXiv:2509.04419 — 清华大学 / 上海人工智能实验室 / 微信 AI
  - 论文链接：https://hf.co/papers/2509.04419
  - PaperScope 解读：https://paperscope.ai/hf/2509.04419
related:
  - "[[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]]"
  - "[[SFT-实战指南|SFT 实战指南]]"
  - "[[RLHF-DPO-2026-技术全景|RLHF-DPO 2026 技术全景]]"
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

---

## 启发思考

**So What**：这篇论文最重要的贡献是把"SFT 与 RL 是两种不同路线"的工程直觉替换成数学精确的陈述——它们是同一策略梯度估计器的特例，区别在于四个组件的偏差-方差选择。HPT 的实用意义：不再需要手动决定"先 SFT 多少步再切 GRPO"，而是让 per-question 验证准确率自动驱动切换。**铁匠比喻精准**：横敲（SFT，确定方向）和竖敲（RL，强化路径），同一工具。

**与 GRPO-Improvement-Panorama 的联系**：HPT 解决的是 SFT↔RL 边界问题，GRPO-Panorama 解决的是 RL 内部方法对比。两者都以"统一框架"为方法论，但层次不同：HPT 在 SFT/RL 边界，GRPO-Panorama 在 RL 各流派内部。

## See Also

- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — Post-Training RL 方向的前沿改进全景
-  — LLM 强化学习（Post-Training 核心范式）全图谱
- [[PEFT 方法对比|PEFT 方法对比]] — Post-Training 中 SFT 的参数高效方法
-  — 大语言模型知识全图谱
- [[RLHF-DPO-2026-技术全景|RLHF-DPO 2026 技术全景]] — Post-Training 的 RLHF/DPO 流派，是 HPT 的背景框架
- [[SFT-实战指南|SFT 实战指南]] — SFT 工程实践，HPT 的 SFT 分支对应

## 推荐阅读

1. **原文**：[arXiv:2509.04419](https://arxiv.org/abs/2509.04419) — Towards a Unified View of LLM Post-Training
2. **PaperScope 解读**：https://paperscope.ai/hf/2509.04419
3. **理论延伸**：[[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — HPT 框架下 RL 分支的内部分化
