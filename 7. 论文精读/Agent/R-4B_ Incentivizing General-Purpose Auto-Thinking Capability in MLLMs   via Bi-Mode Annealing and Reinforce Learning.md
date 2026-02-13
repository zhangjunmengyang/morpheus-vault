---
title: "R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs   via Bi-Mode Annealing and Reinforce Learning"
category: "论文精读"
tags: [Agent, LLM, MLLM, OCR, Qwen]
created: "2026-02-13"
updated: "2026-02-13"
---

# R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs   via Bi-Mode Annealing and Reinforce Learning

by：腾讯混元团队和中科院自动化

论文链接：https://hf.co/papers/2508.21113

PaperScope.ai 解读：https://paperscope.ai/hf/2508.21113

由腾讯混元团队和中科院自动化所提出了R-4B，该工作通过双模退火和强化学习实现多模态大语言模型的自适应思考能力，使模型能够**根据问题复杂度动态切换思考与直接回答模式**，在保持推理性能的同时显著降低计算成本。

研究针对现有模型在简单问题上冗余思考导致的效率问题，创新性地设计了双模退火训练策略，通过构建包含549万条推理数据和1087万条直接回答数据的混合数据集，使模型同时掌握两种响应模式。在此基础上提出的 **双模策略优化（BPO）算法**，采用数学领域规则奖励信号驱动强化学习，在无需复杂奖励工程的情况下，通过强制生成思考与非思考双路径响应，有效解决了模型偏好非思考模式的"思考萎缩"问题。

实验表明，R-4B-RL在MMMU-val等25项基准测试中超越Qwen2.5-VL-7B，并在数学推理和图表理解任务中达到与16B参数模型Kimi-VL-A3B-Thinking相当的水平，同时推理效率提升40%。该模型在保持70亿参数规模下，通过动态调整思考模式，在OCR等简单任务中输出token量仅66个（非思考模式57个），而在MathVista等复杂任务中自动扩展至996个token，实现了推理性能与计算成本的最优平衡。研究提出的双模训练框架和策略优化方法为构建高效智能的多模态模型提供了新范式。
