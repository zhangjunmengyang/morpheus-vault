---
brief: "LLaVA-Critic-R1——Critic 模型可以同时是 Policy 模型的实证研究；通过 RL 训练评判模型兼具批判能力和生成能力，无需维护独立 Critic；多模态 RL 对齐中 Critic-as-Policy 的实验验证。"
title: "LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model

直观理解是在视觉模型上仿照 R1 GRPO 的方式，同时保留模型的 critic 和 生成能力

by：马里兰大学、俄亥俄州立大学和新加坡国立大学

该工作通过重构带偏好标签的批评数据为可验证的强化学习任务，**直接在基础生成模型上进行RL训练**，使模型同时具备卓越的批评判断能力和策略生成能力。

研究发现，仅用40k偏好标注数据训练的LLaVA-Critic-R1在26个视觉推理基准上平均超越基础模型5.7%，在MMMU等任务中达到71.9的SOTA性能。进一步将该方法应用于ThinkLite-VL-7B等强推理模型，得到的LLaVA-Critic-R1+在保持批评能力的同时，策略性能提升至71.9（MMMU），在数学推理、图表理解等任务中表现尤为突出。该方法的核心创新在于：

1. 通过剥离GPT生成的评估标准，迫使模型自主构建判断逻辑
1. 采用Group Relative Policy Optimization（GRPO）进行训练，通过偏好奖励和格式奖励的平衡提升模型能力；
1. 提出测试时自我批评机制，通过Best-of-128策略在5个基准上平均提升13.8%。
1. 
实验表明，**批评能力提升与策略性能增强存在强正相关**，模型在视觉感知和结构化推理两方面均显著优化。该研究揭示了批评数据蕴含的生成能力提升潜力，为构建兼具评估与生成能力的统一模型提供了新范式，其测试时自我改进机制也为开发自进化多模态系统指明了方向。

## See Also

- [[AI/3-LLM/MLLM/目录|MLLM MOC]] — 多模态大模型全景索引
- [[GRPO 深度理解|GRPO]] — LLaVA-Critic-R1 核心训练算法
- [[多模态大模型-2026-技术全景|多模态大模型 2026 全景]] — MLLM 宏观综述
- [[REMuL-CoT-Faithfulness-Multi-Listener-RL|REMuL]] — 同样研究"critic能力与policy性能正相关"（多模态版 vs 纯文本版）
- [[BLIP-2|BLIP-2]] — 另一个 critic-style 评估设计先驱
