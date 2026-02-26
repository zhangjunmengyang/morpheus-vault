---
brief: "KTO TRL 实践——Kahneman-Tversky Optimization 的 TRL 工程指南；不需要成对偏好数据（只需 chosen 或 rejected 之一），KTOTrainer 配置/数据格式；适合偏好数据采集困难场景的对齐方案。"
title: "KTO"
type: project
domain: ai/llm/rl/kto
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/kto
  - type/project
---
# KTO

# 材料

https://huggingface.co/docs/trl/v0.21.0/en/kto_trainer

# 指南

## 学习率建议

每个 `beta` 值都有一个它能容忍的最大学习率，超过这个学习率会导致学习性能下降。对于默认设置 `beta = 0.1`，大多数模型的学习率通常不应超过 `1e-6`。随着 `beta` 的减小，学习率也应相应降低。一般来说，我们强烈建议将学习率保持在 `5e-7` 和 `5e-6` 之间。即使数据集很小，我们也建议不要使用这个范围之外的学习率。相反，可以选择更多的训练轮数来获得更好的结果。

## 样本不均衡处理

[KTOConfig](https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftrl%2Fv0.21.0%2Fen%2Fkto_trainer%23trl.KTOConfig) 的 `desirable_weight` 和 `undesirable_weight` 指的是对期望/正面和期望之外/负面样本的损失所施加的权重。默认情况下，它们都为 1。但是，如果你有更多的某一种或另一种，那么你应该提高较少见类型的权重，使得 `desirable_weight` 和 `undesirable_weight` 的比值（即 `desirable_weight` ×× 正面数量）与（`undesirable_weight` ×× 负面数量）的比例在 1:1 到 4:3 的范围内。

# 日志记录

- `rewards/chosen_sum`: 所选回答的政策模型的日志概率之和，乘以 beta
- `rewards/rejected_sum`: 所拒绝回答的政策模型的日志概率之和，乘以 beta
- `logps/chosen_sum`: 所选补全的日志概率之和
- `logps/rejected_sum`: 被拒绝的补全的日志概率之和
- `logits/chosen_sum`: 所选补全的 logits 之和
- `logits/rejected_sum`: 被拒绝的补全的 logits 之和
- `count/chosen`: 批次中选择的样本数量
- `count/rejected`: 批次中被拒绝的样本数量

## 相关

- [[AI/3-LLM/RL/DPO/DPO-TRL实践|DPO-TRL实践]]
- [[AI/3-LLM/Frameworks/TRL/TRL 概述|TRL 概述]]
- [[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]]
