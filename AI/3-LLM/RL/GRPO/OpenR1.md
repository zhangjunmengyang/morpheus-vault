---
brief: "OpenR1——HuggingFace 开源复现 DeepSeek-R1 的社区项目；基于 TRL + GRPO 复现 R1 的推理训练流水线，提供完整的数据集/训练脚本/评估代码；是入门 GRPO 工程实践的首选参考。"
title: "OpenR1"
type: concept
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/concept
---
# OpenR1

## 概述

OpenR1 是 HuggingFace 在 2025 年 1 月发起的开源项目，目标是**完整复现 DeepSeek-R1 的训练流程**。项目地址：`huggingface/open-r1`。

R1 论文虽然开源了模型权重，但训练细节（数据配比、超参数、训练节奏等）有大量留白。OpenR1 的价值在于填补这些空白，为社区提供一个可复现的 reasoning model 训练 pipeline。

## 项目组成

OpenR1 不只是一个训练脚本，而是一个生态：

### 1. Open-R1 训练代码

基于 TRL（HuggingFace 的训练框架），实现了完整的 R1 训练流程：
- SFT（cold start 阶段）
- GRPO RL 训练
- 评估脚本（基于 lighteval）

核心依赖：
```bash
# 主要依赖
trl >= 0.14
transformers
vllm  # 用于 RL 阶段的快速采样
```

### 2. Open-R1 数据集

社区构建了多个配套数据集：

- **OpenR1-Math-220k**：约 22 万条数学推理数据，包含 prompt + 多个 CoT response
- **OpenR1-SFT**：用于 cold start 的高质量 SFT 数据
- 数据来源：主要通过 DeepSeek-R1 蒸馏 + 人工筛选

### 3. 评估体系

标准化的评估 benchmark：
- MATH-500
- AIME 2024
- GSM8K
- GPQA（研究生级别科学问题）
- LiveCodeBench

## 训练流程复现

OpenR1 把 R1 的 4 阶段训练简化为更实用的流程：

### Stage 1: SFT（Cold Start）

```bash
# 典型的 SFT 配置
accelerate launch \
  --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
  src/open_r1/sft.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --dataset_name open-r1/OpenR1-SFT \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 16384
```

关键点：
- 使用较长的 `max_seq_length`（推理链可能很长）
- 学习率较小，避免破坏预训练知识
- 数据中包含 `<think>` 和 `</think>` 标签分隔推理和回答

### Stage 2: GRPO RL

```bash
accelerate launch \
  --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
  src/open_r1/grpo.py \
  --model_name_or_path <sft_checkpoint> \
  --reward_funcs accuracy format \
  --num_generations 16 \
  --max_completion_length 8192 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --beta 0.01 \
  --learning_rate 1e-6
```

关键超参数：
- `num_generations=16`：每个 prompt 采样 16 个 response（R1 原文用 64，但显存限制）
- `beta=0.01`：KL 惩罚系数，太大会限制探索，太小会导致模型偏离过远
- `max_completion_length=8192`：推理链需要足够长的生成空间

### Reward Function 设计

OpenR1 的 reward function 遵循 R1 的极简原则：

```python
def accuracy_reward(response, ground_truth):
    """提取答案并验证正确性"""
    answer = extract_answer(response)  # 从 \boxed{} 中提取
    return 1.0 if answer == ground_truth else 0.0

def format_reward(response):
    """检查是否遵循 <think>...</think> 格式"""
    has_think = "<think>" in response and "</think>" in response
    return 1.0 if has_think else 0.0
```

## 社区复现成果

截至 2025 年中，OpenR1 社区已经取得了不少成果：

1. **7B 模型**：基于 Qwen2.5-7B 训练的 reasoning model，在 MATH-500 上达到 ~80%+
2. **1.5B 模型**：基于 Qwen2.5-1.5B 的小规模验证，证明流程可行
3. **多语言尝试**：社区成员在中文数学数据上的实验

但也暴露了一些挑战：
- **计算资源门槛**：即使 7B 模型，GRPO 训练（16 个 generation）也需要 8×A100
- **训练不稳定性**：reward hacking、KL 散度爆炸等问题时有发生
- **数据质量**：蒸馏数据的质量直接决定 SFT 阶段的效果

## 与其他复现项目的对比

| 项目 | 维护者 | 框架 | 特点 |
|------|--------|------|------|
| OpenR1 | HuggingFace | TRL | 社区最大、文档最全 |
| SimpleRL | 学术界 | 自定义 | 极简实现，学习用 |
| Logic-RL | 社区 | verl | 关注逻辑推理任务 |

OpenR1 的最大优势是背靠 HuggingFace 生态（TRL、datasets、accelerate），上手门槛最低。

## 我的观点

1. **复现的意义不在于追平 R1**：7B 模型不可能追平 671B，但复现过程中积累的工程经验和数据洞察是无价的
2. **GRPO 训练的 engineering 细节**比算法本身更重要：采样效率、显存优化、reward 设计的 edge case 处理
3. **数据 > 算法**：社区实验反复证明，SFT 阶段的数据质量对最终效果影响巨大
4. **开源的力量**：OpenR1 推动了整个 reasoning model 领域的民主化

## 相关

- [[DeepSeek-R1|DeepSeek-R1]] — 被复现的原始论文
- [[GRPO 深度理解|GRPO 深度理解]] — GRPO 算法详解
- [[DeepSeek-Math|DeepSeek-Math]] — GRPO 的起源
- [[TRL 概述|TRL 概述]] — OpenR1 依赖的训练框架
- [[GRPO-TRL实践|GRPO-TRL实践]] — TRL 中 GRPO 的实践
- [[vLLM|vLLM]] — RL 采样阶段使用的推理引擎
- [[verl 概述|verl 概述]]
