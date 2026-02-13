---
title: "trl"
type: reference
domain: ai/llm/frameworks/trl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/trl
  - type/reference
---
# trl

## 概述

TRL（Transformer Reinforcement Learning）是 HuggingFace 维护的 post-training 框架，覆盖了 LLM 对齐训练的全流程：**SFT → Reward Modeling → RLHF（PPO/GRPO/DPO 等）**。

它是目前社区使用最广泛的 LLM 对齐训练框架之一，优势在于：
- 与 HuggingFace 生态（transformers、datasets、accelerate、peft）无缝集成
- API 设计统一，不同算法的切换成本很低
- 社区活跃，新算法跟进快

项目地址：`huggingface/trl`

## 核心组件

### Trainer 体系

TRL 为每种对齐算法提供专门的 Trainer：

```
SFTTrainer        → 监督微调
RewardTrainer     → 奖励模型训练
PPOTrainer        → PPO 算法
GRPOTrainer       → GRPO 算法（DeepSeek-R1 同款）
DPOTrainer        → Direct Preference Optimization
KTOTrainer        → Kahneman-Tversky Optimization
RLOOTrainer       → REINFORCE Leave-One-Out
ORPOTrainer       → Odds Ratio Preference Optimization
```

所有 Trainer 都继承自 `transformers.Trainer`，所以 HuggingFace Trainer 的所有功能（checkpoint、logging、distributed training）都能直接用。

### 统一的 Config 模式

每个 Trainer 有对应的 Config：

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo-output",
    num_generations=16,
    max_completion_length=8192,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    beta=0.01,
    bf16=True,
)

trainer = GRPOTrainer(
    model="./sft-checkpoint",
    config=config,
    reward_funcs=[accuracy_reward, format_reward],
    train_dataset=dataset,
)
trainer.train()
```

## 典型训练流程

### 完整的对齐 Pipeline

```
Base Model (e.g., Qwen2.5-7B)
    ↓ SFTTrainer (指令微调)
SFT Model
    ↓ 选择对齐策略:
    ├── DPOTrainer (偏好学习，不需要 reward model)
    ├── GRPOTrainer (RL 训练，需要 reward function)
    └── PPOTrainer (经典 RLHF)
Aligned Model
```

### SFT 阶段

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    args=SFTConfig(
        max_seq_length=4096,
        packing=True,
        num_train_epochs=3,
    ),
    train_dataset=sft_dataset,
    peft_config=lora_config,  # 可选，用 LoRA 省显存
)
trainer.train()
```

### DPO 阶段

DPO 不需要 reward model，直接从偏好数据学习：

```python
from trl import DPOTrainer, DPOConfig

# 数据格式：每条包含 chosen 和 rejected response
# {"prompt": "...", "chosen": "好的回复", "rejected": "差的回复"}

trainer = DPOTrainer(
    model="./sft-checkpoint",
    args=DPOConfig(
        beta=0.1,  # KL 惩罚系数
        max_length=2048,
        max_prompt_length=1024,
    ),
    train_dataset=preference_dataset,
)
trainer.train()
```

### GRPO 阶段

GRPO 需要 reward function（但不需要 learned reward model）：

```python
from trl import GRPOTrainer, GRPOConfig

def accuracy_reward(completions, **kwargs):
    """Rule-based reward: 答案正确得 1 分"""
    rewards = []
    for completion, answer in zip(completions, kwargs["answer"]):
        extracted = extract_answer(completion)
        rewards.append(1.0 if extracted == answer else 0.0)
    return rewards

trainer = GRPOTrainer(
    model="./sft-checkpoint",
    args=GRPOConfig(
        num_generations=16,      # 每个 prompt 采样 16 个
        max_completion_length=8192,
        beta=0.01,
    ),
    reward_funcs=[accuracy_reward],
    train_dataset=math_dataset,
)
trainer.train()
```

## 与 vLLM 的集成

TRL 的 GRPO/PPO Trainer 在 generation 阶段可以使用 vLLM 加速：

```python
config = GRPOConfig(
    use_vllm=True,                    # 启用 vLLM
    vllm_gpu_utilization=0.3,         # vLLM 占用 30% GPU 显存
    num_generations=16,
)
```

这对 GRPO 特别重要，因为每个 prompt 要采样 16+ 个 response，生成速度直接决定训练效率。

## 与 PEFT 的集成

所有 Trainer 都原生支持 LoRA：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 传给任何 Trainer
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-7B",
    peft_config=lora_config,
    ...
)
```

## 分布式训练支持

TRL 通过 Accelerate 支持多种分布式策略：

```bash
# FSDP
accelerate launch --config_file fsdp.yaml train.py

# DeepSpeed ZeRO-3
accelerate launch --config_file ds_z3.yaml train.py

# 多节点
accelerate launch --multi_gpu --num_processes 16 train.py
```

## TRL 的局限

1. **大规模 RL 训练效率**：对于 70B+ 模型的 GRPO/PPO 训练，TRL 的效率不如 verl/OpenRLHF 等专门的 RL 框架。主要瓶颈在 generation 和 training 阶段的 GPU 切换。
2. **资源调度**：TRL 在一个进程中同时管理训练和推理，资源分配不够灵活。verl 用 Ray 做更细粒度的资源编排。
3. **自定义空间**：Trainer 封装较深，自定义 reward、custom loss 需要继承 Trainer 重写方法，有一定学习曲线。

## TRL vs verl vs OpenRLHF

| 维度 | TRL | verl | OpenRLHF |
|------|-----|------|----------|
| 维护方 | HuggingFace | 字节 / 开源社区 | 开源社区 |
| 算法覆盖 | SFT/DPO/PPO/GRPO/KTO/... | GRPO/PPO/DAPO/... | PPO/DPO/GRPO |
| 分布式 | Accelerate (FSDP/DS) | Ray + FSDP | Ray + DeepSpeed |
| 大模型效率 | 中等 | 高 | 高 |
| 上手门槛 | 低 | 中 | 中 |
| 生态集成 | 最好（HF 全家桶） | 好 | 中等 |

**选择建议**：
- 7B 以下模型、快速实验 → TRL
- 大规模 RL 训练（70B+） → verl 或 OpenRLHF
- 需要最全的算法覆盖 → TRL

## 相关

- [[SFT-TRL实践]] — TRL SFT 实践
- [[DPO-TRL实践]] — TRL DPO 实践
- [[GRPO-TRL实践]] — TRL GRPO 实践
- [[PPO-TRL实践]] — TRL PPO 实践
- [[KTO-TRL实践]] — TRL KTO 实践
- [[RLOO-TRL实践]] — TRL RLOO 实践
- [[verl 概述]] — 对比框架
- [[OpenRLHF]] — 对比框架
- [[vLLM]] — TRL 集成的推理引擎
- [[LoRA]] — 参数高效训练
