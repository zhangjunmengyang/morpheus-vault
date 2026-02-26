---
brief: "Unsloth Checkpoint 管理——训练中断恢复和模型保存策略；LoRA adapter 的保存格式（.safetensors）、合并导出（merge_to_16bit）、GGUF 量化导出的完整流程。"
title: "Checkpoint"
type: concept
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/concept
---
# Checkpoint 管理

> 文档：https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint

训练中断是常态——OOM、节点故障、手动停止调参。Checkpoint 机制让你不用从头来过。

## 基本配置

Unsloth 底层用的是 HuggingFace Trainer，checkpoint 配置通过 `TrainingArguments` 传入：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./outputs",
    
    # Checkpoint 策略
    save_strategy="steps",       # "steps" | "epoch" | "no"
    save_steps=100,              # 每 100 步保存一次
    save_total_limit=3,          # 最多保留 3 个 checkpoint（节省磁盘）
    
    # 评估（用于选最优 checkpoint）
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True, # 训练结束自动加载最优 checkpoint
    metric_for_best_model="eval_loss",
    
    # 其他
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)
```

## 从 Checkpoint 恢复训练

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=args,
)

# 从最新 checkpoint 恢复
trainer.train(resume_from_checkpoint=True)

# 或者指定特定 checkpoint
trainer.train(resume_from_checkpoint="./outputs/checkpoint-300")
```

恢复时会自动还原：
- 模型权重
- Optimizer 状态
- Learning rate scheduler 状态
- 随机数种子
- 训练步数

## Checkpoint 存储了什么

```
outputs/checkpoint-300/
├── adapter_model.safetensors   # LoRA 权重（通常几十 MB）
├── adapter_config.json          # LoRA 配置
├── optimizer.pt                 # 优化器状态（可能很大）
├── scheduler.pt                 # LR scheduler 状态
├── trainer_state.json           # 训练状态（loss 历史等）
├── rng_state.pth               # 随机数状态
└── training_args.bin           # 训练参数
```

⚠️ **磁盘占用**：`optimizer.pt` 可能比模型权重还大（AdamW 有两个 moment buffer）。如果磁盘紧张，设置 `save_total_limit` 限制数量。

## 实用技巧

### 1. 只保存 LoRA 权重

如果只需要恢复模型（不需要继续训练），可以只保存 adapter：

```python
# 训练结束后
model.save_pretrained("./final_model")  # 只保存 LoRA adapter
tokenizer.save_pretrained("./final_model")
```

### 2. 选择最优 Checkpoint

不要盲目用最后一个 checkpoint——过拟合很常见：

```python
# trainer_state.json 中记录了每个 checkpoint 的 eval_loss
import json
with open("outputs/trainer_state.json") as f:
    state = json.load(f)

for log in state["log_history"]:
    if "eval_loss" in log:
        print(f"Step {log['step']}: eval_loss={log['eval_loss']:.4f}")
```

### 3. Checkpoint 到 HuggingFace Hub

```python
# 直接推送到 Hub
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")

# 或者用 trainer
trainer.push_to_hub()
```

### 4. 合并 LoRA 到基座模型

部署时通常需要合并：

```python
# Unsloth 的合并方式
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_16bit",  # 或 "merged_4bit_forced"
)
```

## 常见问题

| 问题 | 解决 |
|------|------|
| 恢复后 loss 突然跳高 | 检查 learning rate 是否正确恢复 |
| OOM 导致 checkpoint 损坏 | 设置 `save_on_each_node=True` |
| 磁盘不够 | 减小 `save_total_limit`，或用 `save_safetensors=True` |
| Eval loss 先降后升 | 典型过拟合，用 early stopping |

## 相关

- [[AI/LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/LLM/Frameworks/Unsloth/运行 & 保存模型|运行 & 保存模型]]
- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/LLM/SFT/LoRA|LoRA]]
