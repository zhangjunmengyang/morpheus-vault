---
brief: "SFT TRL 实践——HuggingFace TRL SFTTrainer 的工程指南；数据格式（ChatML/Alpaca/ShareGPT）/DataCollator/packing/gradient checkpointing 配置；从数据到模型的 SFT 完整工程流程。"
title: "SFT"
type: project
domain: ai/llm/sft
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/sft
  - type/project
---
# SFT

## 概述

SFT（Supervised Fine-Tuning）是 post-training 的第一步，把一个 base model 变成能 follow instruction 的模型。用 TRL 做 SFT 是目前最主流的方案之一——API 简洁、与 HuggingFace 生态无缝集成、支持 LoRA/QLoRA 等参数高效方法。

本文是 TRL SFTTrainer 的实践记录。

## 数据准备

### 对话格式

TRL SFTTrainer 期望的数据格式：

```python
# 标准 conversational format
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的..."}
    ]
}
```

也支持更简单的 instruction format：

```python
# instruction format
{
    "prompt": "什么是 Transformer？",
    "completion": "Transformer 是一种基于自注意力机制的..."
}
```

### 数据加载

```python
from datasets import load_dataset

# HuggingFace Hub 数据集
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

# 本地 JSONL
dataset = load_dataset("json", data_files="train.jsonl")

# 本地 Parquet
dataset = load_dataset("parquet", data_files="train.parquet")
```

### 数据质量清洗

SFT 数据质量直接决定模型表现。常见清洗策略：

```python
def filter_low_quality(example):
    messages = example["messages"]
    # 过滤太短的回复
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if any(len(m["content"]) < 50 for m in assistant_msgs):
        return False
    # 过滤没有实质内容的
    if any("I cannot" in m["content"][:20] for m in assistant_msgs):
        return False
    return True

dataset = dataset.filter(filter_low_quality)
```

## 基本训练流程

### Full Fine-Tuning

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    max_seq_length=4096,
    packing=True,  # 启用 packing
    dataset_kwargs={"add_special_tokens": False},
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```

### LoRA Fine-Tuning

大多数场景用 LoRA 就够了——显存省 60-80%，效果接近 full fine-tuning：

```python
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

peft_config = LoraConfig(
    r=16,                      # rank，越大越接近 full FT
    lora_alpha=32,             # 通常设为 2*r
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="./sft-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # LoRA 显存小，batch 可以更大
    gradient_accumulation_steps=2,
    learning_rate=2e-4,  # LoRA 学习率通常比 full FT 大 10x
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    max_seq_length=4096,
    packing=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset["train"],
)

trainer.train()
# 保存 LoRA adapter
trainer.save_model()
```

### QLoRA（4-bit 量化 + LoRA）

进一步省显存——base model 用 4-bit 加载，只训练 LoRA 参数：

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # 双重量化，再省一点
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    args=training_args,
    peft_config=peft_config,
    train_dataset=dataset["train"],
    model_init_kwargs={"quantization_config": bnb_config},
)
```

QLoRA 的 7B 模型在单张 24GB GPU（如 RTX 4090）上就能训练。

## 关键技巧

### Packing

TRL 的 `packing=True` 会把多个短样本拼接到一个 `max_seq_length` 的序列中：

```
不 packing: [样本1 + padding][样本2 + padding][样本3 + padding]
packing:    [样本1 | 样本2 | 样本3]  ← 一个序列，无 padding 浪费
```

好处：GPU 利用率高，训练速度快 2-3x。
注意：packing 时需要用 attention mask 确保不同样本之间不会互相 attend。TRL 自动处理了这个。

### Chat Template

确保使用正确的 chat template：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# TRL SFTTrainer 会自动用 tokenizer 的 chat template
# 把 messages 格式化成模型期望的格式
```

不同模型的 chat template 差异很大（ChatML、Llama 格式等），用错了效果会很差。

### Loss Masking

只在 assistant 回复部分计算 loss，不在 user/system 部分计算：

```python
# TRL 自动处理 loss masking
# 在 SFTConfig 中确保 dataset_text_field 不要手动拼接
# 而是传入 messages 格式，让 TRL 自己处理
```

### 学习率与训练轮数

经验值：
- Full FT：`lr=1e-5 ~ 5e-5`，1-3 epochs
- LoRA：`lr=1e-4 ~ 5e-4`，2-5 epochs
- 数据量少（< 10k）→ 少 epoch + 大 lr
- 数据量大（> 100k）→ 1 epoch 可能就够

过拟合信号：train loss 持续降但 eval loss 上升。SFT 特别容易过拟合，尤其是小数据集。

## 分布式训练

### 多 GPU（FSDP / DeepSpeed）

```bash
# FSDP
accelerate launch --config_file fsdp_config.yaml train_sft.py

# DeepSpeed ZeRO-2
accelerate launch --config_file ds_z2_config.yaml train_sft.py
```

SFT 阶段通常用 ZeRO Stage 2 / FSDP SHARD_GRAD_OP 就够了，不需要 Stage 3。

## 评估

SFT 后的简单评估：

```python
# 推理测试
from transformers import pipeline

pipe = pipeline("text-generation", model="./sft-output", device=0)
result = pipe("Explain what a neural network is.", max_new_tokens=512)
print(result[0]["generated_text"])
```

更系统的评估用 lm-evaluation-harness 或 lighteval。

## 常见坑

1. **忘记设 `max_seq_length`**：默认可能很短，长样本被截断
2. **Chat template 不匹配**：base model vs instruct model 的 template 不同
3. **学习率过大**：模型 "忘记" 预训练知识（catastrophic forgetting）
4. **数据泄漏**：训练数据与评估 benchmark 重叠
5. **LoRA target modules 不全**：只加在 attention 上可能不够，加上 FFN 层通常更好

## 相关

- [[SFT 原理]] — SFT 的理论基础
- [[LoRA]] — LoRA 技术详解
- [[TRL 概述|TRL 概述]] — TRL 框架总览
- [[DPO-TRL实践|DPO-TRL实践]] — SFT 之后的 DPO 对齐
- [[GRPO-TRL实践|GRPO-TRL实践]] — SFT 之后的 GRPO 训练
- [[FSDP|FSDP]] — 分布式 SFT 训练
- [[DeepSpeed|DeepSpeed]] — 分布式 SFT 训练
