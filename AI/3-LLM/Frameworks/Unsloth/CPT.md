---
brief: "Unsloth CPT（Continued Pre-Training）——继续预训练的 Unsloth 工程指南；大规模文本数据的高效预训练配置，打包策略（packing）提升 GPU 利用率；领域适应预训练的 Unsloth 实践。"
title: "CPT（continual pretraining）"
type: concept
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/concept
---
# CPT（Continual Pretraining）

> 文档：https://docs.unsloth.ai/basics/continued-pretraining

CPT 是在已有预训练模型基础上，用领域数据继续做预训练。跟 SFT 的区别是：CPT 用的是 next-token prediction 的原始目标，不需要指令格式的数据。

## 什么时候用 CPT

**适合的场景：**
- 模型对你的领域知识严重不足（医疗、法律、金融术语）
- 你有大量非结构化的领域文档（内部 wiki、论文、规范文档）
- SFT 后模型"知道怎么回答"但"不知道领域知识"

**不适合的场景：**
- 只是想让模型学会特定的回答风格 → 用 SFT
- 数据量太少（< 10MB） → CPT 效果不明显

## 典型 Pipeline

```
领域文档 → 清洗/分块 → CPT → SFT → (可选) RLHF/DPO → 部署
```

CPT 是在 SFT 之前做的。先让模型"学会领域知识"，再通过 SFT 让模型"学会如何利用这些知识回答问题"。

## Unsloth 实现

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",  # 注意用 base 模型，不是 Instruct
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,          # CPT 建议用稍大的 rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",
                     "embed_tokens", "lm_head"],  # CPT 要训练 embedding 层
    lora_alpha=32,
)
```

### 数据准备

CPT 的数据格式很简单——纯文本，用 EOS token 分隔文档：

```python
from datasets import load_dataset

def format_for_cpt(example):
    # 直接拼接文本，末尾加 EOS
    return {"text": example["content"] + tokenizer.eos_token}

dataset = load_dataset("json", data_files="domain_docs.jsonl")
dataset = dataset.map(format_for_cpt)
```

### 训练配置

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(
        output_dir="./cpt_output",
        num_train_epochs=2,           # CPT 通常 1-3 个 epoch
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,           # 比 SFT 稍高
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
    ),
    max_seq_length=4096,
    packing=True,  # 多个短文档打包到一个 sequence，提高 GPU 利用率
)

trainer.train()
```

## 关键参数差异（CPT vs SFT）

| 参数 | CPT | SFT |
|------|-----|-----|
| 基座模型 | Base 模型 | Base 或 Instruct |
| LoRA rank | 32-64（更大） | 8-16 |
| 训练 embedding | 是 | 通常不用 |
| Learning rate | 5e-5 ~ 1e-4 | 1e-5 ~ 5e-5 |
| 数据格式 | 纯文本 | 指令/对话格式 |
| Packing | 强烈推荐 | 可选 |

## 注意事项

1. **灾难性遗忘**：CPT 可能让模型忘掉原来会的东西。缓解方法是混入少量通用数据（如 Wikipedia）
2. **数据质量**：垃圾数据做 CPT 比不做更糟。必须清洗干净
3. **训练量**：不要过度训练。1-2 epoch 足够，过多会过拟合
4. **评估**：用领域相关的 perplexity 评估，同时监控通用能力（如 MMLU）是否退化

## 相关

- [[AI/3-LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/3-LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/3-LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[AI/3-LLM/SFT/SFT 原理|SFT 原理]]
- [[AI/3-LLM/SFT/LoRA|LoRA]]

## See Also

- [[AI/3-LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]] — 框架入口：本文是 Unsloth 框架下的专项训练配置
- [[AI/3-LLM/Frameworks/Unsloth/TTS 训练|TTS 训练]] — 同框架其他训练类型
