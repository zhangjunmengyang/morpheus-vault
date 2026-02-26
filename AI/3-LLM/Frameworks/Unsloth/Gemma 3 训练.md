---
brief: "Unsloth Gemma 3 训练——Google Gemma 3 系列模型的 Unsloth 专项配置；模型加载/LoRA 设置/训练稳定性调参；Gemma 3 特有的 tokenizer 和位置编码适配注意事项。"
title: "Gemma 3"
type: project
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/project
---
# Gemma 3 Unsloth 训练

> 文档：https://docs.unsloth.ai/models/gemma-3-how-to-run-and-fine-tune

Gemma 3 是 Google 开源的第三代模型，主打小而精。4B 和 12B 两个规格在同参数量级表现出色，特别是多语言和指令遵循能力。

## 模型规格

| 模型 | 参数量 | 4bit 显存 | 特点 |
|------|--------|-----------|------|
| Gemma-3-1B | 1B | ~1.5GB | 极轻量 |
| Gemma-3-4B | 4B | ~4GB | 性价比王 |
| Gemma-3-12B | 12B | ~8GB | 综合能力强 |
| Gemma-3-27B | 27B | ~16GB | 接近 70B 级水平 |

## 快速上手

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

## Gemma 3 的特殊点

### Chat Template

Gemma 3 使用自己独特的模板格式：

```
<start_of_turn>user
你好<end_of_turn>
<start_of_turn>model
你好！有什么可以帮你的？<end_of_turn>
```

```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
```

### 注意事项

1. **License**：Gemma 有自己的使用协议（Gemma License），商用需要确认合规
2. **System Prompt**：Gemma 3 对 system prompt 的支持不如 Qwen/Llama 稳定，复杂 system prompt 效果可能打折
3. **中文能力**：虽然支持多语言，但中文表现不如 Qwen 系列

## 训练配置

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./gemma3_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
    ),
    max_seq_length=4096,
)

trainer.train()
```

## 和其他模型的对比

在 Unsloth 训练框架下的实际体验：

| 维度 | Gemma 3 4B | Qwen3 4B | Llama 3.2 3B |
|------|-----------|----------|-------------|
| 英文指令遵循 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 中文能力 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 代码生成 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 训练稳定性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 微调数据效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

如果主要做中文场景，首选 Qwen。如果做英文或多语言场景，Gemma 3 是很好的选择。

## 相关

- [[AI/3-LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/3-LLM/Frameworks/Unsloth/Qwen3 训练|Qwen3 训练]]
- [[AI/3-LLM/Frameworks/Unsloth/Chat Templates|Chat Templates]]
- [[AI/3-LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/3-LLM/Frameworks/Unsloth/notebook 合集|Notebook 合集]]
