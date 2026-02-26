---
brief: "Unsloth Qwen3 训练——Qwen3 系列模型（包括 MoE）的 Unsloth 专项配置；QLoRA 显存配置/MoE 的 expert 并行设置/Qwen3 tokenizer 特殊 token 处理；实测显存和速度数据。"
title: "Qwen3"
type: project
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/project
---
# Qwen3 Unsloth 训练

> 文档：https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune

Qwen3 是阿里开源的第三代大语言模型，在中文场景和代码生成上表现非常强。用 Unsloth 微调 Qwen3 是目前中文 LLM 最高效的方案之一。

## 模型选择

| 模型 | 参数量 | 4bit 显存 | 适用 |
|------|--------|-----------|------|
| Qwen3-0.6B | 0.6B | ~1GB | 极端资源受限/嵌入式 |
| Qwen3-1.7B | 1.7B | ~2GB | 轻量级任务 |
| Qwen3-4B | 4B | ~4GB | 性价比最高 |
| Qwen3-8B | 8B | ~6GB | 通用任务推荐 |
| Qwen3-14B | 14B | ~10GB | 复杂推理 |
| Qwen3-32B | 32B | ~20GB | 高质量需求 |

## 快速上手

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
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

## Qwen3 的特殊之处

### Thinking Mode

Qwen3 支持"思考模式"——类似 DeepSeek-R1 的 `<think>` 标签。微调时需要注意：

```python
# 如果你的训练数据包含思考过程
messages = [
    {"role": "user", "content": "证明 √2 是无理数"},
    {"role": "assistant", "content": "<think>\n假设 √2 是有理数...\n</think>\n\n√2 是无理数，证明如下..."},
]
```

如果不需要思考模式，确保训练数据中没有 `<think>` 标签，否则模型推理时会尝试思考导致延迟增加。

### Chat Template

Qwen3 使用 ChatML 格式：

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
```

```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
# Qwen3 和 Qwen2.5 共用模板
```

## 训练配置

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./qwen3_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        optim="adamw_8bit",
    ),
    max_seq_length=4096,
)

trainer.train()
```

## 推理与部署

```python
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "写一个 Spark UDF 计算两点间的地理距离"}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=1024, temperature=0.6)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

导出为 GGUF 后可直接在 Ollama 中使用：

```python
model.save_pretrained_gguf("qwen3_gguf", tokenizer, quantization_method="q4_k_m")
```

## 相关

- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/LLM/Frameworks/Unsloth/Chat Templates|Chat Templates]]
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/LLM/Frameworks/Unsloth/Gemma 3 训练|Gemma 3 训练]]
- [[AI/LLM/Frameworks/Unsloth/gpt-oss 训练|gpt-oss 训练]]
