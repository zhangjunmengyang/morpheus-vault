---
title: "unsloth"
type: reference
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/reference
---
# Unsloth 概述

[Unsloth](https://github.com/unslothai/unsloth) 是一个专注于大模型高效微调的开源框架。核心卖点：**2x faster, 50% less memory**，通过手写 Triton/CUDA kernel 优化常见操作实现。

## 为什么选 Unsloth

在 HuggingFace Transformers + PEFT 生态中做 LoRA/QLoRA 微调时，默认实现有大量冗余计算和内存浪费。Unsloth 的优化点：

1. **Fused kernels** —— 把 RoPE、RMSNorm、CrossEntropyLoss、SwiGLU 等操作融合成单个 kernel，减少 GPU 内存读写
2. **手动反向传播** —— 不用 autograd，手写 backward pass，省中间激活的内存
3. **智能 gradient checkpointing** —— 比 HF 默认方案更细粒度的 checkpoint 策略
4. **4-bit 训练优化** —— 自研 4-bit dequant kernel，比 bitsandbytes 更快

## 基本使用

### 安装

```bash
pip install unsloth
# 或者指定 CUDA 版本
pip install "unsloth[cu121]"
```

### 微调流程

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# 2. 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. 准备数据
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def format_data(example):
    messages = [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# 4. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.map(format_data),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs",
    ),
    dataset_text_field="text",
    max_seq_length=4096,
)
trainer.train()
```

## 支持的模型

Unsloth 对以下模型做了深度优化（手写 kernel）：

- **Llama 系列**：Llama 2/3/3.1/3.2/3.3
- **Qwen 系列**：Qwen 2/2.5
- **Mistral / Mixtral**
- **Gemma 2/3**
- **Phi-3/4**
- **DeepSeek-V2/V3**（MoE 架构）

不在列表中的模型也能用，只是回退到 HF 默认实现，没有加速。

## 性能对比

在 RTX 4090 上用 Qwen2.5-7B 做 QLoRA 微调（batch_size=2, seq_len=4096）：

| 框架 | 显存占用 | 训练速度 |
|------|---------|---------|
| HF + PEFT | ~18 GB | 1x |
| Unsloth | ~11 GB | 2.1x |

显存省 ~40%，速度翻倍。这意味着**同样的 GPU 可以训练更大的模型或用更大的 batch size**。

## 局限性

1. **单卡为主** —— Unsloth 的多卡支持（DDP/FSDP）是后来加的，不如原生框架成熟
2. **只做微调** —— 不支持预训练（token 数太大不划算）
3. **Kernel 适配** —— 每个新模型架构都需要手写 kernel，支持速度受限于开发团队
4. **与其他库的兼容** —— 修改了底层 attention/norm 实现，可能与某些第三方库冲突

## 我的看法

Unsloth 是个人开发者和小团队的利器 —— 一张 4090 就能微调 7B 模型，性价比极高。但它本质上是一个「优化层」而非完整框架，大规模训练还是需要 Megatron-LM 或 DeepSpeed。建议：**用 Unsloth 快速实验，验证效果后再切到生产级框架部署**。

## 相关

- [[量化]]
- [[Chat Templates]]
- [[Checkpoint]]
- [[运行 & 保存模型]]
- [[多卡并行]]
- [[训练示例概述]]
