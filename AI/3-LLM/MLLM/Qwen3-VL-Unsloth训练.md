---
brief: "Qwen3-VL Unsloth 训练——Qwen3 视觉语言模型的 Unsloth 微调工程指南；多模态数据格式/图像 token 处理/显存优化配置；低显存环境下微调 VLM 的实践参考。"
title: "Qwen3-VL"
type: project
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/project
---
# Qwen3-VL-Unsloth 训练

> 参考：https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning
> VLM 微调指南：https://docs.unsloth.ai/basics/vision-fine-tuning

## Qwen3-VL 概述

Qwen3-VL 是通义千问的第三代视觉语言模型，相比 Qwen2.5-VL 主要改进：
- 更好的 OCR 和文档理解
- 动态分辨率支持增强
- 视频理解能力提升

## Unsloth 微调 VLM 的核心流程

### 1. 环境准备

```bash
pip install unsloth
pip install pillow torchvision
```

### 2. 加载模型

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen3-VL-7B",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
```

⚠️ VLM 和纯文本模型不一样，用的是 `FastVisionModel` 而不是 `FastLanguageModel`。

### 3. LoRA 配置

```python
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    
    # VLM 特有: 是否微调视觉编码器
    finetune_vision_layers=False,     # 一般不动 vision encoder
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

**要不要微调 vision encoder？**
- 通常不需要 — 预训练的 vision encoder 已经很强
- 如果你的任务视觉特征很特殊（如医学影像、卫星图），可以 `finetune_vision_layers=True`
- 微调 vision encoder 显存占用会增加 30-50%

### 4. 数据准备

VLM 微调数据格式需要包含图片引用：

```python
from datasets import Dataset
from PIL import Image

def create_vlm_dataset():
    """构造 VLM 训练数据"""
    samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "path/to/chart.png"},
                        {"type": "text", "text": "这张图表的主要趋势是什么？"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "这张折线图显示了2020-2024年的销售增长趋势..."
                }
            ]
        },
        # ... more samples
    ]
    return Dataset.from_list(samples)

# 对于 Qwen3-VL 的 chat template 格式
def format_for_qwen3vl(sample):
    """转为 Qwen3-VL 的原生格式"""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": sample["answer"]
            }
        ]
    }
```

### 5. 训练配置

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="qwen3vl_finetuned",
    
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    learning_rate=2e-5,  # VLM 可以用比纯文本略大的 LR
    num_train_epochs=3,
    
    bf16=True,
    max_seq_length=4096,
    
    # VLM 特有
    dataset_text_field="",           # 不使用纯文本字段
    remove_unused_columns=False,     # 保留图片列
    
    logging_steps=5,
    save_steps=100,
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=None,  # Unsloth 自动处理
)

trainer.train()
```

## GRPO 做视觉 RL

Qwen3-VL + RL 的场景：比如训练模型更好地做数学题图片的 OCR + 解题。

```python
from trl import GRPOConfig, GRPOTrainer

def visual_math_reward(completions, images=None, ground_truths=None, **kwargs):
    """视觉数学的 reward function"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion[0]["content"]
        
        # 提取答案
        answer = extract_boxed_answer(text)
        
        if answer is None:
            rewards.append(-1.0)
        elif answer == gt:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    
    return rewards

grpo_config = GRPOConfig(
    output_dir="qwen3vl_grpo",
    num_generations=4,        # VLM 的 group_size 建议小一些（图片占显存）
    max_new_tokens=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=[visual_math_reward],
    tokenizer=tokenizer,
)
```

## 显存估算

VLM 比纯文本模型显存大得多，因为图片 token 很多：

```python
# Qwen3-VL 的图片 token 数量估算
# 默认: 每张图片 ~256-1280 tokens (取决于分辨率)
# 
# 纯文本 7B + LoRA + 4bit: ~8GB
# VLM 7B + LoRA + 4bit:    ~12GB (vision encoder 额外 ~4GB)
#
# 加上图片 token 的 KV cache:
# 1 张 1024×1024 图片 ≈ 1024 tokens ≈ 额外 0.5GB
#
# 推荐配置:
# - SFT: 单卡 A100-40G 可以跑 (batch_size=2, 4-bit)
# - GRPO: 单卡 A100-80G (group_size=4, 4-bit)
# - 全量微调: 4×A100-80G 起步
```

## 踩坑经验

1. **图片路径**：确保训练时图片路径可访问。用绝对路径或把图片 base64 编码进数据集
2. **分辨率统一**：Qwen3-VL 支持动态分辨率，但训练时最好限制 `max_pixels` 防止 OOM
3. **vision encoder 冻结**：除非有充分理由，否则冻结 vision encoder。解冻后训练不稳定且慢
4. **GRPO group_size**：VLM 的 group_size 要比纯文本小，因为每个 response 的 prefill 都包含图片 token
5. **eval 时切推理模式**：`FastVisionModel.for_inference(model)` 切换到推理模式

## 相关

- [[Qwen-VL|Qwen-VL]]
- [[Qwen 2.5 VL-Unsloth训练|Qwen 2.5 VL-Unsloth训练]]
- [[MLLM 概述|MLLM 概述]]
- [[Unsloth 概述|Unsloth 概述]]
- [[GRPO 深度理解|GRPO 深度理解]]
