---
brief: "Qwen2.5-VL Unsloth 训练——Qwen2.5 视觉语言模型的 Unsloth 微调工程指南；动态分辨率图像的 token 处理/多模态指令数据格式/QLoRA 显存配置；开源 VLM 微调的主流工程实践。"
title: "Qwen 2.5 VL"
type: project
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/project
---
# Qwen 2.5 VL Unsloth 训练

> 用 Unsloth 对 Qwen 2.5 VL 做视觉微调与 VLM RL 训练。
> 参考：https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl

## Qwen 2.5 VL 概览

Qwen 2.5 VL 是 Qwen 系列的视觉语言版本，支持图像和视频理解。模型矩阵：

| 模型 | 参数量 | 推理显存 | 微调显存(QLoRA) |
|------|--------|---------|----------------|
| Qwen2.5-VL-3B | 3B | ~8GB | ~12GB |
| Qwen2.5-VL-7B | 7B | ~16GB | ~18GB |
| Qwen2.5-VL-72B | 72B | ~144GB | ~48GB(4-bit) |

其核心特色是 **Naive Dynamic Resolution** — 不像 InternVL 那样分 tile，而是直接处理原始分辨率图片（通过 ViT 的 patch 机制自适应）。

## SFT 微调

### 模型加载

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",  # 4-bit 量化版
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    finetune_vision_layers=True,     # 关键：微调视觉层
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

### 数据格式

```python
# Qwen VL 的对话格式
def format_sample(image_path, question, answer):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
    }

# 示例
sample = format_sample(
    "medical_xray.png",
    "请分析这张X光片，描述你观察到的异常。",
    "在右肺中叶区域观察到一处约2cm的阴影..."
)
```

### 训练

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./qwen25vl-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,  # VLM 微调 lr 偏低
        fp16=True,
        save_strategy="steps",
        save_steps=100,
    ),
)
trainer.train()
```

## Vision RL（VLM 强化学习）

这是 Qwen 2.5 VL 微调的高级玩法 — 用 GRPO 等 RL 算法提升视觉推理能力：

```python
from trl import GRPOTrainer, GRPOConfig

# 定义奖励函数
def reward_fn(completions, images, ground_truths):
    """
    根据模型输出和真实答案计算奖励
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # 提取模型的最终答案
        answer = extract_answer(completion)
        # 二值奖励：正确 +1，错误 -1
        reward = 1.0 if answer == gt else -1.0
        rewards.append(reward)
    return rewards

# GRPO 训练配置
config = GRPOConfig(
    output_dir="./qwen25vl-grpo",
    num_generations=4,       # 每个问题采样 4 个回答
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,      # RL 阶段 lr 更低
    max_completion_length=512,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=rl_dataset,
    reward_funcs=[reward_fn],
)
trainer.train()
```

**VLM RL 的典型应用场景**：
- 数学题图片理解（几何、图表分析）
- 医学影像诊断
- 文档信息提取（准确率奖励）
- GUI agent 操作（任务完成奖励）

## 实战 Tips

1. **Vision layers 的微调**：`finetune_vision_layers=True` 对 OCR/细粒度识别很重要，但通用场景可以关掉以节省显存
2. **图片分辨率**：Qwen 2.5 VL 对分辨率比较敏感，训练和推理时保持一致
3. **LoRA rank 选择**：视觉微调建议 r=16-32，比纯文本任务稍高
4. **RL 数据量**：VLM RL 不需要太多数据，1000-5000 条高质量样本即可
5. **评估要分维度**：视觉理解和语言生成分开评估，防止视觉能力提升但语言退化

## 相关

- [[Qwen3-VL-Unsloth训练|Qwen3-VL 训练]] — 下一代 Qwen VLM
- [[DeepSeek-OCR-Unsloth实践|DeepSeek-OCR Unsloth 实践]] — 竞品方案
- [[GRPO 深度理解|GRPO 深度理解]] — RL 算法原理
- [[训练示例概述|Unsloth 训练示例]]
- [[Checkpoint|Unsloth Checkpoint]]
