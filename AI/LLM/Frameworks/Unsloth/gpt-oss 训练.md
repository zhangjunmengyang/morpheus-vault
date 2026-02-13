---
title: "gpt-oss"
type: project
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/project
---
# gpt-oss 训练

> 参考：
> - https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning
> - https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune

## 什么是 gpt-oss

gpt-oss 是一系列开源模型的非正式称呼，指的是在能力上追赶 GPT-4 级别的开源模型。Unsloth 对这些模型提供了优化的训练支持，包括 SFT 和 RL。

具体来说，Unsloth 目前支持的 "gpt-oss" 级别模型包括：
- Qwen2.5 系列 (7B/14B/72B)
- Llama 3.1/3.2/3.3 系列
- DeepSeek 系列
- Gemma 3 系列
- Phi-4 系列

## RL 训练流程

### 1. 基础设置

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # gpt-oss 级别可以用更大的 rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)
```

### 2. Reward Function 设计

这是 RL 训练最核心的部分。Unsloth 支持多种 reward function 组合：

```python
def format_reward(completions, **kwargs):
    """格式奖励: 鼓励模型输出结构化格式"""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        
        score = 0.0
        # 检查是否有思考过程
        if "<think>" in text and "</think>" in text:
            score += 0.3
        
        # 检查是否有最终答案标记
        if "\\boxed{" in text:
            score += 0.2
        
        # 惩罚过长输出
        if len(text) > 3000:
            score -= 0.2
        
        rewards.append(score)
    return rewards

def correctness_reward(completions, ground_truths=None, **kwargs):
    """正确性奖励"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion[0]["content"]
        answer = extract_answer(text)
        
        if answer is not None and answer == gt:
            rewards.append(1.0)
        elif answer is not None:
            rewards.append(-0.5)
        else:
            rewards.append(-1.0)
    
    return rewards

# 组合多个 reward
reward_functions = [format_reward, correctness_reward]
reward_weights = [0.3, 0.7]
```

### 3. GRPO 训练

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="gpt_oss_grpo",
    
    # 生成参数
    num_generations=8,
    max_new_tokens=2048,
    temperature=0.9,
    
    # 训练参数
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_steps=500,
    
    # 优化
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    
    # 日志
    logging_steps=1,
    save_steps=50,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    reward_funcs=reward_functions,
    tokenizer=tokenizer,
)

trainer.train()
```

## 不同模型的训练差异

### Qwen2.5 系列

```python
# Qwen 的 chat template 格式
# <|im_start|>system\nYou are...<|im_end|>
# <|im_start|>user\n...<|im_end|>
# <|im_start|>assistant\n...<|im_end|>

# 注意: Qwen 的 tokenizer 默认会加 system prompt
# 如果你的 reward function 检查格式，要知道这一点
```

### Llama 3 系列

```python
# Llama 3 用不同的 chat template
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# ...<|eot_id|>

# 注意: Llama 3 的 tokenizer 没有 BOS token 陷阱
# 但有些版本的 chat template 会自动加 generation prompt
```

### DeepSeek 系列

```python
# DeepSeek 的 thinking 模式
# <think>...</think> 标签内是 chain-of-thought
# 训练时要确保 reward function 能正确处理 think 标签
```

## 训练技巧

### LoRA rank 选择

```python
# gpt-oss 级别模型的 LoRA rank 经验:
# r=16:  轻量级，适合简单任务 (格式调整、风格迁移)
# r=32:  中等，大多数场景的平衡点
# r=64:  重量级，适合能力提升 (数学推理、代码)
# r=128: 接近全参数，训练很慢，仅研究用

# alpha 一般等于 r，或 2*r
# 别忘了: rank 越大，LoRA 适配器的参数量越大
# r=64 对 7B 模型 ≈ 增加 ~200M 可训练参数
```

### 数据混合策略

```python
# RL 训练数据不需要太多，但要多样
# 推荐混合比例 (数学推理为主的场景):
# - 数学题: 60%
# - 逻辑推理: 20%
# - 代码: 15%
# - 通用对话: 5% (防止遗忘)

# 实现方式: 在 dataset 中标记 ability
# reward function 可以根据 ability 调整评分标准
```

### 模型保存与导出

```python
# 保存 LoRA
model.save_pretrained("gpt_oss_lora")

# 合并 LoRA 到基座模型
model.save_pretrained_merged(
    "gpt_oss_merged",
    tokenizer,
    save_method="merged_16bit",  # 或 "merged_4bit" 用于量化部署
)

# 导出 GGUF (用于 Ollama / llama.cpp)
model.save_pretrained_gguf(
    "gpt_oss_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

## 相关

- [[AI/LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|训练示例概述]]
- [[AI/LLM/Frameworks/Unsloth/Qwen3 训练|Qwen3 训练]]
- [[AI/LLM/RL/GRPO/GRPO|GRPO]]
- [[AI/LLM/Frameworks/Unsloth/运行 & 保存模型|运行 & 保存模型]]
- [[AI/LLM/Frameworks/Unsloth/量化 & 显存预估|量化 & 显存预估]]
