---
brief: "Unsloth Chat Templates——聊天模板配置指南；Llama/Qwen/Gemma 等不同模型的对话格式规范，tokenizer 的 apply_chat_template 用法；错误的 chat template 是 SFT/RL 训练常见坑点。"
title: "Chat Templates"
type: concept
domain: ai/llm/frameworks/unsloth
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/unsloth
  - type/concept
rating: ★★★☆
sources:
  - "Unsloth Chat Templates 官方文档: https://docs.unsloth.ai/basics/chat-templates"
  - "HuggingFace apply_chat_template: https://huggingface.co/docs/transformers/chat_templating"
---
# Chat Templates

> 文档：https://docs.unsloth.ai/basics/chat-templates

Chat Template 是 LLM 微调中最容易被忽视但最容易踩坑的环节。用错模板轻则训练效果差，重则模型输出乱码。

## 什么是 Chat Template

Chat Template 定义了多轮对话的格式化方式——模型怎么区分 system/user/assistant 的消息边界。

不同模型家族用不同的模板：

```
# ChatML 格式（Qwen、Yi 等）
<|im_start|>system
你是一个助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的？<|im_end|>

# Llama 3 格式
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
你好<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
你好！有什么可以帮你的？<|eot_id|>
```

## Unsloth 中的使用

Unsloth 封装了主流模型的 chat template，大部分情况下自动处理：

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
)

# 自动检测并应用 chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
```

### 支持的模板类型

| 模型 | 模板名 | 格式 |
|------|--------|------|
| Qwen 2.5/3 | `qwen-2.5` | ChatML |
| Llama 3/3.1 | `llama-3.1` | Llama 格式 |
| Gemma 2/3 | `gemma-3` | Gemma 格式 |
| Mistral | `mistral` | Mistral 格式 |
| Phi-3/4 | `phi-4` | ChatML 变体 |

### 数据格式化

```python
def format_conversation(example):
    messages = [
        {"role": "system", "content": "你是一位数据工程师助手。"},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    # tokenizer.apply_chat_template 自动处理格式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_conversation)
```

## 常见坑

### 1. 模板不匹配

用 ChatML 模板训练的模型，推理时用了 Llama 模板，输出会很混乱。**训练和推理必须用同一个模板**。

### 2. Special Token 没有训练

```python
# ❌ 错误：新 token 的 embedding 没有训练
tokenizer.add_special_tokens({"additional_special_tokens": ["<tool>"]})

# ✅ 正确：添加后要 resize embedding 并训练
model.resize_token_embeddings(len(tokenizer))
```

### 3. EOS Token 丢失

如果 EOS token 没有正确设置，模型推理时不会停止生成：

```python
# 检查 EOS token
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# 确保生成时使用正确的停止条件
outputs = model.generate(
    **inputs,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
)
```

### 4. 多轮对话的 Label Masking

训练时应该只对 assistant 的回复计算 loss，user 和 system 的 token 不参与：

```python
# Unsloth 的 trainer 自动处理了这个问题
# 但如果自己处理数据，要注意设置 labels
# user/system tokens 的 label 设为 -100（ignore_index）
```

## 调试技巧

```python
# 打印格式化后的文本，肉眼检查
messages = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(repr(formatted))  # 看 special tokens 是否正确
```

## 相关

- [[Unsloth 概述|Unsloth 概述]]
- [[训练示例概述|训练示例概述]]
- [[AI/3-LLM/Frameworks/Unsloth/数据合成|Unsloth 数据合成]]
- [[Qwen3 训练|Qwen3 训练]]
- [[SFT 原理|SFT 原理]]
