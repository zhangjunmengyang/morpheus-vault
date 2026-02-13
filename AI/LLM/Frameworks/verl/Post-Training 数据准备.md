---
title: "Post-Training 数据准备"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# Post-Training 数据准备

> 参考：https://verl.readthedocs.io/en/latest/preparation/prepare_data.html

## 数据是 RL 训练的基础

不同于 SFT 需要 (input, output) 对，RL 训练的数据形式取决于算法：

| 算法 | 数据格式 | 说明 |
|------|---------|------|
| PPO / GRPO | prompt only | 只需要 prompt，response 由 rollout 生成 |
| DPO | (prompt, chosen, rejected) | 偏好对 |
| KTO | (prompt, response, label) | 二元标注：好/坏 |

verl 主要服务 on-policy 算法，所以核心数据格式是 **prompt-only**。

## verl 数据格式

verl 使用 Parquet 格式，必须包含一个 `data` 列，里面是符合 chat template 的 message list：

```python
import pandas as pd

# 基本格式
data = [
    {
        "data": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+3?"}
        ],
        "ability": "math",  # 可选: 数据分类标签
        "reward_model": {
            "ground_truth": "5"  # 可选: 用于 reward 计算
        }
    },
    # ... more samples
]

df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

### 完整的数据处理 pipeline

```python
from datasets import load_dataset
import json

def prepare_verl_data(dataset_name, output_path, split="train"):
    """将 HuggingFace 数据集转为 verl 格式"""
    ds = load_dataset(dataset_name, split=split)
    
    records = []
    for item in ds:
        record = {
            "data": format_messages(item),  # 转为 chat messages
        }
        
        # 如果有 ground truth，加入 reward_model 字段
        if "answer" in item:
            record["reward_model"] = {
                "ground_truth": item["answer"]
            }
        
        # 可选的元数据
        if "category" in item:
            record["ability"] = item["category"]
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path)
    print(f"Saved {len(records)} samples to {output_path}")
    return df

def format_messages(item):
    """根据数据集格式转为 message list"""
    messages = []
    
    # 添加 system prompt（可选但推荐）
    messages.append({
        "role": "system",
        "content": "Please reason step by step, and put your final answer within \\boxed{}."
    })
    
    # 添加 user prompt
    messages.append({
        "role": "user",
        "content": item["question"]
    })
    
    return messages
```

## 常用数据集与转换

### 1. GSM8K（数学推理）

```python
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main", split="train")

records = []
for item in ds:
    # 提取最终答案
    answer = item["answer"].split("####")[-1].strip()
    records.append({
        "data": [
            {"role": "system", "content": "Solve the math problem step by step."},
            {"role": "user", "content": item["question"]}
        ],
        "reward_model": {"ground_truth": answer}
    })
```

### 2. MATH（竞赛数学）

```python
ds = load_dataset("hendrycks/competition_math", split="train")

records = []
for item in ds:
    records.append({
        "data": [
            {"role": "user", "content": item["problem"]}
        ],
        "reward_model": {
            "ground_truth": item["solution"],
            "answer": item["answer"],
            "level": item["level"]
        },
        "ability": item["type"]  # algebra, geometry, etc.
    })
```

### 3. 代码生成

```python
ds = load_dataset("openai/openai_humaneval", split="test")

records = []
for item in ds:
    records.append({
        "data": [
            {"role": "user", "content": f"Complete the following Python function:\n\n{item['prompt']}"}
        ],
        "reward_model": {
            "test_cases": item["test"],
            "entry_point": item["entry_point"]
        }
    })
```

## 数据质量检查

```python
def validate_verl_data(parquet_path):
    """验证数据格式"""
    df = pd.read_parquet(parquet_path)
    
    errors = []
    for idx, row in df.iterrows():
        data = row["data"]
        
        # 检查 message 格式
        if not isinstance(data, list):
            errors.append(f"Row {idx}: data is not a list")
            continue
        
        for msg in data:
            if "role" not in msg or "content" not in msg:
                errors.append(f"Row {idx}: missing role or content")
        
        # 最后一条必须是 user（因为 RL 要模型生成 assistant response）
        if data[-1]["role"] != "user":
            errors.append(f"Row {idx}: last message is not from user")
        
        # prompt 不要太长（会吃掉 response 空间）
        prompt_len = sum(len(m["content"]) for m in data)
        if prompt_len > 4096:
            errors.append(f"Row {idx}: prompt too long ({prompt_len} chars)")
    
    print(f"Total: {len(df)}, Errors: {len(errors)}")
    for e in errors[:10]:
        print(f"  - {e}")
    
    return len(errors) == 0
```

## 数据划分策略

RL 训练中的一个实际问题：你不需要太多数据。

```python
# 经验值：
# - SFT: 越多越好，10k-1M
# - RL (GRPO): 5k-50k prompts 就够了
#   因为每个 prompt 会生成 group_size 个 response
#   group_size=8, 10k prompts → 80k responses per epoch

# 建议：
# - 训练集: 10k-30k prompts
# - 验证集: 500-1k prompts
# - 保证多样性比数量更重要
```

## 相关

- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/verl/verl 训练参数|verl 训练参数]]
- [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]]
- [[AI/LLM/Frameworks/verl/配置文件|配置文件]]
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|Unsloth 训练示例]]
