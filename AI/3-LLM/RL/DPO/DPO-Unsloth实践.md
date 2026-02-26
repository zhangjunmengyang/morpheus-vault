---
brief: "DPO Unsloth 实践——Unsloth 加速 DPO 训练；QLoRA + DPO 在低显存环境下的偏好对齐；相比 PPO 减少 4 个模型为 2 个，单卡可运行 7B-13B 的 DPO 全流程。"
title: "DPO"
type: project
domain: ai/llm/rl/dpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/dpo
  - type/project
---
# DPO Unsloth 实践

> 用 Unsloth 跑 DPO/ORPO/KTO 偏好对齐训练。
> 参考：https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/reinforcement-learning-dpo-orpo-and-kto

## DPO 原理速览

DPO（Direct Preference Optimization）把 RLHF 中的 reward model + PPO 两步简化为一步：

```
传统 RLHF:  SFT → 训练 Reward Model → PPO 训练
DPO:        SFT → 直接用偏好数据训练（无需 Reward Model）
```

核心思想：偏好数据 `(prompt, chosen, rejected)` 本身就包含了奖励信号，不需要单独训练一个 reward model。

损失函数：

```python
# DPO loss（简化版）
def dpo_loss(model, ref_model, chosen, rejected, beta=0.1):
    # 计算 log probability
    log_prob_chosen = model.log_prob(chosen)
    log_prob_rejected = model.log_prob(rejected)
    ref_log_prob_chosen = ref_model.log_prob(chosen)
    ref_log_prob_rejected = ref_model.log_prob(rejected)
    
    # 隐式奖励差
    log_ratio_chosen = log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = log_prob_rejected - ref_log_prob_rejected
    
    # DPO loss
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss.mean()
```

## 数据准备

DPO 需要偏好数据（chosen vs rejected）：

```python
# 数据格式
sample = {
    "prompt": "如何解释量子纠缠？",
    "chosen": "量子纠缠是一种量子力学现象...(详细、准确的回答)",
    "rejected": "量子纠缠就是两个粒子心灵感应...(不准确的回答)"
}

# 从 HuggingFace 加载现成数据集
from datasets import load_dataset
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
```

**自建偏好数据的方法**：

```python
# 方法1：用强模型给弱模型的输出排序
def generate_preference_data(prompts, strong_model, weak_model):
    data = []
    for prompt in prompts:
        # 弱模型生成多个回答
        responses = weak_model.generate(prompt, num_return=4)
        # 强模型给排序
        ranked = strong_model.rank(prompt, responses)
        data.append({
            "prompt": prompt,
            "chosen": ranked[0],     # 最好的
            "rejected": ranked[-1],  # 最差的
        })
    return data

# 方法2：人工标注（质量最高但成本高）
# 方法3：规则过滤（如长度、格式、关键词）
```

## Unsloth 训练

```python
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# DPO 训练
trainer = DPOTrainer(
    model=model,
    ref_model=None,      # Unsloth 自动处理 ref model
    args=DPOConfig(
        output_dir="./dpo-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,          # DPO 通常只需 1 epoch
        learning_rate=5e-7,          # DPO lr 要很低
        beta=0.1,                    # DPO 温度参数
        loss_type="sigmoid",         # sigmoid / hinge / ipo
        fp16=True,
        warmup_ratio=0.1,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## DPO 变体

### ORPO（Odds Ratio Preference Optimization）

ORPO 不需要 reference model，更简洁：

```python
from trl import ORPOTrainer, ORPOConfig

trainer = ORPOTrainer(
    model=model,
    args=ORPOConfig(
        output_dir="./orpo-output",
        beta=0.1,
        learning_rate=8e-6,  # ORPO 可以用稍高的 lr
        num_train_epochs=1,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

### KTO（Kahneman-Tversky Optimization）

KTO 不需要成对数据，只需要 "好/坏" 标签：

```python
# KTO 数据格式 — 不需要 chosen/rejected 配对
kto_sample = {
    "prompt": "解释机器学习",
    "completion": "机器学习是...",
    "label": True,  # True=好, False=坏
}
```

## 调参经验

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| `beta` | 0.05-0.5 | 越大越保守，越小越激进 |
| `learning_rate` | 1e-7 ~ 5e-6 | DPO 对 lr 非常敏感 |
| `epochs` | 1-2 | 多了容易过拟合偏好 |
| `max_length` | 1024-2048 | 偏好数据一般不会太长 |

**常见坑**：

1. **lr 太高** → chosen 和 rejected 的 log prob 都下降（模型「忘了」怎么说话）
2. **beta 太小** → 模型偏离 ref model 太远，输出质量变差
3. **数据不均衡** → chosen 和 rejected 的分布差异太大，训练不稳定
4. **训练太久** → 过拟合偏好数据，泛化能力下降

## 我的观点

DPO 的价值在于简洁——不需要训 reward model、不需要复杂的 PPO pipeline，一步到位。但它的局限也很明显：**偏好数据的质量决定了一切**。垃圾数据上跑 DPO，结果只会更差。

在实际项目中的选择建议：
- **通用对齐**：DPO，数据容易获取
- **数学/代码等可验证任务**：GRPO > DPO（有客观奖励信号）
- **没有成对数据**：KTO
- **快速实验**：ORPO（最简单）

## 相关

- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — 对比 on-policy 方法
- [[AI/3-LLM/RL/GRPO/GRPO-Unsloth实践|GRPO Unsloth 实践]] — on-policy RL 方案
- [[AI/3-LLM/Frameworks/Unsloth/训练示例概述|Unsloth 训练示例]]
- [[AI/3-LLM/Application/Synthetic-Data/Synthetic Data|Synthetic Data]] — 数据合成
- [[AI/3-LLM/RL/RLOO/RLOO-TRL实践|RLOO 实践]] — 另一种 off-policy 方法
- [[AI/3-LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/3-LLM/RL/DPO/DPO-TRL实践|DPO-TRL实践]]
- [[AI/3-LLM/Frameworks/Unsloth/Unsloth 概述|Unsloth 概述]]
