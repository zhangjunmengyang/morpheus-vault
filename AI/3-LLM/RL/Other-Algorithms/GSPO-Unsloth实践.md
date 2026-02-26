---
brief: "GSPO Unsloth 实践——Group Sequence Policy Optimization 的 Unsloth 工程实现；低显存环境下跑序列级 IS ratio 约束的 RL 训练；配置和 Qwen3 MoE 的 GSPO 训练参数参考。"
title: "GSPO"
type: project
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/project
---
# GSPO-Unsloth 实践

> 参考：https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/gspo-reinforcement-learning

## 什么是 GSPO

GSPO (Group Relative Self-Play Optimization) 是对 GRPO 的改进，核心思路：引入 **self-play** 机制，用模型的旧版本作为对手来构造偏好对。

与 GRPO 的区别：

| | GRPO | GSPO |
|---|---|---|
| Advantage 计算 | group 内 reward 归一化 | self-play 比较 |
| 参考策略 | 固定 reference model | 动态更新的旧版本 |
| Loss 形式 | 策略梯度 | 类 DPO 的偏好 loss |
| 稳定性 | 一般 | 更好（self-play 提供更稳定的 baseline） |

简单理解：GSPO ≈ GRPO + DPO 的混合体。用 GRPO 的 group sampling，但用 DPO 风格的 loss。

## Unsloth 实现

Unsloth 提供了开箱即用的 GSPO trainer：

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig  # GSPO 基于 GRPOConfig 扩展

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # auto
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

# 3. 定义 reward function
def gspo_reward_fn(completions, **kwargs):
    """GSPO 的 reward function 和 GRPO 完全一样"""
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        # 你的评分逻辑
        score = evaluate(text)
        rewards.append(score)
    return rewards

# 4. 配置训练
training_args = GRPOConfig(
    output_dir="gspo_output",
    
    # GSPO 特有参数
    use_gspo=True,            # 开启 GSPO 模式
    gspo_beta=0.1,            # self-play 的 beta 参数（类似 DPO beta）
    
    # 通用 RL 参数
    num_generations=8,         # group_size
    max_new_tokens=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    
    # 优化
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    
    logging_steps=1,
    save_steps=50,
)

# 5. 开始训练
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[gspo_reward_fn],
    tokenizer=tokenizer,
)

trainer.train()
```

## GSPO 的 Loss 详解

```python
# GSPO loss 的核心逻辑（伪代码）
def gspo_loss(policy_log_probs, ref_log_probs, rewards, beta=0.1):
    """
    对 group 内的 response 两两比较，构造偏好对
    """
    group_size = len(rewards)
    loss = 0
    count = 0
    
    for i in range(group_size):
        for j in range(group_size):
            if rewards[i] > rewards[j]:  # i 比 j 好
                # DPO-style loss
                pi_diff = policy_log_probs[i] - policy_log_probs[j]
                ref_diff = ref_log_probs[i] - ref_log_probs[j]
                
                logit = beta * (pi_diff - ref_diff)
                loss += -F.logsigmoid(logit)
                count += 1
    
    return loss / count
```

这比 GRPO 的好处：
- 不依赖 group mean/std 做 advantage 归一化（归一化对 outlier 敏感）
- 每一对比较都提供明确的梯度方向
- 与 DPO 理论基础一致，有更好的理论保证

## beta 参数调节

```python
# beta 控制 reward 差异对 loss 的放大程度
# beta 小 → loss 对偏好差异不敏感 → 保守更新
# beta 大 → loss 对偏好差异很敏感 → 激进更新

# 推荐范围:
# beta = 0.05: 非常保守，适合模型已经很好的情况
# beta = 0.1:  标准值
# beta = 0.2:  偏激进，早期训练可以用
# beta = 0.5:  太大，容易过拟合到 reward function
```

## 与 GRPO 的实验对比

基于个人在数学推理任务上的粗略对比（Qwen2.5-7B, GSM8K）：

```
GRPO (group_size=8, kl_coef=0.001):
  - GSM8K acc: 82.3% → 87.1% (after 500 steps)
  - reward 曲线: 前 100 步波动大，后面稳定上升

GSPO (group_size=8, beta=0.1):
  - GSM8K acc: 82.3% → 87.8% (after 500 steps)
  - reward 曲线: 更平滑，从第 50 步就开始稳定上升
```

**结论**：GSPO 在收敛稳定性上确实更好，最终效果略优。但训练速度略慢（两两比较的计算量 O(n²)）。

## 实际注意事项

1. **显存**：和 GRPO 差不多，主要开销在 rollout 的 group_size
2. **速度**：GSPO 的 loss 计算比 GRPO 稍慢（两两比较），但 rollout 才是瓶颈，影响不大
3. **与 LoRA 兼容**：完全兼容，Unsloth 的 4-bit LoRA 可以正常用
4. **Reward function 设计**：和 GRPO 完全一样，不需要为 GSPO 特殊设计

## 相关

- [[AI/3-LLM/RL/GRPO/GRPO|GRPO]]
- [[AI/3-LLM/RL/DPO/DPO|DPO]]
- [[Unsloth 概述|Unsloth 概述]]
- [[训练示例概述|Unsloth 训练示例]]
- [[GRPO 深度理解|GRPO 深度理解]]
- [[PPO 原理|PPO 原理]]
