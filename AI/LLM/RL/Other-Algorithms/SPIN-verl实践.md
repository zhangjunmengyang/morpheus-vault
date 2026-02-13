---
title: "SPIN"
type: project
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/project
---
# SPIN verl 实践

> Self-Play Fine-Tuning — 用自我对弈提升模型能力。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/spin.html

## SPIN 是什么

SPIN（Self-Play Fine-Tuning）的核心思想来自博弈论：让模型和自己的旧版本对弈，持续提升。

```
迭代 0: 模型_0 生成回答 → 与 ground truth 比较 → 训练出 模型_1
迭代 1: 模型_1 生成回答 → 与 ground truth 比较 → 训练出 模型_2
...
当模型生成的回答 ≈ ground truth 时收敛
```

**与 DPO 的区别**：DPO 的 rejected 来自外部标注，SPIN 的 rejected 来自模型自身——即当前模型的输出就是 rejected，ground truth 是 chosen。

## 训练流程

```python
# SPIN 的核心循环
def spin_iteration(model, dataset, num_iterations=3):
    for iteration in range(num_iterations):
        # 1. 用当前模型生成回答（作为 rejected）
        generated = []
        for prompt in dataset:
            response = model.generate(prompt)
            generated.append(response)
        
        # 2. 构造偏好对
        preference_data = [
            {
                "prompt": prompt,
                "chosen": ground_truth,    # 真实回答
                "rejected": generated_resp  # 模型当前输出
            }
            for prompt, ground_truth, generated_resp 
            in zip(dataset.prompts, dataset.answers, generated)
        ]
        
        # 3. DPO-style 训练
        model = dpo_train(model, preference_data)
    
    return model
```

## verl 配置

```yaml
algorithm:
  name: spin
  num_iterations: 3       # 自我对弈迭代次数
  beta: 0.1               # DPO 温度
  
actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 5e-7

rollout:
  engine: vllm
  temperature: 0.7
  max_new_tokens: 1024
```

## 适用场景

SPIN 的独特优势：**不需要 reward model，不需要人工标注偏好，只需要 (prompt, answer) 对**。

适合的场景：
- 有大量 QA 对但没有偏好数据
- 想在 SFT 的基础上进一步提升
- 数据标注预算有限

不适合的场景：
- 开放式生成任务（没有明确的 ground truth）
- 模型已经很强（生成的回答 ≈ ground truth，SPIN 无法继续提升）

## 我的观点

SPIN 是一个很巧妙的「穷人版 RLHF」—— 用自我对弈代替了昂贵的人工偏好标注。但收敛后就停滞了，不像 GRPO 那样可以通过更复杂的奖励函数持续提升。适合作为 SFT → RL 之间的过渡方案。

## 相关

- [[AI/LLM/RL/DPO/DPO-Unsloth实践|DPO Unsloth 实践]] — SPIN 的 loss 基于 DPO
- [[AI/LLM/RL/Other-Algorithms/SPPO-verl实践|SPPO verl 实践]] — 类似的自我对弈思路
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO verl 实践]] — on-policy RL 方案
- [[AI/LLM/Frameworks/verl/实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
