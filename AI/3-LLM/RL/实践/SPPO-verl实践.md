---
brief: "SPPO verl 实践——Self-Play Preference Optimization 的 verl 工程实现；通过 self-play 迭代生成更难的对比样本，逐步提升模型能力上限；适合偏好学习的持续自改进训练流水线。"
title: "SPPO"
type: project
domain: ai/llm/rl/other-algorithms
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/other-algorithms
  - type/project
---
# SPPO verl 实践

> Self-Play Preference Optimization — 自我对弈 + 偏好优化的结合。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/sppo.html

## SPPO 概述

SPPO 结合了 SPIN 的自我对弈和 DPO 的偏好优化，但引入了更精细的胜率估计：

```
SPIN: 模型生成 vs Ground Truth → DPO loss
SPPO: 模型生成多个回答 → 互相比较胜率 → 按胜率加权训练
```

核心创新：**不是二元的 chosen/rejected，而是连续的胜率分数**。

```python
# SPPO 的胜率计算
def compute_win_rates(responses, reward_model):
    """
    计算每个回答相对于其他回答的胜率
    """
    K = len(responses)
    scores = reward_model.score(responses)
    
    win_rates = []
    for i in range(K):
        # 与其他所有回答比较
        wins = sum(1 for j in range(K) if scores[i] > scores[j] and i != j)
        win_rate = wins / (K - 1)
        win_rates.append(win_rate)
    
    return win_rates  # [0.0 ~ 1.0]

# 高胜率的回答被强化，低胜率的被抑制
# 比二元的 chosen/rejected 提供了更丰富的梯度信号
```

## verl 配置

```yaml
algorithm:
  name: sppo
  group_size: 8
  beta: 0.1
  win_rate_threshold: 0.6  # 胜率阈值

actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 5e-7

rollout:
  engine: vllm
  temperature: 0.8
  max_new_tokens: 1024

reward:
  type: model
  model_name: "reward-model-path"
```

## SPPO vs SPIN vs DPO

| 维度 | DPO | SPIN | SPPO |
|------|-----|------|------|
| 偏好来源 | 人工标注 | 模型 vs GT | 模型互比 |
| 需要 GT | ❌ | ✅ | ❌ |
| 需要 RM | ❌ | ❌ | ✅ |
| 信号粒度 | 二元 | 二元 | 连续 |
| 自我提升 | ❌ | ✅ | ✅ |

## 我的观点

SPPO 的连续胜率信号确实比二元偏好更细腻，但代价是需要 reward model。在 RM 准确的场景下，SPPO 效果可能优于 DPO/SPIN。但如果 RM 本身有偏差，胜率估计就不可靠了。

实际选择建议：
- 有高质量 RM → SPPO 或 OPO
- 有 GT 答案 → SPIN（简单场景）或 GRPO（复杂场景）
- 只有偏好数据 → DPO

## 相关

- [[AI/3-LLM/RL/实践/SPIN-verl实践|SPIN verl 实践]] — 简化版自我对弈
- [[AI/3-LLM/RL/实践/OPO-verl实践|OPO verl 实践]] — 类似在线偏好方案
- [[AI/3-LLM/RL/实践/DPO-Unsloth实践|DPO 实践]] — 离线方案
- [[AI/3-LLM/Frameworks/verl/实现其他 RL 方法|verl 实现其他 RL 方法]]
- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]]
- [[AI/3-LLM/RL/算法/PPO 原理|PPO 原理]]
- [[AI/3-LLM/Frameworks/verl/verl 概述|verl 概述]]
