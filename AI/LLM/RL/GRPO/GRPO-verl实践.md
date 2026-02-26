---
brief: "GRPO verl 实践——基于 verl 框架的 GRPO 分布式训练工程指南；HybridFlow 架构下的 GRPO 配置/reward function 接入/多节点扩展；面向大规模 RLVR 训练的生产级参考。"
title: "GRPO"
type: project
domain: ai/llm/rl/grpo
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/grpo
  - type/project
---
# GRPO verl 实践

> 用 verl 框架跑 GRPO 训练的实战指南。
> 官方文档：https://verl.readthedocs.io/en/latest/algo/grpo.html

## 概述

verl 是字节跳动开源的 RL 训练框架，原生支持 GRPO。相比用 TRL 跑 GRPO，verl 的优势在于：

- **分布式训练**：原生支持多节点多卡，适合大规模训练
- **Actor-Critic 分离**：Actor 和 Rollout worker 可以部署在不同 GPU 上
- **高效推理**：集成 vLLM/SGLang 做 rollout，比 HuggingFace generate 快很多

## 环境搭建

```bash
# 安装 verl
pip install verl

# 或从源码安装（推荐，可获取最新算法）
git clone https://github.com/volcengine/verl.git
cd verl && pip install -e .

# 验证安装
python -c "import verl; print(verl.__version__)"
```

## 训练配置

verl 使用 Hydra 配置系统，GRPO 的核心配置：

```yaml
# config/grpo.yaml
algorithm:
  name: grpo
  group_size: 8            # 每个 prompt 采样的响应数
  clip_range: 0.2          # PPO-style clipping
  kl_coef: 0.0             # KL 惩罚系数（GRPO 通常设为 0）
  entropy_coef: 0.0        # 熵正则
  
actor:
  model_name: Qwen/Qwen2.5-7B-Instruct
  learning_rate: 1e-6
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  
rollout:
  engine: vllm             # vllm 或 sglang
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 1024
  gpu_memory_utilization: 0.4  # rollout 引擎占用的显存比例

reward:
  type: function            # function-based reward（非模型）
  reward_fn: "rewards.math_reward"
  
data:
  train_path: "data/math_train.parquet"
  max_prompt_length: 512
  
trainer:
  total_epochs: 3
  save_steps: 100
  micro_batch_size: 2
  gradient_accumulation: 4
```

## 奖励函数设计

```python
# rewards/math_reward.py
import re

def math_reward(completions: list[str], ground_truths: list[str]) -> list[float]:
    """数学任务的奖励函数"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # 提取 \boxed{} 中的答案
        pred = extract_boxed_answer(completion)
        
        if pred is None:
            rewards.append(-0.5)  # 格式不对扣分
        elif pred == gt:
            rewards.append(1.0)   # 答对满分
        else:
            rewards.append(-1.0)  # 答错扣分
    return rewards

def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r'\\boxed\{(.+?)\}', text)
    return match.group(1).strip() if match else None
```

**奖励设计的几个要点**：
1. 奖励要有区分度 — 不能所有答案都给同样的分
2. 格式奖励很重要 — 鼓励模型按照特定格式输出（如 `\boxed{}`）
3. 部分分可以给 — 思考过程正确但最终答案错误，可以给小奖励

## 运行训练

```bash
# 单机多卡
torchrun --nproc_per_node=8 \
    -m verl.trainer.main_ppo \
    --config-path config \
    --config-name grpo

# 多机训练
torchrun --nproc_per_node=8 --nnodes=2 \
    --node_rank=$RANK --master_addr=$MASTER \
    -m verl.trainer.main_ppo \
    --config-path config \
    --config-name grpo
```

## 资源分配策略

verl 的资源分配比较灵活，但需要合理规划：

```
8 卡 A100 80GB 的典型分配：
┌──────────────────────────────────────────┐
│ GPU 0-3: Actor (训练)                     │
│   - 模型参数 + 优化器状态                   │
│   - FSDP/DeepSpeed 切分                   │
├──────────────────────────────────────────┤
│ GPU 4-7: Rollout (推理)                   │
│   - vLLM 引擎                            │
│   - 生成响应                              │
└──────────────────────────────────────────┘

或者 colocate 模式（省卡但要注意显存）：
┌──────────────────────────────────────────┐
│ GPU 0-7: Actor + Rollout (共享)           │
│   - 训练时加载训练框架                      │
│   - 推理时切换到 vLLM                      │
│   - 需要仔细管理显存                        │
└──────────────────────────────────────────┘
```

## 常见问题

1. **OOM**：降低 `gpu_memory_utilization`，减小 `group_size`，开启 gradient checkpointing
2. **训练不稳定**：降低 lr（1e-7），增大 `clip_range`（0.3），加入 KL 惩罚
3. **奖励 hacking**：模型学会了投机取巧拿高分但输出质量差 → 加入格式/长度惩罚
4. **Rollout 太慢**：增加 rollout worker 数量，或换用 SGLang

## 相关

- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — 算法原理
- [[AI/LLM/RL/GRPO/GRPO-Unsloth实践|GRPO Unsloth 实践]] — 轻量级方案
- [[AI/LLM/RL/GRPO/GRPO-TRL实践|GRPO TRL 实践]] — TRL 框架方案
- [[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO verl 实践]] — GRPO 的改进版
- [[AI/LLM/Frameworks/verl/verl 训练参数|verl 训练参数]]
- [[AI/LLM/Frameworks/verl/硬件资源预估|硬件资源预估]]
