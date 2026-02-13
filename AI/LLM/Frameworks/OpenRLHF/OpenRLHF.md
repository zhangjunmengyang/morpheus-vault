---
title: "OpenRLHF"
type: concept
domain: ai/llm/frameworks/openrlhf
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/openrlhf
  - type/concept
---
# OpenRLHF

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 是一个开源的 RLHF（Reinforcement Learning from Human Feedback）训练框架，基于 Ray 和 vLLM 构建，支持 PPO、DPO、GRPO 等多种对齐算法。

## 架构设计

OpenRLHF 的核心设计理念是**用 Ray 做分布式调度，把 RLHF 的四个模型（Actor、Critic、Reward、Reference）拆分到不同的 GPU 组上**。

```
┌─────────────────────────────────────────────┐
│                 Ray Cluster                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Actor   │  │  Critic  │  │  Reward  │  │
│  │ (GPU 0-3)│  │ (GPU 4-5)│  │ (GPU 6-7)│  │
│  └──────────┘  └──────────┘  └──────────┘  │
│  ┌──────────┐  ┌──────────┐               │
│  │ Reference│  │  vLLM    │               │
│  │ (GPU 6-7)│  │ Generate │               │
│  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────┘
```

关键设计：

1. **模型分离** —— 4 个模型可以用不同的并行策略和 GPU 数量
2. **vLLM 加速生成** —— 用 vLLM 做 rollout generation，比原生 HF generate 快几倍
3. **Ray 调度** —— 自动处理模型间的参数同步和数据传输

## 支持的算法

| 算法 | 类型 | 是否需要 Reward Model |
|------|------|---------------------|
| PPO | On-policy RL | 是 |
| GRPO | On-policy RL | 否（rule-based reward） |
| DPO | Offline | 否 |
| KTO | Offline | 否 |
| RAFT | Rejection Sampling | 是 |
| Iterative DPO | Semi-online | 是 |

## 基本使用

### 安装

```bash
pip install openrlhf
# 或者从源码
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF && pip install -e .
```

### PPO 训练示例

```bash
ray start --head --num-gpus 8

openrlhf.cli.train_ppo \
    --pretrain meta-llama/Llama-3.1-8B-Instruct \
    --reward_pretrain reward_model_path \
    --save_path output/ppo_model \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 128 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 5e-6 \
    --init_kl_coef 0.01 \
    --vllm_num_engines 2 \
    --actor_num_gpus_per_node 4 \
    --critic_num_gpus_per_node 2 \
    --reward_num_gpus_per_node 2 \
    --ref_num_gpus_per_node 2
```

### GRPO 训练

GRPO 不需要单独的 reward model，用 rule-based reward 即可：

```python
# 自定义 reward function
def math_reward(prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        answer = extract_answer(response)
        ground_truth = extract_ground_truth(prompt)
        if answer == ground_truth:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
```

## 与 verl 的比较

| 维度 | OpenRLHF | verl |
|------|----------|------|
| 调度 | Ray | Ray / 自研 |
| 生成加速 | vLLM | vLLM / SGLang |
| 架构 | 模型分离 | HybridFlow（灵活切换） |
| 算法支持 | 多 | 多，且自定义更方便 |
| 文档 | 一般 | 较好 |
| 社区活跃度 | 高 | 高 |

## 踩坑记录

1. **Ray 版本兼容性** —— OpenRLHF 对 Ray 版本敏感，建议用它指定的版本
2. **显存 OOM** —— 4 个模型同时在 GPU 上，显存规划要仔细，建议先用小 batch 测试
3. **vLLM 和训练的 GPU 冲突** —— vLLM 用的 GPU 不能和 Actor 训练用同一组，否则会 OOM
4. **Reward Model 质量** —— PPO 的效果上限取决于 Reward Model，垃圾进垃圾出

## 我的看法

OpenRLHF 是目前开源 RLHF 框架中工程完成度较高的一个。它的模型分离架构思路清晰，适合有多卡集群的团队。但如果只有单机 8 卡，资源调度会比较紧张 —— 这种情况下 verl 的 HybridFlow 架构可能更合适，因为它可以让同一组 GPU 在不同阶段扮演不同角色。

## 相关

- [[AI/LLM/Frameworks/verl/verl 训练参数|verl 训练参数]]
- [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]]
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO-verl实践]]
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/LLM/Infra/Ray|Ray]]
