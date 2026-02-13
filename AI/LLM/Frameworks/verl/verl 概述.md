---
title: "verl"
type: reference
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/reference
---
# verl

## 概述

verl（Volcano Engine Reinforcement Learning for LLMs）是字节跳动开源的 LLM RL 训练框架，专门为**大规模 RLHF/GRPO 训练**设计。核心优势是用 Ray 做资源编排，实现训练（FSDP/Megatron）和推理（vLLM）的灵活调度。

项目地址：`volcengine/verl`

TRL 适合快速实验，verl 适合正经上规模。如果你要在 70B 模型上跑 GRPO，verl 是目前的首选之一。

## 核心设计：HybridFlow

verl 的灵魂是 **HybridFlow** 架构——把 RL 训练中的不同阶段（generation、training、reward computation）拆分成独立的 worker group，用 Ray 调度。

```
                    Ray Cluster
                   ┌─────────────────────────────────┐
 Rollout Worker    │  vLLM Engines (GPU Group A)      │
 (Generation)      │  负责采样 response                │
                   ├─────────────────────────────────┤
 Training Worker   │  FSDP/Megatron (GPU Group B)     │
 (Policy Update)   │  负责梯度更新                     │
                   ├─────────────────────────────────┤
 Reward Worker     │  Reward Model / Rule-based       │
 (Scoring)         │  负责打分                         │
                   └─────────────────────────────────┘
```

关键设计选择：

1. **同一组 GPU 复用**：Generation 和 Training 可以用同一组 GPU，通过 weight offload 切换。不需要两倍 GPU。
2. **异步流水线**：当 Batch N 在 Training 时，Batch N+1 可以同时做 Generation（Off-Policy 模式）。
3. **灵活的资源分配**：可以根据 bottleneck 调整 generation vs training 的 GPU 比例。

### vs TRL 的资源模型

```
TRL:  [一个进程] → 串行执行 generate → compute_reward → train
                    GPU 在 generate 阶段跑推理
                    GPU 在 train 阶段跑训练
                    资源利用率 ~50%

verl: [Ray 调度] → generation workers / training workers / reward workers
                   可以重叠执行
                   资源利用率 ~80%+
```

## 支持的算法

verl 目前支持的 RL 算法：

- **GRPO**：主推算法，DeepSeek-R1 同款
- **PPO**：经典 RLHF
- **DAPO**：GRPO 的改进版（去 KL、动态采样等）
- **DPO**：直接偏好优化
- **RLOO**：REINFORCE Leave-One-Out
- 以及社区贡献的多种变体（GPG、OPO、SPPO 等）

## 训练流程

### 配置方式

verl 用 Hydra 管理配置：

```yaml
# config.yaml
model:
  name_or_path: Qwen/Qwen2.5-7B
  
algorithm:
  name: grpo
  grpo:
    num_generations: 16
    beta: 0.01
    clip_ratio: 0.2

training:
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-6
  max_grad_norm: 1.0
  
generation:
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.9

reward:
  type: function  # or model
  functions:
    - accuracy
    - format
```

### 启动训练

```bash
# 单节点多 GPU
python -m verl.trainer.main_ppo \
    --config-path config/ \
    --config-name grpo_qwen7b

# 多节点（通过 Ray）
ray start --head  # 主节点
ray start --address=<head_ip>:6379  # 工作节点
python -m verl.trainer.main_ppo --config-name grpo_qwen7b
```

### Reward Function 定义

```python
# verl 的 reward function 接口
def compute_reward(data_batch):
    """
    data_batch 包含:
    - prompts: list of str
    - responses: list of str
    - ground_truth: list of str (如果有)
    
    返回: list of float (reward scores)
    """
    rewards = []
    for response, answer in zip(data_batch["responses"], data_batch["ground_truth"]):
        extracted = extract_answer(response)
        reward = 1.0 if extracted == answer else 0.0
        rewards.append(reward)
    return rewards
```

## 关键特性

### 1. Weight Resharding

在 generation 和 training 之间切换时，模型权重需要在不同的并行策略之间转换：

```
Generation 阶段: Tensor Parallel (vLLM 风格)
Training 阶段:   FSDP (PyTorch 风格)

→ 需要 resharding: TP layout → FSDP layout
```

verl 实现了高效的 weight resharding，避免了完整的 weight gather + scatter。

### 2. 多种训练后端

```
Training:
  - FSDP (PyTorch 原生)
  - Megatron-LM (适合更大模型)

Generation:
  - vLLM
  - SGLang (实验性)
```

### 3. Sandbox Fusion

verl 集成了 sandbox 环境，支持代码执行类的 reward：

```python
# 代码类任务的 reward
# verl 可以在 sandbox 中执行生成的代码，用测试用例验证
reward = sandbox_execute(generated_code, test_cases)
```

### 4. 多轮交互 RL

支持 Agent 场景的多轮对话 RL 训练，模型与环境交互多轮后才计算 reward。

## 资源预估

| 模型规模 | GPU 配置 | Generation | Training |
|---------|---------|------------|----------|
| 7B | 8×A100 80G | 4 GPU (vLLM) | 4 GPU (FSDP) |
| 32B | 16×A100 80G | 8 GPU (vLLM TP=4) | 8 GPU (FSDP) |
| 70B | 32×A100 80G | 16 GPU (vLLM TP=8) | 16 GPU (FSDP) |

注意：同一组 GPU 可以在 generation 和 training 之间切换，所以实际需要的 GPU 可以更少。

## 与 TRL 的互补

两个框架不是竞争关系，更多是互补：

```
实验阶段 → TRL（快速迭代、Jupyter 友好）
    ↓ 验证可行
规模化 → verl（资源高效、多节点）
```

很多团队的工作流：先用 TRL 在 7B 模型上验证算法和数据，再用 verl scale 到 70B。

## 相关

- [[算法概述]] — verl 支持的算法详解
- [[HybridFlow]] — verl 的核心架构
- [[AI/LLM/Frameworks/TRL/TRL 概述|TRL 概述]] — 对比框架
- [[AI/LLM/Frameworks/OpenRLHF/OpenRLHF|OpenRLHF]] — 另一个大规模 RL 框架
- [[AI/LLM/RL/GRPO/GRPO-verl实践|GRPO-verl实践]] — verl 中 GRPO 实践
- [[AI/LLM/RL/PPO/PPO-verl实践|PPO-verl实践]] — verl 中 PPO 实践
- [[AI/LLM/RL/DAPO/DAPO-verl实践|DAPO-verl实践]] — verl 中 DAPO 实践
- [[AI/LLM/Infra/FSDP|FSDP]] — verl 的训练后端
- [[AI/LLM/Inference/vLLM|vLLM]] — verl 的推理后端
- [[AI/LLM/Infra/Ray|Ray]] — verl 的资源编排
- [[Reward Function]] — 自定义 reward 设计
- [[verl 训练参数]] — 详细参数配置
- [[性能调优]] — 性能优化技巧
- [[硬件资源预估]] — 资源规划
