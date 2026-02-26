---
brief: "verl 训练参数完整参考——覆盖 actor_rollout/critic/reward_model/ref_policy 各模块的关键超参；包含显存优化/并行策略/KL 系数/clip ratio 等生产调参经验，verl 工程上手必备。"
title: "verl 训练参数"
type: project
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/project
---
# verl 训练参数

> 材料：https://github.com/chunhuizhang/llm_rl/blob/main/tutorials/infra/verl/verl_config_perf.ipynb

## 参数全景

verl 的配置用 Hydra（YAML + CLI override），参数分为几大块：

```yaml
# 完整配置结构
actor_rollout_ref:    # Actor + Rollout + Reference 共享配置
  model: ...
  actor: ...
  rollout: ...
  ref: ...

critic:               # Critic 模型配置 (PPO 用)
  model: ...
  optim: ...

reward_model:         # Reward 模型配置
  model: ...

algorithm:            # 算法超参数
  ...

trainer:              # 训练器控制
  ...

data:                 # 数据配置
  ...
```

## 核心参数详解

### Actor 参数

```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 1e-6                    # 学习率，RL 阶段要比 SFT 小很多
      lr_warmup_steps_ratio: 0.05 # warmup 比例
      min_lr_ratio: 0.1           # cosine decay 最小 LR 比例
      weight_decay: 0.01
    
    ppo_mini_batch_size: 128      # PPO mini-batch
    ppo_micro_batch_size: 16      # gradient accumulation 的微 batch
    
    gradient_checkpointing: true  # 用计算换显存
    max_grad_norm: 1.0            # 梯度裁剪，RL 必备
    
    # FSDP / ZeRO 配置
    fsdp_config:
      sharding_strategy: "FULL_SHARD"  # ZeRO-3
      cpu_offload: false               # CPU offload
```

**LR 选择经验**：
```
SFT:  1e-5 ~ 5e-5
GRPO: 1e-6 ~ 5e-6  (SFT 的 1/10)
PPO:  5e-7 ~ 2e-6  (更保守)
```

RL 阶段 LR 太大 → 策略剧烈变化 → KL 散度爆炸 → reward hacking。

### Rollout 参数

```yaml
actor_rollout_ref:
  rollout:
    name: "sglang"                # 推理引擎: sglang / vllm
    
    temperature: 1.0              # 生成温度
    top_p: 0.9
    top_k: 50
    
    max_new_tokens: 2048          # 最大生成长度
    
    # GRPO 特有
    n: 8                          # group_size, 每个 prompt 生成几个 response
    
    # SGLang 引擎参数
    gpu_memory_utilization: 0.85  # GPU 显存占比
    tensor_parallel_size: 1       # TP 并行度
    
    # 性能参数
    max_num_batched_tokens: 8192
    max_num_seqs: 128
```

**group_size (n) 的影响**：
```python
# n=4:  variance 高，reward signal 弱，但省显存
# n=8:  标准配置，效果和效率的平衡点
# n=16: variance 低，但 rollout 显存翻倍
# n=32: 研究用，生产中太贵

# GRPO 的 advantage 计算:
# advantage_i = (reward_i - mean(rewards)) / std(rewards)
# n 越大，mean 和 std 估计越准，advantage 越稳定
```

### 算法参数

```yaml
algorithm:
  gamma: 1.0                    # discount factor（LLM RL 通常设 1.0）
  lam: 0.95                     # GAE lambda
  
  kl_ctrl:
    type: "fixed"               # KL penalty 类型: fixed / adaptive
    kl_coef: 0.001              # KL 系数
  
  # GRPO 特有
  adv_estimator: "grpo"         # advantage 估计器
  
  # PPO 特有
  clip_ratio: 0.2               # PPO clip range
  vf_coef: 0.5                  # value function loss 系数
  entropy_coef: 0.01            # entropy bonus
  
  # DAPO 变体
  clip_ratio_low: 0.2           # 负 advantage 的 clip
  clip_ratio_high: 0.28         # 正 advantage 的 clip
  token_level_loss: true        # token 级别 loss vs sequence 级别
```

**KL 系数调节**：
```
kl_coef = 0       → 无约束，容易 reward hack
kl_coef = 0.001   → 轻约束，常用起步值
kl_coef = 0.01    → 中等约束
kl_coef = 0.1     → 强约束，策略几乎不动
```

### Trainer 参数

```yaml
trainer:
  total_epochs: 3               # 总训练 epoch
  save_freq: 50                 # 每 N 步保存 checkpoint
  test_freq: 25                 # 每 N 步做验证
  
  project_name: "my_rl_exp"     # wandb 项目名
  experiment_name: "qwen7b_grpo"
  
  logger: ["wandb", "console"]
  
  total_training_steps: null    # 如果设置，覆盖 epoch 计算
  
  n_gpus_per_node: 8
  nnodes: 1
```

### Data 参数

```yaml
data:
  train_files: ["data/train.parquet"]
  val_files: ["data/val.parquet"]
  
  train_batch_size: 256         # rollout batch size
  val_batch_size: 128
  
  max_prompt_length: 1024       # prompt 截断长度
  max_response_length: 2048     # response 最大生成长度
  
  # 过滤
  filter_overlong_prompts: true
  
  # 打包
  packing: false                # RL 一般不开 packing
```

## 参数调优 Cheatsheet

```bash
# 场景 1: reward 不涨
→ 检查 reward function 是否有 bug (最常见原因)
→ 增大 group_size (n=8 → n=16)
→ 增大 LR (1e-6 → 5e-6)
→ 减小 kl_coef

# 场景 2: reward 涨了但 KL 爆炸
→ 增大 kl_coef
→ 减小 LR
→ 减小 clip_ratio

# 场景 3: OOM
→ 减小 train_batch_size
→ 减小 max_response_length
→ 开启 gradient_checkpointing
→ 增大 ppo_micro_batch_size 的 gradient accumulation

# 场景 4: 训练太慢
→ 增大 ppo_micro_batch_size
→ 用 sglang 替代 vllm
→ 开启 tensor_parallel
→ 减小 max_response_length
```

## CLI Override 示例

Hydra 支持在命令行覆盖任何参数：

```bash
python -m verl.trainer.main \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.rollout.n=16 \
  algorithm.kl_ctrl.kl_coef=0.005 \
  trainer.total_epochs=5 \
  data.train_batch_size=512
```

## 相关

- [[verl 概述|verl 概述]]
- [[配置文件|配置文件]]
- [[性能调优|性能调优]]
- [[硬件资源预估|硬件资源预估]]
- [[算法概述|算法概述]]
- [[grafana 看板|grafana 看板]]
