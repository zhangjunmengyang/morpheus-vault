---
title: "RL with Lora"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# RL with LoRA

> 在 verl 中使用 LoRA 做 RL 训练，用更少的显存训更大的模型。

文档：https://verl.readthedocs.io/en/latest/advance/ppo_lora.html

## 为什么 RL + LoRA

RL 训练的显存需求比 SFT 大得多：

```
SFT:  model + optimizer + gradients + activations
RL:   actor + critic + ref_model + reward_model + KV_cache + ...

# 全量参数 RL 训练一个 7B 模型：
# Actor (bf16): 14GB
# Actor optimizer (fp32 states): 56GB
# Gradients: 14GB
# Activations + KV cache: 20-40GB
# 还没算 Critic, Ref, Reward...
# 总计: 轻松 150GB+, 需要多卡
```

LoRA 的作用：**只训练低秩矩阵，冻结主体权重**，大幅降低 optimizer states 和梯度的显存。

```python
# LoRA 参数量
# rank=16, target_modules=["q_proj", "v_proj"]
# 每个 module: 2 * hidden_dim * rank 参数
# 7B 模型约有 32 层 × 2 modules = 64 个 LoRA 适配器
# 总可训练参数: ~20M (vs 7B 全量)
# 显存节省: optimizer states 从 56GB → ~160MB
```

## verl 中的 LoRA 配置

```yaml
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
  actor:
    # LoRA 配置
    use_lora: true
    lora_config:
      r: 16                    # rank
      lora_alpha: 32           # alpha/r = scaling factor
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
      lora_dropout: 0.05
    
    # LoRA 下可以用更大的学习率
    lr: 1e-4  # 全量用 1e-6, LoRA 可以用 1e-4 ~ 5e-5
    
    # 其他训练参数不变
    bf16: true
    gradient_checkpointing: true
```

## LoRA RL 的特殊考量

### 1. Reference Model 的处理

RL 训练需要 reference model 计算 KL divergence。用 LoRA 时：

```python
# 方案 1: Reference = 基座模型（不加 LoRA）
# KL(π_θ || π_ref) 中 π_ref 就是冻结的基座
# 优势: 不需要额外存储 ref model
# 因为 base weights 本身就在那里，去掉 LoRA adapter 即可

# 方案 2: Reference = SFT 后的 LoRA
# 保存一份 SFT LoRA weights 作为 reference
# 训练时同时加载两套 LoRA: ref_lora (frozen) + train_lora (可训练)
```

verl 默认使用方案 1 — 基座模型即 reference。

### 2. Rollout 的兼容性

```python
# LoRA 模型做 rollout 时需要 merge weights
# 因为推理引擎（vLLM/SGLang）通常不原生支持 LoRA inference
# 流程：
# 1. merge: base_weights + lora_weights → full_weights
# 2. 用 full_weights 做 rollout
# 3. 训练时再切回 LoRA 模式

# 这个 merge/unmerge 过程有开销，但只在每个 RL step 开始时做一次
```

### 3. 效果对比

```
任务: GSM8K 数学推理, Qwen2.5-7B-Instruct

全量 GRPO:  acc 78.2%  (8xA100, 需要 TP=4)
LoRA GRPO:  acc 75.8%  (4xA100, TP=2 即可)
LoRA GRPO (r=64): acc 77.1%

# 结论：
# 1. LoRA 比全量差 2-3 个点，可以接受
# 2. 增大 rank 可以缩小差距
# 3. 如果 GPU 资源有限，LoRA 是非常好的折衷
```

### 4. QLoRA (量化 + LoRA)

```yaml
# 进一步省显存：基座用 4-bit 量化
actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "4bit"  # NF4 量化
  actor:
    use_lora: true
    lora_config:
      r: 16
      
# 显存估算（7B 模型）：
# 全量 bf16: 14GB (weights) + 56GB (optimizer)
# LoRA bf16: 14GB (weights) + 0.16GB (optimizer)
# QLoRA 4bit: 3.5GB (weights) + 0.16GB (optimizer)
# → 单张 24GB 消费级卡就能跑 RL 训练!
```

## 实践建议

1. **Rank 选择**：数学/推理任务用 r=16-32 足够；通用对话/创意写作建议 r=64+
2. **Target modules**：至少包含 attention（q/k/v/o），加上 MLP（gate/up/down）效果更好
3. **学习率**：LoRA 学习率比全量大 10-100 倍是正常的
4. **不建议对 Critic 也用 LoRA**：Critic 需要精确的 value 估计，LoRA 的表达能力可能不够
5. **保存和加载**：只保存 LoRA adapter weights，体积很小

## 相关

- [[AI/LLM/Frameworks/verl/配置文件|配置文件]] — 完整配置参考
- [[AI/LLM/Frameworks/verl/性能调优|性能调优]] — 显存和速度优化
- [[AI/LLM/Frameworks/verl/硬件资源预估|硬件资源预估]] — 资源需求估算
- [[AI/LLM/Frameworks/Unsloth/训练示例概述|Unsloth 训练示例]] — Unsloth 的 LoRA 训练
