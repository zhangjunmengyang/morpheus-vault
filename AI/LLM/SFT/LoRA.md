---
title: "Lora"
type: paper
domain: ai/llm/sft
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/sft
  - type/paper
---
# LoRA

LoRA（Low-Rank Adaptation of Large Language Models）是微软在 2021 年提出的参数高效微调（PEFT）方法。它的核心思想极其简洁：**大模型的权重更新是低秩的**，因此可以用两个小矩阵的乘积来近似全量微调的效果。这个看似简单的想法，彻底改变了大模型微调的经济学。

## 动机：为什么需要 LoRA

全量微调（Full Fine-Tuning）一个 7B 模型需要：
- 存储：7B × 4 bytes（fp32）= 28GB 仅模型权重
- 训练：加上优化器状态（Adam 需要 2 倍）、梯度，至少需要 ~112GB 显存
- 部署：每个任务一个完整的模型副本

这在实际工程中几乎不可行，尤其是当你需要为多个任务分别微调时。

在 LoRA 之前，已有一些参数高效微调方法：
- **Adapter**：在每个 Transformer 层中插入小的瓶颈模块，但会引入推理延迟
- **Prefix Tuning**：在输入前加可学习的 virtual tokens，但压缩了有效上下文长度
- **BitFit**：只微调 bias 参数，太受限

LoRA 的优势在于：**没有额外推理延迟**，因为训练后可以将 LoRA 权重合并回原始权重。

## 核心原理

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将权重更新分解为两个低秩矩阵的乘积：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

训练时冻结 $W_0$，只训练 $A$ 和 $B$。前向传播变为：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

关键设计：
- **$A$ 用 Kaiming 初始化，$B$ 初始化为零**：确保训练开始时 $\Delta W = 0$，不改变原始模型行为
- **缩放因子 $\alpha / r$**：控制 LoRA 权重的影响程度。$\alpha$ 是一个常数，通常设为 $r$ 或 $2r$

### 为什么低秩假设成立？

论文的关键发现：预训练模型在适配到下游任务时，权重变化矩阵 $\Delta W$ 的**本征维度（intrinsic dimension）** 很低。直觉上，预训练已经学到了丰富的通用表示，微调只需要在一个很小的子空间里做调整。

实验表明，即使 $r = 4$ 或 $r = 8$，LoRA 就能达到接近全量微调的效果。这意味着对于一个 $4096 \times 4096$ 的权重矩阵（约 16M 参数），LoRA 只需要 $4096 \times 8 \times 2 = 65K$ 个可训练参数——**压缩比超过 200 倍**。

## 实践要点

### 应用到哪些层？

原论文主要在 attention 的 $W_q$ 和 $W_v$ 上应用 LoRA。后续实践发现：

```python
# 常见的 target_modules 配置
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP (SwiGLU)
]
```

在 MLP 层也加 LoRA 通常能进一步提升效果，尤其是对于需要注入新知识的任务。

### rank 和 alpha 怎么选？

- **r = 8~64** 是最常用的范围
- 简单任务（风格迁移、格式调整）：$r = 8$ 就够了
- 复杂任务（领域知识注入）：$r = 32$ 或 $r = 64$
- **$\alpha$ 通常设为 $r$ 或 $2r$**

### 推理时的合并

训练完成后，将 LoRA 权重合并回原始权重：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

合并后的模型和原始模型结构完全一致，没有任何额外的推理开销。

## LoRA 的变体

### QLoRA

QLoRA（2023，华盛顿大学）是 LoRA 最重要的实践改进：
- 将基座模型量化到 **4-bit**（NF4 量化）
- 在 4-bit 模型上训练 LoRA adapter（fp16/bf16）
- 引入双重量化（Double Quantization）进一步节省内存
- 引入分页优化器（Paged Optimizer）处理 GPU 内存溢出

QLoRA 使得在**单张 24GB 消费级 GPU 上微调 65B 模型**成为可能。

### DoRA（Weight-Decomposed Low-Rank Adaptation）

DoRA 将权重分解为**幅度（magnitude）和方向（direction）** 两个分量，然后只用 LoRA 适配方向分量。这个简单的修改在多数基准上都能带来 1-2% 的提升。

### LoRA+

发现 $A$ 和 $B$ 使用不同学习率效果更好（$B$ 的学习率应该是 $A$ 的 2-8 倍）。

### rsLoRA

提出将缩放因子从 $\alpha/r$ 改为 $\alpha/\sqrt{r}$，在更大的 rank 上表现更稳定。

## 工程实践建议

1. **先用 LoRA 验证可行性**，效果不够再考虑全量微调
2. **数据质量 >> rank 大小**：200 条高质量数据 + LoRA 通常优于 10000 条低质量数据 + 全量微调
3. **多 LoRA 服务**：可以用一个基座模型同时挂载多个 LoRA adapter（vLLM 和 Unsloth 都支持），实现多任务服务
4. **LoRA 的局限**：对于需要大幅修改模型行为（如学习新语言）的任务，LoRA 可能不够，需要 CPT + SFT

## 相关

- [[Unsloth 概述]]
- [[LLaMA]]
- [[SFT-TRL实践]]
- [[分布式训练]]
- [[量化]]
