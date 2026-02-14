---
title: "Layer Normalization 详解"
date: 2026-02-14
tags:
  - architecture
  - normalization
  - interview
type: note
---

# Layer Normalization 详解

## 1. 为什么需要 Normalization

深度网络训练时，每层输入的分布会随参数更新而漂移（Internal Covariate Shift），导致：
- 梯度不稳定（vanishing / exploding）
- 需要极小的学习率，训练缓慢
- 深层网络难以收敛

Normalization 的核心思想：**将激活值重新拉回均值 0、方差 1 的标准分布**，再通过可学习参数 $\gamma, \beta$ 恢复表达能力。

---

## 2. Batch Norm vs Layer Norm

### 2.1 计算维度对比

| 特性 | Batch Norm (BN) | Layer Norm (LN) |
|------|-----------------|-----------------|
| **归一化维度** | 沿 batch 维度（对同一特征，跨所有样本） | 沿 feature 维度（对同一样本，跨所有特征） |
| **统计量依赖** | 依赖 mini-batch 的统计量 | 仅依赖当前样本自身 |
| **推理时行为** | 使用训练时的 running mean/var | 与训练时完全一致 |
| **对 batch size 敏感** | ✅ 小 batch 统计量不稳定 | ❌ 与 batch size 无关 |

### 2.2 为什么 LLM 用 LN 而不用 BN

**核心原因：序列长度可变 + 自回归生成**

1. **变长序列**：NLP 中每个 batch 内句子长度不同，padding 位置的特征没有意义，BN 对这些位置求统计量会引入噪声
2. **自回归推理**：生成时 batch size = 1，BN 的 batch 统计量退化为单样本，完全失效
3. **因果掩码**：Decoder 中每个 token 只能看到前面的 token，不同位置的 "有效 batch" 大小不同
4. **分布式训练**：BN 需要跨设备同步统计量（SyncBN），通信开销大；LN 完全 sample-local

> **面试一句话**：BN 假设同一特征在 batch 内同分布，这在变长序列和自回归场景下不成立；LN 对单个样本的所有特征做归一化，天然适配 NLP。

---

## 3. Pre-Norm vs Post-Norm

### 3.1 结构对比

```
Post-Norm (原始 Transformer):          Pre-Norm (GPT-2+):
x → Attention → Add(x) → LN           x → LN → Attention → Add(x)
  → FFN      → Add(x) → LN              → LN → FFN      → Add(x)
```

即：
- **Post-Norm**：$x_{l+1} = \text{LN}(x_l + F(x_l))$ — 先残差再归一化
- **Pre-Norm**：$x_{l+1} = x_l + F(\text{LN}(x_l))$ — 先归一化再子层

### 3.2 为什么现代 LLM 都用 Pre-Norm

| 维度 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| **梯度流** | 梯度需穿过 LN，深层梯度衰减 | 残差路径直通，梯度无障碍回传 |
| **训练稳定性** | 深层（>24L）需要 warmup，否则易发散 | 无需 warmup 也能稳定训练 |
| **最终性能** | 收敛后理论上限略高 | 略低于 Post-Norm（实践中差距很小） |
| **工程可行性** | 超深网络难以训练 | 可轻松扩展到 100+ 层 |

**关键洞察**：Pre-Norm 中残差连接形成了一条 "梯度高速公路"——输入可以不经任何非线性变换直接加到输出上，保证梯度在反向传播时不被衰减。

> **面试要点**：Post-Norm 理论性能天花板更高（LN 在残差之后对合并信号做归一化，保留了更多表达），但 Pre-Norm 训练更稳定，工程上更实用。现代 LLM 优先选择 **能训起来** 的方案。

---

## 4. RMSNorm（LLaMA 的选择）

### 4.1 原理

标准 Layer Norm：
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

RMSNorm 去掉了 **均值中心化**（re-centering）和 **偏置 $\beta$**：
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

其中 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$ 即 Root Mean Square。

### 4.2 为什么比 LayerNorm 快

1. **省掉均值计算**：不需要先求 $\mu$，再求 $\sigma^2$（两次 reduce），RMSNorm 只需一次 reduce（求平方均值）
2. **省掉偏置参数**：参数量减半（只有 $\gamma$，没有 $\beta$）
3. **计算量减少 ~15-30%**：在大规模模型中，每层的 Norm 被调用多次（Attention 前、FFN 前），节省累积可观
4. **内存带宽友好**：更少的内存读写，对 GPU memory-bound 操作有利

### 4.3 为什么有效

论文（Zhang & Sennrich, 2019）的核心论点：**LayerNorm 的成功主要归功于 re-scaling（缩放不变性），而非 re-centering（均值偏移）**。RMSNorm 保留了前者，去掉了后者，实验表明性能几乎无损。

---

## 5. DeepNorm（用于极深网络的变体）

### 5.1 背景

微软在训练 1000 层 Transformer 时提出（论文：DeepNet, 2022），解决超深 Post-Norm 网络的训练稳定性问题。

### 5.2 核心思想

$$\text{DeepNorm}(x) = \text{LayerNorm}(\alpha \cdot x + F(x))$$

两个关键改动：
1. **残差缩放**：给残差连接乘以系数 $\alpha > 1$，增强恒等映射的权重
2. **参数初始化缩放**：子层参数用 $\beta < 1$ 缩放初始化，抑制子层输出的方差

$\alpha$ 和 $\beta$ 的值由网络深度 $N$ 决定：
- Encoder：$\alpha = (2N)^{1/4}$，$\beta = (8N)^{-1/4}$
- Decoder：$\alpha = (2N)^{1/4}$，$\beta = (8N)^{-1/4}$

### 5.3 效果

- 成功训练 1000 层 Transformer（此前 Post-Norm 极限约 18-24 层）
- 保留了 Post-Norm 的性能优势
- 在极深网络场景下优于 Pre-Norm

---

## 6. 主流模型的 Norm 选择对比

| 模型 | Norm 类型 | Norm 位置 | 备注 |
|------|-----------|-----------|------|
| **GPT-2** | LayerNorm | Pre-Norm | 最早采用 Pre-Norm 的主流模型之一 |
| **GPT-3** | LayerNorm | Pre-Norm | 沿用 GPT-2 架构 |
| **LLaMA / LLaMA 2/3** | RMSNorm | Pre-Norm | 追求训练效率，去掉均值中心化 |
| **Qwen / Qwen2** | RMSNorm | Pre-Norm | 与 LLaMA 架构对齐 |
| **DeepSeek-V2/V3** | RMSNorm | Pre-Norm | MLA 架构，Norm 策略与 LLaMA 一致 |
| **Gemma / Gemma 2** | RMSNorm | Pre-Norm + Post-Norm | Gemma 2 引入 Pre-Post Norm（两处都加） |
| **BERT** | LayerNorm | Post-Norm | 原始 Transformer Encoder 架构 |
| **T5** | RMSNorm | Pre-Norm | 较早采用 RMSNorm 的模型 |

**趋势**：2023 年后的主流 LLM 几乎全部采用 **RMSNorm + Pre-Norm** 组合。

---

## 7. 补充：其他 Norm 变体

| 变体 | 核心思想 | 代表模型 |
|------|----------|----------|
| **Group Norm** | 将 channels 分组做 Norm | CV 领域（ResNeXt） |
| **Instance Norm** | 对单样本单通道做 Norm | 风格迁移 |
| **QK-Norm** | 对 Attention 的 Q、K 做 Norm | Gemma 2、某些 ViT |
| **Sandwich Norm** | 子层前后各加一次 Norm | CogView |

---

## 8. 面试常见问题及回答要点

### Q1: LayerNorm 和 BatchNorm 的核心区别？
> **答**：归一化维度不同。BN 沿 batch 维度对同一特征归一化，依赖 batch 统计量；LN 沿 feature 维度对同一样本归一化，与 batch 无关。NLP 用 LN 是因为变长序列和自回归推理使得 BN 的 batch 统计量不可靠。

### Q2: Pre-Norm 为什么训练更稳定？
> **答**：Pre-Norm 中残差路径不经过任何非线性变换（$x_{l+1} = x_l + F(\text{LN}(x_l))$），形成梯度直通通道，反向传播时梯度不被 LN 的 Jacobian 衰减。Post-Norm 的梯度必须穿过 LN 层，深层时会产生梯度消失。

### Q3: RMSNorm 为什么能替代 LayerNorm？
> **答**：实验表明 LayerNorm 的效果主要来自 re-scaling（方差归一化），而非 re-centering（减均值）。RMSNorm 只做 re-scaling，省去均值计算，参数量减半，速度提升 15-30%，性能几乎无损。

### Q4: 为什么不所有模型都用 DeepNorm？
> **答**：DeepNorm 是为超深网络（100+ 层）设计的，通过残差缩放 + 初始化缩放使 Post-Norm 在极深架构下可训练。但当前主流 LLM 深度通常在 32-80 层，Pre-Norm + RMSNorm 已经足够稳定，无需 DeepNorm 的额外复杂度。

### Q5: Gemma 2 的 Pre-Post Norm 是什么？
> **答**：Gemma 2 在子层前后都加了 RMSNorm（Pre-Norm 保证输入稳定，Post-Norm 保证输出稳定），结合了两种策略的优点。这是一种折中方案，增加了少量计算但提高了训练稳定性。

### Q6: Norm 层的参数量是多少？对总参数量影响大吗？
> **答**：LayerNorm 有 $2d$ 参数（$\gamma$ 和 $\beta$），RMSNorm 有 $d$ 参数（只有 $\gamma$）。以 LLaMA-7B（$d=4096$，32 层，每层 2 个 Norm）为例，RMSNorm 参数约 $32 \times 2 \times 4096 = 262K$，占总参数量 7B 的 0.004%，几乎可忽略。Norm 的开销主要在计算（每次前向都要做 reduce），不在参数。
