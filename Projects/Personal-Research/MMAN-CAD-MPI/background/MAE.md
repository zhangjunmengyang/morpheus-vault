---
brief: "MAE（Masked Autoencoder，He et al.，2022）——75% 图像遮盖+重建的自监督视觉预训练；比 MoCo/DINO 更简单有效；ViT 的最佳预训练方式；是现代 VLM 视觉 encoder 预训练的核心范式之一。"
title: "MAE"
type: paper
domain: ai/cv
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/cv
  - type/paper
---
# MAE (Masked Autoencoders)

MAE（Masked Autoencoders Are Scalable Self-Supervised Learners）是何恺明在 Meta 时期的代表作（2021），将 NLP 中成功的 masked prediction 思想优雅地引入 CV 领域。它证明了一个直觉上简单但效果惊人的方法：**随机遮盖图像的大部分 patch，然后训练模型重建被遮盖的内容**。

## 核心设计

### 高遮盖率

MAE 最关键的设计决策是使用 **75% 的遮盖率**——远高于 BERT 在 NLP 中使用的 15%。为什么？

原因是图像和语言的信息密度不同。语言是高度语义化的离散符号，每个 token 都包含大量信息，遮盖 15% 就足够构成一个有意义的预测任务。而图像具有高度的空间冗余——相邻 patch 之间有很强的相关性，如果只遮盖 15%，模型可以轻松通过插值解决，不需要学习高层语义。

75% 的遮盖率迫使模型必须理解图像的全局结构才能完成重建。

### 非对称 Encoder-Decoder

```
输入: 224×224 图像 → 196 个 patch（16×16）
随机遮盖 75% → 49 个可见 patch

Encoder (ViT-Large):  只处理 49 个可见 patch → 高效！
Decoder (轻量):       可见 patch embedding + mask tokens → 重建 196 个 patch
```

这个非对称设计是工程上的巧妙之处：

1. **Encoder 不处理 mask tokens**：计算量降低到全量的 ~25%，训练速度大幅提升
2. **Decoder 很轻量**：只有 8 层，隐藏维度 512（Encoder 是 24 层，1024 维）
3. **预训练完成后 Decoder 丢弃**：只用 Encoder 做下游任务

### 重建目标

MAE 的重建目标是**像素级 MSE loss**：

$$L = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} ||x_i - \hat{x}_i||^2$$

只在被遮盖的 patch 上计算损失（和 BERT 的 MLM 一致）。

论文也尝试了将 patch 归一化后再计算 MSE（per-patch normalization），效果更好。有意思的是，重建的并不需要很精确——模糊的重建就已经说明模型学到了语义。

## 实验结果

### ImageNet 线性探测和微调

| 方法 | 线性探测 | 微调 |
|------|----------|------|
| MoCo v3 (ViT-B) | 76.7% | 83.2% |
| BEiT (ViT-B) | 56.7% | 83.2% |
| MAE (ViT-B) | 67.8% | 83.6% |
| MAE (ViT-L) | 75.8% | **85.9%** |
| MAE (ViT-H) | 77.2% | **86.9%** |

关键发现：
- MAE 的微调效果极其出色，ViT-H 达到 86.9% 的 ImageNet 准确率
- 但线性探测表现不如对比学习方法（MoCo v3），说明 MAE 学到的表示可能需要非线性变换才能充分利用
- **MAE 的 scaling 表现出色**：模型越大效果越好，没有饱和迹象

### 训练效率

MAE 只处理 25% 的 patch，加上 decoder 很轻量，**训练速度是 ViT 常规训练的 3-4 倍**。这使得在大规模 ViT 上做自监督预训练变得经济可行。

## 为什么 MAE 重要

### 1. 揭示了 CV 自监督的正确方向

在 MAE 之前，CV 的自监督学习被对比学习（SimCLR、MoCo、BYOL）主导。这些方法需要精心设计数据增强、负样本策略、动量编码器等。MAE 表明，**简单的 masked reconstruction 就足够了**，甚至更好。

### 2. 证明了大 ViT 的可训练性

在 MAE 之前，人们对 ViT-Large/Huge 在没有超大标注数据集的情况下能否有效训练持怀疑态度。MAE 证明了通过自监督预训练，可以充分释放大 ViT 的潜力。

### 3. 连接了 CV 和 NLP 的预训练范式

MAE 本质上就是 CV 版的 BERT。这种跨领域的方法论统一，为后续多模态模型的发展铺平了道路。

## 与其他方法的对比

| 方法 | 思路 | 缺点 |
|------|------|------|
| MAE | Masked Image Modeling | 线性探测较弱 |
| MoCo/SimCLR | 对比学习 | 需要负样本/大batch |
| DINO/DINOv2 | 自蒸馏 | 训练trick多 |
| BEiT | Masked visual tokens | 需要 tokenizer |

DINOv2 后来被证明是更通用的视觉基础模型，但 MAE 的设计简洁性和训练效率至今无出其右。

## 相关

- [[ViT]]
- [[AI/LLM/Architecture/BERT|BERT]]
- [[AI/MLLM/CLIP|CLIP]]
- [[AI/Foundations/DL-Basics/深度学习|深度学习]]
