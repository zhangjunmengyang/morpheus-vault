---
brief: "ViT（Vision Transformer，Dosovitskiy 2021）——将图像分割为 patch 序列用 Transformer 处理；打破 CNN 在视觉的垄断；DeiT/BEiT/MAE 的基础架构；现代 VLM（CLIP/LLaVA/InternVL）的视觉 encoder 标配。"
title: "Vit"
type: paper
domain: ai/cv
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/cv
  - type/paper
---
# ViT (Vision Transformer)

ViT（An Image is Worth 16x16 Words）是 Google 于 2020 年提出的将 Transformer 直接应用于图像分类的模型。它的核心信息是：**在足够大的数据集上，纯 Transformer 可以完全替代 CNN，不需要任何卷积操作**。这篇论文是 CV 领域从 CNN 到 Transformer 范式转变的起点。

## 核心思想

ViT 的设计思路极其简洁：把图像当作一个"句子"来处理。

1. **图像分块（Patch Embedding）**：将图像切成固定大小的 patch（通常 16×16），每个 patch 展平后通过线性投影映射到 embedding 空间

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{\text{pos}}$$

其中 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是 patch embedding 矩阵，$N = HW/P^2$ 是 patch 数量。

2. **加上 [CLS] token**：和 BERT 一样，用一个可学习的特殊 token 做分类
3. **加上位置编码**：可学习的 1D 位置编码
4. **标准 Transformer Encoder**：多层 Multi-Head Self-Attention + FFN
5. **分类头**：[CLS] token 的输出接 MLP Head

就这么简单。没有任何 CV 领域的归纳偏置——没有卷积、没有池化、没有局部连接。

## 关键发现

### 数据规模决定一切

这是 ViT 最重要的结论：

- **在 ImageNet-1K（1.3M 图像）上训练**：ViT 不如同等规模的 CNN（如 ResNet）
- **在 ImageNet-21K（14M 图像）上训练**：ViT 开始追平 CNN
- **在 JFT-300M（300M 图像）上训练**：ViT 全面超越 CNN

原因：CNN 的卷积操作包含了**局部性**和**平移不变性**两个归纳偏置，在小数据上是有利的。但 Transformer 没有这些偏置，需要从数据中学习这些结构。数据足够多时，从数据中学到的结构比人为设计的归纳偏置更好。

### 模型规模

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

### 位置编码的发现

实验发现 1D 位置编码和 2D 位置编码效果差不多。这说明模型能够自己从数据中学习到 2D 空间结构——又一个支持"数据驱动 > 归纳偏置"的证据。

## 后续发展

### DeiT（Data-efficient Image Transformers）

Facebook 的工作，证明通过更好的训练策略（知识蒸馏、强数据增强、正则化），ViT 可以在 **ImageNet-1K 上就获得优秀表现**，不需要 JFT-300M 这样的超大数据集。

关键技巧：
- 使用 RegNet 作为 teacher 做蒸馏
- RandAugment + Mixup + CutMix + Random Erasing
- Repeated Augmentation

### Swin Transformer

微软提出的层次化 ViT，重新引入了 CNN 的一些设计哲学：
- **窗口注意力**：在局部窗口内计算注意力，降低复杂度从 $O(n^2)$ 到 $O(n)$
- **移位窗口（Shifted Window）**：让相邻窗口之间的信息交流
- **层次化特征图**：类似 ResNet 的多尺度特征金字塔

Swin Transformer 在 CV 各种任务上全面刷新 SOTA，成为了 CV 领域的"新 ResNet"。

### BEiT 系列

将 BERT 的 MLM 思想引入 CV，用视觉 token 做 masked image modeling。BEiT v2/v3 进一步统一了多模态预训练。

## ViT 的历史意义

ViT 的影响远超图像分类本身：

1. **统一了 CV 和 NLP 的架构**：同一个 Transformer 可以处理文本和图像，这直接催生了多模态模型（CLIP、BLIP-2、LLaVA）
2. **证明了 Scaling Law 在 CV 中同样成立**：模型越大、数据越多、效果越好
3. **简化了模型设计**：不需要再为每种任务设计专门的架构，一个 Transformer 搞定

但也要承认，ViT 的成功很大程度上依赖于大规模数据和算力。在数据受限的场景下，CNN 的归纳偏置仍然有价值。

## 相关

- [[AI/LLM/Architecture/BERT|BERT]]
- [[MAE]]
- [[AI/MLLM/CLIP|CLIP]]
- [[AI/Foundations/DL-Basics/深度学习|深度学习]]
- [[ControlNet]]
- [[AI/Foundations/DL-Basics/Transformer 通识|Transformer 通识]]
- [[AI/LLM/Architecture/Attention 变体综述|Attention 详解]]
