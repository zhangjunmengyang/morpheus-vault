---
brief: "CLIP（OpenAI，2021）——对比学习训练图文对齐的里程碑模型；Image Encoder + Text Encoder 对比预训练；Zero-shot 图像分类的首个成功实现；几乎所有现代 VLM 的视觉 encoder 基础，理解多模态对齐的起点。"
title: "CLIP"
date: 2021-01-05
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-21"
tags:
  - ai/mllm
  - type/paper
arxiv: "2103.00020"
rating: ★★★★★
status: active
---
# CLIP

CLIP（Contrastive Language-Image Pre-training）是 OpenAI 于 2021 年提出的视觉-语言对比学习模型。它的核心想法是：**用自然语言监督来学习视觉表示，而不是用传统的分类标签**。这个想法听起来简单，但效果极其强大——CLIP 在 zero-shot 设定下在 ImageNet 上达到了 ResNet-50 有监督训练的水平。

## 核心设计

### 对比学习框架

CLIP 包含两个编码器：
- **Image Encoder**：可以是 ResNet 或 ViT
- **Text Encoder**：标准 Transformer

训练目标是**对比损失**：将匹配的图文对拉近，不匹配的推远。

具体地说，对一个 batch 中的 $N$ 个图文对 $(I_i, T_i)$，计算 $N \times N$ 的相似度矩阵，然后用对称的 cross-entropy loss：

$$L = -\frac{1}{2N} \sum_{i=1}^{N} \left[\log \frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N} \exp(s_{i,j}/\tau)} + \log \frac{\exp(s_{i,i}/\tau)}{\sum_{j=1}^{N} \exp(s_{j,i}/\tau)}\right]$$

其中 $s_{i,j} = \text{sim}(I_i, T_j) = \frac{f_I(I_i) \cdot f_T(T_j)}{||f_I(I_i)|| \cdot ||f_T(T_j)||}$ 是 cosine similarity，$\tau$ 是可学习的温度参数。

### 训练数据

CLIP 的数据规模是其成功的关键：**4 亿个图文对**（WIT-400M），从互联网上收集。这个数据规模远超当时所有有标注的视觉数据集。

论文的一个重要洞察：与其费力标注数据，不如利用互联网上自然存在的图文配对关系。这种"弱监督"的数据量几乎是无限的。

### Zero-Shot 分类

CLIP 做 zero-shot 分类的方式极其优雅：

1. 把每个类别名转换成自然语言 prompt：`"a photo of a {class}"`
2. 用 Text Encoder 编码所有类别的 prompt
3. 用 Image Encoder 编码目标图像
4. 计算图像 embedding 和所有类别 embedding 的相似度
5. 取最高相似度的类别

```python
# 伪代码
image_features = image_encoder(images)         # [B, D]
text_features = text_encoder(class_prompts)    # [C, D]
similarities = image_features @ text_features.T  # [B, C]
predictions = similarities.argmax(dim=-1)
```

这意味着 CLIP 可以识别**任何可以用语言描述的概念**，不受限于预定义的类别集合。

### Prompt Engineering for CLIP

不同的 prompt 模板对 CLIP 的效果影响很大：

- `"a photo of a {class}"` 是基线
- `"a photo of a big {class}"` / `"a photo of a small {class}"` 可以指定大小
- 使用多个 prompt 模板取平均（prompt ensemble）可以提升 2-5%

## 为什么 CLIP 如此重要

### 1. 开创了视觉-语言对齐

CLIP 证明了一个统一的 embedding 空间可以同时容纳视觉和语言信息。这个思想直接影响了后续几乎所有多模态模型：

- **BLIP-2**：用 Q-Former 对齐 frozen 视觉和语言模型
- **LLaVA**：用线性投影将 CLIP 视觉特征接入 LLM
- **Qwen-VL**：类似的 vision encoder + LLM 架构

可以说，**CLIP 的 Image Encoder 成为了多模态模型的标准视觉组件**。

### 2. Zero-Shot 泛化能力

在 27 个数据集的评估中，CLIP 在 16 个上的 zero-shot 性能超过了有监督的线性探测基线。这说明自然语言监督学到的表示比分类标签更通用。

### 3. 分布偏移鲁棒性

CLIP 在各种分布偏移测试（ImageNet-V2、ImageNet-Sketch、ImageNet-A 等）上表现远优于常规训练的模型。有监督模型在 ImageNet 上过拟合了特定的数据分布，而 CLIP 学到了更本质的概念。

## 局限性

1. **抽象和复杂推理能力有限**：CLIP 擅长识别具体物体，但对于计数、空间关系、抽象概念的理解较弱
2. **训练成本极高**：4 亿图文对 + 大规模 batch size（32768），需要大量 GPU
3. **数据偏见**：互联网数据包含各种偏见，CLIP 不可避免地会学习到这些偏见
4. **细粒度理解不足**：对于需要局部细节的任务（如 OCR、细粒度分类），CLIP 的全局 embedding 不够用

## 后续发展

- **OpenCLIP**：开源复现，支持在自定义数据上训练
- **EVA-CLIP**：通过更好的训练策略和 ViT 改进，刷新 CLIP 的性能
- **SigLIP**：用 sigmoid loss 替代 softmax loss，去掉了对大 batch size 的依赖
- **CLIPA**：通过 masking 和渐进式分辨率训练，大幅降低训练成本

## 相关

- [[AI/CV/ViT|ViT]] — CLIP 的 Image Encoder 选项之一
- [[AI/3-LLM/MLLM/BLIP-2|BLIP-2]] — 继承 CLIP 对比学习范式，用 Q-Former 解决 frozen encoder 对齐
- [[AI/3-LLM/MLLM/Qwen-VL|Qwen-VL]] — 阿里 MLLM，使用 CLIP-style Vision Encoder 接 LLM
- [[AI/3-LLM/Architecture/BERT|BERT]] — Transformer 编码器，CLIP Text Encoder 的架构基础
- [[AI/1-Foundations/DL-Basics/深度学习|深度学习]]
- [[AI/3-LLM/MLLM/InternVL3|InternVL3]] — 后续 MLLM，同样基于 CLIP-style 视觉-语言对齐
- [[AI/3-LLM/MLLM/DeepSeek-VL|DeepSeek-VL]] — DeepSeek 多模态，使用 SigLIP（CLIP 改进版）作 Vision Encoder
- [[AI/3-LLM/MLLM/MLLM 概述|MLLM 概述]] — 多模态大模型整体框架综述
- [[AI/3-LLM/MLLM/非文本的模态对齐|非文本的模态对齐]] — CLIP 的核心贡献正是"非文本模态对齐"的奠基工作
