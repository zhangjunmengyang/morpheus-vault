---
brief: "BLIP-2（Salesforce，2023）——用 Q-Former 桥接冻结视觉 encoder 和 LLM；两阶段预训练（视觉语言对齐→语言生成）；参数高效的多模态预训练范式；LLaVA 等后续工作的重要参照基线。"
title: "BLIP-2"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# BLIP-2

BLIP-2（Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models）是 Salesforce 于 2023 年提出的多模态模型，核心贡献是提出了 **Q-Former**——一个轻量级的桥接模块，用来连接冻结的视觉编码器和冻结的大语言模型。这个设计巧妙地解决了多模态训练的效率问题：**不需要端到端训练整个大模型，只需要训练一个小的桥接模块**。

## 动机

在 BLIP-2 之前，多模态模型的训练面临两难：

1. **端到端训练**（如 Flamingo）：效果好但计算成本极高
2. **冻结+线性投影**：便宜但效果受限

BLIP-2 的洞察是：视觉编码器和 LLM 各自都已经很强了，关键问题是如何让它们有效沟通。

## Q-Former：核心桥接模块

Q-Former 是一个包含 **32 个可学习 query tokens** 的 Transformer 模块，它同时与图像特征和文本进行交互：

```
图像 (ViT) → Image Features (257 tokens)
                    ↕ Cross-Attention
Q-Former Queries (32 tokens) ← 可学习
                    ↕ Self-Attention
                  Query 输出 (32 tokens)
                    ↓ Linear Projection
                  LLM 输入
```

Q-Former 的设计精华在于：它把变长的视觉特征（通常 257 个 token）**压缩成固定 32 个 token**，大幅减少了 LLM 需要处理的视觉 token 数量。

## 两阶段训练

### 阶段 1：视觉-语言表示学习

冻结 Image Encoder，训练 Q-Former 学习与视觉相关的文本表示。使用三个损失函数：

1. **Image-Text Contrastive（ITC）**：对比学习，拉近匹配的图文对
2. **Image-grounded Text Generation（ITG）**：给定图像生成文本描述
3. **Image-Text Matching（ITM）**：二分类，判断图文是否匹配

这三个目标分别对应 CLIP、生成模型、判别模型的训练思路。通过共享 Q-Former 参数（但使用不同的 attention mask），一个模块同时学习了三种能力。

### 阶段 2：视觉到语言的生成学习

冻结 Image Encoder + LLM，用全连接层将 Q-Former 的输出投影到 LLM 的输入空间。

训练目标：language modeling loss，即给定图像生成对应的文本。

```
[Image] → ViT → Q-Former → FC → [LLM Prefix]  "A dog sitting on..."
                                     ↓
                                  LLM 生成
```

## 实验效果

BLIP-2 用少于 Flamingo 80B 总参数量 54 倍的可训练参数，在 zero-shot VQAv2 上超越了 Flamingo 80B：

| 模型 | 可训练参数 | VQAv2 (0-shot) |
|------|-----------|----------------|
| Flamingo 80B | 10.2B | 56.3 |
| BLIP-2 (ViT-G + FlanT5-XXL) | 107M | **65.0** |

这个结果令人震惊：**107M vs 10.2B 的可训练参数，效果还更好**。

## 设计的优缺点

### 优点

1. **训练效率极高**：只训练 Q-Former（~107M 参数），基座模型全部冻结
2. **模块化设计**：可以灵活替换 Image Encoder 和 LLM
3. **信息瓶颈**：32 个 query 天然形成信息压缩，迫使 Q-Former 提取最重要的视觉信息

### 缺点

1. **信息损失**：32 个 token 难以保留所有视觉细节，细粒度任务（OCR、密集物体检测）表现受限
2. **两阶段训练增加复杂性**
3. **冻结 LLM 限制了深度对齐**：后来的 LLaVA 证明，微调 LLM 可以获得更好的指令跟随能力

## 在多模态模型演进中的位置

BLIP-2 处于多模态模型从"方法论探索"到"工程产品化"的转折点：

- **BLIP-2 之前**：各种探索性架构（Flamingo、CoCa、PaLI）
- **BLIP-2 开创**：冻结基座 + 轻量桥接的范式
- **LLaVA 简化**：直接用线性投影替代 Q-Former，微调 LLM
- **现代 MLLM**：Qwen-VL、InternVL3 等基本都采用"视觉编码器 + 投影层 + LLM"的架构

BLIP-2 的 Q-Former 虽然在后续工作中被更简单的投影层替代，但它提出的"桥接冻结模型"的思路和"两阶段对齐"的训练策略，深刻影响了整个领域的发展方向。

## 相关

- [[AI/3-LLM/MLLM/CLIP]]
- [[AI/3-LLM/MLLM/Qwen-VL]]
- [[AI/3-LLM/MLLM/CLIP|ViT（CLIP image encoder）]]
- [[AI/4-模型/OpenAI/GPT|GPT]]
- [[AI/1-Foundations/DL-Basics/深度学习|深度学习]]
- [[AI/3-LLM/MLLM/InternVL3|InternVL3]]
- [[AI/3-LLM/MLLM/DeepSeek-VL|DeepSeek-VL]]
- [[AI/3-LLM/MLLM/MLLM 概述|MLLM 概述]]
- [[AI/4-模型/Meta/LLaMA|LLaMA]]
- [[AI/4-模型/Google/T5|T5]]
