---
brief: "InternVL3——上海 AI Lab 的多模态大模型系列；InternViT 视觉 encoder + InternLM2 语言模型；在 OCR/图表理解/多图推理等任务上保持开源 SOTA；高分辨率动态切分策略处理细粒度视觉任务。"
title: "InternVL3"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# InternVL3

> 书生·万象 —— 从 InternVL 到 InternVL3 的演进之路。
> 模型：https://huggingface.co/OpenGVLab/InternVL3-78B
> 笔记：https://github.com/Tramac/paper-reading-note/blob/main/notes/037_internvl.md

## 系列演进

```
InternVL (2023.12)  →  InternVL 1.5 (2024.04)  →  InternVL 2 (2024.07)  →  InternVL3 (2025.04)
    │                       │                          │                        │
 基础架构                动态分辨率               多模型矩阵               RL + 思维链
```

### InternVL (v1)

核心贡献是 **InternViT-6B** — 当时最大的开源 Vision Transformer。架构：

```
InternViT-6B → QLLaMA → 文本输出
```

用 contrastive learning + generative learning 联合训练，证明了放大 vision encoder 的有效性。

### InternVL 1.5

关键改进 — **Dynamic High Resolution**：

```python
# 动态高分辨率方案
def dynamic_high_resolution(image, min_tiles=1, max_tiles=12):
    """
    将图片动态分割成 448×448 的 tile
    tile 数量根据图片分辨率和宽高比自动决定
    """
    w, h = image.size
    # 枚举所有可能的网格组合
    candidates = [(i, j) for i in range(1, max_tiles+1) 
                  for j in range(1, max_tiles+1) if i*j <= max_tiles]
    
    # 选择最接近原始宽高比的网格
    best_grid = min(candidates, key=lambda g: aspect_ratio_diff(g, w, h))
    
    # 分割 + 全局缩略图
    tiles = split_to_tiles(image, best_grid, tile_size=448)
    thumbnail = resize(image, (448, 448))
    return [thumbnail] + tiles
```

这个方案后来被 DeepSeek-VL2、Qwen-VL 等模型广泛借鉴。

### InternVL 2

从单一模型走向模型矩阵：

| 模型 | Vision Encoder | LLM | 参数量 |
|------|---------------|-----|--------|
| InternVL2-1B | InternViT-300M | InternLM2-1.8B | 2B |
| InternVL2-4B | InternViT-300M | Phi-3-mini | 4B |
| InternVL2-8B | InternViT-300M | InternLM2-7B | 8B |
| InternVL2-26B | InternViT-6B | InternLM2-20B | 26B |
| InternVL2-76B | InternViT-6B | Hermes-2-Llama-3.1-70B | 76B |

一个重要设计：小模型用 InternViT-300M（从 6B 蒸馏而来），大模型用完整的 InternViT-6B。

## InternVL3 核心改进

InternVL3 的最大亮点是引入了 **RL（强化学习）** 到多模态训练中：

1. **Native Multimodal Pre-Training**：不再是先训文本再接视觉，而是从预训练开始就是多模态的
2. **Test-Time Scaling**：通过思维链让模型在推理时做更多思考
3. **RL 后训练**：在数学推理、代码生成等任务上用 GRPO 做强化

```
训练流程:
Pre-training (多模态原生) → SFT → RLHF (GRPO)
                                      ↓
                              数学/代码/推理能力显著提升
```

## Benchmark 表现

InternVL3-78B 在主流 benchmark 上的亮点：
- **MathVista**: 72.7（对标 GPT-4o 的 63.8）
- **DocVQA**: 96.1（文档理解）
- **OCRBench**: 880+（OCR 能力）
- **MMMU**: 72.2（多学科理解）

## 我的观点

InternVL 系列的成功可以归结为几点：

1. **Vision Encoder 足够强** — InternViT-6B 是开源社区最强的视觉编码器之一，大量下游工作都在用
2. **工程扎实** — 动态分辨率方案设计优雅，几乎成了行业标配
3. **开源生态好** — 完整的模型矩阵（1B 到 78B），覆盖各种部署场景
4. **RL 是正确方向** — InternVL3 引入 GRPO 是关键一步，多模态 RL 是 2025 年的重要趋势

从工程师角度看，InternVL2-8B 是性价比最高的选择——效果接近 GPT-4V，但可以在单卡上推理和微调。

## 相关

- [[AI/3-LLM/MLLM/DeepSeek-VL|DeepSeek-VL]] — DeepSeek 多模态系列
- [[AI/3-LLM/MLLM/Seed1.5-VL|Seed1.5-VL]] — 字节多模态方案
- [[AI/3-LLM/MLLM/Qwen 2.5 VL-Unsloth训练|Qwen 2.5 VL 训练]] — Qwen VLM 微调
- [[AI/CV/ViT|ViT]] — Vision Transformer 基础
- [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO]] — InternVL3 使用的 RL 算法
- [[AI/3-LLM/MLLM/CLIP|CLIP]]
- [[AI/3-LLM/MLLM/BLIP-2|BLIP-2]]
- [[AI/3-LLM/MLLM/Qwen-VL|Qwen-VL]]
- [[AI/3-LLM/MLLM/MLLM 概述|MLLM 概述]]
- [[LLaMA|LLaMA]]
