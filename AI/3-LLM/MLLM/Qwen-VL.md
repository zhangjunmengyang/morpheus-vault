---
brief: "Qwen-VL——阿里 Qwen 多模态系列（Qwen-VL → Qwen2.5-VL → Qwen3-VL）；视觉 token 压缩/动态分辨率/文档理解特化的技术演进；国内最强开源 VLM 系列，在 OCR 和文档理解上表现突出。"
title: "Qwen-VL"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# Qwen-VL

Qwen-VL 是阿里通义千问系列中的多模态大模型，从 Qwen-VL 到 Qwen2-VL 再到 Qwen2.5-VL，架构经历了几次关键迭代。核心思路一直是：**在强大的 LLM backbone 上接一个视觉编码器，用尽量少的额外参数把视觉信号对齐到语言空间**。

## 架构演进

### Qwen-VL（初版）

- Vision Encoder: ViT-bigG (OpenCLIP)，固定 224×224 分辨率
- 通过一个 single-layer cross-attention（Resampler）把视觉 token 压缩到 256 个
- 三阶段训练：预训练 → 多任务微调 → 对话 SFT

初版最大的限制是**固定分辨率**，OCR 和细粒度理解表现一般。

### Qwen2-VL

关键改进：**Naive Dynamic Resolution** → 输入图片不再被 resize 到固定尺寸，而是：

1. 将原图按 patch 切分（14×14 pixels/patch）
2. 引入 2D-RoPE 替代 1D position embedding，让模型感知二维空间关系
3. 支持视频输入（视频帧序列 + 时间维度 RoPE）

```python
# Qwen2-VL 的动态分辨率处理伪代码
def process_image(image, min_pixels=256*28*28, max_pixels=1280*28*28):
    # 保持宽高比，将图片 resize 到 [min_pixels, max_pixels] 范围内
    # 然后按 14x14 patch 切分
    h, w = image.size
    scale = find_best_scale(h, w, min_pixels, max_pixels)
    patches = split_to_patches(image.resize(scale), patch_size=14)
    return patches  # shape: (n_patches, 14, 14, 3)
```

2D-RoPE 是 Qwen2-VL 的亮点设计 —— 传统 ViT 用 1D positional embedding 把 2D patch 拉成序列，丢失了空间信息。Qwen2-VL 给每个 patch 分配 (row, col) 坐标，分别在 RoPE 的不同频率维度上编码。

### Qwen2.5-VL

在 Qwen2-VL 基础上的主要改进：

- **更强的 OCR/文档理解**：针对文档场景做了大量数据增强
- **Grounding 能力**：输出 bounding box 坐标，支持 referring expression
- **Agent 能力**：增加了 tool use 和 function calling 的多模态版本
- **视频理解增强**：支持更长的视频帧序列，时间定位更准

## 实际使用体验

在实际业务中用 Qwen2.5-VL 做 OCR，几个要点：

1. **分辨率直接影响效果** —— `min_pixels` 和 `max_pixels` 参数很关键，文档类任务建议调高
2. **中文 OCR 明显优于同级别开源模型** —— 特别是手写体、表格、竖排文字
3. **推理速度** —— 动态分辨率意味着高分辨率图片的 token 数暴增，注意 GPU 显存

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=256*28*28,
    max_pixels=1280*28*28  # 文档场景可以开到更大
)
```

## 我的看法

Qwen-VL 系列是目前开源 MLLM 中工程完成度最高的之一。几个值得关注的设计决策：

- 2D-RoPE 优于传统的绝对位置编码，理论上支持任意分辨率外推
- 没有走 LLaVA 那种简单 MLP projector 的路线，而是在视觉编码侧做了更重的工程
- 开源策略激进 —— 7B/72B 都开源，对社区非常友好

但也有不足：高分辨率输入时 token 数量剧增（一张 4K 图可以产生上千 visual tokens），推理成本高。未来可能需要类似 token pruning 或 early exit 的优化。

## 相关

- [[AI/3-LLM/MLLM/DeepSeek-VL]]
- [[AI/3-LLM/MLLM/InternVL3]]
- [[AI/3-LLM/MLLM/Qwen 2.5 VL-Unsloth训练]]
- [[AI/3-LLM/MLLM/Universal Multimodal Retrieval]]
- [[AI/3-LLM/MLLM/CLIP|CLIP]]
- [[AI/3-LLM/MLLM/BLIP-2|BLIP-2]]
- [[AI/3-LLM/MLLM/MLLM 概述|MLLM 概述]]
- [[AI/3-LLM/MLLM/CLIP|ViT（CLIP image encoder）]]
- [[LLaMA|LLaMA]]
- [[Transformer 位置编码详解|Transformer 位置编码详解]]
