---
brief: "DeepSeek-VL——DeepSeek 的视觉语言模型系列；hybrid vision encoder（高低分辨率并行）+ DeepSeek LLM；特别强化文档/图表/代码截图的理解能力；DeepSeek-VL2 引入 MoE 架构大幅提升效率。"
title: "DeepSeek-VL"
type: paper
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/paper
---
# DeepSeek-VL

> DeepSeek 的多模态大模型系列，从 VL 到 VL2 的架构演进。
> 参考：https://github.com/Tramac/paper-reading-note/blob/main/notes/036_deepseekvl.md

## DeepSeek-VL 架构

DeepSeek-VL 采用经典的三段式 MLLM 架构：

```
Vision Encoder  →  Vision-Language Adaptor  →  LLM Backbone
(SigLIP-L)         (2-layer MLP)               (DeepSeek-LLM)
```

### Vision Encoder

采用混合视觉编码策略，这是 DeepSeek-VL 的一个亮点：

- **SigLIP-L**：处理高级语义特征（384×384）
- **SAM-B encoder**：处理低级细节特征（1024×1024），用于需要高分辨率理解的场景

两路特征通过 adaptor 融合后送入 LLM。这种双 encoder 设计在当时是比较少见的，后来很多工作都采用了类似思路。

### 动态分辨率

```python
# DeepSeek-VL 的动态分辨率策略
# 将高分辨率图片分割成多个 384×384 的 tile
def dynamic_resolution(image, max_tiles=12):
    w, h = image.size
    # 计算最佳 tile 网格
    best_grid = find_best_grid(w, h, max_tiles)
    # 分割 + 缩略图
    tiles = split_image(image, best_grid)
    thumbnail = resize(image, (384, 384))
    return [thumbnail] + tiles  # 缩略图提供全局信息
```

## DeepSeek-VL2 改进

VL2 在 VL 基础上做了几个关键改进：

1. **MoE 架构**：LLM backbone 换成 DeepSeek-MoE，推理效率大幅提升
2. **改进的动态分辨率**：更灵活的 tile 划分策略
3. **多图理解**：原生支持多图输入和跨图推理

```
VL2 模型矩阵:
┌──────────────┬──────────────┬──────────────┐
│ VL2-Tiny     │ VL2-Small    │ VL2          │
│ 3B (1B act)  │ 16B (2.8B)   │ 27B (4.5B)  │
│ 轻量部署     │ 均衡选择     │ 最强性能     │
└──────────────┴──────────────┴──────────────┘
act = 激活参数量（MoE 只激活部分专家）
```

## 训练流程

DeepSeek-VL 的训练分三个阶段：

| 阶段 | 目标 | 训练数据 | 解冻模块 |
|------|------|---------|---------|
| Stage 1 | 视觉-语言对齐 | 图文配对数据 | 仅 Adaptor |
| Stage 2 | 联合预训练 | 混合多模态数据 | 全部 |
| Stage 3 | 指令微调 | 高质量指令数据 | 全部 |

关键 insight：**Stage 2 混入了大量纯文本数据**（约 70%），防止多模态训练导致语言能力退化。这个比例的选择很有讲究，太少语言能力掉，太多视觉能力上不去。

## 数据工程

DeepSeek-VL 在数据上下了大功夫：

```python
training_data_composition = {
    "stage_2": {
        "纯文本": "70% — 保持语言能力",
        "图文配对": "15% — 基础视觉理解",
        "OCR 数据": "5% — 文档理解",
        "表格/图表": "5% — 结构化信息理解",
        "多轮对话": "5% — 交互能力",
    },
    "stage_3": {
        "通用 VQA": "40%",
        "文档理解": "20%",
        "推理类": "20%",
        "创意类": "10%",
        "代码相关": "10%",
    }
}
```

## 与同期模型对比

在论文发布时（2024 年初）的表现：

- **文档理解**：在 DocVQA/ChartQA 上表现突出，双 encoder 对高分辨率场景有明显优势
- **通用 VQA**：与 LLaVA-NeXT-34B 持平，但参数量更小（7B）
- **数学推理**：受益于 DeepSeek 系列的数学能力，在 MathVista 上有优势

## 我的观点

DeepSeek-VL 系列的核心竞争力在于：

1. **工程质量高** — 数据配比、训练策略都经过精心调优，不是简单堆数据
2. **开源诚意足** — 代码、权重、训练细节都开放，可复现性强
3. **MoE 路线正确** — VL2 的 MoE 架构让部署成本大幅降低，3B 模型激活参数只有 1B

从 VL 到 VL2，最大的变化不是模型变大了，而是「怎么用更少的计算做更多的事」。MoE 在多模态模型上的应用是必然趋势。

## 相关

- [[AI/3-LLM/MLLM/DeepSeek-OCR 原理|DeepSeek-OCR 原理]] — 基于 VL 的 OCR 应用
- [[AI/3-LLM/MLLM/DeepSeek-OCR-Unsloth实践|DeepSeek-OCR Unsloth 实践]] — 微调实战
- [[AI/3-LLM/MLLM/InternVL3|InternVL3]] — 同期竞品对比
- [[AI/3-LLM/MLLM/Seed1.5-VL|Seed1.5-VL]] — 字节的多模态方案
- [[AI/3-LLM/MLLM/CLIP|CLIP]]
- [[AI/3-LLM/MLLM/BLIP-2|BLIP-2]]
- [[AI/3-LLM/MLLM/Qwen-VL|Qwen-VL]]
- [[AI/3-LLM/MLLM/MLLM 概述|MLLM 概述]]
- [[AI/CV/ViT|ViT]]
- [[AI/3-LLM/Architecture/DeepSeek-R1|DeepSeek-R1]]
