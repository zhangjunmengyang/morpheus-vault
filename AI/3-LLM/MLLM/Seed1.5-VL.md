---
brief: "Seed1.5-VL（字节）——字节跳动多模态大模型；专注长视频理解/GUI 交互/细粒度视觉推理；HybridVision encoder 混合分辨率处理；在长视频理解 benchmark 上达到 SOTA，与 Gemini 1.5 Pro 竞争。"
title: "Seed1.5-VL"
type: concept
domain: ai/mllm
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/mllm
  - type/concept
---
# Seed1.5-VL

> 字节跳动的多模态大模型，以视频理解和 agent 能力见长。
> 参考：https://mp.weixin.qq.com/s/sB-wNEyPZnQOFl9QR7cYFQ

## 定位

Seed1.5-VL 是字节豆包系列的视觉语言模型，对标 GPT-4o 和 Gemini 1.5 Pro。与其他 MLLM 相比，它的差异化在于：

1. **超长视频理解** — 支持数小时级别的视频输入
2. **GUI Agent 能力** — 原生支持屏幕操作（点击、滑动等）
3. **思维链推理** — 结合 o1-style 的推理能力处理复杂视觉任务

## 架构概览

```
Video/Image → Vision Encoder → Dynamic Token Compression → LLM Backbone
                                      ↓
                              根据内容复杂度动态调整 token 数
```

核心设计：

- **Vision Encoder**：基于 ViT，支持任意分辨率输入
- **Dynamic Token Compression**：这是关键创新——不是所有图像区域都需要同样多的 token。简单区域（纯色背景）压缩更多，复杂区域（文字、细节）保留更多 token
- **LLM Backbone**：基于字节自研的大模型

## 视频理解能力

传统 MLLM 处理视频的方式是均匀采样帧，这对长视频非常低效。Seed1.5-VL 的策略：

```python
# 伪代码：智能帧采样
def smart_frame_sampling(video, max_tokens=32768):
    # 1. 场景检测 — 找到关键转场点
    scene_boundaries = detect_scenes(video)
    
    # 2. 关键帧提取 — 每个场景取代表帧
    keyframes = []
    for scene in scenes:
        # 信息量大的帧多取，静态场景少取
        n_frames = estimate_complexity(scene)
        keyframes.extend(sample_frames(scene, n_frames))
    
    # 3. Token 预算分配
    tokens_per_frame = max_tokens // len(keyframes)
    # 高分辨率帧分配更多 token
    return dynamic_encode(keyframes, tokens_per_frame)
```

这使得模型能处理 1 小时以上的视频，而不是只能看几秒钟的 clip。

## GUI Agent

Seed1.5-VL 的 GUI agent 能力是其最大卖点之一：

```
输入：截屏图像 + 用户指令
输出：结构化操作序列

示例：
用户："帮我在淘宝搜索 iPhone 16 Pro 并按价格排序"
模型输出：
  1. {"action": "click", "element": "搜索框", "bbox": [120, 45, 380, 75]}
  2. {"action": "type", "text": "iPhone 16 Pro"}
  3. {"action": "click", "element": "搜索按钮", "bbox": [400, 45, 450, 75]}
  4. {"action": "click", "element": "价格排序", "bbox": [200, 120, 280, 140]}
```

这需要模型具备：
- **UI 元素检测**：识别按钮、输入框、链接等
- **空间定位**：精确输出 bounding box 坐标
- **多步规划**：理解完成任务需要哪些步骤

## 训练策略亮点

1. **三阶段训练**：视觉对齐 → 多模态预训练 → 指令微调 + RLHF
2. **混合数据**：图像、视频、文档、GUI 截图混合训练
3. **后训练强化**：在 agent 任务上用 RL 做了大量强化训练，这也是 GUI 能力强的主要原因

## 我的观点

Seed1.5-VL 代表了 MLLM 发展的一个方向：**从「看图说话」走向「看图做事」**。视频理解和 GUI Agent 是两个非常有实用价值的方向：

- 视频理解的 dynamic token compression 解决了长上下文的效率问题，这个思路后来被很多模型借鉴
- GUI Agent 是多模态模型离「落地」最近的应用场景之一，比通用 VQA benchmark 刷分有意义得多

不过需要注意，Seed1.5-VL 目前并未完全开源权重，主要通过字节的 API 和豆包产品提供服务。想要复现或微调需要等待社区方案。

## 相关

- [[AI/3-LLM/MLLM/DeepSeek-VL|DeepSeek-VL]] — DeepSeek 的开源多模态方案
- [[AI/3-LLM/MLLM/InternVL3|InternVL3]] — 另一个强力开源 MLLM
- [[AI/3-LLM/MLLM/Qwen 2.5 VL-Unsloth训练|Qwen 2.5 VL]] — Qwen 的多模态方案（Unsloth 训练实战）
- [[AI/3-LLM/MLLM/玩玩多模态大模型|多模态大模型实践]]
