---
title: "计算机视觉"
type: moc
domain: ai/cv
tags:
  - ai/cv
  - type/reference
---

# 👁️ 计算机视觉

> CV 基础模型与技术 — 从传统视觉到 Vision Transformer 的范式迁移

## 综合全景
- [[AI/CV/计算机视觉基础与前沿-2026技术全景|计算机视觉基础与前沿 2026 全景]] ⭐ — 面试武器版，1600行，经典架构→目标检测→语义分割→ViT→多模态→3D→生成模型→自监督全覆盖 ★★★★★

## 发展脉络

```
CNN 时代 (2012-2020)          Transformer 时代 (2020-)
AlexNet → VGG → ResNet        ViT → DeiT → Swin
    ↓                            ↓
目标检测: RCNN → YOLO        DETR → DINO
    ↓                            ↓
语义分割: FCN → U-Net        SegGPT → SAM
    ↓                            ↓
自监督: SimCLR → MoCo        MAE → DINOv2
    ↓                            ↓
生成: GAN → VAE              Diffusion → ControlNet
```

## 核心模型

### Vision Transformer

- [[AI/CV/ViT|ViT]] — Vision Transformer，开启 CV Transformer 时代
  - 将图像切成 16×16 patch，当作 token 序列送入 Transformer
  - 关键发现：**数据量足够大时，ViT 超越 CNN**
  - 不足：小数据集上不如 CNN（缺少归纳偏置）

### 自监督学习

- [[AI/CV/MAE|MAE]] — Masked Autoencoders
  - 受 BERT 启发，随机遮掩 75% 的图像 patch，让模型重建
  - 核心洞察：**视觉信号冗余度高，高遮掩率才能学到好特征**
  - 预训练效率极高：不需要标注数据，不需要数据增强

### 可控生成

- [[AI/MLLM/ControlNet|ControlNet]] — 给 Diffusion Model 加条件控制
  - 输入边缘图 / 深度图 / 骨架图 → 生成符合条件的图像
  - 核心技术：zero convolution（训练开始时不影响原模型）
  - 实用场景：精确控制图像生成的构图、姿态、边缘

## CV 当前的几个重要方向

### 1. Foundation Models
- **SAM (Segment Anything)**：通用分割，任意图像、任意目标
- **DINOv2**：自监督视觉特征提取器，迁移能力极强
- **CLIP**：视觉-语言对齐，zero-shot 分类

### 2. 与 LLM 的融合
- 视觉编码器 + LLM = 多模态大模型（VLM）
- 典型架构：ViT/SigLIP → Projection → LLM
- 代表模型：LLaVA、Qwen-VL、InternVL

### 3. 视频理解
- 时序建模：从图像理解到视频理解
- 长视频：如何高效处理数百帧的信息

## 我的观点

CV 正在被 LLM 吞噬。纯 CV 任务（分类、检测、分割）已经很成熟，增量空间有限。真正的增长点在 **多模态融合** — 让 LLM 理解视觉信息。所以学 CV 不是学 ResNet 调参，而是理解 **视觉特征如何与语言特征对齐**。

## 相关 MOC

- ↑ 上级：[[AI/目录]]
- → 相关：[[AI/MLLM/目录|多模态大模型]]（VLM、视觉-语言融合）
- → 相关：[[AI/LLM/目录|大语言模型]]（LLM 基础）
