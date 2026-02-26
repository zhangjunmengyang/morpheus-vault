---
title: "计算机视觉"
type: moc
domain: ai/cv
tags:
  - ai/cv
  - type/reference
---
## MMAN：多模态注意力网络 · CAD诊断（IEEE JBHI）

> **状态**：手稿 / 发表中 · **角色**：第二作者（一作：Xiaohong Wang）

- [[MMAN-CAD-MPI-论文手稿|📄 论文手稿]] — 完整手稿内容（MMAN · ICCA · CDGA · MAE预训练 · AUC=0.8790）
- [[项目概览|🗂️ 项目概览]] — 背景、方法、So What、面试价值
- [[Projects/MMAN-CAD-MPI/_MOC|📚 背景知识索引]]
  - [[Projects/MMAN-CAD-MPI/ViT|ViT]] — 视觉 Transformer，图像编码器基础
  - [[MAE|MAE]] — 自监督视觉预训练，医疗图像少标注场景
  - [[计算机视觉基础与前沿-2026技术全景|CV 全景]] — 多模态/注意力/检测完整知识图谱

### 核心贡献速查（面试用）
- **ICCA 模块**：图像相关交叉注意力，融合压力/静息 MPI 互补信息
- **CDGA 模块**：临床数据引导注意力，引入 39 项非影像临床变量
- **MAE 预训练**：医疗图像标注稀缺问题的解法
- **结果**：AUC=0.8790，优于对比方法，真实临床数据集

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


## 我的观点

CV 正在被 LLM 吞噬。纯 CV 任务（分类、检测、分割）已经很成熟，增量空间有限。真正的增长点在 **多模态融合** — 让 LLM 理解视觉信息。所以学 CV 不是学 ResNet 调参，而是理解 **视觉特征如何与语言特征对齐**。