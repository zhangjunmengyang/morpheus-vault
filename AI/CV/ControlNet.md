---
title: "ControlNet"
type: paper
domain: ai/cv
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/cv
  - type/paper
---
# ControlNet

> Adding Conditional Control to Text-to-Image Diffusion Models — 让 Diffusion 模型听话的关键一步。

## 核心问题

Stable Diffusion 等 text-to-image 模型在「生成什么」上做得很好，但在「怎么生成」上缺乏精确控制。你想要一个特定姿势的人物、一个特定边缘结构的建筑，光靠 prompt 是不够的。ControlNet 的目标就是：**在不破坏预训练模型能力的前提下，注入空间条件控制**。

## 架构设计

ControlNet 的核心思想极其优雅：

1. **复制一份 encoder**：把 Stable Diffusion U-Net 的 encoder 部分完整复制一份，作为 trainable copy
2. **冻结原始权重**：原始的 SD 模型参数全部 frozen，保证不丢失已有能力
3. **Zero Convolution 连接**：用初始化为零的 1×1 卷积层连接 trainable copy 和原始网络

```
原始 SD Encoder (frozen)
    │
    ├── block_1 ──────────────────── + ← zero_conv(trainable_block_1(condition))
    ├── block_2 ──────────────────── + ← zero_conv(trainable_block_2(...))
    ├── ...
    └── block_n ──────────────────── + ← zero_conv(trainable_block_n(...))
    │
    ▼
  Decoder (frozen)
```

**为什么用 Zero Convolution？** 训练开始时 zero conv 输出全为 0，相当于 ControlNet 还没有产生任何影响，模型行为等价于原始 SD。随着训练推进，zero conv 的参数逐渐学到有意义的值，控制信号被平滑地注入。这个设计避免了训练初期条件信号带来的噪声干扰，是个非常巧妙的 warm-start 策略。

## 条件类型

ControlNet 支持多种空间条件输入：

| 条件类型 | 说明 | 典型用途 |
|---------|------|---------|
| Canny Edge | 边缘检测图 | 保持结构轮廓 |
| OpenPose | 人体关键点 | 控制人物姿势 |
| Depth Map | 深度图 | 控制空间布局 |
| Segmentation | 语义分割图 | 精确区域控制 |
| Scribble | 涂鸦/草图 | 草图生成 |
| Normal Map | 法线图 | 控制表面朝向 |
| M-LSD Lines | 直线检测 | 建筑/室内设计 |

每种条件需要单独训练一个 ControlNet，但可以组合使用（multi-ControlNet）。

## 训练细节

```python
# 伪代码：ControlNet 训练循环
for batch in dataloader:
    # batch 包含: image, text_prompt, condition_image (如 canny edge)
    
    # 1. 原始 SD encoder 处理 (frozen)
    with torch.no_grad():
        sd_features = sd_encoder(noisy_latent, timestep, text_embedding)
    
    # 2. ControlNet 处理条件输入 (trainable)
    control_features = controlnet(noisy_latent, timestep, text_embedding, condition_image)
    
    # 3. Zero conv 后加到 SD features 上
    combined = sd_features + zero_conv(control_features)
    
    # 4. Decoder 生成预测噪声
    pred_noise = sd_decoder(combined)
    
    # 5. 标准 diffusion loss
    loss = F.mse_loss(pred_noise, target_noise)
```

训练数据量不大也能出效果。论文中 Canny edge 条件只用了约 3M 图文对，在 8 张 A100 上训练约 600 GPU-hours。对于特定领域，甚至几千张标注数据就能训练出可用的 ControlNet。

## 实践要点

**推理时的 conditioning_scale**：这是最重要的超参。值越大条件控制越强，但会牺牲生成质量和多样性。一般 0.5-1.5 之间调：

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

image = pipe(
    "a beautiful house",
    image=canny_image,
    controlnet_conditioning_scale=1.0,  # 关键参数
    num_inference_steps=30,
).images[0]
```

**Multi-ControlNet 组合**：可以同时使用多个条件，比如 Depth + Canny，各自设置不同的 scale：

```python
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_depth]
)
image = pipe(
    prompt,
    image=[canny_img, depth_img],
    controlnet_conditioning_scale=[0.8, 0.5],
).images[0]
```

## 我的观点

ControlNet 之所以成功，在于它找到了一个绝佳的平衡点：不修改原始模型（保住通用能力），同时用极少的参数量注入了精确的空间控制。Zero Convolution 的设计看似简单，实际上解决了条件注入训练不稳定的核心难题。

从工程角度看，ControlNet 开启了 Diffusion 模型「可控生成」的大门。后来的 IP-Adapter、T2I-Adapter、InstantID 等工作都受到了它的启发。在生产环境中，ControlNet 可能是 Stable Diffusion 生态中应用最广泛的组件之一。

## 相关

- [[AI/CV/ViT|ViT]] — Transformer 在视觉领域的基础架构
- [[AI/CV/MAE|MAE]] — 自监督视觉预训练
- [[AI/CV/_MOC|计算机视觉 MOC]]
