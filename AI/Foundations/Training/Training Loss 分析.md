---
brief: "Training Loss 分析——LLM 训练中 Loss 曲线的正常模式和异常诊断；Loss spike/不收敛/过拟合的根因分析方法；梯度范数/学习率/batch size 对 loss 的影响；生产级 LLM 训练的调试参考手册。"
title: "训练 Loss 曲线分析与调参"
date: 2026-02-14
tags: [training, debugging, loss, interview]
type: note
---

# 训练 Loss 曲线分析与调参

## Loss 曲线的正常形态

- **快速下降期**：前 5-10% 步数，loss 急剧下降
- **平稳下降期**：主要训练阶段，loss 缓慢持续下降
- **收敛期**：末期趋于平坦，下降速度接近零
- **验证 loss**：应与训练 loss 走势相近，gap 不宜过大（过大=过拟合）

## 常见异常及诊断

### Loss Spike（突刺）
- **现象**：loss 突然大幅上升后回落
- **原因**：数据中的异常样本、学习率过大、数据顺序问题
- **修复**：
  - 梯度裁剪（gradient clipping, 通常 max_norm=1.0）
  - 检查对应 batch 的数据质量
  - 降低学习率
  - 使用 z-loss regularization（DeepSeek 方案）

### Loss Divergence（发散）
- **现象**：loss 持续上升不回落
- **原因**：学习率过大、模型初始化不当、数据 bug
- **修复**：降 LR 50%、检查数据 pipeline、检查 loss 计算

### Loss Plateau（平台期）
- **现象**：loss 不再下降但未达预期
- **原因**：LR 过小、模型容量不够、数据质量/多样性不足
- **修复**：尝试 warmup restart、增大模型、增加数据多样性

### NaN Loss
- **现象**：loss 变成 NaN
- **原因**：数值溢出（FP16 常见）、除零、梯度爆炸
- **修复**：
  - 检查 loss scaling（FP16 需要 dynamic loss scaling）
  - 用 BF16 替代 FP16（更大动态范围）
  - 加梯度裁剪
  - 检查数据中的异常值

## 学习率调度策略

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| **Cosine Decay** | 余弦曲线衰减到 min_lr | 最常用，LLaMA/GPT 系列标配 |
| **WSD (Warmup-Stable-Decay)** | warmup → 恒定 → 衰减 | MiniCPM 提出，适合不确定总步数 |
| **Linear Decay** | 线性衰减 | 简单但效果一般 |
| **Constant + Decay** | 先恒定再衰减 | 适合 fine-tuning |
| **Inverse sqrt** | 1/√step 衰减 | 原始 Transformer 论文 |

### Warmup
- 目的：避免训练初期梯度过大导致不稳定
- 典型比例：总步数的 0.5-2%
- LLaMA-2 使用 2000 步 warmup

## Batch Size 与 LR 的关系

- **线性缩放规则**：batch size 翻倍 → LR 翻倍（Linear Scaling Rule）
- **平方根缩放**：batch size 翻倍 → LR × √2（更保守，大 batch 常用）
- 实践中 batch size 增大到一定程度后，LR 不再线性增长
- Gradient Accumulation 等效增大 batch size（micro_batch × accumulation_steps）

## 权重衰减（Weight Decay）

- 典型值：0.01-0.1
- AdamW 中 weight decay 与 L2 regularization 等效但实现不同
- 通常只对权重矩阵衰减，**不对 bias 和 LayerNorm 参数衰减**
- 过大会欠拟合，过小会过拟合

## 梯度裁剪（Gradient Clipping）

- **Max Norm Clipping**：||g|| > max_norm 时按比例缩放，常用 max_norm=1.0
- **Value Clipping**：裁剪单个梯度值，不常用
- 对训练稳定性至关重要，几乎所有 LLM 训练都使用

## 面试常见问题

### Q1: 训练中出现 loss spike 怎么办？
**答**：首先检查对应 batch 的数据质量（可能有异常样本）。短期措施：确认梯度裁剪已启用且阈值合理（通常1.0）。如果频繁出现，考虑降低学习率或加入 z-loss regularization。记录 spike 对应的 step 和数据，排查数据 pipeline。

### Q2: BF16 和 FP16 训练怎么选？
**答**：优先 BF16。BF16 动态范围与 FP32 相同（8位指数），不需要 loss scaling，训练更稳定。FP16 精度更高但动态范围小，容易溢出，需要 dynamic loss scaling。前提是硬件支持（A100+/H100 支持 BF16）。

### Q3: 学习率怎么选？
**答**：常见范围 1e-4 到 3e-4（预训练），1e-5 到 5e-5（fine-tuning）。一般从 3e-4 开始，配合 cosine decay + warmup。Batch size 变化时按线性/平方根规则调整。最可靠的方法是做小规模 LR sweep（3-5 个值跑 1000 步看 loss 曲线）。

### Q4: 训练 loss 下降但验证 loss 不降，怎么办？
**答**：过拟合信号。增大 weight decay、增加 dropout（预训练一般不用 dropout，fine-tuning 可加 0.05-0.1）、增大训练数据量或数据多样性、减小模型或用 LoRA 等参数高效方法。

### Q5: 如何估算预训练需要的计算量？
**答**：经验公式 C ≈ 6 × N × D，其中 N 是参数量，D 是训练 token 数。例如 7B 模型训练 2T tokens：C = 6 × 7e9 × 2e12 = 8.4e22 FLOPs。按 H100 GPU 的有效吞吐 ~400 TFLOPS，需要 8.4e22 / 4e14 ≈ 2.1e8 秒 ≈ 6700 GPU 小时。8 卡节点约 838 小时 ≈ 35 天。

---

## See Also

- [[AI/Foundations/Training/Scaling Laws|Scaling Laws]] — loss 曲线宏观规律：参数量/数据量/计算量的幂律
- [[AI/Foundations/Math/向量微积分|向量微积分]] — 梯度下降、反向传播的数学基础
- [[AI/LLM/Frameworks/verl/verl 概述|verl 框架]] — 分布式 RL 训练中的 loss 监控
- [[AI/Foundations/目录|Foundations MOC]] — 训练基础全图谱
