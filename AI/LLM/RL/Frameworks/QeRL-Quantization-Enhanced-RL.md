---
brief: "QeRL——量化增强 RL 训练框架；核心反直觉发现：适度量化（如 INT8）有助于 RL 训练的正则化效果，防止策略过拟合；与 Jet-RL（量化须一致）形成理论对立，两篇形成重要交叉链接。"
title: "QeRL: Quantization-Enhanced Reinforcement Learning"
type: note
domain: ai/llm/rl
tags:
  - ai/llm/rl
  - quantization
  - rl-training
  - iclr-2026
  - type/paper
date: 2026-02-20
---

# QeRL: Beyond Efficiency — Quantization-enhanced Reinforcement Learning for LLMs

**arXiv**: 2510.11696  
**提交日期**: 2025-10-13  
**会议**: ICLR 2026  
**作者**: Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao, Huizi Mao, Yujun Lin, Hanrong Ye, Sifei Liu, Ka Chun Cheung, Hongxu Yin, Yao Lu, Xiaojuan Qi, **Song Han**, Yukang Chen  
**机构**: NVIDIA / MIT / HKU / THU  
**代码**: https://github.com/NVlabs/QeRL  
**评分**: ★★★★☆

---

## 一句话

QeRL 发现了一个反直觉的现象：**量化噪声在 RL 中是有益的**——它增加 policy entropy，促进探索，使 4-bit 量化 + LoRA 的 RL 训练不仅比 16-bit LoRA 更快，还**在多项基准上超越** 16-bit LoRA，同时达到 1.5× 端到端加速。

---

## 与 Jet-RL 的关系

> **关键**：QeRL（Oct 2025）是前作，Jet-RL（Jan 2026）是进化版，同为 Song Han lab。

| | QeRL | Jet-RL |
|---|---|---|
| 量化精度 | NVFP4（4-bit 权重）| FP8（8-bit 权重+激活）|
| 训练策略 | LoRA 参数高效微调 | 全参数（rollout 和 train 统一精度）|
| 核心贡献 | 量化噪声 → 探索增益 | 量化引入 off-policy，统一 flow 修复 |
| 内存节省 | ~70%（vs BF16 LoRA）| ~40%（vs BF16 全参数）|
| 速度提升 | rollout 1.5×，E2E 1.8× | rollout +33%，train +41%，E2E +16% |
| 32B 可行性 | **单 H100 80GB 可训** | 需多卡 |

**演化路径**：QeRL 从"量化可以提升探索"出发（副作用变特性），Jet-RL 从"量化引入 off-policy 偏差"出发（修复副作用）。两者视角互补——QeRL 关注量化的好处，Jet-RL 关注量化的代价，并在系统层面统一解决。

---

## 核心发现：量化噪声 = 免费的探索增益

### 反直觉之处

传统观点：量化 → 精度损失 → 性能下降（SFT 中确实如此）  
QeRL 发现：在 RL 中，量化噪声 → entropy 增加 → 更好的探索 → 更快的 reward 增长

### 机制分析

量化误差 Δε = Q(W) - W 是一种参数空间噪声：

```
(θ̃ + θ_lora) - (θ + θ_lora) = Q(θ) - θ = Δε
```

这与 RL 中著名的 Parameter Space Noise (Plappert et al., 2017) 效果等价：
- 权重加噪 → logits 被扰动 → softmax 输出分布"变平" → entropy 增加
- 更高 entropy = 更高探索概率 = 更快发现 high-reward 路径

**实证**：NVFP4 量化模型的初始 entropy > BF16 模型；reward 曲线在 200 步内快速上升，而 BF16 LoRA 需 500+ 步才开始显著提升。

### 为什么 SFT 中量化有害而 RL 中有益？

- **SFT**：目标是模仿真实数据分布 → 噪声 = 偏离目标分布 = 有害
- **RL**：目标是探索高 reward 区域 → 噪声 = 鼓励偏离当前策略 = 有益

这是**问题结构**的差异，不是量化本身的优劣。

---

## 技术方案

### 量化格式选择

比较了三种 4-bit 格式：
- **NF4**（QLoRA）：lookup table 解包，慢（0.7-0.8× vs BF16）
- **MXFP4**：FP8(E8M0) 共享因子，block size 32
- **NVFP4**（QeRL 选择）：FP8(E4M3) 细粒度因子，block size 16，Marlin kernel 加速

NVFP4 反量化：
```
Ŵ = Dequant(W̃) = S_FP32 · (S_E4M3 ⊙ W̃)
```

关键：NVFP4 通过 Marlin kernel 在 H100 上实现硬件加速，比 NF4 快得多。

### Adaptive Quantization Noise (AQN)

**问题**：静态量化噪声在训练后期无法适应探索-利用 tradeoff  
**解决**：在 LayerNorm 参数中注入动态噪声，指数衰减调度

噪声向量 Z_noisy ~ N(0, σ²I)，注入 RMSNorm：
```
w_noise = Z_noise + w
RMSNorm_noise(x) = w_noise ⊙ x / RMS(x)
```

等价于对权重矩阵施加**行乘性噪声** Z_noise/w + I（乘性噪声在 RL 中已知比加性噪声更有效）。

噪声衰减：指数调度 σ(k) = σ_start · (σ_end/σ_start)^((k-1)/(K-1))
- 训练初期：高噪声 → 强探索
- 训练后期：低噪声 → 稳定利用

**技巧**：噪声向量直接合并到 LayerNorm 参数中，**零额外参数开销**（不需要额外的噪声层）。

---

## 实验结果

### GSM8K（7B Qwen2.5-Instruct，GRPO）

| Method | Precision | GSM8K | vs BF16 baseline |
|--------|-----------|-------|----------|
| BF16 Full FT | BF16 | 91.2 | +14.9 |
| BF16 LoRA | BF16 | 88.1 | +11.8 |
| **QeRL + AQN** | **NVFP4** | **90.8** | **+13.5** |
| QLoRA | NF4 | 85.0 | +8.7 |

**结论**：QeRL（NVFP4+AQN）以 ~40% 显存达到接近全参数训练的性能，并超越 BF16 LoRA。

### BigMath 跨模型规模（DAPO）

| Model | Method | Avg↑ |
|-------|--------|------|
| 7B | BF16 Full | 37.3 |
| 7B | BF16 LoRA | 35.7 |
| 7B | **QeRL+AQN** | **36.4** |
| 14B | BF16 Full | 43.3 |
| 14B | **QeRL+AQN** | **42.0** |
| 32B | BF16 Full | 46.2 |
| 32B | **QeRL+AQN** | **45.6** |

### 内存与速度（7B 模型，H100 80GB）

| Method | 显存 | E2E Speedup（batch=8）|
|--------|------|----------------------|
| BF16 LoRA | 15.2 GB | baseline |
| QLoRA (NF4) | 5.7 GB | **0.7×** （更慢！）|
| **QeRL (NVFP4)** | **5.9 GB** | **1.2-1.5×** |

**关键**：QLoRA 节省了内存，但因为 NF4 的 lookup table 开销反而更慢；QeRL 同时节省内存且更快。

### 里程碑：单卡 H100 训练 32B

QeRL 是**首个在单张 H100 80GB GPU 上完成 32B LLM RL 训练**的框架。

---

## 我的分析

### 最重要的洞察

**量化噪声在 SFT 和 RL 中产生截然相反的效果，原因在于优化目标的本质不同。**

SFT 是分布匹配问题（最小化 KL 散度），任何噪声都是干扰。  
RL 是探索优化问题（最大化 expected reward），适度噪声是助力。

这个区别以前没被认真对待过。QeRL 把它从"副作用"变成了可控的特性（AQN），这是真正有价值的贡献。

### 与 DEEP-GRPO 的隐藏联系

DEEP-GRPO（2602.14169）发现 GRPO 的 root sampling 导致探索不足（只在高概率区域采样）。QeRL 的量化噪声提供了一种互补的解法：**从 policy 参数层面增加熵，而不是改变采样策略**。

两者都在解决同一个问题（GRPO 探索不足），但路径完全不同：
- DEEP-GRPO：改变 **采样位置**（pivot-driven resampling）
- QeRL：改变 **采样分布**（量化噪声增加 entropy）

理论上可以组合——但需要小心 double counting（两者都增加探索，过度可能不稳定）。

### 与 Jet-RL 的深层对比

QeRL 和 Jet-RL 都是 Song Han 组做的，但解决了量化在 RL 中的**不同问题**：

- **QeRL 问题**：如何在 RL 中高效使用量化（找到对 RL 友好的量化格式 + 利用量化噪声）
- **Jet-RL 问题**：量化 rollout 精度和训练精度不一致导致 off-policy 偏差怎么办（统一 FP8 flow）

Jet-RL 引用了 "FlashRL"（即 QeRL 的直接竞争对手），但没有直接引用 QeRL——说明 Jet-RL 的出发点是 FlashRL 的问题（FP8/FP16 双精度带来的 off-policy），而 QeRL 更关注 FP4 的参数高效路径。

**演化关系假说**：QeRL → "嗯，量化有好处但也有 off-policy 问题" → Jet-RL 解决 off-policy → 未来可能有 QeRL-v2 = FP4量化 + on-policy 统一（Jet-RL 思路 + QeRL 探索增益）

### 局限性

1. **只测了数学推理**：GSM8K/MATH/AIME。对代码生成、Agent 任务的量化探索效应未验证
2. **LoRA 参数量限制**：QeRL 依赖 LoRA，参数更新受限，可能无法完全捕捉全参数训练的能力（在 AIME 这类难题上差距更明显）
3. **AQN 超参敏感**：σ_start/σ_end 的选择对结果影响较大，需要精调
4. **NF4 vs NVFP4**：NVFP4 需要 Hopper/Blackwell GPU 支持（H100+），工业落地有硬件依赖

---

## 技术连接

- [[AI/LLM/RL/Frameworks/Jet-RL-FP8-On-Policy-RL-Training|Jet-RL]] — 同组后作，FP8 精度 + on-policy 统一，解决量化 off-policy 问题；两篇合看：量化在 RL 中有益（QeRL）但须精度一致（Jet-RL）
- [[AI/LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — 同样解决 GRPO 探索不足，但从采样策略而非参数噪声角度
- [[AI/LLM/RL/Other-Algorithms/MASPO-Mass-Adaptive-Soft-Policy-Optimization|MASPO]] — 解决 GRPO 正负样本概率质量不平衡，与 QeRL 的探索方向正交
- [[AI/LLM/RL/Theory/RL-Training-Stability-2026-Unified-Analysis|RL 训练稳定性 2026 统一分析]] — 归属系统效率维度

---

## 总结

QeRL 的核心贡献是把"量化噪声在 RL 中促进探索"这个意外发现系统化，并设计了 AQN 将其变成可控机制。这不是 trick，是对 RL 探索本质的洞察。

实用价值：单 H100 训练 32B 模型的可行性，对资源受限团队意义重大。

在 Jet-RL（系统精度统一）、QeRL（量化探索增益）、DEEP-GRPO（采样策略探索）三篇的框架里，量化对 RL 的影响已经被从多个角度拆解清楚。这是一个值得综合的方向。
