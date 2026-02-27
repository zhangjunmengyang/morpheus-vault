---
title: "MTP + ConfAdapt: Multi-Token Prediction via Self-Distillation"
brief: "arXiv:2602.06019：把已有 pretrained autoregressive LLM 通过 Online Self-Distillation 转为 MTP 模型，不需要额外 draft model / verifier / 特殊推理代码；ConfAdapt 推理策略动态选择每步输出 token 数；GSM8K >3x 解码加速，精度损失 <5%。与 DeepSeek V3 MTP 头的关键区别：这是 post-hoc 转换，不是预训练阶段的设计。"
type: paper
date: 2026-02-28
tags:
  - inference
  - multi-token-prediction
  - speculative-decoding
  - self-distillation
  - inference-speedup
  - test-time-compute
sources:
  - "arXiv:2602.06019 | University of Maryland + JHU | Feb 2026"
verdict: "★★★★"
related:
  - "[[AI/3-LLM/Inference/vLLM|vLLM]]"
  - "[[AI/3-LLM/Inference/推理优化|推理优化]]"
  - "[[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek V3]]"
  - "[[AI/3-LLM/Inference/SNDT-Stitching-Noisy-Diffusion-Thoughts-Test-Time-Scaling|SNDT]]"
---

# MTP + ConfAdapt: Multi-Token Prediction via Self-Distillation

> arXiv:2602.06019 | Feb 2026 | ★★★★
> **一句话**：把现成的 autoregressive LLM 转成"每步能吐多个 token"的 MTP 模型，不改架构，不加辅助模型，GSM8K 上 >3x 推理加速。

---

## 一、问题与动机

### 1.1 推理加速的三条路线

| 路线 | 代表方案 | 代价 |
|------|---------|------|
| **Speculative Decoding** | Eagle / Medusa / EAGLE-2 | 需要额外 draft model，占用额外显存和计算 |
| **预训练时 MTP** | DeepSeek V3（预训练加 MTP 头）| 需要从头设计，无法迁移到已有模型 |
| **Post-hoc MTP**（本文）| Self-Distillation → MTP + ConfAdapt | 直接转换已有 checkpoint，无需额外组件 |

**本文的独特价值**：已经有大量 pretrained checkpoint，能否**不改架构、不加额外 verifier**，直接把它们变成更快的 MTP 模型？

### 1.2 关键洞察：confidence 与预测质量的相关性

论文发现：**一个 autoregressive 模型对下一个 token 的置信度，与其能正确预测多步 token 的能力高度相关**。

- 高置信度（top token 概率高）→ 当前位置的预测空间收窄 → 下几步也更可预测
- 低置信度（top token 概率低）→ 多步预测不可靠，只能老实用单步

这个洞察直接导出了 ConfAdapt 策略的设计。

---

## 二、方法

### 2.1 Online Self-Distillation（训练阶段）

**目标**：让 model 学会"在置信度高时，直接预测后续 k 个 token"。

**训练流程**：
```
1. 对训练文本，前向计算得到 next-token 分布
2. 计算 top-token 概率（confidence score）
3. 高 confidence 位置：用 self-distillation 训练 k-step ahead prediction
   - Teacher：原始 next-token prediction（ground truth 或自回归采样）
   - Student：同一模型在当前位置预测 +1, +2, ..., +k token
4. 低 confidence 位置：正常 next-token loss，不加 MTP 目标
```

**关键**：使用 **online** distillation——teacher 就是模型自身（不是外部 teacher），不需要第二个模型。

训练后，模型**保持完全相同的架构和权重格式**——只是权重被更新了，让它学会了"在高置信时多步预测"的能力。

### 2.2 ConfAdapt（推理阶段）

**动态决定每步输出多少 token**：

```python
for each decoding step:
    logits = model.forward(context)
    top_prob = softmax(logits).max()  # 置信度
    
    if top_prob > threshold_k:        # 高置信
        output k tokens directly
        context = context + k_tokens
    elif top_prob > threshold_2:      # 中等置信
        output 2 tokens
        ...
    else:                             # 低置信
        output 1 token（安全）
```

**confidence threshold** 是可调超参数，论文在 threshold 上做了 sweep，找到精度-速度的 Pareto 最优点。

**与 speculative decoding 的本质区别**：
- Speculative decoding：draft model 生成候选 → target model 验证接受/拒绝（需要两个模型）
- ConfAdapt：同一个模型自主决定是否多步输出，**无验证步骤**，accept 率≠100% 但接近

---

## 三、实验结果

### 3.1 主要结果（GSM8K）

- **>3x 平均解码速度**（相比 greedy single-token decoding）
- **精度损失 <5%**（相比 baseline）
- 在 confidence threshold sweep 中，可以选择 Pareto 最优点（高速度 or 高精度）

### 3.2 与 Static 方案的对比

论文对比了两种方案：
- **Static k**：固定每步输出 k 个 token（k=2 或 k=3）
- **ConfAdapt**：动态选择 k

结果：ConfAdapt 在相同平均 k 值下，精度显著高于 Static——因为它在低置信位置自动回退到 k=1，避免了错误积累。

### 3.3 Coverage

评测了多个 benchmark（GSM8K / HumanEval 等）和多个模型规模，结果一致。

---

## 四、与 DeepSeek V3 MTP 的区别

| 维度 | DeepSeek V3 MTP | 本文（Self-Distillation MTP） |
|------|----------------|------------------------------|
| **阶段** | 预训练时设计 | Post-hoc 转换已有 checkpoint |
| **架构变化** | 额外 MTP 预测头（独立参数） | 无架构变化（只更新现有权重） |
| **目标** | 辅助训练（主要目标是更好表征） | 直接推理加速 |
| **推理时** | 用 MTP 头做 speculative decoding | ConfAdapt 直接多步输出 |
| **适用范围** | 需要从头训练 / 很大规模 | 可应用于任何已有 checkpoint |

**核心区分**：DeepSeek V3 的 MTP 头是**训练辅助**，不直接用于推理加速（推理时仍用 speculative decoding 的逻辑）；本文的 MTP 是**直接推理机制**，通过 confidence 自适应地多步输出。

---

## 五、批判性评估

### 优点
- **架构无侵入性**：不改变 checkpoint 格式，工程上最容易部署
- **无额外模型**：比 speculative decoding 的内存开销低，单卡可用
- **自适应**：ConfAdapt 的动态策略比固定 k 更鲁棒

### 局限
- **精度损失 <5% 但不为零**：对精度敏感的任务（数学证明、代码生成）可能不可接受
- **Self-Distillation 质量依赖原模型**：如果原模型的 confidence calibration 差，ConfAdapt 的效果会显著下降
- **训练成本**：虽然不需要额外模型，但 online distillation 需要额外的 fine-tuning（几百到数千步）
- **与 Reasoning 的兼容性未知**：推理模型（DeepSeek-R1 类）的 long CoT 步骤置信度分布可能和普通模型很不同

### 工程判断

对**部署端推理加速**（不需要从头训练、预算有限的场景），MTP + ConfAdapt 是一个非常实用的工具箱。适用场景：已有 SFT/RLHF 后的 checkpoint，需要降低推理延迟但无法重训。

---

## See Also

- [[AI/3-LLM/Inference/推理优化|推理优化]] — 推理优化全景：Speculative Decoding / PagedKV / Chunked Prefill / PD Disaggregation
- [[AI/4-模型/DeepSeek/DeepSeek-R1|DeepSeek V3]] — MTP 头的预训练阶段设计，与本文的 post-hoc 转换对比
- [[AI/3-LLM/Inference/vLLM|vLLM]] — v0.16 的 Unified Parallel Drafting 与 MTP 类方案的兼容性
- [[AI/3-LLM/Inference/SNDT-Stitching-Noisy-Diffusion-Thoughts-Test-Time-Scaling|SNDT]] — 另一个"多路径利用"的推理加速方向，dLLM 场景

*写作时间：2026-02-28 07:20 | arXiv:2602.06019 | ★★★★*
