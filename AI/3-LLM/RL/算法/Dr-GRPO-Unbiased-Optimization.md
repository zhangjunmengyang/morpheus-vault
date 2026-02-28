---
title: "Dr. GRPO: Unbiased Optimization — 修复 GRPO 的 Length Bias 和 Difficulty Bias"
brief: Dr. GRPO 发现 GRPO 存在两个系统性偏差：length bias（错误回答会被优化得更长）和 difficulty bias（简单问题的 std 接近 0 导致 advantage 爆炸）。修复只需去掉两个 normalization 项，COLM 2025，AIME 2024 43.3%。
date: 2026-02-23
type: paper-note
tags:
  - grpo
  - bias-correction
  - optimization
  - rlvr
  - reasoning
rating: ★★★★☆
sources:
  - arXiv:2503.20783 (Zichen Liu et al., Sea AI Lab, COLM 2025)
  - https://github.com/sail-sg/understand-r1-zero
venue: COLM 2025
related:
  - "[[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]]"
  - "[[AI/3-LLM/RL/算法/REINFORCE-Plus-Plus-Global-Advantage-Normalization|REINFORCE++]]"
  - "[[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO-Improvement-Panorama-2026]]"
  - "[[AI/3-LLM/RL/算法/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]]"
---

# Dr. GRPO: 修复 GRPO 的两个系统性偏差

> **一句话**：GRPO 在实现上存在两个 bug 级别的偏差，导致错误回答越训练越长、简单题的梯度爆炸。Dr. GRPO 用两行代码修复，并额外发现 Qwen2.5 base 不需要 prompt template 就有强推理能力。

## GRPO 的两个偏差

### 偏差 1：Length Bias（长度偏差）

GRPO 的 loss 按 response length 归一化：

$$\mathcal{L}_{\text{GRPO}} = \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left(s_t(\theta) A_t, \text{clip}(s_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)$$

**问题**：对于 **错误回答**（$A_t < 0$），length normalization 会让模型倾向于生成**更长的错误回答**——因为更长意味着每 token 的负损失更小（分母更大），模型发现"错得更啰嗦"可以减小 loss。

**实证**：训练过程中，incorrect responses 的长度比 correct responses 增长更快——这完全是优化目标的 artifact，不是泛化能力的提升。

**修复**：去掉 length normalization，直接对所有 token 的 loss 求和（不除以 $|o|$）。

### 偏差 2：Difficulty Bias（难度偏差）

GRPO 的 advantage 用 group 内 std 归一化：

$$A_t = \frac{R_i - \text{mean}_{\text{group}}}{\text{std}_{\text{group}} + \epsilon}$$

**问题**：对于极难或极简单的题（group 内所有回答都对或都错），$\text{std}_{\text{group}} \approx 0$，advantage 爆炸（被 $\epsilon$ 救活但数值极大）。这些题的梯度噪声会掩盖有意义 prompt 的信号。

**修复**：去掉 std 归一化，只做 mean-centering：

$$A_t = R_i - \text{mean}_{\text{group}}$$

## Dr. GRPO 的改动（两行）

```python
# GRPO (原始)
advantage = (reward - mean_group) / (std_group + eps)
loss = -advantage * log_prob / response_length  # length normalization

# Dr. GRPO (修复后)
advantage = reward - mean_group              # 去掉 std normalization
loss = -advantage * log_prob                 # 去掉 length normalization
```

就这两行。

## 与 PPO/REINFORCE++ 的关系

这个发现和 REINFORCE++ 的发现部分重叠：
- REINFORCE++ 发现 group-level std normalization 导致不稳定 → 改为 global batch normalization
- Dr. GRPO 发现 std normalization 直接导致极难/极易题的 advantage 爆炸 → 直接去掉

两者都是去掉 std，但出发角度不同（REINFORCE++ 换成 global；Dr. GRPO 直接删掉）。

## 副产品发现

论文还分析了 base model 对 R1-Zero-like training 的影响：

- **DeepSeek-V3-Base 已经有 "Aha moment"**：不用 RL 就已经有一定推理能力（pretraining 中学到）
- **Qwen2.5 base 不需要 prompt template**：直接输入问题就能出 CoT 格式，说明 Qwen2.5 的 pretraining 数据对 RL 非常友好
- **PPO 的实现也有 length bias**：检查了多个主流开源 PPO 实现，发现都有 loss/response_length 归一化，这是普遍存在的实现错误

## 实验结果

7B base model + Dr. GRPO + 极简 R1-Zero recipe：
- AIME 2024：**43.3%**（与 ToRL-7B 持平，是当时 7B SOTA）
- 同时 response length 更短（token efficiency 提升）

## 关键洞察

**Bug 还是 feature？** Length bias 在某些情况下不一定是坏事——更长的思维链有时确实更准确。但 Dr. GRPO 证明这种"更长"主要是对错误回答发生的，是 optimization artifact，不是能力提升。

**Difficulty bias 的危害**：极难题（全错）/ 极易题（全对）的梯度方向是噪声，不是信号——这些 prompt 应该被 curriculum 过滤掉，而不是因为 std ≈ 0 就放大梯度。这与 Goldilocks / PACED-RL 的 difficulty curriculum 思路是同一问题的不同解法。

## 落地应用

**直接 drop-in 替换 GRPO：** 任何用 GRPO 的地方，去掉两个 normalization 即可得到 Dr. GRPO。

**什么时候特别重要：**
- 训练数据难度分布不均（混合了很难和很简单的题）
- 发现 incorrect responses 越来越长时（length bias 症状）
- Response length 无法下降（过度惩罚短错误导致模型学习 workaround）

**面试问法：**
- "GRPO 有什么已知问题？" → length bias + difficulty bias，Dr. GRPO 修复
- "为什么 GRPO 的错误回答会越来越长？" → length normalization 让更长的错误 loss 更小
- "GRPO 实现中 std normalization 有什么问题？" → 极难/易题的 std ≈ 0，advantage 爆炸

## 推荐阅读

- 原论文：[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)（COLM 2025）
- 代码：[sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)
- 相关修复：[[AI/3-LLM/RL/算法/REINFORCE-Plus-Plus-Global-Advantage-Normalization|REINFORCE++]]（不同角度的同类修复）
- GRPO 基础：[[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]]
- 全景：[[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO-Improvement-Panorama-2026]]

---

## See Also

### GRPO 修复谱系（同维度算法对照）
- [[AI/3-LLM/RL/算法/REINFORCE-Plus-Plus-Global-Advantage-Normalization|REINFORCE++]] — 同类修复但角度不同：REINFORCE++ 用全局 batch 归一化解决 advantage 估计问题；Dr. GRPO 直接对 loss 函数去偏（length/difficulty 两个维度）；两者可组合
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景]] — Dr. GRPO 在七维改进框架中属于"优化偏差修正"维度，全景视角定位
- [[AI/3-LLM/RL/算法/GRPO 深度理解|GRPO 深度理解]] — 理论基础：理解 Dr. GRPO 修复的前提是理解原版 GRPO 的 advantage 计算缺陷

### 相关偏差问题
- [[AI/3-LLM/RL/实践/DAPO-verl实践|DAPO]] — 同为 GRPO 稳定性修复，DAPO 解决 entropy collapse 和 clip 不对称；Dr. GRPO 解决 length/difficulty bias；两者互补（熵稳定 + 梯度去偏）
- [[AI/2-Agent/Agentic-RL/SeeUPO-Sequence-Level-Agentic-RL-Convergence-Guarantees|SeeUPO]] — ⚠️ 理论边界：GRPO（含 Dr. GRPO）的修复在单轮场景有效；multi-turn agent 训练需要 SeeUPO 逆序更新框架

### 跨域实证（difficulty bias 的通用性验证）
- [[AI/3-LLM/RL/算法/NoRD-Dr-GRPO-Reasoning-Free-VLA-Autonomous-Driving|NoRD（CVPR 2026，Applied Intuition + UC Berkeley）]] — **首次在自动驾驶 VLA 域验证 difficulty bias 机制**：弱 SFT + high variance 中等难度样本被系统压制，Dr. GRPO 修复后 PDM score +11.68%（vs GRPO +0.67%）；确认 difficulty bias 不是 LLM 推理特有问题，而是"reward 分布极化 + std 归一化"的通用组合失效
