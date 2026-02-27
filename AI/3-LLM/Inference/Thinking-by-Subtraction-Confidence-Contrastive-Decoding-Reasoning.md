---
title: "Thinking by Subtraction: Confidence-Driven Contrastive Decoding for LLM Reasoning"
brief: "arXiv:2602.18232：推理不确定性是局部化的——少量低置信度 token 贡献了大多数推理错误。Thinking by Subtraction 构建 contrastive reference（高置信 token 替换为 placeholder），仅在低置信位置做减法精炼预测。相比 best-of-N 多路径，单条轨迹质量提升，compute 不增加。"
type: paper
date: 2026-02-28
tags:
  - inference
  - contrastive-decoding
  - test-time-compute
  - reasoning
  - confidence
  - LLM-reasoning
sources:
  - "arXiv:2602.18232 | Lexiang Tang 等 | Feb 2026"
verdict: "★★★★"
related:
  - "[[AI/3-LLM/Inference/推理优化|推理优化]]"
  - "[[AI/3-LLM/Inference/MTP-ConfAdapt-Self-Distillation-Inference-Speedup|MTP+ConfAdapt]]"
  - "[[AI/3-LLM/Inference/SNDT-Stitching-Noisy-Diffusion-Thoughts-Test-Time-Scaling|SNDT]]"
---

# Thinking by Subtraction

> arXiv:2602.18232 | Feb 2026 | ★★★★
> **一句话**：推理不确定性是局部化的——只在低置信度 token 处做 contrastive 减法，单条轨迹质量提升，不需要 best-of-N 的多倍计算开销。

---

## 一、核心洞察：推理不确定性的局部化分布

### 1.1 打破 test-time scaling 的均匀假设

**主流假设**：更多推理 compute（更长 CoT、更多 rollout）均匀地提升正确率。

**实证发现**（论文对 2880 条推理轨迹的分析）：
- 推理错误**不是均匀分布**在整个 chain-of-thought 里
- 只有**一小部分低置信度 token** 造成了大多数推理错误
- 这些 token 同时也是"不必要的输出膨胀"的来源（reasoning models 的 token 浪费问题）

**推论**：如果能精确定位这些低置信度位置并在那里做精炼，可以在**不增加全局 compute** 的情况下提升质量。

### 1.2 与现有方法的对比

| 方法 | 思路 | compute 开销 |
|------|------|-------------|
| Best-of-N | 生成 N 条轨迹取最优 | N× |
| Tree Search（MCTS/beam） | 在关键节点分支 | 2-10× |
| Self-Consistency | 多数投票 | N× |
| **Thinking by Subtraction** | 单轨迹内精炼低置信位置 | **≈1×**（略多） |

---

## 二、方法：置信度驱动的 Contrastive Decoding

### 2.1 Contrastive Decoding 背景

原始 CD（Li et al., 2022）：用强模型分布减去弱模型分布，消除弱模型的"常规但错误"的预测偏好。

$$p_{CD}(y_t) \propto p_{strong}(y_t) - \alpha \cdot p_{weak}(y_t)$$

**问题**：全局减法对高置信位置（本来就对的步骤）没有帮助，反而可能引入噪声。

### 2.2 Thinking by Subtraction 的关键改进

**步骤 1：置信度检测**

对每个 token 位置 $t$，计算置信度：

$$\text{conf}(t) = \max_v p_\theta(v | y_{<t}, x)$$

低置信位置集合：$\mathcal{L} = \{t : \text{conf}(t) < \tau\}$

**步骤 2：构建 Contrastive Reference**

对原始推理序列，把**高置信 token** 替换为 minimal placeholder（`[MASK]` 或 `_`），保留低置信位置的 context。这得到一个"缺失了确定性信息"的退化版本。

直觉：高置信 token 代表"已经确定的推理步骤"，把它们 mask 掉后，模型在低置信位置的预测只能依赖"不确定的信息"——这个退化分布代表了"无效推理路径"。

**步骤 3：选择性减法**

只在低置信位置 $t \in \mathcal{L}$ 处应用减法：

$$\log p_{refined}(y_t) = \log p_{original}(y_t) - \alpha \cdot \log p_{contrast}(y_t)$$

高置信位置（$t \notin \mathcal{L}$）：直接用 original 分布，不做任何修改。

**关键：** 减法只在"需要帮助"的位置激活，不干扰已经正确的步骤。

### 2.3 为什么这能 work？

**直觉解释**：

低置信位置的预测困难 = 模型在"高不确定推理空间"中随机游走。Contrastive reference 捕捉了这种"随机游走的先验"，减去它相当于：

$$p_{refined} \propto \frac{p_{original}}{p_{contrast}^\alpha}$$

即放大了"有确定性信息支持下的差异预测"，抑制了"仅靠模糊 context 的通用预测"。

**和经典 CD 的本质区别**：不是"强模型 vs 弱模型"，而是"完整 context 的模型 vs 缺失确定步骤的模型"——都是同一个模型，difference 来自 context 质量不同。

---

## 三、实验结果

- **基准**：MATH500、AIME 等数学推理；Code 推理 benchmark
- **相比 greedy 解码**：准确率提升，token 数量相近或降低（减少了 token 膨胀）
- **相比 best-of-N**：在相同 compute budget 下，单轨迹精炼 ≥ 多轨迹投票的提升量
- **Thinking-Answer 一致性**：DAR 发现的 format hacking 问题（thinking 和 answer 不一致）在这里也有缓解——减法精炼使 reasoning 更连贯

---

## 四、批判性评估

### 优点

- **实证洞察坚实**：2880 条轨迹分析，"不确定性局部化"有充分数据支撑
- **单轨迹效率**：不需要 N× compute，实用性高（特别是 latency-sensitive 场景）
- **无训练**：纯推理时方法，直接应用于任何现有模型
- **理论自洽**：CD 框架有理论基础，选择性激活的设计合理

### 局限

- **τ 阈值敏感性**：低置信 threshold 如何选？论文是否给了鲁棒的调参建议？
- **额外前向计算**：构建 contrastive reference 需要额外的 forward pass（约 1× 额外 compute），不是完全免费
- **placeholder 设计**：高置信 token 替换成什么 placeholder 影响 contrast 质量，这个设计选择的分析需要验证
- **推理 vs 事实性**：对事实性问题（需要从记忆中提取，不是推导），方法是否适用存疑

### 工程判断

**适用场景**：reasoning model（R1/QwQ/DeepSeek-R1-32B 类）的推理服务。这类模型本来就有 token 膨胀问题（长 CoT 里很多废话），选择性减法可以同时提升准确率和降低输出长度。

**不适用**：需要确定性输出的场景（代码生成的确定性比推理更重要）；或者本来置信度就很高的任务（减法没有目标位置）。

---

## 五、与其他方法的关系

### 5.1 在 Test-Time Compute 方法谱系中的位置

```
全局多路径（高 compute）：
  Best-of-N → 取最高 reward 轨迹
  MCTS/Tree Search → 动态分支
  SC → 多数投票
单轨迹精炼（低 compute）：
  ★ Thinking by Subtraction → 低置信位置 contrastive 减法
  MTP+ConfAdapt → 高置信位置多步输出（token 层面）
  VeriThinker → 验证减少推理长度
```

### 5.2 与 MTP+ConfAdapt 的互补关系

两篇论文都基于"置信度"做出决策，但方向相反：

- **MTP+ConfAdapt（2602.06019）**：高置信 → 输出更多 token（加速）
- **Thinking by Subtraction（2602.18232）**：低置信 → 精炼预测（提质）

组合可能性：先用 ConfAdapt 在高置信处快速输出，再用 Thinking by Subtraction 在低置信处精炼——两者正交，可能在 compute budget 内同时提升速度和质量。

### 5.3 与 SNDT 的对比

**SNDT（2602.22871）**：跨多条 dLLM 轨迹 stitch 最优步骤（多路径→拼接）

**Thinking by Subtraction**：单条轨迹内精炼（单路径→精炼）

适用场景互补：compute 充足用 SNDT，compute 受限用 Thinking by Subtraction。

---

## See Also

- [[AI/3-LLM/Inference/推理优化|推理优化]] — 推理优化全景，test-time compute 方向的完整图谱
- [[AI/3-LLM/Inference/MTP-ConfAdapt-Self-Distillation-Inference-Speedup|MTP+ConfAdapt]] — 高置信位置加速（与本文低置信精炼正交互补）
- [[AI/3-LLM/Inference/SNDT-Stitching-Noisy-Diffusion-Thoughts-Test-Time-Scaling|SNDT]] — 多轨迹 stitch，compute 充足场景的对应方案
- [[AI/3-LLM/RL/算法/DAR-Dual-Regularized-Advantage-Regression-Unifying-RLHF|DAR]] — Thinking-Answer 不一致（format hacking）和本文发现的 token 膨胀问题同源

*写作时间：2026-02-28 08:12 | arXiv:2602.18232 | ★★★★*
