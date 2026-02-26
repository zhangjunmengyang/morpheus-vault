---
brief: "ProGRPO（arXiv:2602.05281）——用概率论重加权替代 GRPO 的硬 clip，缓解 entropy collapse；引入温度参数控制策略多样性，在 RLVR 数学推理任务上提升推理多样性 15%+，同时保持准确率。"
title: "ProGRPO: Probabilistic Advantage Reweighting for GRPO"
date: 2026-02-05
tags: [GRPO改进, 策略优化, entropy-collapse, RLVR, 推理多样性]
domain: AI/LLM/RL/GRPO
arxiv: "2602.05281"
rating: 4
status: permanent
---

# ProGRPO: Probabilistic Group Relative Policy Optimization

> arXiv: 2602.05281 | 提交: 2026-02-05 (v2: 2026-02-06)
> 作者: Pengyi Li et al.
> 机构: 未标注（cs.LG + cs.CL）

## 评分: ★★★★☆

## 一句话

GRPO 的 entropy collapse 不是 bug 而是结构性缺陷——标准 advantage 把最高概率路径越训越强，合理但低频的推理链被压死。ProGRPO 在 advantage 层面引入概率置信度重加权，用模型自身的概率信号来「扶弱抑强」，Pass@32 比 GRPO 高 13.9%。

---

## 问题诊断：GRPO 为什么 entropy collapse？

标准 GRPO advantage 公式：

```
A_i = (R_i - mean(R)) / std(R)
```

问题出在这里：**advantage 只依赖 reward，与该路径的生成概率无关**。

后果：
- 高概率路径（dominant solution）每次被采样，每次正确，每次得正 advantage，梯度持续强化它
- 低概率路径即使正确，被采样概率低，更新机会少，概率越来越低
- 训练收敛后：pass@1 还行（只走概率最高路径），pass@k 暴跌（多样性死了）

这是 **reward-weighted likelihood maximization 的内在矛盾**：目标函数最大化期望 reward，最优解是把所有概率质量堆到一条最优路径上。但多步推理问题往往有多条等效解，压死它们是过拟合。

---

## 方法：ARM (Advantage Re-weighting Mechanism)

### 核心 idea

引入两个置信度信号：
- `c_θ(q)` — **Prompt 置信度**：模型对该问题有多熟悉？（问题难不难）
- `c_θ(o|q)` — **Answer 置信度**：模型生成这个答案有多自信？（路径有多「主流」）

重加权公式：

```
Ã_i = A_i + α * (c_θ(q) - c_θ(o_i|q))
```

直觉解读：
- 如果 `c_θ(o|q)` 很高（模型很确信这条路径）→ `c_θ(q) - c_θ(o|q)` 为负 → advantage 打折扣 → 减弱对 dominant solution 的强化
- 如果 `c_θ(o|q)` 很低（模型对这条路径不自信，低频但正确）→ `c_θ(q) - c_θ(o|q)` 为正 → advantage 加权 → 鼓励探索 under-explored 路径
- `c_θ(q)` 作为难度调节器：问题很难（c 低）时，整体重加权幅度收窄，避免对 hard sample 过度探索

### 关键细节：Low-Probability Token Length Normalization

计算置信度时，**不对所有 token 做长度归一化**，只对低概率 token（约 20%）归一化。

原因：推理过程中约 80% 的 token 是「trivial」的（高置信，top-1 概率 > 0.9，比如连接词、推理套语）。如果全部平均，有效信号被稀释。只取低概率 token，真实反映模型的「不确定时刻」。

```python
T_low = top-20% most uncertain token positions in o_i
c_θ(o|q) = exp(mean_{t ∈ T_low} log p_θ(o_t | q, o_{<t}))
```

### ARM 激活条件

- 若 group 全对（所有 r=1）或全错（所有 r=0）：不激活 ARM，用原始 A_i（避免无意义重加权）
- 只在有对有错时激活：此时 ARM 的差异化信号才有意义

### 完整目标函数

```
J_ProGRPO(θ) = E[1/Σ|o_i| * ΣΣ min(r_{i,t} * Ã_i, clip(r_{i,t}, 1-ε_low, 1+ε_high) * Ã_i)]
```

注：用了 asymmetric clip (DAPO 风格)，ε_low ≠ ε_high，clip range [0.8, 1.28]。

---

## 实验结果

### 数学推理（Qwen2.5-7B，Pass@1 / Pass@32 平均）

| Method | Pass@1 | Pass@32 |
|--------|--------|---------|
| GRPO   | 37.6%  | 54.7%   |
| FlowRL | 35.3%  | 61.0%   |
| **ProGRPO** | **43.3%** | **68.5%** |

Pass@32 差距更大（+13.8% vs GRPO）—— 恰好印证多样性的改善。

### 代码生成（CodeForces rating）

| Method | Rating |
|--------|--------|
| GRPO   | ~1242  |
| FlowRL | ~1129  |
| **ProGRPO** | **1422** |

+180 rating over GRPO，+293 over FlowRL。

### Qwen2.5-32B

ProGRPO Pass@1 = 52.7%，+4.8% over GRPO——大模型同样有效。

### DeepSeek-R1-Distill-Qwen-1.5B

Pass@1 49.4% → 58.3%，+8.9%——小模型效果更显著。

### OOD 泛化（GPQA / MMLU-Pro）

同样优于 GRPO，表明 diversity 改善不以牺牲泛化为代价。

---

## Entropy 动态

训练中 entropy 的变化：
1. **先降**（早期：快速学习最优路径，entropy 自然下降）
2. **后升**（中期：ARM 重加权开始起效，低频路径被鼓励探索）
3. **稳定**（后期：在更高 entropy 水平收敛）

这与 GRPO 单调下降 entropy 形成对比。

---

## 我的分析

### 为什么这个问题重要？

Pass@1 vs Pass@k 的 gap 是 real-world LLM 应用的核心问题：
- Agent 场景：第一次失败可以 retry，需要多样性
- Best-of-N 推理：N 大时，多样性决定上限
- 科研助手：覆盖多种推理路径才能发现 edge case

### 技术上有多新颖？

- **核心 idea 不复杂**：advantage 里加个置信度调节项
- **真正 novel 的是 insight**：识别 entropy collapse 根源在 advantage 而非 reward，并从概率信号内部解决
- 和 entropy regularization 方法（加 entropy bonus）的区别：entropy bonus 是外部强制多样性，ARM 是内部调整，更 principled

### 潜在问题

1. **α 超参敏感性**：ARM 的强度 α 需要调，过大可能损害 Pass@1
2. **Low-prob token 选取**：20% 阈值是如何确定的？论文未详细 ablation
3. **计算开销**：需要额外计算 c_θ(q) 和 c_θ(o|q)，overhead 约 10-15%
4. **动态效果**：在 easy task 上（group 全对概率高），ARM 基本不激活，退化为标准 GRPO

### 与其他 GRPO 改进的对比

| 方法 | 解决 entropy collapse？ | 机制 |
|------|------------------------|------|
| Clip-Higher (DAPO) | 部分 | 不对称 clip，间接鼓励低概率 |
| Entropy Regularization | 是，但粗暴 | 加 H(π) 项，全局强制 |
| FlowRL | 否 | 关注 reward matching |
| VESPO | 部分 | 序列级分布约束 |
| **ProGRPO** | 是，精细 | 概率信号调节 advantage |

ProGRPO 的优势：直接在 advantage 层面手术，不改变整体 objective 结构，干扰最小。

---

## 连接

- 前驱：GRPO（Shao 2024）、DAPO（Yu 2025，asymmetric clip）
- 竞争：Entropy Reg.（Wang 2025）、FlowRL、VESPO
- 应用方向：Best-of-N 推理、科研 agent、代码生成

## Tags
`#GRPO` `#entropy-collapse` `#policy-optimization` `#advantage-reweighting` `#diversity` `#RL-training` `#2026-02`

---

## See Also

- [[GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — ProGRPO 在七维框架的 Diversity 维度
- [[RePO-Rephrasing-Policy-Optimization|RePO]] — 同在 Diversity 维度，两者互补
- [[GRPO 深度理解|GRPO 深度理解]] — ProGRPO 的算法基础
- [[AI/3-LLM/RL/目录|RL MOC]] — LLM 强化学习全图谱
