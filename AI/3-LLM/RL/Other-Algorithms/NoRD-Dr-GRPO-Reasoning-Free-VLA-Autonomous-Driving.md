---
title: "NoRD: 无推理 VLA + Dr. GRPO 解决自动驾驶训练中的 Difficulty Bias"
brief: 自动驾驶 VLA 首次实证 GRPO difficulty bias 机制：弱 SFT 策略下高方差中等难度样本被 std 归一化系统性压制梯度，Dr. GRPO（去掉 std 归一化）将 PDM score 从 +0.67% → +11.68%；NoRD 架构：无推理标注 + 80k 数据（vs 212k+）达到 SOTA 竞争力；弱 SFT + 强 RL 在有 dense 验证器的任务上可行的实证。
type: paper-note
domain: ai/llm/rl
date: 2026-02-25
arxiv: "2602.21172"
venue: CVPR 2026
authors: Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan
affiliation: Applied Intuition + Texas A&M + UC Berkeley
rating: ★★★★☆
tags:
  - ai/llm/rl
  - grpo
  - dr-grpo
  - difficulty-bias
  - vla
  - autonomous-driving
  - rl-post-training
  - type/paper-note
sources:
  - arXiv:2602.21172 (Rawal, Gupta, Hu, Zhan; Applied Intuition + Texas A&M + UC Berkeley, CVPR 2026)
  - https://arxiv.org/abs/2602.21172
related:
  - "[[AI/3-LLM/RL/GRPO/Dr-GRPO-Unbiased-Optimization|Dr. GRPO（原始论文，COLM 2025）]]"
  - "[[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景]]"
  - "[[AI/3-LLM/RL/Other-Algorithms/SAPO-Soft-Adaptive-Policy-Optimization|SAPO（同解 advantage 问题，不同路径）]]"
  - "[[AI/3-LLM/RL/Other-Algorithms/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL（advantage 计算修改对比）]]"
---

# NoRD: 无推理 VLA + Dr. GRPO 解决自动驾驶训练中的 Difficulty Bias

> **arXiv**: 2602.21172 · **会议**: CVPR 2026 · **机构**: Applied Intuition + Texas A&M + UC Berkeley · **★★★★☆**

---

## TL;DR

在自动驾驶 VLA 上发现了 GRPO 的一个根本性失败模式：当 SFT 策略较弱（数据少、无推理标注）时，训练集中的中等难度样本具有高 intra-group reward variance，而 GRPO 的 std 归一化会系统性地抑制这些样本的梯度信号——即 **difficulty bias**。用 Dr. GRPO（去掉 std 归一化）替换后，性能从 +0.67% 跃升至 +11.68%，且仅用 <40% 的数据、零推理标注，达到 SOTA 竞争力。

---

## 动机：三个非扩展性成本

当前 reasoning-based VLA（如 AutoVLA）的训练范式：
1. 大规模驾驶数据集（200k+ 样本）
2. Chain-of-Thought 推理标注（Teacher model 生成）
3. SFT → RL 两阶段（GRPO 对齐 PDM/RFS 分数）

**三个问题**：
- 数据收集成本（场景多样性，长尾覆盖）
- 推理标注成本（teacher 生成 + 过滤）
- 推理 token 推理延迟（实时驾驶不可接受 CoT 延迟）

**核心假设**：推理可能是计划的 byproduct 而非因果决定因素（Reasoning-Planning Decoupling Hypothesis）。理论上，reasoning-free + 小数据 + 强 RL 应该可行。

---

## 关键发现：GRPO 在弱 SFT 策略上的系统性失败

### 实验设计

- 基础模型：Qwen-2.5VL-3B-Instruct
- SFT：仅用 80,000 NAVSIM 样本（vs. AutoVLA 的 212k+）、无推理标注
- RL：GRPO 优化 PDM score（Waymo/NAVSIM 综合驾驶质量指标）
- 结果：GRPO 带来 **+0.67%** 的 PDM 提升（从 76.66 → 77.18）

这是灾难性的——AutoVLA 用强 SFT + GRPO 能提升 9%。

### 诊断：Polarized Reward Distribution

对训练集每个样本做 8 次 rollout，分析 group-mean reward 和 group std 的关系：

```
group-mean PDM ≤ 0.15 → 低 std（极难样本，模型全失败，low mean/low var）
group-mean PDM ≥ 0.8  → 低 std（简单样本，模型全成功，high mean/low var）
group-mean PDM ∈ [0.2, 0.65] → 高 std（中等难度，成功率不稳定，high var）
```

**关键观察**：GRPO 训练过程中，PDM 分布的中间区域（[0.2, 0.65]）的样本密度**全程几乎不变**，只有低方差的高分区域（≥0.8）样本密度在增加。

### 机制：为什么 GRPO 忽略高方差样本？

GRPO 的 advantage 计算：

$$\hat{A}_{i,t}^{\text{GRPO}} = \frac{r(o_i|x) - \frac{1}{G}\sum_{j=1}^{G}r(o_j|x)}{\text{std}_{j=1,...,G}(r(o_j|x))}$$

当 group 内 std 很大（高方差区间）时，分母很大 → advantage 被大幅**压缩**。

反之，当 std 很小（低方差区间，即简单/极难样本）时，advantage 被**放大**。

**结论**：GRPO 实际上更新的几乎都是"easy samples"（高分低方差）和极端失败样本（低分低方差），而**中等难度的关键样本**（sharp turns、lane changes 等）被系统性地忽视了。

这在弱 SFT 策略上尤为致命——弱模型的 majority 样本都落在中等难度区间（高方差）。强 SFT 策略中，大部分样本已经接近成功，不存在此问题。

---

## 解决方案：Dr. GRPO

**Dr. GRPO 的修改极其简单**：去掉 std 归一化项。

$$\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_i|x) - \frac{1}{G}\sum_{j=1}^{G}r(o_i|x)$$

完整训练目标（加了 DAPO 风格非对称 clipping）：

$$L_{\text{DrGRPO}} = \sum_{t=1}^{|o_i|}\min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}\hat{A}^{\text{DrGRPO}},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}},\ 1-\epsilon_l,\ 1+\epsilon_h\right)\hat{A}^{\text{DrGRPO}}\right)$$

额外设计选择：
- **DAPO 风格非对称 clipping**：$\epsilon_h > \epsilon_l$，防止 entropy collapse
- **无 KL 散度正则化**（与 Liu et al. 2029 一致）

效果：Dr. GRPO 后，PDM 分布中的中间区域（[0.2, 0.65]）样本逐渐向高分区迁移，模型真正学会了急弯、换道等复杂驾驶行为。

---

## 实验结果

### NAVSIM（navtest subset）

| 模型 | 无推理数据 | 训练样本数 | PDM Score |
|------|-----------|-----------|-----------|
| AutoVLA | ✗ | 212k | 89.1 |
| RecogDrive | ✗ | 12.8M | 89.6 |
| NoRD-base | ✓ | 80k | 76.66 |
| NoRD-base + GRPO | ✓ | 80k | 77.18 (+0.67%) |
| **NoRD（Dr.GRPO）** | ✓ | 80k | **85.62 (+11.68%)** |
| NoRD-BoN（×6） | ✓ | 80k | 92.4（超越 AutoVLA-BoN 92.1）|

NoRD 是 NAVSIM 上**唯一**同时满足：无推理数据、无 LiDAR、仅 3 帧 RGB、Pareto 高效的 VLA。

### WaymoE2E

| 模型 | 无推理 | RFS ↑ | ADE@3 ↓ |
|------|--------|-------|---------|
| Poutine | ✗ | **7.986** | 1.2055 |
| HMVLM | ✗ | 7.736 | 1.3269 |
| **NoRD** | ✓ | **7.709** | **1.2504** |
| AutoVLA | ✗ | 7.556 | 1.3507 |

NoRD 在 WaymoE2E 上用 12,000 训练样本（vs. Poutine 的 204k，**17x 少**），RFS 仅差 0.277。ADE@3 超越所有带推理的方法。

---

## 效率对比

- **Token 效率**：NoRD 是最低 token 数的 VLA（无 CoT token）
- **推理延迟**：最低延迟（直接输出 trajectory tokens，无 reasoning 步骤）
- **数据效率**：NAVSIM 用 80k（vs. 210k+），WaymoE2E 用 12k（vs. 204k+）

---

## 核心洞察与批判性评价

### 洞察 1：Difficulty Bias 的通用性

Dr. GRPO 最初来自 LLM 数学推理领域（Liu et al., 2029）。NoRD 是首次在自动驾驶领域验证这个机制。

关键点：**difficulty bias 本质上是因为 GRPO 的 std 归一化对 reward variance 的处理不当**。这个问题在任何 reward 分布高度极化（bimodal 或 multimodal）的任务中都会出现——不只是数学推理或自动驾驶。

### 洞察 2：弱 SFT + 强 RL 的可行性条件

- ✅ 可行：reward signal 可靠（PDM score 计算确定性高）、任务可分解为 trajectory prediction
- ❌ 不一定可行：reward sparse 且 long-horizon（比如 web agent、open-ended 任务）

NoRD 成立的一个重要前提：PDM score 是 **dense** 且 **simulation-based** 的，每次 rollout 都能得到信号。对于 sparse reward 的 agent RL，弱 SFT + Dr. GRPO 能否同样奏效还需验证。

### 洞察 3：推理是否必要？

NoRD 明确表态："不是说 VLA 不能从语言推理中获益，而是高效 VLA 不**必须**依赖推理和大数据。"

这跟 [[AI/2-Agent/Agentic-RL/ERL-Experiential-Reinforcement-Learning|ERL（把反思内化到 base policy）]]的方向有趣对立：
- ERL：让推理可训练后消失（distill into base behavior）
- NoRD：一开始就不要推理

两者都在降低推理的**部署成本**，但路径不同。

### 局限性

1. 任务特定：k-disc trajectory tokenization 是 driving-specific，不通用
2. Dr. GRPO 仍不完美：对于极端高方差样本（真正的 out-of-distribution）效果仍有限
3. 没有和 iStar / SeeUPO 等 process reward 方法对比（这些方法也能提升对弱 SFT 策略的帮助）

---

## See Also

**Dr. GRPO 体系（difficulty bias 修复）**
- [[AI/3-LLM/RL/GRPO/Dr-GRPO-Unbiased-Optimization|Dr. GRPO（原始论文，COLM 2025）]] — NoRD 所采用的核心修复方案：去掉 std 归一化消除 difficulty bias；原始论文在 LLM 数学推理域验证，NoRD 首次在自动驾驶 VLA 跨域实证其通用性
- [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景 2026]] — NoRD 的 Dr. GRPO 属于全景「difficulty_debiasing」分支；与 DAPO 非对称 clipping 组合使用（NoRD 同时采用）

**同解 GRPO advantage 问题的不同路径（对比理解）**
- [[AI/3-LLM/RL/Other-Algorithms/SAPO-Soft-Adaptive-Policy-Optimization|SAPO（Qwen 团队）]] — 改 clip 函数形状（hard→sigmoid 软衰减），解决 advantage 过度截断；与 Dr. GRPO 正交：SAPO 改 clip，Dr. GRPO 改 std 归一化；两者理论上可叠加
- [[AI/3-LLM/RL/Other-Algorithms/OAPL-Off-Policy-RL-LLM-Reasoning|OAPL（Cornell+Databricks+Harvard）]] — 从 KL-reg RL 第一性原理推导 squared loss 替代 clip；与 Dr. GRPO 正交：OAPL 改 IS-ratio 处理，Dr. GRPO 改 std 归一化

**Rollout 质量维度（正交可叠加）**
- [[AI/2-Agent/Agentic-RL/TSR-Trajectory-Search-Rollouts-Multi-Turn-RL|TSR（ICML 2026，TU Munich）]] — 训练时树搜索选优 rollout；与 Dr. GRPO 正交：TSR 改 rollout 采样策略，Dr. GRPO 改 advantage 计算；两者组合是"好的 rollout + 好的更新信号"的完整方案

**VLA 与多模态 RL**
- [[AI/3-LLM/RL/Other-Algorithms/AT-RL-Anchor-Token-Reinforcement-Learning-Multimodal|AT-RL（视觉锚点）]] — 多模态 credit assignment，与 NoRD 的 advantage 计算修复正交；AT-RL 解决"哪些视觉 token 值得信用"，NoRD 解决"哪些样本难度应该有梯度"

---

## 推荐阅读

1. [原文（arXiv:2602.21172）](https://arxiv.org/abs/2602.21172) — CVPR 2026 完整实验
2. [[AI/3-LLM/RL/GRPO/Dr-GRPO-Unbiased-Optimization|Dr. GRPO（COLM 2025）]] — difficulty bias 的理论基础
3. [[AI/3-LLM/RL/Theory/GRPO-改进七维框架分析|GRPO 改进全景]] — NoRD 在 GRPO 改进谱系中的完整定位
4. [[AI/3-LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]] — 理解 std 归一化为何导致 difficulty bias 的数学基础

---

## GRPO 改进全景的新维度

**Difficulty Bias 分支**（`difficulty_debiasing`，之前未单独列出）：

| 方法 | 操作 | 适用场景 |
|------|------|---------|
| Dr. GRPO | 移除 std 归一化 | 弱 SFT + 高方差 reward |
| DAPO | 非对称 clipping + token-level policy entropy | entropy collapse |
| DEEP-GRPO | 无 Dr. GRPO 细节，但包含类似机制 | 通用 |

Dr. GRPO 是 GRPO 改进全景中**修改最小、效果最显著**的变体之一——一行改动，PDM +11%。
