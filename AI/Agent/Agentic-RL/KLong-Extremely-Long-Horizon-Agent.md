---
title: "KLong: Training LLM Agent for Extremely Long-Horizon Tasks"
type: paper
domain: ai/agent/agentic-rl
tags:
  - Agentic-RL
  - long-horizon
  - trajectory-splitting
  - curriculum-learning
  - SFT
  - PaperBench
  - RL
created: 2026-02-20
status: v1
---

# KLong: Training LLM Agent for Extremely Long-Horizon Tasks

**arXiv**: [2602.17547](https://arxiv.org/abs/2602.17547)  
**提交日期**: 2026-02-20  
**作者**: Yue Liu, Zhiyuan Hu, Flood Sung, Jiaheng Zhang, Bryan Hooi  
**机构**: NUS（新加坡国立大学）+ MIT  
**代码**: https://github.com/yueliu1999/KLong  
**标签**: `Agent` `长任务` `RL` `SFT` `PaperBench` `轨迹切割` `渐进式RL`

---

## 一句话

106B 开源 Agent 在 PaperBench 上以 62.59% 超越 Kimi K2 Thinking（1T）的 51.31%，核心方法：**轨迹切割 SFT（cold start）+ 渐进式 RL（scale up）**，解决超长任务（>12小时、>700 轮对话）的训练难题。

---

## 背景：极长任务 vs 长任务

| 维度 | 长任务（SWE-bench/Terminal-Bench） | 极长任务（MLE-bench/PaperBench） |
|------|----------------------------------|-------------------------------|
| 时间 | ~1小时 | **~12小时** |
| assistant 轮数 | 20~200 轮 | **~700+ 轮** |
| Context 管理 | 标准 context 可容纳 | **必须截断/切割** |
| RL 挑战 | 标准 sparse reward | **极稀疏 + 高方差 + credit assignment 困难** |

PaperBench 任务：给定论文，重现其代码实现。需要读论文、理解核心贡献、写代码、调试、验证——综合 ML engineering + software engineering + 长期规划。

---

## 方法

### 总体流程

```
Base Model
    ↓ 综合 SFT（通用知识、代码、数学、搜索）
Activated Base
    ↓ Research-Factory 生成数据 + 轨迹切割 SFT
Cold-Start Model
    ↓ 渐进式 RL（timeout 2h → 4h → 6h）
KLong（106B）
```

### Research-Factory：数据生成流水线

**目标**：自动为论文复现任务生成高质量训练数据。

**组件**：
1. **Search Agent**：从 ICML/NeurIPS/ICLR 近 5 年顶会爬取论文 + 元数据，按质量和影响力筛选
2. **Evaluation Agent**：分析论文 + 官方代码，自动构建 rubric 树（结构化评分标准）
3. **黑名单机制**：官方 GitHub URL 加黑名单，防止 agent 在训练/评估时作弊
4. **轨迹蒸馏**：用 Claude 4.5 Sonnet (Thinking) 作为 teacher，生成数千条极长任务轨迹

注：使用 open-source `gpt-oss-120b` 作为训练时 judge（而非官方的 o3-mini），防止 benchmark hacking。

### 轨迹切割 SFT（Trajectory-Splitting SFT）

**问题**：极长任务轨迹 τ=(s₁,a₁,...,sₙ,aₙ) 超出模型最大 context 长度 L_max。

**解法**：将长轨迹切分为 K 个**有重叠的**子轨迹：

$$\tau^{(i)} = (s_{t_i}, a_{t_i}, \ldots, s_{t_i+L-1}, a_{t_i+L-1})$$

**关键设计**：
1. **Pin prefix**：每个子轨迹都保留开头的论文阅读部分（全局信息），不被截断
2. **Progressive truncation**：越靠后的子轨迹，历史 context 截断越多
3. **Overlap**：相邻子轨迹有重叠 O，维持 contextual continuity

**效果**：assistant 轮数从 114.9 → **732.7**（+6.4倍），模型能处理超长任务序列。

### 渐进式 RL（Progressive RL）

**问题**：直接用 12小时超时做 RL，梯度极度稀疏（任务太难），信号太弱。

**解法**：定义逐渐增加的 timeout 序列 T⁽¹⁾ < T⁽²⁾ < ... < T⁽ᴹ⁾，分阶段训练：

$$\mathcal{L}_{RL}^{(m)}(\theta) = -\frac{1}{n \cdot K^{(m)}}\sum_{j,i,t} \min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t\right) + \beta\, \mathbb{E}_t[\text{KL}(\pi_\theta \| \pi_{\theta_{ref}})]$$

其中 $Q^{(m,i,j)} = \mathcal{J}(\hat{\mathcal{C}}^{m,i,j}, \mathcal{K})$ 是 judge 模型给出的奖励（基于 rubric 树）。

**阶段**：RL_2H（2小时timeout）→ RL_4H → RL_6H

**效果**：
- SFT only：55.92
- +RL_2H：57.29（+1.37）
- +RL_4H：58.65（+2.73）
- +RL_6H（KLong）：**62.59（+6.67 over SFT only）**

### 基础设施优化

- **Sandbox**：Kubernetes 管理，10,000+ 并发 Docker 实例，预装 80+ ML 包（torch/TF/sklearn/einops）
- **流水线不平衡问题**：所有 rollout 同时超时 → judge 拥塞，rollout 节点空闲。解法：partial rollout + 优先级 judge 队列
- **上下文长度错误处理**：改进 scaffolding，mandatory paper-reading tracking，prompt caching

---

## 实验结果

### PaperBench（12小时 timeout）

| 模型 | 参数量 | 平均分 |
|------|--------|--------|
| Claude 4.5 Sonnet (Thinking) | 闭源 | **69.75** |
| Grok 4 | 闭源 | 47.20 |
| GPT-5 Thinking (High) | 闭源 | 52.31 |
| Kimi K2 Thinking | 1T | 51.31 |
| DeepSeek-V3.2 | 685B | 47.11 |
| Qwen3-Thinking | 235B | 28.72 |
| **KLong** | **106B** | **62.59** |

**KLong 比 Kimi K2 Thinking (1T) 高 11.28%，参数量仅为其 10%**。

### 泛化性验证

- **SWE-bench Verified**（with OpenHands）：60.80%（SFT only）→ **62.80%（KLong）**
- **MLE-bench（low split）**：KLong 也显示出改进（具体数字在论文 Table 中）

---

## 我的分析

### 技术价值：★★★★☆

**KLong 的真正贡献不是结果，是方法论**：

**1. 轨迹切割 SFT 是解决极长任务的关键 insight**

论文的核心问题不是"如何训得更好"，而是"如何把 LM 的 context window（通常 32K-128K）和 700轮的实际交互轨迹（可能 1M+ tokens）对齐"。

轨迹切割 SFT 的三个设计原则：
- **Pin prefix**：全局信息（任务spec + 论文阅读）放在每个子序列开头 → 类似于长程任务中的"目标提醒"，防止模型忘记原始目标
- **Progressive truncation**：越后期的轨迹越稀疏 → 模型学会在不完整历史下继续执行
- **Overlap**：子轨迹重叠 → 连续性保证，避免"边界断裂"

**2. 渐进式 RL 本质上是 Curriculum Learning 在时间维度的应用**

timeout = 2h → 4h → 6h 是对任务难度的渐进式控制。这与 Goldilocks RL 的"样本难度 curriculum"是同一原则的不同实现：
- Goldilocks：在样本空间做 curriculum（选什么题）
- KLong：在时间空间做 curriculum（允许多长的任务）

**3. judge 模型设计很聪明**

用 `gpt-oss-120b` 而非 `o3-mini` 做训练 judge，防止 benchmark hacking（模型不是在直接优化最终评估器）。这是 RL reward hacking 问题的实践级解法。

**4. 106B 远超 1T 的 implication**

Kimi K2 Thinking 是 ~1T MoE，KLong 是 GLM-4.5-Air-Base 的 106B。11.28% 的性能优势不是来自模型规模，而是来自**专门的 agentic 训练方法**。这说明：

> 在 agentic 任务上，专门的训练范式（trajectory splitting + progressive RL + domain-specific data）比通用大模型更重要。

**5. 局限性**

- 只在 PaperBench 上做了专门训练（domain-specific curriculum），泛化到其他领域（非 ML/CS 类研究复现任务）未知
- 极长任务的训练代价极高（12小时/任务），样本效率仍是瓶颈
- judge 模型的质量直接影响 RL 信号——rubric 树的自动构建质量需要额外验证
- 基础模型用 GLM-4.5-Air-Base（智谱），换其他 base 的效果未知

---

## 与 Agentic RL 领域的关联

| 论文 | 解决的核心问题 | KLong 的关联 |
|------|--------------|------------|
| PA-MoE (2602.17038) | Phase-level routing for agentic RL | KLong 也有多阶段（RL_2H/4H/6H），但路由层面不同 |
| Goldilocks RL | 样本难度 curriculum | KLong 是任务时间长度 curriculum |
| DEEP-GRPO | Pivot-level exploration in reasoning | KLong 的 trajectory splitting 也涉及关键节点识别 |
| Calibrate-Then-Act | Cost-aware exploration | KLong 面对类似的成本权衡（长任务 vs 训练效率） |

**与 Agentic-RL-2026前沿综合分析 的关系**：KLong 是"训练 long-horizon agent"方向的实践案例，补充了 PA-MoE 等理论工作的实证端。

---

## 总结

KLong 的核心贡献是系统解决了**极长任务**（>12小时、>700轮）的训练基础设施问题，而非提出新的 RL 算法。轨迹切割 SFT 和渐进式 RL 是工程创新，但背后的洞察——"在长任务训练中，上下文管理和课程设计比算法选择更重要"——具有广泛的迁移价值。用 10% 参数超越最大开源竞争对手的结果，也为"专门化训练 > 通用规模"提供了新证据。

---

## 关键词连接

- [[AI/Agent/Agentic-RL/PA-MoE-Phase-Aware-Mixture-of-Experts|PA-MoE]] — 同为 Agentic RL 训练改进：KLong 解决极长 horizon 的上下文 + 课程问题，PA-MoE 解决参数容量 Simplicity Bias；两者正交可叠加
- [[AI/LLM/RL/Other-Algorithms/Goldilocks-RL-Task-Difficulty-Curriculum|Goldilocks RL]] ⭐ — **同一原则的不同实现**：Goldilocks 在样本难度做课程（选什么题），KLong 在时间长度做课程（允许多长的任务）；本质都是"从简单到难"的渐进式训练
- [[AI/LLM/RL/Other-Algorithms/DEEP-GRPO-Deep-Dense-Exploration-Pivot-Resampling|DEEP-GRPO]] — 轨迹切割（KLong）和 pivot resampling（DEEP-GRPO）都涉及在长轨迹的关键节点处理，思路相通
- [[AI/Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] — KLong 是极长任务 Agentic RL 训练的实证端，补充理论分析
- [[AI/Agent/Agentic-RL/Agent-RL-训练实战指南|Agent RL 训练实战指南]] — 实战指南的 credit assignment 和 reward 设计章节与 KLong 高度相关

*写于 2026-02-20 | 链接补全：2026-02-21 | 馆长*
