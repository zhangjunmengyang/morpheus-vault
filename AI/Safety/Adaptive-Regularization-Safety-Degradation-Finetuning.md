---
title: "Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning"
date: 2026-02-19
arxiv: "2602.17546"
domain: AI/Safety
tags:
  - safety-alignment
  - harmful-finetuning
  - adaptive-regularization
  - kl-regularization
  - activation-probing
  - type/paper
rating: 4
status: permanent
---

# Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning

> **评分**: ★★★★☆  
> **一句话**: Harmful intent 在生成前就线性可预测（AUROC>0.9）；基于此，用 safety critic 动态调整 KL 正则强度，将 fine-tuning 攻击成功率从 97% 压回 1-9%，同时不损失 task performance。

---

## 基本信息

- **arXiv**: 2602.17546
- **提交时间**: 2026-02-19（30 pages，work in progress）
- **Venue**: cs.CL + cs.LG
- **代码**: https://github.com/gjyotin305/adaptive-ai-safety-align
- **关联任务**: Harmful Fine-Tuning Defense / Safety Alignment Preservation

---

## 核心问题

**Harmful fine-tuning attack（有害微调攻击）的威胁**：

> 对对齐的 LLM 仅用 10-100 个有害样本进行 fine-tuning，就能让 safety guardrails 灾难性崩溃，同时保留所有 task capabilities。

这是一个真实的 production 威胁场景：开放模型（Llama/Qwen）被用户恶意 fine-tune。

**现有防御的根本限制**：
- **固定 KL 约束（Constrained SFT）**：统一强度不区分 benign/harmful 数据 → 要么保护不足，要么过度约束损 utility
- **后验修复方法**（如 Antidote）：在 safety 已经崩溃后才介入，被动

**MARS 的核心洞察**（与本文类比）：MARS 把增强 budget 集中在 reward model 最不确定的地方；本文把 KL 正则强度集中在 safety 风险最高的 batch。都是「adaptive, model-aware 资源分配」。

---

## 关键科学发现

### Harmful Intent 的线性可预测性

**实验设置**：取模型 forward pass 中「最后一个 input token 生成之前」的 hidden state，训练 logistic regression probe。

**结论**：
- AUROC > **0.9** in-distribution 和 held-out 均成立
- Early layers（0-2层）和 final layers 都有效（害处信息在 forward pass 全程存在）
- 不同模型家族（Phi-3.5 / Llama-3.1 / Qwen2.5）均复现

**含义**：模型在生成任何 token 之前，其激活空间已经编码了"我即将产生有害输出"的信号，且该信号是**线性可分的**。这对盾卫设计有直接价值。

---

## 框架设计

### 核心公式：Adaptive Alignment Objective

训练 loss 是 NLL 和 KL 的动态加权和：

$$\mathcal{L}_{\text{tot}}^{(t)} = \alpha_t \mathcal{L}_{\text{NLL}}^{(t)} + \beta_t \mathcal{L}_{\text{KL}}^{(t)}$$

动态权重由 safety critic 的实时信号 $s_t \in [0,1]$ 控制：

$$\beta_t = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \cdot s_t$$
$$\alpha_t = 1 - \beta_t$$

- $s_t$ 高（高风险）→ $\beta_t$ 大 → KL 主导 → 模型被拉回 reference policy
- $s_t$ 低（低风险）→ $\alpha_t$ 大 → NLL 主导 → 正常 task 学习

可选 EMA 平滑防止系数剧变：$\tilde{s}_t = \lambda \tilde{s}_{t-1} + (1-\lambda)s_t$

### 两种 Safety Critic 实例

#### (1) Activation-Based Critic（预生成）
- 取 reference model 的 pre-generation hidden states
- Layer-wise 提取后做 **activation pooling**（跨层聚合，提升跨架构鲁棒性）
- 轻量 logistic regression 输出 $s_t \in [0,1]$
- **优点**：零推理开销（训练结束后 probe 可丢弃）；速度快；早发现

#### (2) Judge-Based Critic（后生成）
- 用 LLM（gpt-oss-20b）评估模型输出的 harmfulness（1-5 分归一化）
- 同时评估 reference model 和 main model 的输出，context-aware
- **优点**：捕获语义层面的违规；**缺点**：计算昂贵，增加训练 latency

---

## 实验结果

### Experiment 1: 全有害数据 fine-tuning（最严苛场景）

数据集：HEx-PHI 300 有害样本，20 epochs

| 模型 | Initial ASR | SFT ASR | C-SFT | Vaccine | LISA | Antidote | **Ours** |
|------|------------|---------|-------|---------|------|----------|---------|
| Phi-3.5-mini | 1.35 | 97.27 | 5.33 | 89.18 | 65.12 | 62.38 | **1.67** |
| Llama-3.1-8B | 0.33 | 96.92 | 4.33 | 86.29 | 68.98 | 61.91 | **3.67** |
| Llama-3.2-3B | 5.00 | 96.27 | 6.67 | 87.90 | 67.28 | 58.59 | **6.67** |
| Qwen2.5-7B | 4.05 | 96.92 | 13.67 | 89.19 | 63.17 | 61.54 | **5.69** |
| Qwen2.5-3B | 8.72 | 96.91 | 14.0 | 89.34 | 68.19 | 63.82 | **9.06** |

**结论**：ASR 从 97% 压回 1-9%，接近 initial baseline。C-SFT 在有些模型上已有效（4-6%），但本文方法在所有 5 个模型上均达到或超越。

### Experiment 2: 混合数据（realistic 场景）

有害样本比例 1-9%，与 Alpaca benign 数据混合

- **ASR 保护**：本文方法在所有比例下维持 1-9% ASR（vs. SFT 的 69-94%）
- **Task Performance（Alpaca Win-Rate）**：本文与标准 SFT 接近（无明显 utility 损失）

### Experiment 3: 学习率鲁棒性

跨 $2 \times 10^{-4}$ 到 $2 \times 10^{-7}$ 四个数量级，本文方法 ASR 始终低，而 C-SFT 在高 lr 下显著退化。

---

## 我的分析

### 最重要的发现：Harmful Intent 的线性可分性

**这是论文里最值得深挖的科学发现**，不是方法本身。

Pre-generation hidden states 中 harmful intent 线性可分（AUROC > 0.9）意味着：

1. **机制存在**：对齐训练给模型写入了某种"harm marker"的内部表示，这个表示在 forward pass 的早期层就稳定存在

2. **对盾卫的直接价值**：
   - 可以在**生成之前**就检测 harmful intent，成本极低
   - 不需要等模型产生输出再判断——这对实时 agent safety 至关重要
   - `memory_guard.py` 可以借鉴这个思路：在 agent 处理 memory 之前先做 activation probe

3. **深层问题**：为什么害处意图是线性可分的？可能是 RLHF 的 "safety direction" 在激活空间形成了清晰的几何结构——与 representation engineering / steering vectors 的研究一致

### 方法设计的优雅性

Adaptive Regularization 的核心思想极为简洁：

$$\text{如果这个 batch 的风险高} \Rightarrow \text{收紧 KL，不让模型漂移}$$
$$\text{如果这个 batch 是 benign} \Rightarrow \text{松开 KL，自由学习任务}$$

这是「data-dependent trust region」的自然实现，等价于 soft constraint RL 中的 adaptive penalty。与 ProGRPO（概率加权）、MARS（margin-aware budget）的思路相通：**资源分配需要 model-aware**。

### 局限性分析

1. **Probe 需要 labeled 有害数据训练**：activation-based critic 要先有标注数据训练 probe，这是 cold-start 问题

2. **对 adversarial probe bypass 的脆弱性**：如果攻击者知道 probe 存在，能否设计绕过 pre-generation detection 的攻击？论文未讨论

3. **Judge-based critic 的 cost**：每个 training step 都要调用 gpt-oss-20b，生产级别的训练 overhead 不小

4. **"Work in progress" 标注**：30 页但还是草稿，某些 ablation 和分析在 appendix 中，理论保证较弱

5. **只测 ASR（attack success rate）**：没有测 broader safety categories（如 truthfulness / bias），以及对 adaptive attacks（攻击者知道 defense 存在）的鲁棒性

### 对盾卫项目的启示

```python
# 盾卫可以借鉴的设计模式：
# 1. Pre-generation activation probe 作为 lightweight safety gate
#    - 在 agent 处理 memory / tool output 之前先做 probe
#    - 低 overhead，早拦截
# 2. Adaptive trust region
#    - 对高风险 input 加强约束，对低风险 input 放开
#    - 与 AgentSpec DSL 的规则引擎互补
```

---

## Tags

#safety-alignment #harmful-finetuning #adaptive-regularization #kl-regularization #activation-probing #linear-probing #safety-critic #agent-safety #盾卫项目

---

## See Also

- [[AI/LLM/RL/Theory/MARS-Margin-Aware-Reward-Modeling-Self-Refinement|MARS]] — 同样是 adaptive, model-aware 训练策略：MARS 按 reward margin 分配增强 budget，本文按 harmful intent score 分配 KL 强度；共同思想：model-aware 资源分配
- [[AI/LLM/RL/Theory/RLRR-Reference-Guided-Alignment-Non-Verifiable|RLRR]] — 对齐的另一个维度：RLRR 解决 non-verifiable 对齐，本文解决 fine-tuning 对对齐的破坏；互补形成安全对齐双线
- [[AI/Safety/AI安全与对齐-2026技术全景|AI 安全与对齐 2026 全景]] — 有害微调攻击在安全全景中的位置
- [[AI/Safety/_MOC|Safety MOC]] — AI 安全知识图谱
- [[AI/LLM/RL/_MOC|RL MOC]] — KL 正则、trust region 在 RL 训练中的理论基础
