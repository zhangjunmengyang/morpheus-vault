---
title: "From Growing to Looping: A Unified View of Iterative Computation in LLMs"
brief: "TU Munich + Google（arXiv:2602.16490）从统一视角重新理解 LLM 迭代计算：Growing（深度扩展，层间信息向上流动）vs Looping（循环复用，同一层反复执行）是同一 computation-depth 图谱上的两端。理论上证明两类架构的等价条件，对 Universal Transformer/Looped Transformer 等混合设计有指导意义。"
date: 2026-02-20
type: paper
domain: ai/llm/architecture
rating: ★★★★☆
arxiv: "2602.16490"
institution: TU Munich + Helmholtz AI + Google (Paradigms of Intelligence Team)
tags:
  - architecture
  - reasoning
  - depth-growing
  - looping
  - iterative-computation
  - test-time-compute
  - training-efficiency
---

# Growing to Looping：LLM 迭代计算的统一理论

> **一句话**：Depth growing（中间层复制 → 浅到深训练）和 Looping（权重绑定 → 同一 block 重复执行）看似不同，其实都是"迭代计算"的不同实现方式，共享相同的深度计算特征，可以组合：先 grow 再 loop，推理时免训练获得最高 2x 提升。

---

## 背景：两种增强推理的独立方法

### Looped Models（Universal Transformer 家族）
- 同一 block 的权重在多个深度位置共享（weight tying）
- `Loop(L×k)` = L 层 unique 权重，unroll k 次，effective depth = L·k
- 参数量与 L 层相同，但计算量是 L·k 层
- **优势**：固定参数预算下提升推理；可在推理时调整 loop 次数（test-time compute scaling）
- **变体**：`Loop(e-L×k-d)` = e 层 encoder（unique）+ L×k 中间循环 + d 层 decoder（unique）

### Depth Growing（MIDAS/LIDAS 家族）
- 从浅模型开始，通过复制中间层逐步增加深度
- **MIDAS**：复制中间 block（block size B=4）
- **LIDAS**：复制精确的层级中点（更对称，推理更强）
- 最终架构与 baseline 相同（untied 权重），但训练 FLOPs 减少约 20%
- **优势**：相同 inference FLOPs 下，更好推理；相同推理性能，更少训练计算

---

## 核心发现一：共享的机械特征（Unified Signatures）

用两种分析工具验证两者的统一性：

**深度利用诊断（Depth Utilization Diagnostics）**：
- Looped 和 depth-grown 模型都把不可或缺的计算**移向深层**（later layers indispensable）
- 标准 baseline 计算更均匀分布在各层

**残差流干预（Residual Stream Interventions）**：
- 两种模型都在 residual updates 和 sublayer contributions 中产生**周期性、block 对齐的模式**
- 这与迭代计算的假设一致：同样的计算逻辑在不同深度重复执行，精化表示

**结论**：这两种方法实现迭代计算的方式不同（隐式 vs 显式），但**结果是相同的计算组织方式**。

---

## 核心发现二：各自的权衡（Trade-offs）

| 约束条件 | Looped 模型 | Depth-grown 模型 |
|----------|------------|----------------|
| 固定参数量 | ✅ 更好推理（参数共享 → 更多计算深度） | ⚡ 需要更大的 unique 模型才能匹配 |
| 固定推理 FLOPs | ≈ 竞争力（和 baseline 相当） | ✅ 轻微优势 |
| 固定训练 FLOPs | ❌ 训练开销等同 | ✅ 减少约 20% 训练 FLOPs 达到相同性能 |
| 鲁棒性 | ❌ 全局 looped 对层序敏感 | ✅ 对层序更鲁棒 |

**实际选择建议**：
- 资源受限（参数预算紧）→ Looped
- 训练成本受限 → Depth-grown
- 最优 → 两者组合（见下）

---

## 核心发现三：可组合性（Grow First, Loop Later）

**最 counter-intuitive 也最 elegant 的发现**：

> 把 depth-grown 模型在推理时对**中间 block 进行 looping**，在 Reasoning Primitives 任务上可获得最高 **2×** 提升——**尽管该模型从未用 weight tying 训练过**。

这说明：depth growing 在初始化时隐式地引导了网络学习某种可迭代的中间表示，而 looping 只是在推理时显式地利用了这个特性。

**"Loop in the Middle" 设计原则**：
- 全局 looped（所有层共享）比只在中间 loop 鲁棒性差
- `Loop(e-L×k-d)` 架构（独立 encoder + 中间循环 + 独立 decoder）在推理上最优
- 这与 depth-grown 模型的中间复制一致：**中间层最适合做迭代计算**，头尾层做特征提取和解码

**最强组合**（匹配数据和推理 FLOPs 的条件下）：
1. LIDAS（depth-grown）
2. cooldown 阶段 retrofit 中间 block 做 looping
3. 高质量数学重型 cooldown 数据混合

→ 所有比较条件下最强推理性能

---

## 核心发现四：可适应性（Adaptability）

Looped 和 depth-grown 模型在两种适应场景下都比标准 baseline 更高效：

1. **In-context Learning（多样本学习）**：给更多示例，两种模型收益更大
2. **Supervised Fine-tuning（SFT）**：用相同数据微调，两种模型性能提升更快

直觉：迭代计算的归纳偏置（同样的层逻辑重复处理）天然适合从 few-shot 示例中抽取规律，也适合 SFT 时快速适应新格式。

**Depth-grown + 高质量数据的额外发现**：
- 用高质量、数学重型的 cooldown 数据混合训练时，depth-grown 模型的推理提升最大
- 这暗示 growing 的归纳偏置与**需要迭代推导的数学推理**特别兼容

---

## 批判性分析

### 为什么这个统一视角重要

在此之前，looping 和 depth growing 是两条独立的研究线，各自有不同的动机（参数效率 vs 训练效率）。这篇论文提供了**统一的机理解释**：两者都是在诱导"迭代计算"这种 forward pass 内的计算组织方式。

这个视角的价值不只是学术的——它直接指导了实践：先 grow（便宜的训练）再 loop（推理时免费提升），是个真正可用的工程策略。

### 有趣的疑问

1. **最优 loop 次数**：推理时的 loop 次数如何选择？论文用固定 k，但 adaptive halting（早停）可能更好
2. **任务特异性**：+2x 是"some reasoning primitives"——是所有推理任务都有这个收益吗？在代码生成、多轮对话上如何？
3. **大模型 scaling**：实验在 360M 和 1.7B，7B/70B 上是否仍然成立？growing 的 20% 训练 FLOPs 节省在大规模预训练中意义重大
4. **与 Test-Time Compute Scaling 的关系**：looping 其实是 "latent space" 的 TTC scaling——在 token level 扩展（CoT）和 latent level 扩展（looping）之间如何取舍和组合？

### 与今天其他工作的连接

- **ReFINE**（今日读）：也是"迭代计算"主题，但方向不同——fast weight 的在线状态更新是另一种迭代机制
- **Engram**（今日读）：解耦知识检索 vs 推理计算，而 looping/growing 是在计算深度上做文章。互补。
- **DTR**（昨日读）：Deep-Thinking Ratio 用 logit lens 衡量 deep layer 的推理质量——而 looping/growing 模型的 "late layers indispensable" 特征直接提高了 DTR

---

## 对 Vault 的贡献

这篇笔记补充了"**推理如何在 Transformer 内部组织**"的视角：
- 不是 token 层面的推理（CoT/RL）
- 不是外部系统层面的推理（Agent/Tool Use）
- 而是 **forward pass 内部的迭代精化**——这是 latent reasoning 研究的核心问题

---

## 与 Vault 其他笔记的连接

- → [[Test-Time-Compute|Test-Time-Compute]] (looping 是 latent TTC 的一种)
- → [[Deep-Thinking-Ratio-DTR|Deep-Thinking-Ratio-DTR]] (late layer indispensability 共同主题)
- → [[AI/3-LLM/Architecture/ReFINE-Fast-Weight-RL-Next-Sequence-Prediction|ReFINE-Fast-Weight-RL-Next-Sequence-Prediction]] (迭代计算的另一视角：fast weight 在线更新)
- → [[AI/3-LLM/RL/Other-Algorithms/GEPA-Reflective-Prompt-Evolution|GEPA-Reflective-Prompt-Evolution]] (外部迭代 vs 内部迭代)
- → [[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析|Agentic-RL-2026前沿综合分析]] (agent 层面的迭代)
