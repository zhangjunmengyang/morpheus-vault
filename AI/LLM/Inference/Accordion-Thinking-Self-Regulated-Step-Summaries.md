---
title: "Accordion-Thinking: Self-Regulated Step Summaries"
date: 2026-02-04
tags: [推理效率, CoT压缩, RL训练, ICML2026, 长推理]
domain: AI/LLM/Inference
arxiv: "2602.03249"
rating: 4
status: permanent
---

# Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning

**arXiv**: 2602.03249  
**提交日期**: 2026-02-04  
**会议**: ICML 2026 投稿  
**作者**: Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, Jing Tang  
**评分**: ★★★★☆

---

## 一句话

Accordion-Thinking 让 LLM 在推理过程中**自主学会压缩**——每步推理后生成一个 summary，然后丢掉原始详细内容只保留 summary，用 RL 训练使模型在压缩上下文中依然能正确推理，实现 **3× throughput，零精度损失**。

---

## 核心问题

长 CoT 推理（o1-like）有两个根本矛盾：

1. **计算复杂度**：attention 是 O(L²)，KV cache 随序列长度线性增长 → 长推理链极耗内存
2. **可读性差**：内部独白冗长、无结构、meandering，人类无法直接理解模型在想什么

已有方案的缺陷：
- **启发式 token eviction**（LighterThinker）：外部规则决定哪些 token 被丢掉，破坏推理流
- **固定大小分块**（Delethink/Markovian Thinker）：静态规则，不自适应，不改善可读性
- 关键缺陷：两者都是**外部施加压缩**，而非让模型**学会压缩**

---

## 技术方案

### 数学结构

将推理链结构化为 K 步，每步包含：
- **d_k**：详细推理段（详细展开、探索、推导）
- **s_k**：步骤 summary（该步的核心状态更新和逻辑结论）

完整序列：`y = [x, d₁, s₁, ..., d_K, s_K, a]`

### 两种推理模式

**Unfold Mode（全上下文，baseline）**：
```
H_k^unfold = [x, d₁, s₁, ..., d_{k-1}, s_{k-1}]
```
计算复杂度：O(|x| + Σ(|d_i| + |s_i|)) — 标准 CoT

**Fold Mode（压缩上下文，核心贡献）**：
```
H_k^fold = [x, s₁, s₂, ..., s_{k-1}]
```
- 当 `s_{k-1}` 生成完毕后，立即从 KV cache 中**丢弃 d_{k-1}**
- 关键：生成 `s_k` 时 `d_k` 仍可见（确保 summary 忠实）；summary 生成完毕后才 fold
- 内存复杂度降至 O(|x| + Σ|s_i|)，由于 |s_i| ≪ |d_i|，大幅节省

视觉上形似手风琴：推理时展开，总结后折叠 → **sawtooth 计算曲线**（每次 fold 后时延骤降）

### 三阶段训练流程

**阶段 1：数据合成（Accordion Data Synthesis）**
1. 从 openr1-math-46k 采样 10,000 条长 CoT 推理轨迹
2. 用 DeepSeek-V3.2 将自由形式 CoT 改写为 `<step>...</step>` 结构化格式
3. 规则过滤：步数 K ∈ [2,6]，每步 ≤ 6,144 tokens，每个 summary ≥ 100 tokens
4. 得到 3,900 samples（Unfold）+ 14,653 samples（Fold 格式）

**阶段 2：Cold-Start SFT**
- 学习率 1e-5，batch size 8，3 epochs
- 作用：让模型学会**格式**（生成 `<step>` 标签）
- 局限：SFT 无法确保 summary **语义完整**——被 fold 后信息丢失严重

**阶段 3：Accordion RL**
- 目标函数：GRPO（不带 KL penalty），采纳 Dr. GRPO 去除 response length 项
- Reward：二元可验证奖励（最终答案是否正确）
- 三种训练策略：
  - **Unfold-RL**：全上下文训练（baseline）
  - **Fold-RL**：强制压缩上下文训练 → 模型必须学会压缩否则得不到奖励
  - **Mix-RL**：同一 step 内同时跑 Fold 和 Unfold，交替更新（best approach）

---

## 关键现象：Gap-Vanishing

这是论文最重要的实证发现：

- **训练初期**：Fold mode 和 Unfold mode 之间有巨大性能 gap
  - Qwen2.5-Math-7B：~27 points gap
  - Qwen3-4B-Base：~42 points gap
- **RL 训练过程中**：Fold mode 的奖励以显著更快的速度提升
- **训练收敛后**：两个 mode 的奖励曲线完全重合，gap **消失**

**理论含义**：当压缩版本的推理能与全上下文版本持平时，说明 summary 已经携带了与完整推理链等量的信息。模型成功将"压缩"内化为推理能力的一部分。

这是**信息论意义上的有损压缩变无损压缩**——不是精度下降可接受，而是真正 lossless。

---

## 实验结果

### 主要结果（Pass@1, Avg@32）

| Method | Mode | AIME24 | AIME25 | MATH500 | AMC | Minerva | Macro |
|--------|------|--------|--------|---------|-----|---------|-------|
| Qwen2.5-Math-7B |
| Zero-RL | Unfold | 25.8 | 18.1 | 82.2 | 58.9 | 37.8 | 44.6 |
| Cold-Start | Unfold | 26.7 | 24.6 | 86.2 | 65.4 | 39.7 | 48.5 |
| Cold-Start | **Fold** | 23.0↓ | 23.1↓ | 82.3↓ | 62.4↓ | 37.6↓ | 45.7↓ |
| Unfold-RL | Unfold | 32.0 | 26.7 | 89.2 | 71.2 | 42.1 | 52.2 |
| Unfold-RL | **Fold** | 29.1↓ | 25.1↓ | 87.3↓ | 70.2↓ | 39.7↓ | 50.3↓ |
| **Fold-RL** | **Fold** | 31.3 | 26.9 | 89.9 | 73.8 | 42.0 | **52.7** |
| **Mix-RL** | **Fold** | 32.2 | 28.3 | 89.6 | 71.9 | 41.8 | **52.8** |

**结论**：Mix-RL/Fold-RL 在 Fold 模式下不仅消除了与 Unfold-RL 的差距，部分基准上还**超越** Unfold-RL（AIME24: 32.2 vs 32.0，AMC: 73.8 vs 71.2）

### 推理效率（Qwen3-4B，48GB GPU）

| Model | Mode | Throughput |
|-------|------|-----------|
| Mix-RL-4B | **Fold** | **5888 token/s** |
| Fold-RL-4B | Fold | 5612 token/s |
| Unfold-RL-4B | Unfold | 1483 token/s |

**3.97× speedup** — 接近 4× throughput。24GB 配置下同样约 3× (3182 vs 1083)。

### 数据合成消融

- **Strict Prompt > Lax Prompt**：严格 prompt 要求 summary 语义完整；Lax prompt 产生 "I calculated the result" 此类空洞 summary → 推理 collapse
- **Filter > No Filter**：质量过滤稳定提升 Fold mode 表现

---

## 我的分析

### 与 Progressive Thought Encoding (PTE) 的对比

| | Accordion-Thinking | Progressive Thought Encoding |
|---|---|---|
| 目标 | CoT 推理链压缩 + 可读性 | KV cache eviction 后的记忆恢复 |
| 机制 | 学习生成 summary，fold 后丢原始 token | cross-attention → LoRA adapter 内化被 evict 的 KV |
| 训练 | 序列级奖励（RL on final answer） | LoRA 参数更新（在线自蒸馏） |
| 压缩位置 | **主动**：模型生成 summary 然后 fold | **被动**：sliding window 强制 evict 后补救 |
| 可读性 | ✅ step summary 是人类可读的推理摘要 | ❌ 状态内化到 LoRA 参数，不可读 |
| Throughput | **3-4×** | ~2× (peak memory -30%) |

**关键洞察**：两者正交，可组合：
- Accordion 负责**结构化推理 + 主动压缩**（模型层面）
- PTE 负责**被动 eviction 后的记忆恢复**（架构层面）
- Jet-RL 负责**量化精度统一**（系统层面）
- 三者可形成完整的高效长推理 pipeline

### 这篇论文 elegant 在哪里

1. **问题建模极简**：把压缩问题转化为 RL 信号设计问题——Fold 失败（信息丢失）→ 得不到奖励 → 模型必须学会压缩。不需要额外的 summarization reward，用最终答案正确性即可
2. **Gap-Vanishing 现象**：这不是"我们的方法 works，精度差距可接受"，而是**精度差距最终完全消失**——更强的实证证明。说明 summary 质量上限和完整 CoT 等价，而不是近似
3. **可读性副产品**：step summary 是 human-interpretable 的，是内部独白的简洁映射。这对模型可解释性有价值——不是事后解释，而是推理过程的在线总结

### 质疑点

1. **仅验证了数学推理**：MATH500 / AIME / AMC 都是封闭式数学题，reward 明确（答案对错）。在代码、科学推理、多步工具调用等开放域，这个框架是否成立？summary 的粒度该如何定义？
2. **步数限制 K ∈ [2,6]**：对超长推理链（20+ 步的 agent 任务）是否够？论文只测试了数学题
3. **SFT 冷启动数据规模较小**：3,900 samples 的结构化数据 + 14,653 fold 格式，能否泛化到更多 domain？DeepSeek-V3.2 做 teacher 的质量依赖于 teacher 能力
4. **与 Delethink 的实质差异**：Delethink 也是分块+丢弃前半，主要区别是 Accordion 通过 RL 学习 summary 质量而非 rule-based。差距有多大？论文未直接比较

### 在 Test-Time Compute Scaling 框架下的定位

ICLR 2026 热点：Test-time compute scaling（257 篇）的核心矛盾是**计算增加 vs 资源限制**。

Accordion-Thinking 提供了一个新路径：**压缩推理链，而非截断推理链**。
- Budget-forcing / LACONIC 类方法：强制压缩总长度 → 丢掉推理步骤
- Accordion：压缩每步的历史 → **不减少推理步数**，只减少 attention 计算量

这使得 Accordion 是少数真正**与 test-time compute 兼容**的效率方案：可以同时跑更多步推理 + 使用更少内存。

---

## 关键词连接

- [[GRPO]] - 采用 GRPO 变体（无 KL，无 length normalization）
- [[Progressive-Thought-Encoding]] - 同类问题（长 CoT 效率），不同路径（主动 vs 被动压缩）
- [[Jet-RL-FP8-On-Policy-RL-Training]] - 系统效率（量化）vs 算法效率（压缩），可组合
- [[RL-Training-Stability-2026-Unified-Analysis]] - 归属 Exploration/Efficiency 维度
- [[Test-Time-Compute-Scaling]] - ICLR 2026 最大热点，Accordion 是 TTC 扩展的资源高效方案

---

## 总结

Accordion-Thinking 是一篇有真实 insight 的高效推理论文。最核心的贡献不是技术 trick，而是**证明了"让模型学会压缩"这件事是可行的**——通过 RL 和足够简洁的信号设计，模型可以在压缩版上下文中达到与全上下文相同的推理能力。

3-4× throughput + 零精度损失，实用价值高。Gap-Vanishing 现象是重要实证发现，说明 summary 的信息密度可以达到原始推理链的等价水平。

待观察：能否推广到非数学域（代码/agent）；与 PTE/Jet-RL 的组合是否有 1+1>2 效果。
