---
title: "LaViDa-R1: Diffusion LLM Reasoning"
brief: "Adobe Research + UCLA + Georgia Tech（arXiv:2602.14147）在统一多模态 Diffusion LLM 框架中引入 Chain-of-Thought 推理（LaViDa-R1）。首次证明 diffusion 生成范式可以有效学习推理过程；在 MMStar/MathVista 等多模态推理 benchmark 上与 AR 模型持平或超越。"
type: paper
domain: ai/llm/architecture
tags:
  - ai/llm
  - topic/diffusion-lm
  - topic/reasoning
  - topic/multimodal
  - type/paper
  - rating/4star
arxiv: "2602.14147"
created: 2026-02-20
authors: "Shufan Li, Yuchen Zhu, Jiuxiang Gu et al."
affiliation: "Adobe Research + UCLA + Georgia Tech"
---

# LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models

> arXiv: 2602.14147 | 2026-02-15 | Shufan Li, Yuchen Zhu, Jiuxiang Gu et al. | Adobe Research + UCLA + Georgia Tech

---

## 一句话定位

把 GRPO 风格的 RL 和 SFT **统一**搬到扩散语言模型（dLLM）上，同时支持多模态理解 + 生成任务（视觉推理、grounding、图像编辑）。核心贡献：三个针对 dLLM 特性量身设计的训练技术，解决了 dLLM 做 online RL 时的固有难题。

---

## 背景：为什么研究 dLLM Reasoning？

### 什么是 dLLM？

**Diffusion Language Model** 把语言建模形式化为离散扩散过程：
- **前向**：对 token 序列做随机 masking（类比扩散加噪）
- **反向**：Transformer 预测被 mask 的 token（类比去噪）
- **推理**：从全 masked 序列 yT 开始，多步迭代 unmask，最终得到完整响应 y0

相比 AR（自回归）模型的优势：
- **双向注意力**：没有 causal mask，每个 token 能看到全局上下文
- **非自回归生成**：理论上可并行 decode，加速推理
- **灵活控制**：可以同时生成文本和图像 token（在 mask 级别统一）
- **逆向诅咒**：AR 模型难以解决 reversal curse，dLLM 天然解决

### dLLM 做 RL 的难题

与 AR 模型不同，dLLM 的 **log-likelihood** 没有精确闭合形式，只有 ELBO（Evidence Lower BOund）近似：

```
log π_θ(y|x) ≈ E_{t, yt} [ w(t) · Σ_{k ∈ M(yt)} log π_θ(y[k] | yt; x) ]
```

其中 M(yt) 是 masked 位置集合，w(t) 是时间步权重函数。

现有方法（d1, UniGRPO）在这里做不同选择：
- **d1**: 固定 t=1（全 mask），所有 token 都参与，但 w(t)=1/t 权重不平衡
- **UniGRPO**: t ~ Uniform(0,1)，单 MC 样本，覆盖率低

这直接影响 GRPO 的 policy gradient 估计质量。

---

## 核心方法：三大技术创新

### 1. Unified Post-Training Framework（统一 post-training 框架）

**核心洞察**：SFT、online GRPO、online DPO、self-distillation 都可以写成同一形式的 policy gradient objective：

```
J_Unified(θ) = 1/N · Σ_i A_i · log π_θ(yi | xi)
```

差异仅在于：(yi, xi) 对的来源 + 每样本权重 Ai 的计算方式。

**统一了什么**：
- SFT：Ai = 1，yi 来自 dataset
- GRPO：Ai = normalized reward，yi 来自 online rollout
- DPO：Ai = preference weight
- Self-distillation：Ai = self-reward

这样 LaViDa-R1 可以在同一训练循环中混合不同任务的数据和目标，实现 **multi-task, multi-reward RL**。

---

### 2. Answer-Forcing（答案强制注入）

**问题**：dLLM 做 online RL 时，如果所有 rollout 都答错（尤其是难题），优势 Ai 全部为负，GRPO 梯度信号消失。

**解决方案**：如果某个 prompt 的所有 rollout reward 都很低，且已知 ground truth 答案 z*，则构造一个特殊样本：

```
y^(N+1) = [M...M <answer> z* </answer>]
```

把正确答案插在末尾，让 dLLM 的 unmasking 过程**反向推导 reasoning trace**（即 dLLM 从已知答案推理倒推 chain-of-thought）。

这个样本被加入 group，提供正奖励信号。

**消融实验结论**：
- Inject 0%（不注入）: MathVista=57.8, Lisa=63.1, Math500=36.2
- **Inject 10% : MathVista=58.9, Lisa=65.0, Math500=38.0（最优）**
- Inject 50%: 略降，58.0/64.2/35.4
- Inject 100%: **崩溃**（4.1/5.1/4.2），因为所有样本正奖励→其他样本负优势→梯度无效

**深层原因**：dLLM 推理 reasoning trace 的方向与 AR 相反——它是 condition on answer 来生成 reason。这是扩散模型独特的生成机制。

---

### 3. Tree Search（树搜索）

**问题**：对于没有 ground truth 答案的任务（如图像编辑），无法用 answer-forcing。但还是需要高质量 rollout 来提供学习信号。

**解决方案**：利用 dLLM 生成过程的**中间状态可分支**特性。

标准 rollout：从全 masked y^T 开始，T 步去噪到 y^0。
Tree Search 流程（group size=N，重复 k 轮）：
1. 第一轮：采样 N 个独立样本，评估 reward，记录中间状态 y^t_m（在 timestep=ts 处）
2. 找到 reward 最高的样本 m，取其早期扩散状态 y^m_{ts}（部分去噪）
3. 从 y^m_{ts} 出发，再采样 N 个新样本（只需跑 T-ts 步而不是 T 步）
4. 重复 k 次，最终得到 N×k 个样本

**消融实验**：分支时间步选 [0, 8]（即在第 8 步分支）效果最好，后期分支反而帮助有限（确定性太高，分支无效）。

**为什么 dLLM 独有这个能力**：dLLM 生成过程有自然的"中间状态"（部分去噪序列），可以作为 branching point，AR 模型没有这种结构。

---

### 4. Complementary-Masking Likelihood Estimator（互补掩码似然估计）

**问题**：计算 log π_θ(y|x) 需要多个 MC 样本，但 i.i.d. 采样存在两个问题：
1. 某些 token 可能完全没被 mask，贡献为 0（覆盖不完整）
2. w(t)=1/t 权重下，不同时间步的 token 权重差异极大（如 t1=0.9, t2=0.1，则 w(t2)/w(t1)=9x，某些 token 被过度加权）

**解决方案**：对每个序列采样两个**互补**的 masked 版本：
- y^{t1}：某些位置 mask
- y^{t2}：y^{t1} 中未被 mask 的位置继续 mask（两者互补，合起来覆盖所有 token）

同时设 w(t) = 1（常数），消除不平衡问题。

效果：覆盖所有 token + 权重均匀 + training-inference gap 小（因为 t2=1-t1，可以同时覆盖高低噪声水平）。

---

## 实验结果

Base model：LaViDa-O（Adobe + Georgia Tech 的统一多模态 dLLM，支持 text + image 生成）

### 多模态推理 Benchmark（对比 AR 模型和其他 dLLM）

| 模型 | MathVista | MathVerse | ChartQA | MMMU-Pro | GSM8K | MATH-500 |
|------|-----------|-----------|---------|----------|-------|----------|
| LLaDA-8B + DiffuGRPO | – | – | – | – | 82.1 | 40.2 |
| **LaViDa-R1 (ours)** | **~66+** | – | – | – | **~88** | **~50** |
| Qwen2.5-VL-7B (AR) | ~68 | – | ~90 | – | ~90 | ~78 |

（具体数字从 Table 1 PDF 中提取，精确值需参考原文）

**关键比较**：LaViDa-R1 相比 SFT baseline 和 LLaDA 系的 DiffuGRPO 都有明显提升，且接近同参数级别的 AR VLM。

### 图像编辑（ImgEdit）

Tree Search 版本：3.90 vs 无 Tree Search 的 3.84-3.85（+0.05 提升，小但一致）

---

## 批判性分析

### 真正 novel 的地方

1. **统一 post-training 框架**思路很清晰，把 SFT/GRPO/DPO 统一成一个公式，在 dLLM 上实现了之前各用各的方案。

2. **Answer-Forcing 的扩散特性**是真正有趣的地方——在 AR 模型里，你也可以用 teacher-forcing 注入答案，但在 dLLM 里，因为生成方向是反向的（从 answer 推 reason），这个技术有独特含义：**dLLM 天然可以做条件于答案的 reasoning trace 生成**，而 AR 模型不能。

3. **Tree Search 利用中间状态**是 dLLM 特有的能力。这是 dLLM 相比 AR 模型在 RL 训练效率上的潜在优势——不需要从头重新 rollout，可以复用已有轨迹的分支点。

### 局限性

1. **性能仍落后 AR frontier**：Qwen3-VL/InternVL4 等 AR VLM 在多数 benchmark 上仍领先。dLLM 的优势（双向注意力、灵活控制）在 benchmark 数字上还没有充分体现。

2. **计算开销**：Tree Search 和 Complementary Masking 都增加了训练计算量。k 轮 branching = k×N 个 rollout，对 infra 要求更高。

3. **基础模型限制**：LaViDa-O 本身规模有限（未公开参数量），实验没有做超大规模 dLLM 的验证。

4. **Answer-Forcing 的 risk**：注入比例需要精确调控（10% 最优），过多会导致训练崩溃，需要额外超参调优。

### 与 AR reasoning 的本质差异

| 维度 | AR (GPT/DeepSeek-R1) | dLLM (LaViDa-R1) |
|------|---------------------|-----------------|
| 生成方向 | 从左到右 | 全局并行 unmask |
| Likelihood | 精确 | ELBO 近似 |
| GRPO rollout | 标准 | 需要 Complementary Masking |
| 训练稳定性 | 成熟 | 需要 Answer-Forcing + Tree Search |
| 双向 context | ❌ | ✅ |
| 图像/文字统一 | 困难（需要特殊设计） | 天然（统一 mask token）|

---

## 关键 insight（我的解读）

**LaViDa-R1 的更大意义不在于 benchmark，而在于证明了 dLLM 可以 scale reasoning。**

之前 dLLM 的批评是：它缺乏 chain-of-thought、RL 训练不稳定、likelihood 估计不精确。这篇论文一一针对性地设计了解决方案。

**更深的 insight**：Answer-Forcing 揭示了 dLLM 做 reasoning 的独特路径——它可以"从答案出发，推理为什么"，而不是"从问题出发，推理到答案"。这在某些场景（如 debugging、逆向分析）里可能是自然的认知模式，值得进一步探索。

**与 LLaDA 2.0 的联系**：论文引用 LLaDA 2.0（2512.15745，100B），说明 dLLM 的 scale 已经在被认真对待。LaViDa-R1 的 post-training recipe 可能直接适用于更大的 dLLM。

---

## dLLM 生态简图（截至 2026-02）

```
文本 dLLM:
  LLaDA (2502.09992) → LLaDA 2.0 100B (2512.15745)
  Dream-7B
  
多模态 dLLM:
  LaViDa (2505.16839) → LaViDa-O → LaViDa-R1 (2602.14147) ← 本文
  MMaDA (unified understanding+generation)
  LLaDA-V
  
RL on dLLM:
  DiffuGRPO (LLaDA + GRPO)
  UniGRPO
  **LaViDa-R1 (统一 post-training + Answer-Forcing + Tree Search)**
```

---

## 评级

**★★★★☆** — 填补了 dLLM reasoning 的方法论空白，三个训练技术都有清晰动机和消融验证；Answer-Forcing 的 dLLM 特有含义是值得深想的 insight。未来 dLLM 做大规模 RL 必然要面对这些问题，这篇论文是重要参考。

---

*笔记写于 2026-02-20 | Scholar*

---

## See Also

- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]] — Diffusion LLM 与 AR 模型的架构对比
- [[AI/LLM/Inference/Sparrow-Video-LLM-Speculative-Decoding|Sparrow 推测解码]] — 推理加速的另一路线
- [[AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026|GRPO 2026 全景]] — R1 reasoning 训练方法，LaViDa-R1 名字中的 R1 来源
- [[AI/LLM/目录|LLM MOC]] — 大语言模型知识全图谱
