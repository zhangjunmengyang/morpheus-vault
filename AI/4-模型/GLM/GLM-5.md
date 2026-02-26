---
brief: "GLM-5（arXiv:2602.15763，智谱AI）——MoE 架构+异步 RL 训练的新一代 GLM；OpenWeights 发布；DSA（动态稀疏 Attention）降低推理成本；在推理/代码/Agent 任务上与 GPT-4o 对标；评分 4★。"
title: "GLM-5 技术报告精读"
date: 2026-02-17
tags: [GLM, 智谱AI, MoE, 异步RL, OpenWeights, SOTA, DSA]
domain: AI/Frontiers
arxiv: "2602.15763"
rating: 4
status: permanent
---

# GLM-5: from Vibe Coding to Agentic Engineering

> arXiv: 2602.15763 | 智谱 AI + 清华大学 | 发布: 2026-02-17
> 标签: #MoE #DSA #AsyncRL #AgentRL #OpenWeights #SOTA

## TL;DR

GLM-5 是智谱的新一代旗舰模型。核心 claim：**open weights 首次在 AA Intelligence Index v4.0 达到 50 分，与 Claude Opus 4.5 / GPT-5.2 处于同一梯队**。技术贡献包含四个维度：DSA (DeepSeek Sparse Attention)、Muon Split、MTP Parameter Sharing、三阶段异步 RL pipeline。

---

## 1. 模型规模与架构

| 参数 | GLM-4.5 | GLM-4.7 | **GLM-5** |
|------|---------|---------|-----------|
| 总参数 | 355B | ~355B | **744B** |
| Active/Token | 32B | ~32B | **40B** |
| Experts | - | - | **256** |
| Layers | - | - | **80** |
| Context | 128K | 128K | **200K** |
| 预训练 tokens | - | - | **28.5T** |

- 80 层设计（vs DeepSeek-V3 的更多层）：**减少 expert parallelism 通信开销**
- 256 experts：更细粒度的 MoE routing

---

## 2. 关键架构创新

### 2.1 DSA (DeepSeek Sparse Attention)

**核心问题**：传统 dense O(L²) attention 在 128K+ context 计算成本急剧增加。

**DSA 方案**：用 content-aware dynamic sparsity 替代 dense attention，"lightning indexer" 动态选择 important tokens，**不丢弃任何 long-range dependency**（关键：lossless by construction）。

**对比实验**（在 GLM-9B 上的 ablation，RULER@128K）：

| 方法 | RULER@64K | RULER@128K | 损失 |
|------|-----------|------------|------|
| Full Attention | 85.35 | 75.28 | baseline |
| SWA Interleave | 65.94 | **44.93** | ↓30.35 |
| SWA Pattern (search) | 83.72 | 69.59 | ↓5.69 |
| GDN | 76.76 | 64.00 | ↓11.28 |
| SimpleGDN | 81.76 | 67.03 | ↓8.25 |
| **DSA** | ≈baseline | ≈baseline | **≈0** |

**关键发现**：所有 SWA/线性注意力方法都有不可避免的 accuracy gap，尤其在 fine-grained retrieval（RULER/RepoQA）。DSA 是唯一 lossless 方案。

**引入方式（重要）**：通过 Continued Pre-Training 从 dense base 转 DSA，两阶段：
1. Warm-up（1000 steps）：只训 indexer，冻结 base model 权重
2. Joint Training：model + indexer 协同训练（GLM-5 用了 20B tokens）

DSA 验证数据：DeepSeek-V3.2 用了 943.7B tokens 才完成 DSA 适配，GLM-5 仅 20B tokens 就达到同等效果——说明 Continued Pre-Training 路线极度高效。

**效率收益**：长序列 attention 计算 **减少 1.5-2×**，128K context agent 任务 GPU 成本减半。

---

### 2.2 Muon Split（MLA + Muon Optimizer 兼容性修复）

**问题**：MLA（Multi-Latent Attention，DeepSeek 提出的 KV cache 压缩方案）与 Muon optimizer 不兼容。GLM-5 实验发现：MLA（576-dim latent KV）在多项 benchmark 上显著弱于 GQA-8（2048-dim KV），如 BBH 差 4.4 分，HumanEval 差 5 分。

**Muon Split 方案**：
- 原始 Muon：对 W^UQ, W^UK, W^UV 做整体 matrix orthogonalization
- Muon Split：**按 attention head 拆分**，对每个 head 独立做 matrix orthogonalization
- 效果：不同 head 的 projection weights 可以以不同 scale 更新 → 解锁 MLA 的性能潜力
- 副效应：attention logits scale 训练全程稳定，无需任何 clipping 策略

| 方法 | MMLU | C-Eval | BBH | HumanEval |
|------|------|--------|-----|-----------|
| GQA-8 (baseline) | 61.2 | 60.0 | 53.3 | 38.5 |
| MLA | 61.5 | 59.7 | 48.9 | 33.5 |
| MLA + Muon Split | **62.5** | **62.1** | **51.8** | **36.7** |
| MLA-256 + Muon Split | 62.0 | 59.9 | 51.3 | 47.5 |

---

### 2.3 MLA-256（Decoding 加速）

**问题**：MLA 在 decode 阶段做 576-dim dot product，比 GQA 的 128-dim 贵很多。

**方案**：
- Head dim 从 192 → **256**
- Head 数量减少 1/3
- 效果：训练 FLOPS 和参数量不变，decode 计算量下降

---

### 2.4 MTP Parameter Sharing（Speculative Decoding 加速）

**问题**：Multi-Token Prediction (MTP) 提升 base model 性能 + 充当 draft model for speculative decoding。但 n 个 MTP layer 意味着 memory 随 speculative steps 线性增长。DeepSeek-V3 的解决方案是单 MTP layer + 推理时预测 2 tokens，训练/推理不一致导致 accept rate 下降。

**GLM-5 方案**：**3 个 MTP layer 共享参数**——memory cost 与 DeepSeek-V3 持平，但训练时 3 个 MTP layer 各自预测，缩小了 training-inference gap。

| 模型 | Accept Length (4 steps) |
|------|-------------------------|
| DeepSeek-V3.2 | 2.55 |
| **GLM-5** | **2.76** |

---

## 3. 预训练

- **Base Model**: 27T tokens，代码和推理数据早期 prioritize
- **Mid-Training**: 逐步扩展 context 4K → 200K，聚焦 long-context agentic 数据
- **数据新增**：引入新 DCLM classifier（sentence embedding based）+ World Knowledge classifier（Wikipedia + LLM-labeled），针对 long-tail knowledge

---

## 4. Post-Training: 三阶段 Sequential RL

```
Base Model → SFT → Reasoning RL → Agentic RL → General RL
                         ↓              ↓             ↓
               On-Policy Cross-Stage Distillation（各阶段间防遗忘）
```

### 4.1 三阶段设计

1. **Reasoning RL**：数学 / 代码推理，verifiable rewards
2. **Agentic RL**：长 horizon agent 交互（核心），异步算法
3. **General RL**：通用对齐，human preference

### 4.2 On-Policy Cross-Stage Distillation

防止 catastrophic forgetting 的关键机制。在每阶段 RL 结束时，用当前策略在线（on-policy）蒸馏给下一阶段起点，保留前一阶段习得的能力。

### 4.3 异步 Agent RL 算法

在 slime 框架基础上升级，**彻底解耦 generation 和 training**。相比 GLM-4.5 的 iterative self-distillation + outcome supervision：
- 新方法支持 diverse long-horizon interactions 的连续学习
- 专门优化 planning 和 self-correction 能力
- 具体算法细节论文中描述有限（待后续拆解）

---

## 5. 国产芯片全栈适配

首日支持 7 大国内芯片平台：

| 厂商 | 芯片 |
|------|------|
| 华为 | 昇腾 (Ascend) |
| 摩尔线程 | Moore Threads |
| 海光 | Hygon |
| 寒武纪 | Cambricon |
| 昆仑芯 | Kunlunxin |
| 沐曦 | MetaX |
| 燧原 | Enflame |

覆盖范围：底层 kernel → 推理框架，全栈优化。

---

## 6. Benchmark 结果

### AA Intelligence Index v4.0
- **GLM-5 = 50**（open weights SOTA，历史首次）
- GLM-4.7 = 42（+8 分）
- 组成：10 个 eval，包括 τ²-Bench Telecom, Terminal-Bench Hard, SciCode, AA-Omniscience, Humanity's Last Exam, GPQA Diamond 等

### LMArena (UC Berkeley，真实人类偏好)
- **#1 open model in both Text Arena & Code Arena**
- 整体与 Claude Opus 4.5 / Gemini 3 Pro 持平

### 8项 ARC Benchmark 对比（GLM-5 vs Claude Opus 4.5 / GPT-5.2 xhigh）

| Benchmark | 类型 | GLM-5 |
|-----------|------|-------|
| Humanity's Last Exam | 知识推理 | ~= Opus 4.5 |
| SWE-bench Verified | 代码 agent | ~= Opus 4.5 |
| SWE-bench Multilingual | 多语言代码 | ~= Opus 4.5 |
| Terminal-Bench 2.0 | 终端 agent | ~= Opus 4.5 |
| BrowseComp | 浏览器 agent | ~= Opus 4.5 |
| MCP-Atlas | MCP tool use | ~= Opus 4.5 |
| τ²-Bench | Telecom agent | ~= Opus 4.5 |
| Vending Bench 2 | 长 horizon | $4,432 (open #1) |

整体比 GLM-4.7 提升 ~20%，好于 Gemini 3 Pro。

---

## 7. 我的批判性评估

### ✅ 真正 Novel 的贡献

1. **DSA ablation 是这篇论文最有价值的部分**：首次在大规模 MoE 上系统对比了 SWA/GDN/DSA，证明了 DSA 的 lossless 性质。这个 ablation 对整个 efficient attention 领域都有参考价值。

2. **Muon Split**：发现并解决 MLA + Muon 的不兼容性。这是一个实用的 recipe 级创新，但独立 novelty 有限。

3. **MTP Parameter Sharing**：用参数共享巧妙解决了 speculative decoding 的 training-inference gap，是 elegant 的工程优化。

4. **三阶段 Sequential RL + Cross-Stage Distillation**：这个 pipeline 设计是 post-training 领域的一个成熟范式 signal，值得关注。但论文对其收益的 ablation 不充分。

### ⚠️ 需要保持怀疑的地方

1. **异步 Agent RL 算法细节不足**：这是论文最重要的 claim 之一（"novel asynchronous agent RL algorithms"），但技术细节几乎没有披露。这是有意为之（保护 IP）还是论文写作不足？

2. **Benchmark 选择有偏**：8 项 ARC benchmark 是智谱自己挑选的，且都是当前模型表现好的领域（coding/agent）。MMLU/MT-Bench 等通用 benchmark 没有展示。

3. **AA Index v4.0 = 50 的含义**：ArtificialAnalysis 的 Intelligence Index 不是标准 academic benchmark，是他们自己的综合评分。"首次 open weights 达到 50" 是 marketing 数字，需要看具体 sub-metrics。

4. **成本数据缺失**：论文几乎没有提到推理成本 vs 性能的 trade-off，而这是 MoE 最关键的维度之一。

### 🔍 与竞品的真实差距

从公开数据看，GLM-5 在 **coding / agentic** 任务上确实接近 Opus 4.5，但在通用知识、指令跟随等维度可能还有差距（这些没在论文中充分展示）。"comparable to Opus 4.5" 在 8 个 cherry-picked benchmarks 上，不等于全面超越。

---

## 8. 与相关工作的关系

- **DSA**: 继承自 DeepSeek-V3.2，GLM-5 是首个在 744B MoE 上落地的公开报告
- **MLA**: 继承自 DeepSeek-V2/V3，Muon Split 是 GLM-5 的增量改进
- **Slime 框架**: [[Slime-RL-Framework]] — GLM 系列专用异步 RL infra
- **Sequential RL**: 与 InstructGPT 的 RLHF pipeline 精神类似，但扩展到三阶段

---

## 9. 对老板的意义

1. **面试相关**："efficient attention for long context" 是热门题目，DSA 的 ablation（SWA 暴跌 vs DSA lossless）是好素材
2. **RL pipeline 设计参考**：三阶段 Sequential RL + Cross-Stage Distillation 是可以借鉴的 post-training 框架
3. **MTP Parameter Sharing**：speculative decoding 面试题的新 answer
4. **工程参考**：Muon Split 说明了 optimizer + architecture 兼容性的重要性

---

## 相关笔记

- [[Slime-RL-Framework]] — 智谱异步 RL 框架
- [[GRPO 深度理解|GRPO]] — GLM-5 使用的 RL 算法族
- [[2026年2月模型潮（这篇毫无价值，哪怕梳理个从 deepseek R1 以来的时间线都比这强）]] — 竞争背景
- [[ICLR-2026-趋势分析]] — 学术趋势背景

---

*Created: 2026-02-18 | Source: arXiv:2602.15763 直接精读 | Confidence: High（论文 primary source）*
