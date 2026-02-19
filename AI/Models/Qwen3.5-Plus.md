---
title: "Qwen3.5-Plus / Qwen3.5-397B-A17B"
date: 2026-02-18
release_date: 2026-02-16
tags: [llm, qwen, moe, linear-attention, multimodal, architecture, agent]
type: note
model_family: Qwen3.5
developer: Alibaba Cloud / Qwen Team
license: Apache-2.0
params_total: 397B
params_active: 17B
context_length: 262144
max_context: 1010000
---

# Qwen3.5-Plus / Qwen3.5-397B-A17B

> **Qwen3.5: Towards Native Multimodal Agents**
> 2026-02-16 发布，Qwen 系列首次引入混合线性注意力架构的旗舰模型。

## 概述

Qwen3.5 是阿里通义千问团队于 2026 年 2 月 16 日（除夕）发布的新一代大模型系列，首发两个版本：

| 版本 | 定位 | 总参数 | 激活参数 | 上下文 | 许可 |
|------|------|--------|----------|--------|------|
| **Qwen3.5-397B-A17B** | 开源旗舰 | 397B | 17B | 256K（原生） | Apache 2.0 |
| **Qwen3.5-Plus** | API 托管版 | 同上 | 同上 | **1M**（扩展） | 商用 API |

核心亮点：**397B 总参但仅激活 17B**，性能匹配万亿参数 Qwen3-Max，部署显存降 60%，推理吞吐最高提升 **19 倍**。

---

## 架构：Qwen3-Next 混合架构

Qwen3.5 基于 **Qwen3-Next** 架构，核心创新是融合线性注意力与稀疏 MoE：

### 整体规格

| 属性 | 值 |
|------|-----|
| 模型类型 | Causal LM + Vision Encoder |
| 总参数 | 397B |
| 激活参数 | 17B |
| Hidden Dimension | 4,096 |
| 层数 | 60 |
| 词表大小 | 248,320（padded） |
| 原生上下文 | 262,144 tokens |
| 最大上下文 | 1,010,000 tokens（Plus 版） |
| 训练模式 | Pre-training + Post-training + RL |
| MTP | Multi-Token Prediction（多步） |

### Hidden Layout（层排布）

```
15 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE))
```

即 60 层中，每 4 层为一组：
- **3 层 Gated DeltaNet + MoE**（线性注意力块）
- **1 层 Gated Attention + MoE**（标准注意力块）
- 重复 15 次

这是一个 **3:1 的线性/标准注意力混合比**，在保持建模能力的同时大幅减少 KV cache 开销。

### Gated DeltaNet（线性注意力）

- **线性注意力 Heads**：V 64 个，QK 16 个
- **Head Dimension**：128
- 基于 Delta Networks 的门控机制，用线性复杂度近似 softmax attention
- 无需存储完整 KV cache → 长序列推理效率极高

### Gated Attention（标准注意力）

- **Q Heads**: 32，**KV Heads**: 2（极端 GQA）
- **Head Dimension**: 256
- **RoPE Dimension**: 64
- 每 4 层才用一次标准注意力，提供全局依赖建模能力

### MoE 结构

| 属性 | 值 |
|------|-----|
| 总专家数 | 512 |
| 每 token 激活专家 | 10 routed + 1 shared = **11** |
| Expert Intermediate Dim | 1,024 |
| 稀疏度 | ~4.3%（17B / 397B） |

对比 Qwen3-235B-A22B（128 专家，8+1 激活），Qwen3.5 的 **专家数翻 4 倍、稀疏度更极端**。

---

## 关键技术创新

### 1. 混合线性注意力架构

**首次在旗舰级模型中大规模采用 Gated DeltaNet**。线性注意力将 self-attention 的 O(n²) 降为 O(n)，配合门控机制保持表达力。3:1 的混合比是实践中性能-效率的最优平衡点。

**影响**：
- 32K 上下文：解码吞吐 = Qwen3-Max 的 **8.6×**，Qwen3-235B 的 **3.5×**
- 256K 上下文：解码吞吐 = Qwen3-Max 的 **19.0×**，Qwen3-235B 的 **7.2×**
- 长序列场景下优势更明显（线性注意力的 O(n) vs softmax 的 O(n²)）

### 2. 极端稀疏 MoE

512 个专家中每 token 仅激活 11 个（~2.1%），实现 397B 模型以 17B 计算量运行。对比：
- Qwen3-235B-A22B：128 专家，9 激活（~7%）
- DeepSeek-V3-671B-A37B：256 专家，8 激活（~3.1%）
- Kimi K2.5-1T-A32B：稀疏度 ~3.25%

### 3. 原生多模态（Early Fusion）

不同于 "先训文本再加视觉" 的后融合方案，Qwen3.5 在预训练阶段就同时使用文本和视觉数据（Early Fusion），使视觉理解能力成为模型的原生能力。在相近规模下 **超越 Qwen3-VL 全系列**。

### 4. Multi-Token Prediction (MTP)

训练时引入多步预测目标，推理时支持 speculative decoding 加速：
```bash
# SGLang MTP 推理示例
--speculative-algo NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4
```

### 5. 扩展词表与多语言

- 词表从 ~151K → **248,320 tokens**
- 语言覆盖从 119 种 → **201 种**语言/方言
- 多数语言编码/解码效率提升 **10–60%**

### 6. 大规模异步 RL

在百万 agent 环境中进行强化学习，使用渐进式复杂任务分布，提升真实世界适应性。异步 RL 框架支持大规模 agent 脚手架和环境编排。

---

## 预训练策略：三维推进

| 维度 | 细节 |
|------|------|
| **能力（Power）** | 更大规模视觉-文本语料，强化中英文/多语言/STEM/推理数据，严格过滤 → 397B 匹配 1T+ 的 Qwen3-Max-Base |
| **效率（Efficiency）** | Qwen3-Next 架构 + 更高稀疏度 MoE + 混合注意力 + MTP |
| **通用性（Versatility）** | Early Fusion 原生多模态，201 种语言，25 万词表 |

训练基础设施实现 **多模态训练效率接近 100%**（相比纯文本训练），这是工程上的重大突破。

---

## Benchmark 评测（部分）

### 语言

| Benchmark | GPT-5.2 | Claude 4.5 Opus | Gemini-3 Pro | **Qwen3.5-397B** |
|-----------|---------|-----------------|--------------|-------------------|
| MMLU-Pro | 87.4 | 89.5 | 89.8 | **87.8** |
| MMLU-Redux | 95.0 | 95.6 | 95.9 | **94.9** |
| IFBench | 75.4 | 58.0 | 70.4 | **76.5** ✨ |
| LiveCodeBench v6 | 87.7 | 84.8 | 90.7 | 83.6 |
| AIME26 | 96.7 | 93.3 | 90.6 | 91.3 |
| TAU2-Bench | 87.1 | 91.6 | 85.4 | **86.7** |
| BrowseComp | 65.8 | 67.8 | 59.2 | **78.6** ✨ |
| SWE-bench Verified | 80.0 | 80.9 | 76.2 | 76.4 |

### 视觉多模态

| Benchmark | GPT-5.2 | Claude 4.5 Opus | Gemini-3 Pro | **Qwen3.5-397B** |
|-----------|---------|-----------------|--------------|-------------------|
| MMMU | 86.7 | 80.7 | 87.2 | **85.0** |
| MathVision | 83.0 | 74.3 | 86.6 | **88.6** ✨ |
| ZEROBench | 9 | 3 | 10 | **12** ✨ |
| OCRBench | 80.7 | 85.8 | 90.4 | **93.1** ✨ |
| OSWorld-Verified | 38.2 | 66.3 | -- | 62.2 |

**特点**：在 Agent（IFBench、BrowseComp、TAU2）和视觉（MathVision、OCRBench、ZEROBench）方面尤其突出，多项超越闭源模型。

---

## 部署

### 硬件需求

- FP16/BF16 全量：~807GB（HuggingFace）→ 需 8×H100/A100-80G
- GGUF 量化版（Unsloth）：Q8 ~462GB，1-bit ~94.2GB
- TP=8 为推荐配置

### 推理框架

```bash
# SGLang（推荐）
python -m sglang.launch_server --model-path Qwen/Qwen3.5-397B-A17B \
  --port 8000 --tp-size 8 --context-length 262144 --reasoning-parser qwen3

# vLLM
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3

# Transformers serve
transformers serve --force-model Qwen/Qwen3.5-397B-A17B --port 8000 --continuous-batching
```

### 推理参数

| 模式 | temperature | top_p | top_k | presence_penalty |
|------|------------|-------|-------|------------------|
| Thinking（默认） | 0.6 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.8 | 20 | 1.5 |

---

## 与前代及竞品对比

| 特性 | Qwen3-235B-A22B | Qwen3-Max (1T+) | **Qwen3.5-397B-A17B** | DeepSeek-V3.2 | Kimi K2.5-1T-A32B |
|------|-----------------|-----------------|----------------------|---------------|-------------------|
| 架构 | Transformer MoE | Dense (推测) | **Hybrid MoE** | MoE + MLA | MoE |
| 注意力 | GQA | GQA | **Gated DeltaNet + Gated Attn** | MLA | -- |
| 总参/激活 | 235B/22B | 1T+/-- | **397B/17B** | 671B/37B | 1T/32B |
| 专家数 | 128 | -- | **512** | 256 | -- |
| 上下文 | 128K | -- | **256K/1M** | 128K | 128K |
| 词表 | 151K | -- | **248K** | 129K | -- |
| 语言 | 119 | -- | **201** | -- | -- |
| 多模态 | 否（文本） | 否 | **原生多模态** | 否 | 否 |
| 许可 | Apache 2.0 | 商用 | **Apache 2.0** | MIT | -- |

---

## Qwen3.5-Plus vs 开源版差异

| 特性 | Qwen3.5-397B-A17B | Qwen3.5-Plus |
|------|-------------------|--------------|
| 模型权重 | 相同 | 相同 |
| 上下文长度 | 256K（原生） | **1M**（扩展） |
| 内置工具 | 无 | Search、Code Interpreter |
| Adaptive Tool Use | 手动配置 | **内置自适应** |
| 访问方式 | 本地部署/第三方 API | Alibaba Cloud Model Studio |

---

## Qwen3.5-Max（暂未发布）

截至 2026-02-18，Qwen3.5-Max **尚未发布**。时间线参考：
- **Qwen3-Max**：2025-09-05 发布，1T+ 参数闭源旗舰
- **Qwen3-Max-Thinking**：2026-01-27 发布，支持推理模式
- Qwen3.5 目前仅发布 397B-A17B 一个尺寸，官方预告 "More sizes are coming"

---

## 技术脉络

```
Qwen3 (2025-04)
  └─ Qwen3-235B-A22B: 128 experts, GQA, 128K
       │
Qwen3-Next (2025-09)
  └─ Qwen3-Next-80B-A3B: 首次引入 Gated DeltaNet 混合注意力
       │
Qwen3.5 (2026-02)
  └─ Qwen3.5-397B-A17B: 512 experts, Gated DeltaNet + Gated Attn, 256K/1M
```

Qwen3-Next 是架构验证版本（80B-A3B），Qwen3.5 是该架构的全面放大版。

---

## 参考链接

- [官方博客：Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
- [GitHub: QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5)
- [HuggingFace: Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Unsloth GGUF 量化版](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF)
- [IT之家报道](https://news.qq.com/rain/a/20260216A05AT800)
- [Simon Willison 博客](https://simonwillison.net/2026/Feb/17/qwen35/)
- [MarkTechPost 分析](https://www.marktechpost.com/2026/02/16/alibaba-qwen-team-releases-qwen3-5-397b-moe-model-with-17b-active-parameters-and-1m-token-context-for-ai-agents/)
- [Reuters 报道](https://www.reuters.com/world/china/alibaba-unveils-new-qwen35-model-agentic-ai-era-2026-02-16/)
- [Latent Space AINews](https://www.latent.space/p/ainews-qwen35-397b-a17b-the-smallest)
- 关联笔记：[[Qwen 系列架构]]
