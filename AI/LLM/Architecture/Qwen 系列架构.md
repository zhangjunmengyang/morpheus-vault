---
brief: "Qwen 系列架构详解——Qwen 1/2/2.5/3 的架构演进：GQA/SwiGLU/RoPE/Tie-Embedding/YaRN 等技术的逐版本引入；MoE 版本（Qwen3-MoE）的 Expert 设计；面试被问国内主流开源 LLM 架构的参考。"
title: "Qwen 系列架构详解"
date: 2026-02-14
tags: [llm, qwen, architecture, interview]
type: note
---

# Qwen 系列架构详解

## 1. 发展脉络

| 版本 | 时间 | 关键变化 |
|------|------|----------|
| **Qwen (1.0)** | 2023-08 | 阿里云首个开源大模型，1.8B–72B，基于 Transformer decoder-only |
| **Qwen1.5** | 2024-02 | 对齐升级（DPO+PPO）、支持 32k 上下文、chat 模板统一为 ChatML |
| **Qwen2** | 2024-06 | 架构迭代：GQA 全面启用、支持 128k 上下文（YaRN）、多语言 27→29 种、新增 57B-A14B MoE |
| **Qwen2.5** | 2024-09 | 全系列开源旗舰：0.5B–72B Dense + 3B-A3B MoE，预训练 18T tokens，代码/数学专项模型 |
| **Qwen3** | 2025-04 | 混合思维模式（thinking/non-thinking 动态切换）、MoE 旗舰 235B-A22B、支持 119 种语言和方言 |
| **Qwen3-Next** | 2025-09 | 架构验证版（80B-A3B），首次引入 Gated DeltaNet 混合线性注意力 |
| **Qwen3.5** | 2026-02 | [[AI/LLM/Architecture/Qwen3.5-Plus\|详细笔记]]。Qwen3-Next 架构放大版：397B-A17B MoE（512 专家），Gated DeltaNet + Gated Attention 3:1 混合，原生多模态 Early Fusion，201 种语言，256K/1M 上下文，吞吐 19× 提升 |

### 核心趋势

- **数据规模**：从 Qwen1 的 2.4T → Qwen2.5 的 18T → Qwen3 的 36T+ tokens
- **对齐方法**：SFT → SFT+RLHF → SFT+DPO → 多阶段 RL（GRPO/DAPO）
- **上下文长度**：2k → 8k → 32k → 128k → 1M（Qwen2.5-Turbo）
- **开源策略**：从仅开放推理到 Apache 2.0 全面开源（Qwen2.5 起）

---

## 2. 架构特点

Qwen 系列基于 **Transformer decoder-only** 架构，核心组件：

### 2.1 GQA (Grouped-Query Attention)

- 介于 MHA 和 MQA 之间：将 query heads 分组，每组共享一套 KV head
- Qwen2/2.5/3 全面采用，KV head 数量显著少于 query head（如 72B：64 query heads / 8 KV heads）
- **优势**：推理时 KV cache 内存降至 MHA 的 1/8，吞吐量提升 30-40%，质量接近 MHA

### 2.2 SwiGLU FFN

- 替代传统 ReLU/GELU FFN：$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$
- 引入门控机制（gating），FFN 从 2 个矩阵变为 3 个（gate/up/down）
- **隐藏层维度**：通常为 $\frac{8}{3}d_{model}$（对齐到 128 的倍数）
- 参数略增但效果显著优于 GELU，LLaMA/Qwen/Mistral 均采用

### 2.3 RMSNorm (Root Mean Square Normalization)

- 简化 LayerNorm：去掉 mean centering，只做 RMS 归一化
- $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$
- 计算量减少 ~15%，训练稳定性不受影响
- Qwen 全系列采用 **Pre-Norm**（Norm 在 attention/FFN 之前）

### 2.4 RoPE (Rotary Position Embedding)

- 通过旋转矩阵将位置信息编码到 query/key 中
- 天然支持相对位置、可外推到更长序列
- Qwen2+ 使用 **YaRN**（Yet another RoPE extensioN）扩展至 128k
- Qwen2.5-Turbo 进一步使用 dual-chunk attention 支持 1M 上下文

### 2.5 其他设计选择

- **词表**：151,643 tokens（含多语言 + 代码），基于 tiktoken（BPE）
- **Embedding 共享**：input/output embedding **不**共享（tied=false）
- **Bias**：attention 中 QKV 有 bias，FFN 无 bias（Qwen2 起）
- **激活函数**：SiLU（Swish）用于 gate projection

---

## 3. Qwen2.5 全系列

### Dense 模型

| 参数量 | Layers | d_model | Heads (Q/KV) | 上下文 | 用途 |
|--------|--------|---------|---------------|--------|------|
| 0.5B | 24 | 896 | 14/2 | 128k | 端侧/嵌入式 |
| 1.5B | 28 | 1,536 | 12/2 | 128k | 手机端 |
| 3B | 36 | 2,048 | 16/2 | 128k | 边缘设备 |
| 7B | 28 | 3,584 | 28/4 | 128k | 通用基准 |
| 14B | 40 | 5,120 | 40/8 | 128k | 平衡性价比 |
| 32B | 64 | 5,120 | 40/8 | 128k | 高性能 |
| 72B | 80 | 8,192 | 64/8 | 128k | 旗舰开源 |

### 专项模型

- **Qwen2.5-Coder**：代码专项，7B/14B/32B，SWE-bench Verified 第一梯队
- **Qwen2.5-Math**：数学推理，支持 CoT 和 tool-integrated reasoning（TIR）
- **Qwen2.5-Turbo**：API-only，1M 上下文，推理速度优化

### MoE 版本

| 模型 | 总参数 | 激活参数 | 专家数 | 激活专家 | 共享专家 |
|------|--------|----------|--------|----------|----------|
| Qwen2-57B-A14B | 57B | 14B | 64 | 8 | — |
| Qwen2.5-3B-A3B (推测) | ~16B | 3B | 64 | 8 | 1 |
| **Qwen3-235B-A22B** | 235B | 22B | 128 | 8 | 1 |
| Qwen3-30B-A3B | 30B | 3B | 128 | 8 | 1 |
| Qwen3-Next-80B-A3B | 80B | 3B | — | — | — |
| **Qwen3.5-397B-A17B** | 397B | 17B | 512 | 10 | 1 |

**MoE 关键设计**：
- **Router**：top-k 路由（通常 k=8），每 token 选择 8 个专家
- **Shared Expert**：Qwen3 引入 1 个全局共享专家，处理通用知识
- **负载均衡**：auxiliary loss + expert-level load balancing
- 推理效率：235B 总参但激活仅 22B，推理 FLOPS 接近 22B Dense 模型

---

## 4. 与 LLaMA / DeepSeek 架构对比

| 特性 | Qwen2.5-72B | LLaMA 3.1-70B | DeepSeek-V3 (671B-A37B) |
|------|-------------|---------------|--------------------------|
| **架构** | Dense Decoder | Dense Decoder | MoE Decoder |
| **注意力** | GQA (64/8) | GQA (64/8) | MLA (Multi-head Latent Attn) |
| **FFN** | SwiGLU | SwiGLU | SwiGLU + DeepSeekMoE |
| **归一化** | RMSNorm (Pre) | RMSNorm (Pre) | RMSNorm (Pre) |
| **位置编码** | RoPE + YaRN | RoPE | RoPE + YaRN |
| **上下文** | 128k | 128k | 128k |
| **词表大小** | 151,643 | 128,256 | 129,280 |
| **训练数据** | 18T tokens | 15T tokens | 14.8T tokens |
| **KV Cache 优化** | GQA 压缩 | GQA 压缩 | MLA 低秩压缩 (93%↓) |
| **特色** | 全面均衡开源 | 社区生态最强 | MLA + 无 aux loss MoE |

### 关键差异分析

- **KV Cache**：DeepSeek MLA 通过低秩投影将 KV cache 压缩到极致（仅需 ~512 维 latent），远优于 GQA
- **MoE 路由**：DeepSeek 使用无辅助损失的负载均衡策略，Qwen3 仍用 auxiliary loss
- **训练效率**：DeepSeek-V3 使用 FP8 训练，Qwen 主要使用 BF16
- **开源程度**：Qwen2.5 Apache 2.0，LLaMA 3.1 自定义许可（限商用），DeepSeek MIT

---

## 5. 多模态系列

### Qwen-VL / Qwen2-VL / Qwen2.5-VL

- **视觉编码器**：ViT（Vision Transformer），Qwen2-VL 使用 naive dynamic resolution（原生动态分辨率）
- **跨模态对齐**：单层 cross-attention adapter（Qwen-VL）→ MLP projector（Qwen2-VL）
- **视频理解**：支持任意长度视频输入，通过动态帧采样
- **Qwen2.5-VL**：支持结构化输出（bbox/JSON），agent 能力（GUI grounding）

### Qwen-Audio / Qwen2-Audio

- 音频编码器：Whisper-large-v3
- 支持语音理解、音频分析、多轮语音对话

### Qwen2.5-Omni

- 全模态模型：文本/图像/音频/视频输入 → 文本/语音输出
- 实时流式处理，端到端语音对话
- "Thinker-Talker" 架构：Thinker 负责多模态推理，Talker 负责语音生成

---

## 6. 面试题

### Q1: Qwen 系列为什么从 MHA 切换到 GQA？对推理有什么影响？

**答**：MHA 中每个 attention head 都有独立的 K、V 投影，KV cache 与 head 数成正比，在长序列推理时成为内存瓶颈。GQA 将多个 query head 分组共享同一组 KV head（如 Qwen2.5-72B 用 64 个 query head 共享 8 组 KV），KV cache 减少为原来的 1/8。这使得：① batch size 可以更大（内存省了）→ 吞吐量提升 30-40%；② 长序列推理（128k）变得可行；③ 质量损失极小（< 0.5% benchmark 差距）。权衡点是 KV group 数越少速度越快但质量越低，Qwen 选择 8 组是经验最优。

### Q2: SwiGLU 相比 GELU FFN 的优势是什么？为什么现在成为主流？

**答**：SwiGLU 引入了门控机制（$\text{Swish}(xW_{gate}) \otimes xW_{up}$），门控信号可以学习性地"关闭"某些维度的信息流，相当于 FFN 内部增加了一层动态特征选择。虽然参数量增加约 50%（2 矩阵 → 3 矩阵），但通过缩小隐藏维度（8/3 d 替代 4d）保持 FLOPS 大致不变。PaLM 论文实验显示 SwiGLU 在同等计算预算下 loss 更低 ~2-3%，该优势在大规模训练中非常显著。LLaMA 引入后成为事实标准，Qwen/Mistral/DeepSeek 均跟进。

### Q3: Qwen3 的"混合思维模式"是什么？与 DeepSeek-R1 的思维链有何区别？

**答**：Qwen3 支持在同一模型中动态切换 **thinking**（显式 CoT 推理，输出 `<think>` 块）和 **non-thinking**（直接回答）模式，用户通过 `enable_thinking` 参数控制，或模型自行判断。与 DeepSeek-R1 的区别：① R1 是专门训练的推理模型，thinking 是默认且唯一模式；② Qwen3 通过统一训练（先 RL 训练 thinking → 再混合 SFT 融合 non-thinking），一个模型兼顾两种场景；③ Qwen3 的 thinking budget 可通过 `thinking_budget` 参数动态分配 token 预算，实现速度-质量权衡。

### Q4: MoE 架构中"共享专家"的作用是什么？Qwen3-235B 为什么只用 1 个共享专家？

**答**：共享专家（shared expert）对所有 token 都激活，负责处理跨领域的通用知识和基础语言能力，避免这些能力被分散到各个路由专家中导致冗余。作用：① 减少专家间的知识重复；② 提供稳定的"基线"能力，使路由专家可以更专注于细分领域；③ 缓解 token dropping 时的质量退化。只用 1 个的原因：更多共享专家意味着更多总是激活的参数（增加推理成本），实验表明 1 个共享专家已经能充分捕获通用模式，增加到 2+ 个边际收益递减，而激活参数从 22B 增加到 25B+ 会显著影响推理效率。

### Q5: 如何将 Qwen2.5-72B 部署到生产环境？需要考虑哪些优化？

**答**：核心考量：
1. **量化**：FP16 需要 ~144GB 显存（≥2×A100-80G），可用 GPTQ/AWQ 4-bit 量化降至 ~40GB（单卡 A100），质量损失约 1-2%
2. **推理框架**：vLLM（PagedAttention 动态 KV cache 管理）或 SGLang（RadixAttention 前缀缓存），支持 continuous batching
3. **KV Cache**：128k 上下文单请求 KV cache 约 5GB（GQA 8 组），需做 context length 限制或 KV cache offloading
4. **Tensor Parallelism**：多卡部署用 TP（同节点）或 PP（跨节点），TP=2/4 是常见配置
5. **Speculative Decoding**：用 Qwen2.5-0.5B 做 draft model 加速 2-3x
6. **前缀缓存**：系统 prompt 较长时用 prefix caching 避免重复计算
7. **监控**：关注 TTFT（首 token 延迟）、TPS（token/s 吞吐）、GPU 利用率、KV cache 命中率

## See Also

- [[AI/LLM/Architecture/MoE 深度解析]] — MoE 架构深度（Qwen3-MoE 的 shared expert + routing 原理）
- [[AI/LLM/Architecture/Attention 变体综述]] — GQA/MQA/MLA 详细对比（Qwen 从 MHA→GQA 的理论基础）
- [[AI/LLM/Inference/LLM-推理优化-2026-全景]] — Qwen2.5-72B 生产部署优化全图（量化/TP/prefix cache）
- [[AI/LLM/Architecture/Transformer架构深度解析-2026技术全景]] — Qwen 所基于的 Decoder-Only Transformer 架构全景
- [[AI/LLM/Architecture/DeepSeek-R1]] — 同期国内主流开源 LLM 对比（架构选择差异：MoE规模/稀疏度/共享专家数）

## 推荐阅读

- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115) — arXiv:2412.15115
- [Qwen3 技术博客](https://qwenlm.github.io/blog/qwen3/) — 混合思维模式设计细节
