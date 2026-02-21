---
title: "KV Cache 原理与优化"
date: 2026-02-14
tags:
  - inference
  - kv-cache
  - optimization
  - interview
type: note
---

> [!warning] 重复笔记
> 同名深入版：[[AI/LLM/Inference/KV Cache]]
> 本篇为 Foundations 面试准备版，建议以 LLM 版为主

# KV Cache 原理与优化

## 1. 什么是 KV Cache？为什么需要它？

### 自回归解码的重复计算问题

Transformer 的文本生成是**自回归（autoregressive）**的：每个 token 的生成依赖于之前所有 token。在标准 Self-Attention 中，生成第 $t$ 个 token 时需要计算：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

如果不做缓存，每生成一个新 token，都要**重新计算之前所有 token 的 K 和 V**。对于长度为 $n$ 的序列，总计算量为 $O(n^2)$ 次 K/V 投影 —— 这些计算中绝大部分是冗余的。

### KV Cache 的核心思想

**缓存已计算的 Key 和 Value 矩阵**，每次只计算新 token 的 $K_t, V_t$ 并追加到缓存中。这样每步的增量计算量从 $O(t \cdot d)$ 降为 $O(d)$（仅一个 token 的投影），总体生成复杂度从 $O(n^2 \cdot d)$ 降为 $O(n \cdot d)$。

## 2. 基本原理

### 工作流程

```
Prefill 阶段（处理 prompt）:
  输入 x_{1:n} → 计算所有层的 K_{1:n}, V_{1:n} → 存入 KV Cache → 输出 token_{n+1}

Decode 阶段（逐 token 生成）:
  Step t:
    1. 输入 token_t
    2. 计算 Q_t, K_t, V_t（仅当前 token）
    3. K_t, V_t 追加到 KV Cache
    4. Q_t 与完整 K_{1:t} 做 attention → 输出 token_{t+1}
```

### 关键细节

- **每层独立缓存**：每个 Transformer 层有自己的 KV Cache
- **每个注意力头独立**：MHA 中每个 head 维护自己的 K/V
- **只在 decoder 中使用**：encoder 一次性处理完整输入，不需要缓存
- **Prefill vs Decode 分离**：Prefill 是 compute-bound（并行计算），Decode 是 memory-bound（逐 token、受显存带宽限制）

## 3. 显存占用计算

### 公式

$$\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_param}$$

其中因子 2 表示 K 和 V 各一份。

### 实例计算

以 **LLaMA-2-70B**（GQA，8 个 KV heads）为例：

| 参数 | 值 |
|------|-----|
| 层数 | 80 |
| KV heads | 8（GQA） |
| head dim | 128 |
| seq_len | 4096 |
| dtype | FP16 (2 bytes) |

$$\text{单请求} = 2 \times 80 \times 8 \times 128 \times 4096 \times 2 \approx 1.34 \text{ GB}$$

> **对比：** 若使用标准 MHA（64 个 heads），则为 $2 \times 80 \times 64 \times 128 \times 4096 \times 2 \approx 10.7 \text{ GB}$

### 为什么 KV Cache 是瓶颈

- 模型权重是**固定的**，KV Cache 随 batch_size × seq_len **线性增长**
- 长上下文（128K+）场景下 KV Cache 轻松超过模型本身的显存占用
- 直接限制了 serving 的**最大并发数**

## 4. 优化方法

### 4.1 Multi-Query Attention（MQA）

**核心思想**：所有 attention heads 共享同一组 K/V，只有 Q 保持多头。

- 来源：Shazeer 2019
- 效果：KV Cache 缩小为原来的 $1/n_{\text{heads}}$
- 代价：少量质量下降（约 0.5% 在大模型上可忽略）
- 典型应用：PaLM、Falcon

### 4.2 Grouped-Query Attention（GQA）

**核心思想**：将 attention heads 分为若干组，每组共享 K/V。是 MHA 和 MQA 的折中。

- 来源：Ainslie et al., 2023
- 关系：MHA → GQA（G 组）→ MQA（1 组）
- 效果：如 LLaMA-2-70B 用 8 个 KV heads / 64 个 Q heads，KV Cache 缩小 8×
- 优势：接近 MHA 的质量 + 接近 MQA 的速度

### 4.3 PagedAttention

**核心思想**：借鉴 OS 虚拟内存的分页机制管理 KV Cache。

- 来源：vLLM（Kwon et al., 2023）
- 问题背景：传统实现为每个请求预分配 max_seq_len 的连续显存 → **内部碎片严重**（平均浪费 60-80%）
- 解决方案：
  - 将 KV Cache 分为固定大小的 **block**（如 16 tokens/block）
  - 通过 **block table**（类似页表）维护逻辑→物理映射
  - 按需分配，用完释放
- 效果：显存利用率接近 100%，throughput 提升 2-4×
- 额外能力：**Copy-on-Write** 实现 beam search 和 prefix sharing 零拷贝

### 4.4 KV Cache 量化

**核心思想**：将缓存的 K/V 从 FP16 量化到 INT8/INT4。

- 方法：
  - **Per-channel 量化**：对每个 head 的 K/V 独立量化
  - **KIVI**（Liu et al., 2024）：K 用 per-channel，V 用 per-token，保持精度
  - **KV Cache 与权重量化正交**：可以叠加使用
- 效果：INT8 量化 KV Cache 缩小 2×，INT4 缩小 4×，精度损失极小
- 在 TensorRT-LLM 中已原生支持 FP8/INT8 KV Cache

### 4.5 Sliding Window Attention

**核心思想**：每个 token 只关注最近 $W$ 个 token，KV Cache 大小固定为 $W$。

- 来源：Mistral（Jiang et al., 2023），Longformer
- 实现：环形缓冲区（Ring Buffer），新 token 覆盖最旧 token
- 效果：KV Cache 从 $O(\text{seq\_len})$ 变为 $O(W)$，支持无限长生成
- 局限：牺牲了全局注意力能力
- 改进：Mistral 通过**层间堆叠**（不同层的窗口错开）实现有效感受野远大于 $W$

### 4.6 其他优化

| 方法 | 思路 |
|------|------|
| **Token Eviction** (H2O, StreamingLLM) | 动态淘汰不重要的 token 的 KV |
| **Multi-head Latent Attention (MLA)** | DeepSeek-V2 提出，对 KV 做低秩压缩后缓存 |
| **Prefix Caching** | 共享 system prompt 的 KV Cache（如 SGLang RadixAttention） |
| **Offloading** | KV Cache 卸载到 CPU/SSD，按需加载 |

## 5. 与 vLLM / TensorRT-LLM 的关系

### vLLM

- 核心创新就是 **PagedAttention**
- 通过分页管理解决了 KV Cache 显存碎片问题
- 支持 **continuous batching**：新请求可以在旧请求完成前加入 batch
- 支持 **prefix caching**：多请求共享相同 prefix 的 KV Cache
- 已成为开源 LLM serving 的事实标准

### TensorRT-LLM

- NVIDIA 官方推理框架，深度优化 GPU kernel
- KV Cache 管理：支持 **Paged KV Cache** + **FP8/INT8 量化**
- **In-flight Batching**：类似 continuous batching
- 与 Triton Inference Server 集成，面向生产环境
- 相比 vLLM 更底层、性能更高，但灵活性和易用性稍差

### 核心关系

```
应用层:        ChatGPT / API Service
                    ↓
Serving 框架:  vLLM / TensorRT-LLM / SGLang
                    ↓
KV Cache 管理: PagedAttention / Paged KV Cache
                    ↓
模型层面优化:   GQA / MLA / Sliding Window
```

## 6. 面试常见问题及回答要点

### Q1: KV Cache 是什么？为什么能加速推理？

**要点**：自回归生成中避免重复计算历史 token 的 K/V 投影。将 decode 阶段每步计算从 $O(t)$ 降为 $O(1)$（投影部分），代价是显存占用。强调 **空间换时间** 的本质。

### Q2: KV Cache 的显存占用怎么算？瓶颈在哪？

**要点**：给出公式 $2 \times L \times h \times d \times s \times b \times \text{dtype}$。强调它随 seq_len 和 batch_size 线性增长，长上下文场景（128K）下可能超过模型权重本身。瓶颈在于限制了最大并发。

### Q3: MQA、GQA、MHA 的区别和 trade-off？

**要点**：三者是一个连续谱。MHA → 质量最高、KV Cache 最大；MQA → KV Cache 最小但质量有损；GQA → 工程上的最优折中（LLaMA-2/3 的选择）。画图说明 heads 的共享关系加分。

### Q4: PagedAttention 解决了什么问题？

**要点**：解决 KV Cache 的**内存碎片**问题。传统方式预分配 max_seq_len 连续内存导致巨大浪费。PagedAttention 借鉴 OS 分页，按需分配 block，利用率接近 100%。提到 copy-on-write 支持 beam search 加分。

### Q5: Prefill 和 Decode 阶段有什么区别？

**要点**：
- Prefill：处理整个 prompt，**compute-bound**，可以高度并行，构建初始 KV Cache
- Decode：逐 token 生成，**memory-bound**，受限于显存带宽读取 KV Cache
- 这一区别决定了优化方向完全不同：prefill 优化算力利用率，decode 优化显存带宽

### Q6: 如何支持长上下文（128K+）推理？

**要点**：多管齐下 —— GQA 减少 KV 头数、KV Cache 量化（FP8/INT8）减少每个元素大小、Sliding Window 限制缓存长度、PagedAttention 减少碎片、Offloading 利用 CPU 内存。提到 Ring Attention（跨设备分布式 KV Cache）更佳。

---

## See Also

- [[AI/LLM/Inference/KV Cache|KV Cache（LLM 深度版）]] — 本文面试版，深度版含 PagedAttention/vLLM/MLA/量化实现
- [[AI/LLM/Architecture/长上下文技术|长上下文技术]] — 128K+ 场景下 KV Cache 是核心瓶颈；GQA/Sliding Window/Ring Attention 均是 KV Cache 的扩展方案
- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]] — MLA/GQA/MQA 从架构层减少 KV heads，是 KV Cache 的根本优化方向
- [[AI/Foundations/Inference/Speculative Decoding|Speculative Decoding（推理加速）]] — KV Cache 优化显存，Speculative Decoding 优化 latency；两者是推理加速的两个独立维度，组合使用
