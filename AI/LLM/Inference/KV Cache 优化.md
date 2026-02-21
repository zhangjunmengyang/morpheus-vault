---
title: "KV Cache 优化全景"
date: 2026-02-13
tags:
  - ai/llm/inference
  - ai/llm/optimization
  - type/concept
  - interview/hot
status: deprecated
---

> [!warning] ⚠️ Deprecated
> 本文为早期精简版（265行）。完整版见 [[AI/LLM/Inference/KV Cache]]（830行，含原理详解 + 优化全景 + 面试版链接）。保留供参考，不再维护。

# KV Cache 优化全景

> PagedAttention、GQA/MQA、FlashAttention — LLM 推理的三大核心优化

## 1. KV Cache 基础：为什么需要缓存？

### 自回归推理的重复计算问题

LLM 推理分两个阶段：
1. **Prefill（预填充）**：处理整个 prompt，一次性生成所有 token 的 KV
2. **Decode（解码）**：逐 token 生成，每步只需计算新 token 的 QKV

```
没有 KV Cache:
Step 1: Attention(Q=[t1], K=[t1], V=[t1])
Step 2: Attention(Q=[t2], K=[t1,t2], V=[t1,t2])  ← t1 重复计算
Step 3: Attention(Q=[t3], K=[t1,t2,t3], V=[t1,t2,t3])  ← t1,t2 重复

有 KV Cache:
Step 1: Attention(Q=[t1], K=[t1], V=[t1]), cache K1,V1
Step 2: Attention(Q=[t2], K=cache+[t2], V=cache+[t2]), 追加缓存
Step 3: Attention(Q=[t3], K=cache+[t3], V=cache+[t3]), 追加缓存
```

### KV Cache 的显存开销

```
KV Cache 大小 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_size

以 LLaMA-70B 为例 (FP16):
= 2 × 80 × 64 × 128 × 4096 × 1 × 2 bytes
= ~5.4 GB (单条序列!)
```

批处理 32 条序列 → **~173 GB KV Cache**。这就是为什么 KV Cache 优化如此关键。

## 2. PagedAttention — vLLM 的核心技术

### 2.1 问题：内存碎片化

传统 KV Cache 为每个序列分配连续内存块，但：
- 序列长度不可预知 → 需要预分配最大长度 → **巨大浪费**
- 不同序列长短不一 → 内存碎片化严重
- 实测浪费率高达 **60-80%**

### 2.2 PagedAttention 方案

借鉴操作系统的虚拟内存/分页机制：

```
传统方式（连续分配）:
Seq 1: [████████████________]  ← 预分配但未用完
Seq 2: [██████______________]  ← 大量浪费
       ← 无法分配给新序列 →

PagedAttention（分页分配）:
Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7]...
Seq 1: B0→B2→B5→B7  (逻辑连续，物理分散)
Seq 2: B1→B3→B6      (按需分配新 block)
```

核心特性：
- **固定大小 Block**（通常 16 tokens/block）
- **动态分配**：序列增长时按需申请新 block
- **Block 共享**：beam search 时多个 beam 共享 prompt 的 KV block（Copy-on-Write）
- **近零浪费**：内存利用率从 20-40% 提升到 **>95%**

### 2.3 vLLM 实现

[[AI/LLM/Inference/vLLM|vLLM]] 是 PagedAttention 的原生实现：

```python
# vLLM 启动示例
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,  # 90% GPU 显存用于 KV Cache
    max_model_len=8192,
    block_size=16,  # PagedAttention block 大小
)

# vLLM 自动管理 KV Cache 的分页分配
outputs = llm.generate(prompts, SamplingParams(max_tokens=512))
```

## 3. GQA / MQA — 从架构层面减少 KV Cache

### 3.1 注意力头变体对比

```
MHA (Multi-Head Attention):
Q: [H1][H2][H3][H4][H5][H6][H7][H8]
K: [H1][H2][H3][H4][H5][H6][H7][H8]  ← 每个 Q 头有独立 KV
V: [H1][H2][H3][H4][H5][H6][H7][H8]
→ KV Cache: 8 × head_dim × 2

GQA (Grouped-Query Attention):   ← LLaMA-3 使用
Q: [H1][H2][H3][H4][H5][H6][H7][H8]
K: [G1    ][G2    ][G3    ][G4    ]  ← 每组共享 KV
V: [G1    ][G2    ][G3    ][G4    ]
→ KV Cache: 4 × head_dim × 2 (减少 50%)

MQA (Multi-Query Attention):     ← PaLM 使用
Q: [H1][H2][H3][H4][H5][H6][H7][H8]
K: [      Single Head            ]  ← 所有 Q 头共享一组 KV
V: [      Single Head            ]
→ KV Cache: 1 × head_dim × 2 (减少 87.5%)
```

### 3.2 实际影响

| 模型 | 注意力类型 | KV Cache 相对大小 |
|------|-----------|------------------|
| GPT-3 | MHA | 100% |
| PaLM | MQA | ~12.5% |
| LLaMA-3-70B | GQA (8 groups) | ~12.5% |
| DeepSeek-V2/V3 | MLA | ~5-10% |

DeepSeek 的 **MLA (Multi-head Latent Attention)** 更激进——将 KV 投影到低秩隐空间：

```python
# MLA 的核心思想（简化版）
class MLA(nn.Module):
    def __init__(self, d_model, d_compressed, num_heads):
        self.down_proj = nn.Linear(d_model, d_compressed)  # 压缩
        self.up_proj_k = nn.Linear(d_compressed, d_model)  # 还原 K
        self.up_proj_v = nn.Linear(d_compressed, d_model)  # 还原 V
    
    def forward(self, x):
        compressed = self.down_proj(x)  # 只缓存这个!
        k = self.up_proj_k(compressed)  # 推理时还原
        v = self.up_proj_v(compressed)
        return k, v
    # KV Cache 只需存 compressed，而非完整 K、V
```

## 4. FlashAttention — IO 感知的注意力计算

### 4.1 核心问题：内存墙

标准注意力的瓶颈不在计算（FLOPs），而在**内存读写（IO）**：

```
标准 Attention 的 IO 模式:
1. 从 HBM 读取 Q, K → 计算 S = QK^T → 写回 HBM  (O(n²) IO)
2. 从 HBM 读取 S → 计算 P = softmax(S) → 写回 HBM  (O(n²) IO)
3. 从 HBM 读取 P, V → 计算 O = PV → 写回 HBM  (O(n²) IO)

总 IO: O(n²) × 3 次 HBM 读写
```

GPU SRAM（片上缓存）速度是 HBM 的 **10-20 倍**，但容量只有几十 MB。

### 4.2 FlashAttention v1/v2 原理

核心思想：**分块计算（Tiling）+ 在线 Softmax**

```
FlashAttention:
将 Q, K, V 分成小块（适配 SRAM 大小）
for each Q_block:
    for each K_block, V_block:
        在 SRAM 中计算 partial attention
        使用 online softmax 增量更新结果
    写出最终结果到 HBM

总 IO: O(n²d / M)，其中 M = SRAM 大小
```

FlashAttention v2 改进：
- 更好的并行策略（沿 sequence 维度并行）
- 减少非矩阵运算（warp 级别优化）
- **速度比 v1 快 2x**

### 4.3 FlashAttention v3 (Hopper 架构)

针对 NVIDIA H100 (Hopper) 的深度优化：

- **WGMMA 指令**：使用 Tensor Core 的异步矩阵乘法
- **Pingpong 调度**：两个 warpgroup 交替执行 GEMM 和 softmax，流水线化
- **FP8 支持**：配合 H100 的 FP8 Tensor Core，吞吐再翻倍
- **Block Quantization**：逐 block 动态量化 + 非相干处理

```
FlashAttention v3 性能 (H100):
- FP16: ~740 TFLOPS (接近硬件上限 989 TFLOPS 的 75%)
- FP8:  ~1.2 PFLOPS
- 比 FlashAttention v2 快 1.5-2x
```

### 4.4 FlashInfer

vLLM 集成的推理优化注意力内核：
- 专为 **Paged KV Cache + GQA** 场景优化
- 融合 decode attention 操作
- 支持 prefetch 和 continuous batching

## 5. KV Cache 压缩与 Offloading

### 5.1 KV Cache 量化

```python
# KV Cache 量化（INT8/FP8）
# 将 KV Cache 从 FP16 量化到 INT8，显存减半
# vLLM 支持 kv_cache_dtype 参数
llm = LLM(
    model="...",
    kv_cache_dtype="fp8",  # 或 "fp8_e5m2"
    quantization="fp8",
)
```

### 5.2 KV Cache Offloading

- **CPU Offload**：将不活跃序列的 KV Cache 移到主机内存
- **NVMe Offload**：NVIDIA BlueField-4 的 KVCache 方案（CES 2026 发布）
- **CXL Memory**：利用 CXL 扩展的远端内存池

### 5.3 其他优化技术

| 技术 | 原理 | 节省比例 |
|------|------|----------|
| Prefix Caching | 共享相同 prompt 前缀的 KV Cache | 40-70% |
| Sliding Window | 只保留最近 N 个 token 的 KV | 固定上限 |
| H2O (Heavy Hitter) | 只保留注意力分数高的 token KV | 50-80% |
| SnapKV | 按注意力模式自动选择保留 | 60-80% |
| StreamingLLM | 保留 sink token + 最近 token | 90%+ |

## 6. 面试常见问题

**Q1: KV Cache 的大小怎么计算？**
A: `2 × L × H × D × S × B × dtype_bytes`。2 是 K+V，L 是层数，H 是 KV 头数（注意 GQA 下头数可能很少），D 是 head_dim，S 是序列长度，B 是 batch_size。

**Q2: PagedAttention 为什么能提升吞吐量？**
A: 核心是提高**内存利用率**。传统方案碎片浪费 60-80% 显存，PagedAttention 近零浪费 → 同等显存能处理更多并发请求。另外 block 共享也节省重复存储。

**Q3: GQA 和 MQA 会损失精度吗？**
A: 轻微。MQA 精度损失更大但 KV Cache 最小。GQA 是折中方案。LLaMA-3 使用 GQA 在精度和效率间取得了良好平衡。实际 GQA 8 组的精度损失几乎可忽略。

**Q4: FlashAttention 能减少计算量吗？**
A: 不能。FlashAttention 的 FLOPs 和标准 Attention 一样，甚至因为重计算略多。它优化的是 **IO**——减少 HBM 读写次数，利用 SRAM 的高带宽。

**Q5: 如何选择 KV Cache 优化策略？**
A: 分层叠加：GQA/MLA（架构层面）→ FlashAttention（计算层面）→ PagedAttention（内存管理）→ 量化/Offload（进一步压缩）。这些不冲突，现代系统全部组合使用。

## 相关链接

- [[AI/LLM/Inference/vLLM|vLLM]] — PagedAttention 原生实现
- [[AI/LLM/Architecture/LLaMA|LLaMA]] — GQA 采用者
- [[AI/LLM/Architecture/DeepSeek-R1|DeepSeek-R1]] — MLA 架构
- [[AI/LLM/Architecture/MoE 深度解析|MoE 深度解析]] — 与 Expert Parallelism 结合
- [[AI/LLM/Inference/推理优化|推理优化]] — 更多优化技术
