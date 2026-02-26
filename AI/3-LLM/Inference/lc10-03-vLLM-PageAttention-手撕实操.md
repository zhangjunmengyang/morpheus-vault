---
title: vLLM PageAttention 手撕实操
brief: 从零实现 PageAttention：把 KV Cache 逻辑序列拆分为固定大小物理 Block，实现 Block 级别内存管理（alloc/free/copy-on-write），解决内存碎片。理解与 OS 虚拟内存管理的类比以及为什么 PageAttention 是 vLLM 的核心创新。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - inference
  - paged-attention
  - vllm
  - lc10
related:
  - "[[AI/LLM/MA-RLHF课程/lc10-推理系统-MOC]]"
  - "[[LLM-推理优化-2026-全景]]"
  - "[[vLLM]]"
---

# vLLM PageAttention 手撕实操

> **来源**: MA-RLHF lc10_inference / vLLM-PageAttention.ipynb
> **系列**: [[AI/LLM/MA-RLHF课程/lc10-推理系统-MOC|lc10-推理系统-MOC]]
> **关联**: [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] | [[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]] | [[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]]
> **日期**: 2026-02-25

---

## TL;DR

PageAttention 是 vLLM 的 Attention Kernel——在**物理上不连续**的 KV page 上直接计算 Attention，避免了 `get_sequence_kvcache` 的内存聚合开销。

**核心创新**：把 batch attention 变成 **request-wise attention**：
- Prefill：`X[total_pages, page_size, dim]`（多请求的 pages 拼成一个 tensor），对每个请求切分后分别做 attention
- Decoding：q 按 `request_num_pages` 复制分发，对不连续 KV pages 做 block attention，再用 **Online Softmax** 聚合 page-wise 结果

**与 FlashAttention 的关系**：正交互补
- FlashAttention：解决单一连续 KV 的 IO 效率问题（SRAM tiling，减少 HBM 读写）
- PageAttention：解决不连续 KV pages 的 Attention 调度问题（block-level 聚合）
- 生产系统：两者叠加使用（PA 调度 + FA 做 block 内计算）

---

## 核心架构

```
PageAttentionDecoderBlock
├── forward_prefill()   # 多请求合并 → request-wise attention → 各自独立计算
└── forward_decoding()  # q 复制分发 → block attention → Online Softmax combine
    └── combine_result()  # 核心：O, M, L 三元组跨 page 合并
    
Attention Backends（可插拔）:
├── MultiHeadsAttention()  # 标准 Attention（reshape 聚合）
└── FlashAttention()       # Block-Tiled Attention（SRAM 友好）
```

---

## 关键概念：输入 Tensor 形状变化

### 传统 Attention（vLLM-PageKVCache）
```
KV 形状: [batch_size, max_total_len, num_heads, head_dim]
         ↑ 聚合后的连续张量，内存拷贝代价 O(L)
```

### PageAttention
```
KV 形状: [num_physical_pages, page_size, num_heads, head_dim]
         ↑ 直接操作物理 page，零拷贝
Prefill: X[total_pages, page_size, dim]  # 所有请求的 pages 拼成一个 tensor
Decoding: KV_Cache = [k_cache, v_cache]  # 物理 KV 存储（不聚合）
```

---

## Prefill：Request-wise Page Attention

```python
def forward_prefill(self, X, request_num_pages, attention_backend, ...):
    """
    Args:
        X: [total_pages, page_size, dim]   # 2 requests → [5, page_size, dim]
        request_num_pages: [3, 2]           # request 0 用 3 pages，request 1 用 2 pages
        attention_backend: MultiHeadsAttention 或 FlashAttention
    """
    B, T, _ = X.shape  # B=total_pages, T=page_size
    H, D = self.num_heads, self.head_dim
    
    Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
    Q = Q.reshape(B, T, H, D).transpose(1, 2)  # [pages, H, T, D]
    K = K.reshape(B, T, H, D).transpose(1, 2)
    V = V.reshape(B, T, H, D).transpose(1, 2)
    O = torch.zeros_like(Q)
    
    # ★ 核心：按 request 切片，各自独立做 attention
    offset = 0
    for t, num_pages in enumerate(request_num_pages):
        Q_ = Q[offset : offset + num_pages]   # [num_pages, H, T, D]
        K_ = K[offset : offset + num_pages]
        V_ = V[offset : offset + num_pages]
        
        # 每个 request 的 attention 完全独立（不会跨 request 混入）
        O_ = attention_backend(Q_, K_, V_)     # [num_pages, H, T, D]
        O[offset : offset + num_pages] = O_
        offset += num_pages
    
    O = O.transpose(1, 2).reshape(B, T, H * D)
    O = self.WO(O)
    O = X + self.act(O)
    return O, [K.transpose(1, 2), V.transpose(1, 2)]
```

**关键设计**：`X` 虽然是 `[total_pages, page_size, dim]`，但 batch 维度是 pages 而非 requests。对每个 request 切 `num_pages` 个 pages，独立计算，写回对应位置。

---

## Backend 1：MultiHeadsAttention（标准 Attention）

```python
def MultiHeadsAttention(Q, K, V, mask=None):
    """
    Args:
        Q/K/V: [num_pages, num_heads, page_size, head_dim]  # 单个 request 的所有 pages
    """
    B, H, T, D = Q.shape  # B=num_pages, T=page_size
    
    # 把 pages 展平为完整序列
    Q = Q.transpose(0, 1).reshape(H, B * T, D)   # [H, total_tokens, D]
    K = K.transpose(0, 1).reshape(H, B * T, D)
    V = V.transpose(0, 1).reshape(H, B * T, D)
    
    S = Q @ K.transpose(1, 2)                      # [H, total_tokens, total_tokens]
    P = F.softmax(S, dim=-1)
    Z = P @ V                                      # [H, total_tokens, D]
    O = Z.reshape(H, B, T, D).transpose(0, 1)     # 恢复 [pages, H, T, D]
    
    return O
```

**问题**：`reshape` 把 pages 合并为连续序列，有 O(B×T) 内存拷贝，且要求 B×T 连续。这在真实 CUDA 中仍需 gather。

---

## Backend 2：FlashAttention（Block-Tiled Attention）

```python
def FlashAttention(Q, K, V, mask=None):
    """
    1 Request，page 级 tiling + Online Softmax，避免完整 softmax
    Args:
        Q/K/V: [num_pages, num_heads, page_size, head_dim]
    """
    N, H, T, D = Q.shape  # N=num_pages
    
    O_global = torch.zeros(N, H, T, D)
    
    for i in range(N):        # Q Loop (遍历 Q 的每个 block)
        O = torch.zeros(1, H, T, 1)
        M = torch.zeros(1, H, T, 1)   # running max（数值稳定）
        L = torch.zeros(1, H, T, 1)   # running sum of exp
        
        Q_ = Q[i]  # 第 i 个 page 的 Q
        
        for j in range(N):    # KV Loop (遍历所有 KV block)
            if j > i:
                continue      # 因果 mask：Q[i] 只看 K[0..i]
            
            K_, V_ = K[j], V[j]
            
            # Block-level attention score
            S_ij = Q_ @ K_.transpose(1, 2)           # [H, T, T]
            M_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
            
            # Online Softmax update（核心公式）
            M_new = torch.maximum(M_ij, M)           # 更新全局 max
            P_ij = torch.exp(S_ij - M_new)
            L_ij = torch.sum(P_ij, dim=-1, keepdim=True)
            L_new = torch.exp(M - M_new) * L + L_ij  # 重缩放旧 L
            O_i = torch.exp(M - M_new) * O + P_ij @ V_  # 重缩放旧 O
            
            M, L = M_new, L_new
        
        # 归一化
        O_global[i] = (O_i / L_new).unsqueeze(0)
    
    return O_global
```

**Online Softmax 公式推导**：

传统 softmax 需要先扫描全部 token 得到 max 再计算 exp。Online Softmax 分块处理，每看到新 block 就更新：

$$M_{new} = \max(M_{old}, M_{block})$$
$$L_{new} = e^{M_{old} - M_{new}} \cdot L_{old} + \sum_j e^{S_j - M_{new}}$$
$$O_{new} = e^{M_{old} - M_{new}} \cdot O_{old} + P_{block} \cdot V_{block}$$

最终 $O_{final} = O_{new} / L_{new}$，数学上等价于全局 softmax。

---

## Decoding：q 复制分发 + Block Combine

Decoding 时每个请求只有 1 个 token（q），但要 attend 到 N 个不连续 pages 的历史 KV。

```python
def forward_decoding(self, X, request_num_pages, KV_Cache):
    """
    Args:
        X: [num_requests, 1, dim]      # 每个请求当前 token
        request_num_pages: [3, 1, 2, 2]  # 每个请求的 KV page 数
        KV_Cache: [k_cache, v_cache]   # [num_physical_pages, page_size, H, D]
    """
    B, T, _ = X.shape   # B=num_requests, T=1
    
    q, k, v = self.WQ(X), self.WK(X), self.WV(X)
    q = q.reshape(B, 1, H, D).transpose(1, 2)  # [B, H, 1, D]
    
    # 初始化：q 自注意力（当前 token 自身）
    S  = q @ k.transpose(2, 3)    # [B, H, 1, 1]
    M_ = S.clone()
    L_ = torch.ones_like(M_)
    O_ = v
    
    # ★ 关键：按 request 的 page 数复制 q（dispatch）
    # request_num_pages = [3, 1, 2, 2] → q 扩展为 [3+1+2+2=8, H, 1, D]
    repeat_tensor = torch.tensor(request_num_pages)
    q_ = torch.repeat_interleave(q, repeat_tensor, dim=0)  # [total_pages, H, 1, D]
    
    # Block Attention：q_ 对应的 KV pages
    K_, V_ = KV_Cache[0], KV_Cache[1]  # [total_pages, page_size, H, D]
    S = q_ @ K_.transpose(1, 2).transpose(2, 3)  # [total_pages, H, 1, page_size]
    M, _ = torch.max(S, dim=-1, keepdim=True)
    L = torch.sum(torch.exp(S - M), dim=-1, keepdim=True)
    P = torch.softmax(S, dim=-1)
    O = P @ V_.transpose(1, 2)    # [total_pages, H, 1, D]
    
    # ★ Reduce：对每个 request，combine 其所有 pages 的结果
    offset = 0
    global_O = torch.zeros_like(O_)
    for i, T in enumerate(request_num_pages):
        Oi = self.combine_result(
            O[offset : offset + T],   # 该 request 的 T 个 page 的 attention 结果
            M[offset : offset + T],
            L[offset : offset + T],
            O_[i], M_[i], L_[i],     # 当前 token 自注意力的结果
        )
        global_O[i] = Oi[0]
        offset += T
    
    O = global_O.transpose(1, 2).reshape(B, 1, H * D)
    O = self.WO(O)
    return X + self.act(O), [k.transpose(1, 2), v.transpose(1, 2)]

def combine_result(self, O, M, L, O_, M_, L_):
    """
    Online Softmax 跨 page 聚合
    O, M, L: [T, H, 1, D/1]  — 来自 T 个 KV pages 的 block attention 结果
    O_, M_, L_: [H, 1, D/1]  — 来自当前 token 自注意力
    """
    # 拼接自注意力结果
    O = torch.cat([O, O_.unsqueeze(0)], dim=0)
    M = torch.cat([M, M_.unsqueeze(0)], dim=0)
    L = torch.cat([L, L_.unsqueeze(0)], dim=0)
    
    # Online Softmax combine
    M_new, _ = torch.max(M, dim=0, keepdim=True)
    L_new = torch.sum(torch.exp(M - M_new) * L, dim=0, keepdim=True)
    O_new = torch.sum((M - M_new) * (L / L_new) * O, keepdim=True, dim=0)
    
    return O_new
```

**repeat_interleave 技巧**：
```
q = [q1, q2, q3, q4]  (4 requests)
num_pages = [3, 1, 2, 2]
repeat_interleave → [q1, q1, q1, q2, q3, q3, q4, q4]  (8 copies)
                      ←req1→  req2  ←req3→  ←req4→
```
这样 `q_[0:3]` 可以直接对 `KV[0:3]`（request 1 的 pages）做 batch 矩阵乘，无需循环。

---

## 对比：标准 Decoding vs PageAttention Decoding

| 维度 | 标准（gather-then-compute）| PageAttention |
|------|--------------------------|---------------|
| KV 访问 | gather 到连续内存，O(L) 拷贝 | 直接 index 物理 pages，无拷贝 |
| 计算 | 连续 KV 的标准 Attention | Block-level + Online Softmax combine |
| CUDA 实现 | 通用 GEMM | 专用 `paged_attention_kernel` |
| 内存效率 | 需要额外 O(L) 临时 buffer | 零额外 buffer |
| FlashAttn 兼容 | 直接 | 需要 block-level FA 变体 |

---

## 面试考点

**Q1: PageAttention 中 `repeat_interleave` 的作用是什么？**
A: Decoding 时每个 request 的 q 只有 1 个 token，但需要与 N 个 KV pages 做 attention。将 q 按 `num_pages` 复制，使 `q_[offset:offset+N]` 可以直接与 `KV[offset:offset+N]` 做 batch 矩阵乘，把 per-request 的 loop 变成一次并行矩阵计算。

**Q2: Online Softmax 的核心公式，为什么数学上等价于全局 softmax？**
A: 关键是 `M_new = max(M_old, M_block)` 和重缩放 `L_new = exp(M_old - M_new) * L_old + L_block`。由于 softmax 的数值稳定性 trick 是 `exp(x - max) / sum(exp(x - max))`，无论 max 用局部值还是全局值，最终归一化后结果相同。Online Softmax 实现了这个性质的分块累积，每个新 block 都正确地更新全局统计量。

**Q3: PageAttention 和 FlashAttention 的核心区别？**
A:  
- **FlashAttention**：针对单一**连续** KV 序列，用 SRAM tiling 减少 HBM 访问（memory bandwidth 优化，IO 复杂度从 O(N²) 降到 O(N)）
- **PageAttention**：针对**不连续** KV pages，用 block-level attention + Online Softmax combine 实现零拷贝聚合（内存管理优化，消除碎片和 gather 开销）
- 两者正交可叠加：PA 调度哪些 pages 参与计算，FA 做每个 block 内的高效计算

**Q4: forward_prefill 中为什么 X 的 batch 维度是 total_pages 而不是 num_requests？**
A: PageAttention 的设计将多请求的 pages 连续存在同一个 tensor 中（内存局部性更好）。通过 `request_num_pages` 列表知道每个 request 占几个 pages，在 attention 计算时切片，使每个 request 的 attention 计算完全独立——不同请求的 token 不会互相 attend。这是在 tensor 维度上的 "虚拟 batch" 设计。

**Q5: combine_result 中为什么要加入当前 token 的自注意力（O_, M_, L_）？**
A: Decoding 时，q 对应当前新 token，而 KV_Cache 里存的是历史 token 的 KV。当前 token 自身的 K/V 也需要参与 attention（因为 self-attention 中 q 要和自己对齐）。所以 block attention 结果（历史 KV）和自注意力结果（当前 token）需要合并——这正是 `combine_result` 最后一步 `torch.cat([O, O_.unsqueeze(0)])` 的作用。

---

## 延伸阅读

- [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] — PA 的前置：物理 page 分配和管理
- [[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]] — FA 的 SRAM tiling 实现，与 PA 正交
- [[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]] — 调度层基础
- [[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] — Prefill 分块调度，使用 PA Kernel
- vLLM 论文（Kwon et al., SOSP 2023）§3.2 PagedAttention Kernel 设计
- vLLM second meetup slides — PageAttention + multi-query/grouped-query attention 融合
