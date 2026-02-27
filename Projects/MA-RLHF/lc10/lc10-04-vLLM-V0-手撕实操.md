---
title: "vLLM 手撕实操"
brief: "vLLM核心机制完整实现：PagedAttention（虚拟内存管理/物理KV块/逻辑→物理块映射）、Continuous Batching、Chunked Prefill、Speculative Decoding、多GPU张量并行推理，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, vllm, paged-attention, inference, serving, pytorch]
related:
  - "[[Projects/MA-RLHF/lc10/lc10-00-FlashAttention-手撕实操|FlashAttention-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-ray-01-Ray-推理系统实操|Ray-推理系统实操]]"
---

# vLLM 推理系统手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

1. [KV Cache 基础](#1-kv-cache-基础)
2. [Continuous Batching](#2-continuous-batching)
3. [PagedKVCache 分页缓存管理](#3-pagedkvcache-分页缓存管理)
4. [PagedAttention Kernel](#4-pagedattention-kernel)
5. [vLLM V0 架构](#5-vllm-v0-架构)
6. [Chunked Prefill](#6-chunked-prefill)
7. [vLLM V1 架构](#7-vllm-v1-架构)
8. [Speculative Decoding](#8-speculative-decoding)
9. [PD Disaggregation](#9-pd-disaggregation)

---

## 1. KV Cache 基础

LLM 推理分为两个阶段：

- **Prefill**（预填充）：处理完整 prompt，计算所有 token 的 KV 并缓存，输出第一个预测 token
- **Decoding**（解码）：每次只输入上一步生成的 token，利用已缓存的 KV 进行增量计算

### 无 KV Cache 的生成

每次生成都需要将整个序列重新送入模型：

```python
idx = {'input_ids': X}  # X: [1, 10]
for i in range(4):
    output = model(**idx)
    logits = output['logits'][:, -1, :]
    idx_next = torch.argmax(logits, dim=1)[0]
    # 每次拼接完整序列重新输入
    idx['input_ids'] = torch.cat((idx['input_ids'], idx_next.unsqueeze(0).unsqueeze(1)), dim=-1)
```

### 带 KV Cache 的生成

缓存历史 K/V，每步只输入新 token：

```python
class xiaodonggua_kv_cache(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.cache_K = self.cache_V = None

    def forward(self, X):
        X = self.Embedding(X)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)

        # KV Cache: 首次存入，后续拼接
        if self.cache_K is None:
            self.cache_K = K
            self.cache_V = V
        else:
            self.cache_K = torch.cat((self.cache_K, K), dim=1)
            self.cache_V = torch.cat((self.cache_V, V), dim=1)
            K = self.cache_K
            V = self.cache_V

        attn = Q @ K.transpose(1, 2) @ V
        return self.lm_head(attn)
```

**关键点**：Decoding 阶段只输入 `next_token`，Q 是 `[B,1,D]`，K/V 从 cache 中取出完整历史。

---

## 2. Continuous Batching

### 动机

批解码（Batch Decoding）中，某些请求提前结束导致 GPU 空闲。Continuous Batching 实现**动态插入新请求**到正在解码的 batch 中。

### 核心组件

| 组件 | 职责 |
|------|------|
| `RequestManager` | 管理请求队列（waiting → running → completed） |
| `KVCacheManager` | 管理 slot-based KV Cache，分配/释放槽位 |
| `ModelWrapper` | 封装 prefill/decode 前向 |
| `ContinueBatchingEngine` | 主循环引擎 |

### KVCacheManager：槽式管理

```python
class KVCacheManager:
    def __init__(self, config):
        # 预分配固定大小的 KV Cache
        self.k_cache = torch.zeros(num_layers, max_batch_size,
                                   max_seq_len, num_heads, head_dim)
        self.v_cache = torch.zeros_like(self.k_cache)

        self.request_to_slot = {}  # request_id -> slot_index
        self.free_slots = set(range(max_batch_size))

    def allocate_slots(self, request_ids):
        """为新请求分配空闲槽位"""
        for request_id in request_ids:
            slot_id = self.free_slots.pop()
            self.request_to_slot[request_id] = slot_id

    def free_slot(self, request_id):
        """请求完成后释放槽位"""
        slot_id = self.request_to_slot[request_id]
        self.free_slots.add(slot_id)
        self.k_cache[:, slot_id, :, :, :] = 0
```

### 主循环伪代码

```python
while engine.has_pending_work():
    # 阶段1：已有请求 → 解码
    if kv_cache_manager.has_active_requests():
        decoding_logits = model.decode_next_tokens(input_tokens, kv_cache)
        # 更新状态，完成的请求释放槽位

    # 阶段2：新请求 → 预填充
    if kv_cache_manager.has_available_slots():
        pending = request_manager.get_pending_requests(available_slots)
        prefill_logits = model.prefill_requests(pending)
```

### 槽式管理的问题

- 当 batch 内请求的 context length 方差大时，短请求产生**内部碎片**
- 例如 `max_seq_len=1024`，实际请求长度 66，浪费 958 个位置

---

## 3. PagedKVCache 分页缓存管理

### 核心思想

借鉴操作系统的**虚拟内存分页**机制：

- 将 KV Cache 切分为固定大小的 **Page（页）**
- 每个请求持有若干逻辑页，通过**页表**映射到物理页
- 请求长度 66，页大小 64 → 只需 2 页（浪费 62），而非槽式的 958

### BlockTable：页表管理

```python
class BlockTable:
    """纯资源管理，与业务无关"""
    def __init__(self, page_size, num_pages):
        self.free_pages = list(range(num_pages))
        self.allocated_pages = set()

    def _allocate_pages(self, num_pages, parent_block_id=-1):
        """分配 N 个页，返回页 ID 列表"""
        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.allocated_pages.update(allocated)
        # 链表连接
        if parent_block_id != -1:
            self.next_page[parent_block_id] = allocated[0]
        return allocated

    def _free_pages(self, page_ids):
        """归还页到空闲池"""
        for page_id in page_ids:
            self.allocated_pages.remove(page_id)
            self.free_pages.append(page_id)
```

### PageKVCacheEngine：分页 KV 缓存引擎

```python
class PageKVCacheEngine:
    def __init__(self, config):
        self.block_table = BlockTable(page_size, num_pages)

        # 物理 KV 存储：[layers, num_pages, page_size, heads, head_dim]
        self.k_cache = torch.zeros(num_layers, num_pages, page_size, num_heads, head_dim)
        self.v_cache = torch.zeros_like(self.k_cache)

        self.request_to_pages = {}  # request_id -> [page_id1, page_id2, ...]
        self.sequence_lengths = {}

    def allocate_request_pages(self, request_id, request_length):
        """按需分配页面"""
        num_needed = (request_length // self.page_size) + 1
        page_ids = self.block_table._allocate_pages(num_needed)
        self.request_to_pages[request_id] = page_ids
        return page_ids

    def update_pages(self, request_id, new_kv_cache):
        """Prefill 填充 / Decoding 追加"""
        seq_len = new_kv_cache[0][0].shape[0]

        if seq_len == 1:  # Decoding: 追加单 token
            length = self.sequence_lengths[request_id]
            pages = self.request_to_pages[request_id]
            # 如果当前页满了，申请新页
            if length % self.page_size == 0:
                new_page = self.block_table._allocate_pages(1, pages[-1])
                self.request_to_pages[request_id].append(new_page[0])
            # 写入最后一页的对应位置
            offset = self.sequence_lengths[request_id] % self.page_size
            self.k_cache[layer, pages[-1], offset] = new_k
        else:  # Prefill: 按页填充
            for k, page_id in enumerate(pages):
                self.k_cache[layer, page_id, :] = kv_chunk[k*T:(k+1)*T]
```

### 分页 vs 槽式对比

| | 槽式 | 分页式 |
|---|---|---|
| 分配粒度 | 固定 max_seq_len | page_size（如 64） |
| 碎片 | 严重内部碎片 | 仅最后一页有碎片 |
| Batch 上限 | 受 max_batch_size 限制 | 受总页数限制，灵活 |

---

## 4. PagedAttention Kernel

### 核心挑战

KV Cache 分散在不同物理页中，Attention 计算需要跨页聚合结果。

### Prefill 阶段：FlashAttention 后端

多个请求的 Page 拼成一个 batch，按请求循环处理：

```python
class PageAttentionDecoderBlock(nn.Module):
    def forward_prefill(self, X, request_num_pages, attention_backend):
        """
        X: [num_pages, page_size, dim]  -- 所有请求的 page 拼接
        request_num_pages: [3, 2]       -- 各请求占几页
        """
        Q, K, V = self.WQ(X), self.WK(X), self.WV(X)
        # ... reshape to [num_pages, num_heads, page_size, head_dim]

        # 按请求循环，每个请求独立做 Attention
        for t in range(request_size):
            offset_i = offset[t]
            N = request_num_pages[t]
            Q_ = Q[offset_i: offset_i+N]
            K_ = K[offset_i: offset_i+N]
            V_ = V[offset_i: offset_i+N]
            O_ = attention_backend(Q_, K_, V_)  # FlashAttention / 标准 Attention
            O[offset_i: offset_i+N] = O_
```

### FlashAttention-V2 后端实现

Block-wise Attention + Online Softmax 聚合：

```python
def FlashAttention(Q, K, V):
    """
    Q/K/V: [num_pages, num_heads, page_size, head_dim]
    逐 Q-block 遍历所有 KV-block，用 online softmax 增量更新
    """
    N, H, T, D = Q.shape
    O_global = torch.zeros(N, H, T, D)

    for i in range(N):  # Q block loop
        O = torch.zeros(1, H, T, 1)
        M = torch.zeros(1, H, T, 1)  # running max
        L = torch.zeros(1, H, T, 1)  # running sum(exp)
        Q_ = Q[i]

        for j in range(N):  # KV block loop
            if j > i:
                continue  # causal mask
            K_, V_ = K[j], V[j]

            S_ij = Q_ @ K_.transpose(1, 2)
            M_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
            M_new = torch.maximum(M_ij, M)

            P_ij = torch.exp(S_ij - M_new)
            L_ij = torch.sum(P_ij, dim=-1, keepdim=True)
            L_new = torch.exp(M - M_new) * L + L_ij
            O_i = torch.exp(M - M_new) * O + P_ij @ V_

            M, L = M_new, L_new

        O_global[i] = (O_i / L_new).unsqueeze(0)  # re-scale

    return O_global
```

### Decoding 阶段：Block Attention + Combine

单个 q 对 N 页 KV 做 block attention，再通过 online softmax 聚合：

```
q6 [k1,k2,k3,k4] -> o6_a, m_a, l_a
q6 [k5,k6,/ ,/ ] -> o6_b, m_b, l_b
o6 = combine(o6_a, o6_b)   # online softmax trick
```

```python
def combine_result(O, L, M):
    """Online Softmax 聚合多个 block 的 attention 输出"""
    M_new, _ = torch.max(M, dim=0, keepdim=True)
    L_new = torch.exp(M - M_new) * L
    L_new = torch.sum(L_new, dim=0, keepdim=True)
    O_new = torch.exp(M - M_new) * (L / L_new) * O
    return O_new.sum(dim=0, keepdim=True)
```

**Decoding 完整流程**：

```python
def forward_decoding(self, X, request_num_pages, KV_Cache):
    q = self.WQ(X)  # [B, 1, D]

    # step1: q self-attention 初始化
    S = q @ k.transpose(2,3)
    M_, L_, O_ = S.clone(), torch.ones_like(S), v

    # step2: dispatch q 到各页
    q_ = torch.repeat_interleave(q, torch.tensor(request_num_pages), dim=0)

    # step3: block attention on KV pages
    S = q_ @ K_.transpose(2,3)
    M = torch.max(S, dim=-1, keepdim=True)
    L = torch.sum(torch.exp(S - M), dim=-1, keepdim=True)
    O = torch.softmax(S, dim=-1) @ V_

    # step4: combine per-request results
    for i, T in enumerate(request_num_pages):
        globle_O[i] = combine_result(O[offset:offset+T], M[...], L[...], O_[i], M_[i], L_[i])
```

---

## 5. vLLM V0 架构

V0 = Continuous Batching + PagedKVCache + PagedAttention

### 架构组件

```
Scheduler ──→ vLLMv0Engine ──→ ModelWrapper ──→ PageToyModel
                  │                                   │
                  └──── PageKVCacheEngine ─────────────┘
```

### 关键变化：page_input_ids

Prefill 时，输入不再是 `[batch, padded_seq_len]`，而是 `[num_pages, page_size]`：

```python
def prefill_requests(self, request_ids, prompts):
    # batch input_ids -> page input_ids
    for request_id, prompt in zip(request_ids, prompts):
        page_ids = self.cacher.allocate_request_pages(request_id, len(prompt))
        request_input_ids = torch.zeros(len(page_ids), page_size, dtype=torch.long)
        for i in range(len(page_ids)):
            if i == len(page_ids) - 1:
                offset = len(prompt) % page_size
                request_input_ids[i, :offset] = torch.tensor(prompt[i*T:len(prompt)])
            else:
                request_input_ids[i, :] = torch.tensor(prompt[i*T:(i+1)*T])
```

### Decoding 的 Page KV Cache 获取

```python
def get_page_kvcache(self, request_ids):
    """获取 request-wise 的 page KV cache"""
    N = sum(len(self.request_to_pages[idx]) for idx in request_ids)
    K = torch.zeros(L, N, T, H, D)

    for idx in request_ids:
        page_ids = self.request_to_pages[idx]
        K[:, b_id:b_id+num_pages] = self.k_cache[:, page_ids]
    return (K, V), num_pages_len, batch_to_page
```

### V0 的局限

- Prefill 和 Decoding 严格分离，先 Decode 再 Prefill
- Decoding 时 KV Cache 需要重组为 batch 形式，存在拷贝开销

---

## 6. Chunked Prefill

### Part 1：投影搭便车

**核心洞察**：Decoding 是 memory-bound（`x[1,d] @ W[d,d]`），Prefill 是 compute-bound（`X[L,d] @ W[d,d]`）。融合后共享权重加载：

```python
def ChunkPrefillLinearForward(W, XP, XD):
    """PD 融合投影：共享一次权重加载"""
    BP, LP, D = XP.shape  # Prefill batch
    BD, LD, D = XD.shape  # Decoding batch

    XP = XP.reshape(BP*LP, D)
    XD = XD.reshape(BD*LD, D)
    XPD = torch.cat((XP, XD), dim=0)  # 拼接
    YPD = W(XPD)  # 一次 matmul

    YP = YPD[:BP*LP].reshape(BP, LP, D)
    YD = YPD[BP*LP:].reshape(BD, LD, D)
    return YP, YD
```

### Part 2：分块 Prefill

长 prompt 切成 chunk，每步只处理一个 chunk，同时带上 Decoding 任务：

```python
def chunk_prefill_method(model, x, page_size, dim):
    x_chunks = x.split(page_size, dim=1)

    for i, x_c in enumerate(x_chunks):
        # 关键：后续 chunk 需要前面 chunk 的 KV Cache
        if i == 0:
            chunk_kv_cache = None
        else:
            chunk_kv_cache = KVCache[:, :, :i*page_size]

        logits, tmp_KVCache = model.forward(x_c, KVCache=chunk_kv_cache)

        # 只有最后一个 chunk 的 logits 才是有效的 next-token 预测
        if i == num_chunks - 1:
            last_token_logits = logits[:, -1, :]
```

### Attention 的特殊处理

Chunked Prefill 的 Attention 模式介于 Prefill 和 Decoding 之间：

| 模式 | 输入 | KV Cache |
|------|------|----------|
| Prefill | `[chunk]` | `None` |
| Decoding | `[next_token]` | `[context]` |
| **Chunked Prefill** | `[chunk]` | `[previous_chunks]` |

```python
class ChunkedPrefillAttention(Attention):
    def forward(self, X, KVCache):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        if KVCache is not None:
            K_ = torch.cat([KVCache[0], K], dim=1)  # 拼接历史 KV
            V_ = torch.cat([KVCache[1], V], dim=1)
        else:
            K_, V_ = K, V
        Z = attention_kernel(Q, K_, V_)
        return self.Wo(Z), [K, V]  # 只返回当前 chunk 的 KV
```

### Part 3：PD 混合 Batch

一个 step 中同时处理 Decoding 请求和一个 Prefill chunk：

```python
class ChunkedPrefillMergeAttention(Attention):
    def forward(self, X, KVCache_P, KVCache_D, batch_len_p, batch_len_d):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)

        # Decoding 请求：q 长度为 1
        if batch_len_d != 0:
            Q_d = Q[0, :num_d].reshape(num_d, 1, dim)
            Z_d = self.forward_pd(Q_d, K_d, V_d, KVCache_D, batch_len_d)

        # Prefill 请求：chunk 长度
        if batch_len_p != 0:
            Z_p = self.forward_pd(Q_p, K_p, V_p, KVCache_P, batch_len_p)

        # 合并输出
        O = torch.cat((O_d, Z_p), dim=0)
```

### 调度策略

1. 短请求优先调度 → 快速进入 Decoding
2. 长请求逐步 Chunked Prefill → 每步带上短请求的 Decoding
3. 最终全部进入纯 Decoding 阶段

---

## 7. vLLM V1 架构

### 核心改进

根据 [vLLM Seventh Meetup](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/) 的设计：

- **消除 P/D 分离概念**：所有请求统一用 Chunked Prefill 处理
- **Token Budget**：每步固定 token 预算，Decoding 优先填入，剩余给 Prefill chunk
- **decode-maximal batching**：Decoding 请求总是优先并入 batch

### V1 执行示例

```
step0: R1(chunk1) + R2(chunk1) + R3(chunk1)    ← 全 Prefill
step1: R1(decode) + R2(decode) + R3(chunk2)     ← 混合 PD
step2: R1(decode) + R2(decode) + R3(chunk3)     ← 混合 PD
step3: R1(decode) + R2(decode) + R3(decode)     ← 全 Decoding
```

### V1 Scheduler

将所有请求拼成一条序列，token budget 决定单步计算量：

```python
class Scheduler:
    def get_requests(self, max_batch_tokens, max_decoding_batch, max_prefill_batch):
        info = SchedulerInfo()

        # Decoding 优先
        for req in running_requests:
            if req.is_decoding():
                info.ids.append(req.id)
                info.chunk_prompts.append([req.generated_tokens[-1]])
                info.chunk_len.append(1)
                count_batch_token += 1

        # Prefill 用剩余 budget
        for req in running_requests:
            if not req.is_decoding():
                available = max_batch_tokens - count_batch_token
                chunk = req.prompt[start:start+available]
                info.chunk_prompts.append(chunk)
                # last_pos == -1 表示该 chunk 不是最后一个（无有效 next token）
                if len(remaining) <= available:
                    info.last_pos.append(len(chunk)-1)  # 有效预测位
                else:
                    info.last_pos.append(-1)  # chunked, 还没完

        # 拼成一条序列
        for chunk in info.chunk_prompts:
            info.merge_prompt.extend(chunk)
        return info
```

### V1 Engine 主循环

```python
class vLLMEngine:
    def step(self, config):
        # 1. 获取融合 batch
        info = self.scheduler.get_requests(
            max_batch_tokens=config.max_batch_tokens,
            max_prefill_batch=config.max_prefill_batch,
            max_decoding_batch=config.max_decoding_batch)

        # 2. 获取 Page KV Cache
        kv_cache, info.kv_page_len = self.cacher.get_kv_cache(info.ids)

        # 3. 拼接为一条序列
        input_ids = torch.tensor([info.merge_prompt], dtype=torch.long)

        # 4. 执行计算
        next_token, kv = self.execute(input_ids, kv_cache, info)

        # 5. 更新请求和 KV Cache
        self.update(next_token, kv, info)
```

### V1 PageKVCacheEngine 改进

支持增量写入，自动扩展页：

```python
class PageKVCacheEngine:
    def update_kv_cache(self, request_id, KV):
        """增量写入 KV，自动分配新页"""
        new_len = KV_token_count
        while new_len != 0:
            page_available = total_pages * page_size - self.kv_len[request_id]
            if page_available == 0:
                # 当前页满，申请新页
                new_page = self.block_table._allocate_pages(1, parent=pages[-1])
                self.request_to_pages[request_id].append(new_page)
            elif page_available >= new_len:
                # 当前页够用，直接写入
                self.kv_cache[:, :, target_page, start:end] = KV[...]
                new_len = 0
            else:
                # 写满当前页，剩余申请新页
                self.kv_cache[:, :, target_page, start:] = KV[..., :available]
                new_len -= page_available
            self.kv_len[request_id] += written
```

### V0 → V1 演进总结

| 特性 | V0 | V1 |
|------|----|----|
| PD 处理 | 分离：先 Decode 再 Prefill | 融合：统一 Chunked Prefill |
| 输入形式 | Page-level batch | 一条拼接序列 + token budget |
| 调度 | 简单 FIFO | Decode-maximal batching |
| Attention | 区分 prefill/decode kernel | 统一 kernel（目标） |
| KV Cache 写入 | 按阶段区分 | 统一增量写入 |

---

## 8. Speculative Decoding

### 核心思想

用小模型（Draft Model）快速猜测多个 token，再用大模型（Target Model）一次性验证。

### Basic Speculative Decoding（贪婪采样）

```python
class SPDecoding:
    def __init__(self, model_target, model_draft, spec_n):
        self.model_target = model_target
        self.model_draft = model_draft
        self.spec_n = spec_n  # 每次猜测 token 数

    def generate_draft(self, spec_n, x):
        """Draft 模型逐步生成 spec_n 个 token"""
        for i in range(spec_n):
            logits = self.model_draft(x)[:, [-1], :]
            next_token = torch.argmax(logits, dim=-1)
            x = torch.cat([x, next_token], dim=1)
        return x, logits_list

    def generate(self, x, max_new_tokens=30):
        for i in range(max_new_tokens):
            # 1. Draft 猜测 spec_n 个 token
            x_spec, logits_draft = self.generate_draft(self.spec_n, x)
            y_spec = x_spec[:, -self.spec_n:]

            # 2. Target 一次性验证（单次 forward）
            logits_target = self.model_target(x_spec)[:, -self.spec_n-1:]
            y_target = torch.argmax(logits_target, dim=-1)

            # 3. 比对：找到第一个不一致的位置
            verify = y_spec == y_target[:, :-1]
            idx1, idx2 = torch.where(verify == False)
            accept_len = self.spec_n if len(idx2) == 0 else idx2[0]

            # 4. 接受 accept_len + 1 个 token（含 target 自身预测）
            x = torch.cat((x, y_target[:, :accept_len+1]), dim=1)
```

### 接受逻辑分析

- **全接受**：Draft 的 5 个 token 全对 → 接受 6 个（5 + target 额外预测 1 个）
- **部分接受**：前 k 个对 → 接受 k+1 个
- **全拒绝**：退化为标准解码 → 仍接受 1 个（target 的预测）

加速比 = `max_new_tokens / actual_steps`

### Speculative Sampling（随机采样版）

当使用随机采样时，不能简单比较 token 是否相等，需要基于概率分布进行接受/拒绝：

```python
for j in range(self.spec_n):
    r = torch.rand(1).item()
    token_id = y_spec[0, j]
    q = F.softmax(logits_target[0, j], dim=-1)  # target 分布
    p = F.softmax(logits_draft[0, j], dim=-1)   # draft 分布

    # 接受概率：min(1, q(x)/p(x))
    if r < min(1, q[token_id] / p[token_id]):
        accept_len += 1
        next_tokens.append(y_spec[0, j])
    else:
        # 拒绝后 re-sampling from adjusted distribution
        q_ = q.clone()
        idx = torch.where(q < p)
        q_[idx] = p[idx]
        next_token = torch.multinomial(q_, num_samples=1)
        next_tokens.append(next_token)
        break
```

**数学保证**：该采样方案保证生成分布与 Target Model 完全一致。

---

## 9. PD Disaggregation

### 动机

- Prefill 是 **compute-bound**，Decoding 是 **memory-bound**
- 不同任务可配置不同硬件：Prefill 节点用高算力 GPU，Decoding 节点用高带宽存储
- 灵活配置各节点的 batch size

### 架构

```
请求 ──→ [Prefill 节点] ──KV Transfer──→ [Decoding 节点] ──→ 输出
              │                              │
          Compute-bound                 Memory-bound
          高算力 GPU                    高带宽 GPU
```

### 实现要点

1. **基础版本**：分离 PD 进程，Prefill 完成后将 KV Cache 传输至 Decoding 节点
2. **分布式版本**：基于 Ray 实现，Prefill 生产者 → KV 传输队列 → Decoding 消费者
3. **通信-计算重叠**：异步 KV 传输，减少 Decoding 等待

### PD 分离的代价

- KV Cache 跨节点传输带来额外延迟
- 但总吞吐可以提升（P/D 各自独立扩展）

### 工程现实

> PD 分离和融合不是非此即彼。复杂推理系统一定是融合和分离兼并的，设计是 case-by-case 的。
>
> — MA-RLHF

更进一步的分离：
- **Attention-FFN 分离**（AF 分离）
- **训练分离**（GRPO 训练 / Agentic-RL / AReal 异步训练）
- "分离"是推理服务优化的新维度

---

## 总结：vLLM 技术栈全景

```
                    ┌─────────────────────┐
                    │   Speculative        │
                    │   Decoding           │
                    └──────┬──────────────┘
                           │
┌──────────┐    ┌──────────┴──────────┐    ┌────────────────┐
│ PD       │    │   vLLM V1           │    │  Chunked       │
│ Disagg.  │◄──│  (统一 PD Batch)     │───►│  Prefill       │
└──────────┘    └──────────┬──────────┘    └────────────────┘
                           │
                ┌──────────┴──────────┐
                │   vLLM V0           │
                │ (PageAttn+PageKV)   │
                └──────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
  ┌───────┴───────┐ ┌─────┴─────┐ ┌───────┴───────┐
  │ PagedAttention│ │ PagedKV   │ │ Continuous    │
  │ Kernel        │ │ Cache     │ │ Batching      │
  └───────────────┘ └───────────┘ └───────────────┘
                           │
                    ┌──────┴──────┐
                    │  KV Cache   │
                    │  基础       │
                    └─────────────┘
```
