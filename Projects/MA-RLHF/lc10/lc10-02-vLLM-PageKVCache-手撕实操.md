---
title: "vLLM PageKVCache 手撕实操"
brief: "vLLM PagedKVCache：借鉴 OS 分页内存管理，把 KVCache 切成固定 page 按需分配，解决 Continue Batching 的内存碎片问题（93.6% 浪费→近零）。BlockTable 逻辑→物理映射，vLLM 核心机制，高频面试必问。"
date: 2026-02-26
type: code-practice
source: "MA-RLHF lc10 推理系统 / vLLM-PageKVCache.ipynb"
tags: ["code-practice", "inference", "vLLM", "KVCache", "PagedAttention", "memory-management"]
related:
  - "[[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-03-vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-04-vLLM-V0-手撕实操|vLLM-手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-05-Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]]"
---

# vLLM PageKVCache 手撕实操

> **来源**：MA-RLHF lc10 推理系统 / vLLM-PageKVCache.ipynb  
> **难度**：★★★★★  
> **面试频率**：★★★★★（vLLM 核心机制，必问）  
> **关联**：[[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]] [[Projects/MA-RLHF/lc10/lc10-03-vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]] [[Projects/MA-RLHF/lc10/lc10-04-vLLM-V0-手撕实操|vLLM-手撕实操]]

---

## 核心问题

**Continue Batching 的遗留问题**：KVCache 按 `batch_size × max_seq_len` 整块预分配，当 batch 内请求长度方差大时产生严重**内部碎片**。

**极端例子**：
- `max_seq_len = 1024`，实际请求只有 66 tokens
- 浪费 = `(1024 - 66) / 1024 = 93.6%`

**解法**：借鉴操作系统的**分页内存管理**思想——把 KVCache 切成固定大小的 page（块），按需分配，逻辑连续但物理离散。

---

## 架构全景

```
vLLM 组件层次：

Request（状态机）
  ↓
Scheduler（调度器，即 RequestManager 升级版）
  ↓
BlockTable（逻辑块表——纯索引，不存数据）
  ↓
PageKVCacheEngine（物理存储 + 分页映射）
  ↓
ModelWrapper（prefill/decode，调用 PageKVCache）
  ↓
vLLMPageCacheEngine（主引擎，step() 循环）
```

**关键设计原则**：
- **逻辑/物理分离**：BlockTable 只管索引，PageKVCacheEngine 管实际数据
- **按需分配**：不预分配 max_seq_len 个 slot，而是分配实际需要的 page 数
- **页内碎片上界**：最多浪费 `(page_size - 1)` 个 token slot

---

## 组件详解

### 1. BlockTable — 逻辑块表（纯索引层）

```python
class BlockTable:
    """逻辑块表管理 - 仅负责分页资源管理，不存 KV 数据"""
    
    def __init__(self, page_size: int, num_pages: int):
        self.page_size = page_size      # 每页能存多少 token 的 KV（如 64）
        self.num_pages = num_pages      # 总页数（如 1024）
        self.free_pages = list(range(num_pages))  # 空闲页 ID 列表
        self.allocated_pages = set()
        
        self.page_usage = [0] * num_pages   # 每页已使用多少个 token slot
        self.next_page = [-1] * num_pages   # 链式结构：当前页满了，next_page 指向下一页
    
    def _allocate_pages(self, num_pages: int) -> Optional[List[int]]:
        """分配 num_pages 个连续 page ID"""
        if len(self.free_pages) < num_pages:
            return None  # OOM
        
        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.allocated_pages.update(allocated)
        
        for page_id in allocated:
            self.page_usage[page_id] = 0
            self.next_page[page_id] = -1
        
        return allocated
    
    def _free_pages(self, page_ids: List[int]):
        """释放页面，归还到 free_pages"""
        for page_id in page_ids:
            self.allocated_pages.discard(page_id)
            self.page_usage[page_id] = 0
            self.next_page[page_id] = -1
        self.free_pages.extend(page_ids)
```

**关键：BlockTable 只是索引，不存实际 KV 数据。真正的 K/V tensor 在 PageKVCacheEngine。**

---

### 2. PageKVCacheEngine — 物理存储层

```python
class PageKVCacheEngine:
    def __init__(self, config):
        # 物理 KV 存储：[num_layers, num_pages, page_size, num_heads, head_dim]
        # 不再是 [layer, batch, seq, head, dim]！
        self.k_cache = torch.zeros(
            config.num_layers,
            config.num_pages,    # 总页数，而非 batch_size
            config.page_size,    # 每页大小，而非 max_seq_len
            config.num_heads,
            config.head_dim
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        self.block_table = BlockTable(config.page_size, config.num_pages)
        self.sequence_lengths = {}      # request_id → 当前长度
        self.request_to_pages = {}      # request_id → [page_id_list]
    
    def allocate_request_pages(self, request_id: int, request_length: int):
        """为 request 分配足够存下 request_length 个 token 的页"""
        num_pages_needed = (request_length + self.block_table.page_size - 1) \
                           // self.block_table.page_size  # 向上取整
        page_ids = self.block_table._allocate_pages(num_pages_needed)
        self.request_to_pages[request_id] = page_ids
        self.sequence_lengths[request_id] = 0
    
    def free_request_pages(self, request_id: int):
        """请求完成，释放所有已分配的页"""
        if request_id in self.request_to_pages:
            page_ids = self.request_to_pages.pop(request_id)
            self.block_table._free_pages(page_ids)
            self.sequence_lengths.pop(request_id, None)
    
    def write_kv(self, layer: int, request_id: int, pos: int, k: torch.Tensor, v: torch.Tensor):
        """在逻辑位置 pos 写入 KV"""
        page_idx = pos // self.block_table.page_size      # 第几页（逻辑）
        offset = pos % self.block_table.page_size          # 页内偏移
        physical_page = self.request_to_pages[request_id][page_idx]  # 物理页 ID
        
        self.k_cache[layer, physical_page, offset] = k
        self.v_cache[layer, physical_page, offset] = v
    
    def get_sequence_kvcache(self, request_ids: List[int]):
        """
        拼接 request 的所有 KV 页，返回 [layer, batch, seq, head, dim]
        （兼容原有 model.forward，但有拼接开销）
        """
        # 实际 vLLM 用 PageAttention 避免这个拼接，直接块内 attention
        ...
```

---

### 3. 逻辑地址 → 物理地址 转换

这是 Page KVCache 的核心抽象：

```
逻辑视角（请求视角）：
  request A 的 KV 序列 = [pos_0, pos_1, ..., pos_66]
  
物理布局（显存视角）：
  page_size = 64
  pos_0~63  → 物理页 #17（第一页，全满）
  pos_64~66 → 物理页 #42（第二页，用了3格）

转换公式：
  page_idx = pos // page_size       # 第几个逻辑页
  offset   = pos % page_size        # 页内偏移
  physical_page = request_to_pages[request_id][page_idx]  # 查表得物理页
```

**碎片计算**：
- 请求 66 tokens，page_size=64 → 需要 2 页
- 第 2 页只用了 2 格（66-64=2），浪费 62 格
- 碎片 = 62 / (2 × 64) = 48.4%（比 Continue Batching 的 93.6% 好得多）
- **最坏情况碎片 = (page_size - 1) / page_size**（趋近于 1 页碎片）

---

### 4. vLLMPageCacheEngine — 主引擎 step()

```python
class vLLMPageCacheEngine:
    def step(self):
        # 阶段1: Decode（已有请求）
        if self.scheduler.get_num_running_requests() > 0:
            request_ids = self.scheduler.get_running_request_ids()
            
            # 拿 KVCache（拼接所有 page）
            layer_kvcaches = self.cacher.get_sequence_kvcache(request_ids)
            current_lens = [self.cacher.sequence_lengths[rid] for rid in request_ids]
            
            # 每个请求输入上一步生成的最后一个 token
            input_tokens = torch.tensor([
                self.scheduler.requests[rid].get_last_token() 
                for rid in request_ids
            ]).unsqueeze(1)  # [batch, 1]
            
            logits, new_kvcaches = self.model_wrapper.model(
                input_tokens, layer_kvcaches, current_lens
            )
            
            # 写回新的 KV
            for i, rid in enumerate(request_ids):
                pos = self.cacher.sequence_lengths[rid]
                for layer in range(num_layers):
                    k_new = new_kvcaches[layer][i]  # [head, dim]
                    v_new = ...
                    self.cacher.write_kv(layer, rid, pos, k_new, v_new)
                self.cacher.sequence_lengths[rid] += 1
            
            # 采样下一 token
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            completed_ids = []
            for rid, token in zip(request_ids, next_tokens):
                self.scheduler.requests[rid].add_token(token.item())
                if self.scheduler.requests[rid].is_finished():
                    completed_ids.append(rid)
            
            # 释放完成请求的页
            for rid in completed_ids:
                self.cacher.free_request_pages(rid)
                self.scheduler.running_requests.discard(rid)
        
        # 阶段2: Prefill（新请求）
        new_reqs = self.scheduler.get_pending_requests(max_count=available)
        for req_id, prompt in new_reqs:
            self.cacher.allocate_request_pages(req_id, len(prompt))
        # ... prefill forward pass
```

---

## Compare: 两种 KVCache 管理方式

| 维度 | Continue Batching KVCache | vLLM PageKVCache |
|------|--------------------------|-----------------|
| 存储形状 | `[layer, batch, max_seq, H, D]` | `[layer, num_pages, page_size, H, D]` |
| 分配时机 | 预分配全量 | 按需分配，request 到来时 |
| 内部碎片 | `max_seq - actual_len` per slot | `≤ page_size - 1` per request |
| 外部碎片 | 无（连续分配）| 无（page 池统一管理）|
| 地址转换 | slot_id → 直接访问 | `page_idx + offset → physical_page` |
| 实现复杂度 | 低 | 高（BlockTable + 两级索引）|
| KV Copy-on-Write | 不支持 | 支持（多请求共享相同 prompt prefix 的 page）|
| Beam Search 支持 | 差 | 好（共享 prefix page，只 copy 分叉处）|

---

## PageAttention：为什么不直接拼接 KV？

Notebook 中 `get_sequence_kvcache()` 把所有 page 拼成连续 tensor 再做 attention，存在**拼接开销**。

真正的 vLLM **PageAttention** 实现：
- Attention 在每个 page 内独立计算（block attention）
- 块间结果用 Flash Attention 风格的 online softmax 合并（类似 Ring Attention 思想）
- 无需跨 page 拼接，直接在分散页面上做计算

```
Block Attention（伪代码）：
for each page_id in request_to_pages[req]:
    K_block = k_cache[:, page_id, :, :]  # [page_size, H, D]
    V_block = v_cache[:, page_id, :, :]
    scores_block = Q @ K_block.T         # [1, page_size]
    O_block, M_block, L_block = flash_attn_block(scores_block, V_block)
    
# 合并：online softmax across blocks
O_final = merge_online_softmax(O_blocks, M_blocks, L_blocks)
```

这部分在 `vLLM-PageAttention.ipynb` 中实现。

---

## 关键设计洞察

### 1. OS 分页的映射关系
```
OS 虚拟内存     ←→    vLLM KV Cache
虚拟页号        ←→    逻辑页索引（pos // page_size）
物理页框        ←→    物理 page ID（block_table 存储）
页表             ←→    request_to_pages[request_id]
缺页中断        ←→    allocate_request_pages（按需分配）
内存释放        ←→    free_request_pages
```

### 2. Prefix Caching 的基础
多个请求共享相同的 system prompt → 它们的前几个 page 完全一样 → 共享物理 page，CoW（写时复制）才有差异。
Continue Batching 无法做到这点（每个 slot 独立）。

### 3. num_pages vs max_batch × max_seq
```python
# Continue Batching
memory = batch_size × max_seq_len × num_layers × num_heads × head_dim × 2 × 2  # KV × bytes
# vLLM
memory = num_pages × page_size × num_layers × num_heads × head_dim × 2 × 2
# 理论上总量相同，但 vLLM 利用率 >> Continue Batching
```

---

## 面试考点

**Q1: vLLM 如何解决 KVCache 内存碎片问题？**
> A: 借鉴操作系统分页管理，把 KVCache 切成固定大小的 page（如 16/32/64 tokens）。每个请求按需分配 page，通过 BlockTable 维护逻辑→物理 page 映射。最大碎片 = page_size - 1，远优于 max_seq_len 级别的内部碎片。

**Q2: BlockTable 和 PageKVCacheEngine 的职责分离是什么？**
> A: BlockTable 只管页面索引（哪些页空闲、哪些已分配），不存 KV 数据。PageKVCacheEngine 管实际的 K/V tensor 存储，通过查 BlockTable 把逻辑地址翻译成物理 page ID + 页内偏移来读写数据。

**Q3: page_size 怎么选？**
> A: 小 page（如 16）碎片少，但 BlockTable 管理开销大，且 PageAttention 的块间合并次数多。大 page（如 512）管理简单，但碎片大。vLLM 默认 block_size=16，是 latency 和 memory 的权衡。

**Q4: PageKVCache 如何支持 Beam Search？**
> A: Beam Search 产生多个候选序列，前缀相同。vLLM 可以让这些 beam 共享前缀的 page（CoW），只在分叉处分配新 page。Continue Batching 每个 beam 需要独立的完整 slot，显存 N 倍消耗。

**Q5: 为什么 Notebook 版本 PageKVCache 最后要做 page 拼接，而真正的 vLLM 不需要？**
> A: Notebook 为了兼容标准 model.forward（需要连续 tensor 输入），所以用 `get_sequence_kvcache` 拼接。真正的 vLLM 实现了 PageAttention 算子（CUDA kernel），直接在分散的 page 上做 block-wise attention + online softmax 合并，完全不需要拼接，消除了额外的内存 copy 开销。

---

## 延伸阅读

- [[Projects/MA-RLHF/lc10/lc10-01-Continue-Batching-手撕实操|Continue-Batching-手撕实操]] — 基础，理解 slot 管理
- [[Projects/MA-RLHF/lc10/lc10-03-vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]] — PageAttention CUDA kernel 实现
- [[Projects/MA-RLHF/lc10/lc10-05-Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] — Prefill/Decode 混合，解决 Prefill 阻塞 Decode
- [[Projects/MA-RLHF/lc10/lc10-07-PD-Disaggregation-手撕实操|PD-Disaggregation-手撕实操]] — Prefill/Decode 异机，极致解耦

---

*笔记来源：MA-RLHF lc10 / vLLM-PageKVCache.ipynb — 2026-02-26*
