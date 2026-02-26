---
title: "Continue Batching 手撕实操"
brief: "Continue Batching（动态批处理）：请求完成后立刻释放槽位，下一 step 填入新请求，GPU 始终满载。vLLM 必问基础，解决静态 Batching 的 slot 浪费问题。配合 PageKVCache 和 Chunked-Prefill 构成 vLLM 三大核心机制。"
date: 2026-02-26
type: code-practice
source: "MA-RLHF lc10 推理系统 / Continue_Batching.ipynb"
tags: ["code-practice", "inference", "vLLM", "continue-batching", "scheduling"]
related:
  - "[[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]]"
  - "[[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]]"
  - "[[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]]"
  - "[[AI/LLM/Inference/LLM-推理优化-2026-全景|LLM推理优化2026全景]]"
---

# Continue Batching 手撕实操

> **来源**：MA-RLHF lc10 推理系统 / Continue_Batching.ipynb  
> **难度**：★★★★☆  
> **面试频率**：★★★★★（vLLM 必问基础）  
> **关联**：[[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] [[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]] [[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]]

---

## 核心问题

**Batching Decoding 的缺陷**：静态 batch 中某些请求提前结束 → 该 slot 空闲，但整个 batch 还在跑，GPU 计算浪费。

**Continue Batching（动态批处理）**：请求完成后立刻释放槽位，下一个 step 马上填入新请求。始终保持 batch 满载，像推理服务"永动机"。

---

## 架构全景

```
请求流
  ↓
RequestManager（请求队列调度）
  ↓
KVCacheManager（静态预分配 [layer, batch, seq, head, dim]）
  ↓
ModelWrapper（prefill / decode 两阶段）
  ↓
ContinueBatchingEngine（主循环 step()）
```

**关键设计**：每次 `step()` 先 decode（已有请求）再 prefill（新请求），保持解码连续性。

---

## 组件详解

### 1. Request — 请求状态机

```python
class Request:
    def __init__(self, request_id, prompt, max_len=2048):
        self.request_id = request_id
        self.prompt = prompt
        self.generated_tokens = []
        self.status = "REQUEST_WAITING"  # → RUNNING → COMPLETED
        self.current_length = len(prompt)
        self.max_length = max_len
        
    def add_token(self, token: int):
        self.generated_tokens.append(token)
        self.current_length += 1
        if self.is_finished():
            self.status = "REQUEST_COMPLETED"
    
    def is_finished(self) -> bool:
        # EOS token 或达到 max_length
        return (len(self.generated_tokens) > 0 and 
                self.generated_tokens[-1] == EOS_TOKEN) or \
               self.current_length >= self.max_length
```

**状态转换**：`WAITING → RUNNING → COMPLETED`

---

### 2. RequestManager — 调度器

```python
class RequestManager:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.requests = {}           # request_id → Request
        self.waiting_queue = deque() # 等待队列
        self.running_requests = set()# 正在运行的 request_id 集合
    
    def get_available_slots(self) -> int:
        # 还能容纳多少新请求
        return self.max_batch_size - len(self.running_requests)
    
    def get_pending_requests(self, max_count) -> List[Tuple[int, List[int]]]:
        # 从 waiting_queue 取最多 max_count 个请求，移入 running
        ...
    
    def handle_completed_requests(self) -> List[int]:
        # 扫描 running_requests，找到 COMPLETED 的，释放槽位
        completed = [rid for rid in self.running_requests 
                     if self.requests[rid].is_finished()]
        for rid in completed:
            self.running_requests.remove(rid)  # 释放槽位！
        return completed
```

**核心**：`handle_completed_requests()` 释放槽位，下一 step 的 `get_pending_requests()` 就能填入新请求。

---

### 3. KVCacheManager — 静态 KV 缓存

```python
class KVCacheManager:
    def __init__(self, config):
        # 预分配固定大小 [layer, batch_size, seq_len, num_heads, head_dim]
        self.k_cache = torch.zeros(
            config.num_layers, config.max_batch_size, 
            config.max_seq_len, config.num_heads, config.head_dim
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        
        self.sequence_lengths = torch.zeros(config.max_batch_size, dtype=torch.long)
        self.request_to_slot = {}  # request_id → slot_index（物理位置）
        self.slot_to_request = {}  # slot_index → request_id
        self.free_slots = set(range(config.max_batch_size))
    
    def allocate_slots(self, request_ids: List[int]) -> List[int]:
        # 从 free_slots 分配，建立 request↔slot 映射
        slot_ids = []
        for req_id in request_ids:
            slot = self.free_slots.pop()
            self.request_to_slot[req_id] = slot
            self.slot_to_request[slot] = req_id
            self.sequence_lengths[slot] = 0
            slot_ids.append(slot)
        return slot_ids
    
    def free_slots_for_requests(self, request_ids: List[int]):
        # 请求完成，释放 slot，归还 free_slots
        for req_id in request_ids:
            slot = self.request_to_slot.pop(req_id)
            self.slot_to_request.pop(slot)
            self.sequence_lengths[slot] = 0
            self.free_slots.add(slot)
```

**关键缺陷（引出 PageKVCache 的动机）**：
- 按 `max_batch_size × max_seq_len` 预分配 → 实际请求远短时，大量内存浪费（内部碎片）
- 长 prompt 超过 `max_seq_len` 被截断或拒绝

---

### 4. ContinueBatchingEngine — 主循环

```python
class ContinueBatchingEngine:
    def step(self):
        """
        t1: prefill req1
        t2: decode req1, prefill req2  
        t3: decode req1,2, prefill req3
        ...
        """
        # 阶段1: Decode（已有请求推进一步）
        if self.kv_cache_manager.has_active_requests():
            active_slots, request_ids = self.kv_cache_manager.get_active_slots()
            next_tokens = self.model_wrapper.decode_requests(active_slots)
            
            # 更新请求状态
            completed_ids = []
            for req_id, token in zip(request_ids, next_tokens):
                self.request_manager.requests[req_id].add_token(token.item())
                if self.request_manager.requests[req_id].is_finished():
                    completed_ids.append(req_id)
            
            # 释放完成的槽位
            self.kv_cache_manager.free_slots_for_requests(completed_ids)
        
        # 阶段2: Prefill（新请求进入 batch）
        available_slots = self.kv_cache_manager.get_available_slots()
        new_requests = self.request_manager.get_pending_requests(available_slots)
        if new_requests:
            self.model_wrapper.prefill_requests(new_requests)
```

**时序图**：
```
Step 1: [prefill A]       → A进入 batch
Step 2: [decode A] [prefill B]  → A推进，B进入 batch  
Step 3: [decode A,B] [prefill C]
Step 4: [decode A,B,C]
Step 5: [decode B,C] (A完成释放) [prefill D]  ← 动态填充的核心
```

---

### 5. Prefill vs Decode 的区别

| | Prefill | Decode |
|---|---|---|
| 输入长度 | 整个 prompt（变长）| 单个新 token |
| KV Cache | 新建，写入所有 prompt tokens 的 K/V | 追加一个位置 |
| 计算量 | O(prompt_len²) attention | O(cached_len) attention |
| batch 维度 | 各 prompt 长度不同，需 padding | 固定 1 token/请求 |

**Prefill 的 padding 问题**：
```python
def prefill_requests(self, requests):
    prompts = [prompt for _, prompt in requests]
    max_len = max(len(p) for p in prompts)  # 找最长
    
    input_ids = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        input_ids[i, :len(prompt)] = torch.tensor(prompt)  # 左对齐，右 padding
```

---

## 关键公式 & 原理

### KV Cache 写入位置
```
Decode step t 时，slot s 的 KV：
  k_cache[layer, s, current_length, :, :] = K_new
  v_cache[layer, s, current_length, :, :] = V_new
  sequence_lengths[s] += 1

Attention 时只 attend 到 [0, current_length) 范围：
  scores = Q @ k_cache[layer, s, :current_length, :, :].T
```

### Continue Batching 利用率
```
静态 Batching 利用率 = 有效 token / (batch_size × max_seq_len)
Continue Batching 利用率 → 接近 1（始终填满空位）
```

---

## 与 vLLM PageKVCache 的关系

| 维度 | Continue Batching | vLLM PageKVCache |
|------|-----------------|-----------------|
| KV 存储方式 | 连续 [batch, seq] 矩阵 | 分页块 [num_pages, page_size] |
| 内存碎片 | 严重（内部碎片） | 极小（2个块以内碎片）|
| 长 prompt 支持 | 受 max_seq_len 限制 | 灵活，按需分配页 |
| 动态 batch | ✅ 支持 | ✅ 支持（更优）|
| 实现复杂度 | 低 | 高（Block Table 管理）|

**Continue Batching 是 vLLM 的基础**，解决了"槽位浪费"问题；PageKVCache 在此基础上进一步解决"显存碎片"问题。

---

## 面试考点

**Q1: Continue Batching 和 Static Batching 的核心区别？**
> A: Static Batching 等整个 batch 完成才开始下一批；Continue Batching 一个请求完成就立刻释放槽位，下一 step 填入新请求。核心在于 `step()` 函数先 decode 再 prefill 的设计。

**Q2: Continue Batching 的瓶颈在哪？**
> A: KVCache 按 `batch_size × max_seq_len` 预分配，当请求长度方差大时，内部碎片严重。长 prompt（如 66 token）仍占用 max_seq_len（如 1024）的 KV Cache 空间，浪费 958/1024 = 93%。

**Q3: Prefill 和 Decode 为什么要分开处理？**
> A: Prefill 是一次性处理整个 prompt，是计算密集（O(n²)）；Decode 每步只生成一个 token，是带宽密集（大量 KV Cache 读取）。两者特征不同，分开调度可以优化算力使用（这也是 PD Disaggregation 的动机）。

**Q4: step() 为什么先 decode 再 prefill？**
> A: 先完成正在运行的请求（decode），完成的释放槽位，然后用空出来的槽位接纳新请求（prefill）。顺序颠倒会导致新请求占用槽位后没有空间让旧请求继续。

**Q5: KVCache 的 request_to_slot 映射有什么作用？**
> A: Continue Batching 的 slot（物理位置）和 request_id（逻辑标识）是分离的。一个 slot 在不同时间可以被不同请求使用。request_to_slot 是这两者的桥接，使得 batch 的物理布局可以动态复用。

---

## 工程要点

1. **EOS 检测**：每个 step 生成的 token 如果是 EOS，立刻标记 COMPLETED，下一 step 释放
2. **padding 对齐**：prefill 时多请求 padding 到相同长度，但 KV Cache 写入要按真实长度
3. **KVCache 清零**：释放 slot 时需要 `sequence_lengths[slot] = 0`（否则旧数据影响新请求）
4. **Batch 大小选择**：max_batch_size 由显存决定，过大 OOM，过小 GPU 利用率低

---

## 延伸阅读

- [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] — 分页管理，解决内存碎片
- [[AI/LLM/Inference/vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]] — 块内 attention 实现
- [[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] — Prefill/Decode 混合调度
- [[AI/LLM/Inference/PD-Disaggregation-手撕实操|PD-Disaggregation-手撕实操]] — Prefill/Decode 异机部署

---

*笔记来源：MA-RLHF lc10 / Continue_Batching.ipynb — 2026-02-26*
