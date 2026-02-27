---
title: "vLLM V0 / V1 完整系统实操"
brief: "vLLM V0/V1 系统级完整实现：V0 = PageKVCache + PageAttention 集成（page-level Prefill 输入转换 + last-token logits 提取）；V1 = Chunked Prefill + SchedulerInfo 统一 PD 调度（merge_prompt + KV.split(chunk_len) 精确分配）；V0→V1 调度粒度从 request 降到 token，来源 MA-RLHF lc10 推理系统专题。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc10_inference)"
tags: [code-practice, vllm, inference, paged-attention, chunked-prefill, v0-v1, ma-rlhf, lc10]
related:
  - "[[AI/3-LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM PageKVCache 手撕实操]]"
  - "[[AI/3-LLM/Inference/vLLM-PageAttention-手撕实操|vLLM PageAttention 手撕实操]]"
  - "[[AI/3-LLM/Inference/Chunked-Prefill-手撕实操|Chunked Prefill 手撕实操]]"
  - "[[AI/3-LLM/Inference/Continue-Batching-手撕实操|Continue Batching 手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc10-推理系统-MOC|lc10 推理系统专题地图]]"
---

# vLLM V0 / V1 完整系统实操

> **来源**: MA-RLHF lc10_inference / vLLM-V0.ipynb + vLLM-V1/
> **系列**: [[AI/3-LLM/MA-RLHF课程/lc10-推理系统-MOC|lc10 推理系统专题地图]]
> **关联**: [[AI/3-LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache]] | [[AI/3-LLM/Inference/vLLM-PageAttention-手撕实操|vLLM-PageAttention]] | [[AI/3-LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill]]
> **日期**: 2026-02-25

---

## TL;DR

vLLM V0 = **PageKVCache + PageAttention 的完整集成**：把分页 KV 管理和不连续 page 上的 attention 计算真正组合成一个可运行的推理引擎。

vLLM V1 = **V0 + Chunked Prefill**：用 `SchedulerInfo` 统一调度 PD 混合 batch，实现"Prefill 分块、Decoding 搭便车"的生产级系统。

**两个版本的核心差异**：

| 维度 | V0 | V1 |
|------|----|----|
| Prefill | 一次性全量 Prefill | Chunked Prefill（按 max_batch_tokens 切块）|
| 调度单位 | request 级别 | token 级别（`chunk_len` 精确到 token 数）|
| PD 混合 | 先 Decode 再 Prefill（分两段）| merge_prompt 拼成一串，统一前向 |
| Scheduler 输出 | request_ids 列表 | `SchedulerInfo`（含 merge_prompt / chunk_len / last_pos）|
| KV 更新 | 按 request 逐个更新 page | 按 `chunk_len` split KV 后分配到各 request |

---

## vLLM V0：完整集成架构

### 系统架构

```
vLLMvOEngine
├── Scheduler           # 请求队列管理（waiting → running → completed）
├── PageKVCacheEngine   # 分页 KV 存储（BlockTable + k/v_cache）
└── ModelWrapper        # Prefill/Decoding 两路封装
    └── PageToyModel    # PageAttentionBlock × L layers
```

### V0 的核心亮点：Prefill 输入格式转换

V0 中 Prefill 的最复杂逻辑：把 request-level 的 prompt 转换为 **page-level 的 input_ids**：

```python
def prefill_requests(self, request_ids, prompts):
    T = self.cacher.page_size
    request_num_pages = []
    input_ids_list = []

    for request_id, prompt in zip(request_ids, prompts):
        # 1. 分配物理 pages
        page_ids = self.cacher.allocate_request_pages(request_id, len(prompt))
        num_pages = len(page_ids)
        request_num_pages.append(num_pages)

        # 2. prompt → page-level tensor [num_pages, page_size]
        requst_input_ids = torch.zeros(num_pages, T, dtype=torch.long)
        for i in range(num_pages):
            if i == num_pages - 1:
                # 最后一页：只写 len(prompt) % T 个 token
                offset = len(prompt) % T
                requst_input_ids[i, :offset] = torch.tensor(prompt[i*T : len(prompt)])
            else:
                requst_input_ids[i, :] = torch.tensor(prompt[i*T : (i+1)*T])
        input_ids_list.append(requst_input_ids)

    # 3. 所有请求的 page-input 拼成大 tensor
    input_ids = torch.cat(input_ids_list, dim=0)  # [total_pages, page_size]

    # 4. PageAttention Prefill 前向
    logits, layer_kvcaches = self.model.forward(input_ids,
                                                 request_num_pages=request_num_pages)

    # 5. 从 page-level logits 提取 request-level 的 last-token logits
    offset = 0
    for i in range(len(request_ids)):
        batch_id = request_num_pages[i] - 1      # 最后一页的索引
        idx = len(prompts[i]) % T                 # 最后一页的最后有效位置
        page_logits[i] = logits[batch_id + offset, idx, :]
        offset += request_num_pages[i]

    return page_logits, layer_kvcaches, request_page_ids
```

**关键点**：`logits[batch_id + offset, idx]`——从所有 pages 展平后的 tensor 里，找到最后一个 page 的最后有效 token 位置，这才是该请求真正的"下一个 token 预测"。

### V0 主循环

```python
class vLLMvOEngine:
    def step(self):
        # 阶段1: Decoding（已有请求）
        if self.scheduler.get_num_running_requests() > 0:
            request_ids = self.scheduler.get_running_request_ids()
            input_tokens = [last_generated_token for each req]
            
            # ★ 关键：直接用 page 索引读 KV，不聚合
            batch_kvcache, num_pages_len, batch_to_page = \
                self.cacher.get_page_kvcache(request_ids)
            
            logits, layer_kvcaches = self.model_wrapper.decode_next_tokens(
                input_tokens, KVCache=batch_kvcache, num_pages_len=num_pages_len,
                current_length=current_length
            )
            # 更新 page KV Cache + 释放完成的 pages
        
        # 阶段2: Prefill（新请求）
        if self.scheduler.get_num_pending_requests() > 0:
            pending = self.scheduler.get_pending_requests(...)
            logits, layer_kvcaches, request_page_ids = \
                self.model_wrapper.prefill_requests(request_ids, prompts)
            self.update_kvcache(request_ids, layer_kvcaches, request_page_ids)
```

---

## vLLM V1：完整生产系统

V1 的核心升级：`SchedulerInfo` 统一描述混合 batch，`merge_prompt` 把所有 PD token 拼成一串，一次前向处理所有请求。

### SchedulerInfo：调度单元的精确描述

```python
@dataclass
class SchedulerInfo:
    ids: List[int]            # 请求 ID 列表（Decoding 优先，Prefill 后跟）
    chunk_prompts: List[...]  # 每个请求的 token chunk
    chunk_idx: List[int]      # 每个请求在 merge_prompt 中的起始位置
    chunk_len: List[int]      # 每个请求的 token 数（精确）
    merge_prompt: List[int]   # 所有 chunk 拼接成一串（这是模型实际看到的输入）
    
    kv_len: List[int]         # 每个请求当前已有的 KV 长度
    kv_page_len: List[int]    # 每个请求占用的 page 数
    
    is_decoding: List[bool]   # 是否是 Decoding 请求
    last_pos: List[int]       # Prefill 请求的最后有效 logit 位置（-1 = chunk 未完成）
    
    decoding_batch: int       # Decoding 请求数量
    prefill_batch: int        # Prefill 请求数量
```

### V1 Scheduler：token-level 精确调度

```python
def get_requests(self, max_batch_tokens=8192, max_decoding_batch=0, max_prefill_batch=100):
    """
    把所有 running 请求的 chunk 打包成 merge_prompt
    Decoding 请求（每个 1 token）优先，Prefill 请求后跟（可被 max_batch_tokens 截断）
    """
    info = SchedulerInfo()
    count_batch_token = 0

    # ===== Decoding 请求（优先）=====
    for req_id in self.running_requests:
        req = self.requests[req_id]
        if req.is_decoding():
            info.ids.append(req_id)
            info.chunk_prompts.append([req.generated_tokens[-1]])  # 只有 1 个 token
            info.chunk_len.append(1)
            info.chunk_idx.append(count_batch_token)
            info.last_pos.append(0)
            info.is_decoding.append(True)
            count_batch_token += 1

    info.decoding_batch = count_batch_token

    # ===== Prefill 请求（Chunked）=====
    for req_id in self.running_requests:
        req = self.requests[req_id]
        if req.is_decoding():
            continue

        start = req.kv_len  # 已处理到哪了
        available = max_batch_tokens - count_batch_token

        if len(req.prompt[start:]) <= available:
            # 剩余 prompt 全部放入（最后一个 chunk，last_pos 有效）
            chunk = req.prompt[start:]
            info.last_pos.append(len(chunk) - 1)
        else:
            # 超出预算，只放 available 个 token（非最后 chunk，last_pos = -1）
            chunk = req.prompt[start : start + available]
            info.last_pos.append(-1)   # 这个 chunk 的 logits 无效，不生成 next token

        info.chunk_prompts.append(chunk)
        info.chunk_len.append(len(chunk))
        info.chunk_idx.append(count_batch_token)
        info.is_decoding.append(False)
        count_batch_token += len(chunk)

    # 所有 chunk 拼成一串
    for chunk in info.chunk_prompts:
        info.merge_prompt.extend(chunk)

    return info
```

**`last_pos = -1` 的含义**：Chunked Prefill 中，只有最后一个 chunk 的 last-token logits 才是有效预测（整个 prompt 处理完）。中间 chunk 的 logits 丢弃（不更新 request 的 generated_tokens）。

### V1 主循环

```python
def step(self, config):
    if self.scheduler.get_available_request() == 0:
        return

    # 1. 获取 merge batch（Decoding 优先 + Prefill chunks 后跟）
    info = self.scheduler.get_requests(
        max_batch_tokens=config.max_batch_tokens,
        max_prefill_batch=config.max_prefill_batch,
        max_decoding_batch=config.max_decoding_batch
    )

    # 2. 获取 page KV Cache
    kv_cache, info.kv_page_len = self.cacher.get_kv_cache(info.ids)

    # 3. 构造 merge_prompt tensor
    input_ids, = self.get_merge_batch(info)  # [1, total_tokens]

    # 4. 一次统一前向（PageAttention 处理 merge input）
    next_tokens, kv = self.execute(input_ids, kv_cache, info)

    # 5. 更新：KV 按 chunk_len split，各自写回对应 request 的 pages
    self.update(next_tokens, kv, info)
```

### V1 更新机制：按 chunk_len 拆分 KV

```python
def update(self, next_token, KV, info: SchedulerInfo):
    # 更新每个请求的 next token
    for bid, token in enumerate(next_token):
        req_id = info.ids[bid]
        if token != -1:  # -1 = 中间 chunk，不更新
            self.scheduler.update_request(req_id, token.item())

    # ★ 关键：KV 按 chunk_len 拆分，分配给各 request 的 page
    reqs_KV = KV.split(info.chunk_len, dim=2)  # 按 seq 维度切
    for req_id, tmp_KV in zip(info.ids, reqs_KV):
        if self.scheduler.requests[req_id].status == "REQUEST_COMPLETED":
            self.cacher.free(req_id)
        else:
            self.cacher.update_kv_cache(req_id, tmp_KV)

    # 更新 kv_len
    for i, req_id in enumerate(info.ids):
        self.scheduler.requests[req_id].kv_len += info.chunk_len[i]
```

`KV.split(info.chunk_len, dim=2)` 是 V1 的精髓：merge_prompt 一次前向产生的 KV，按各 request 的 chunk 长度精确拆分，每段写回对应 request 的 page KV Cache。

### V1 代码结构

```
vLLM-V1/
├── config.py        # vLLMEngineConfig（含 max_batch_tokens / max_prefill_batch / max_decoding_batch）
├── request.py       # Request（含 kv_len 字段追踪已处理 token 数）
├── scheduler.py     # Scheduler + SchedulerInfo（核心：get_requests() 生成 merge_prompt）
├── kvcache.py       # PageKVCacheEngine（含 get_kv_cache / update_kv_cache / free）
├── kernel.py        # PageAttention Kernel（prefill / decoding 两路 kernel）
├── model.py         # PageToyModel（PageAttentionBlock × L layers）
├── wrapper.py       # ModelWrapper（forward 封装）
├── engine.py        # vLLMEngine（主引擎：step / execute / update）
├── server.py        # HTTP 服务层（请求接入 / 结果返回）
└── user.py          # 客户端模拟（并发请求生成）
```

---

## V0 → V1 升级的本质

```
V0 的 step():                    V1 的 step():
┌─────────────────────────┐      ┌─────────────────────────────┐
│ if decoding:            │      │ info = scheduler.get_requests│
│   decode_batch()        │  →   │ // Decoding + Chunked Prefill│
│ if prefill:             │      │ merged into ONE forward pass │
│   prefill_batch()       │      │ execute(merge_input)         │
└─────────────────────────┘      │ update with chunk_len split  │
  两段分开处理                    └─────────────────────────────┘
                                    统一处理，token-level 精确调度
```

V1 的调度粒度从 **request** 降到 **token**：
- V0：Prefill 一个 request，一次性全部；Decoding 一个 batch
- V1：每步的 token 预算 `max_batch_tokens` 精确分配，Prefill 可被截断（Chunked）

---

## 面试考点

**Q1: vLLM V0 中，Prefill 的 page-level input 和 request-level logits 如何转换？**
A: 输入转换：prompt 按 page_size 切块，填入 `[num_pages, page_size]` tensor，最后一页补 padding。多个请求拼成 `[total_pages, page_size]`。输出提取：模型输出 `[total_pages, page_size, vocab_size]`，取每个请求最后一页(`request_num_pages[i]-1`)的最后有效位置(`len(prompt) % page_size`)的 logits 作为 next-token 预测。

**Q2: V1 中 `last_pos = -1` 的含义？**
A: 当一个 Prefill 请求因 `max_batch_tokens` 限制只处理了部分 prompt（Chunked Prefill 中间 chunk），该 chunk 的输出 logits 无效——因为 attention 只覆盖了 prompt 的前一部分，预测的下一个 token 是错的。`last_pos = -1` 标记这种情况，`execute()` 看到后返回 `-1` token，`update()` 看到 `-1` 就跳过 request 的 next-token 更新，只更新 KV Cache 和 kv_len。

**Q3: V1 中 `KV.split(info.chunk_len, dim=2)` 为什么是 `dim=2`？**
A: V1 的 KV 格式是 `[2, num_layers, total_tokens, num_heads, head_dim]`，其中 dim=2 是 token/序列维度。`KV.split(chunk_len, dim=2)` 按各 request 的 chunk 长度切开，得到每个 request 自己的 KV，再写入对应的 page KV Cache。

**Q4: V0 和 V1 中 Decoding 请求的 KV 如何组织以支持 PageAttention？**
A: Decoding 时调用 `get_page_kvcache(request_ids)`，返回的不是聚合后的连续 KV，而是按 page 索引直接引用物理 page 的 KV tensor，同时返回每个请求的 `num_pages_len`。PageAttention Decoding Kernel 接收这个信息，用 `repeat_interleave(q, num_pages)` 分发 query 到对应 pages，做 block attention 后 Online Softmax combine——零内存聚合。

**Q5: vLLM V1 相比 V0 对 TTFT 有什么影响？**
A: V1 引入 Chunked Prefill 后，长 prompt 的 Prefill 被切块，TTFT 增加（需要多步才能完成 prefill）。但 TPOT 更稳定（每步都有 Decoding 推进，不会被长 Prefill 阻塞）。V1 还通过 `max_batch_tokens` 参数让用户在 TTFT 和 TPOT 之间灵活权衡：增大 `max_batch_tokens` → chunk 更大 → TTFT 更短但 TPOT 抖动更大。

---

## See Also

- [[AI/3-LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM PageKVCache 手撕实操]] — V0/V1 的 KV 存储基础：Block Table 分配 + page 级别 KV 写入/读取
- [[AI/3-LLM/Inference/vLLM-PageAttention-手撕实操|vLLM PageAttention 手撕实操]] — V0/V1 的 Attention Kernel：不连续 page 上的 Online Softmax + zero-copy attention
- [[AI/3-LLM/Inference/Chunked-Prefill-手撕实操|Chunked Prefill 手撕实操]] — V1 引入的核心特性：三阶段 Proj 搭便车 + Mix-PD batch 三路调度
- [[AI/3-LLM/Inference/Continue-Batching-手撕实操|Continue Batching 手撕实操]] — V0/V1 调度层的基础思路：Decode 完成即插入新请求，消灭 padding
- [[LLM-推理优化-2026-全景|LLM 推理优化 2026 全景]] — 系统级视角：vLLM V0/V1 在推理优化全貌中的位置
- [[AI/3-LLM/MA-RLHF课程/lc10-推理系统-MOC|lc10 推理系统专题地图]] — 课程 MOC 入口，V0/V1 在 Step 4 & 6 位置

## 推荐阅读

- [vLLM 论文（SOSP 2023）](https://arxiv.org/abs/2309.06180) — V0 的完整系统设计原文，PageAttention 核心思路
- [vLLM V1 博客（2024.12）](https://blog.vllm.ai/2024/12/11/roadmap-v1.html) — V1 重构的设计动机和性能对比
- [Chunked Prefill 技术博客](https://www.usenix.org/system/files/nsdi24-agrawal.pdf) — Sarathi-Serve，Chunked Prefill 理论基础
