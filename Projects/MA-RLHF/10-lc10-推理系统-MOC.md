---
title: "lc10 · 推理系统专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc10_inference"
tags: [moc, ma-rlhf, inference, vllm, lc10]
---

# lc10 · 推理系统专题地图

> **核心问题**：如何把一个 LLM 做成一个**高吞吐、低延迟、生产可用**的推理服务？  
> **学习路线**：从「单请求推理」出发，逐步解决吞吐瓶颈 → 显存碎片 → Kernel 效率 → 系统架构 → 极限优化。

---

## 学习路线（必须按顺序）

```
Step 1  理解瓶颈          Continue Batching          ← 从这里开始
   ↓
Step 2  解决显存碎片       PageKVCache                ← 理解分页管理
   ↓
Step 3  Kernel 优化       PageAttention              ← 不连续块上的 Attention
   ↓
Step 4  系统集成 V0        vLLM-V0                   ← PageKV + PageAttn 组合
   ↓
Step 5  混合批处理         Chunked Prefill            ← TTFT vs TPOT 平衡
   ↓
Step 6  完整系统 V1        vLLM-V1                   ← Chunked Prefill + PageAttn
   ↓
Step 7  推理加速           Speculative Decoding       ← Draft + Verify 机制
   ↓
Step 8  集群级架构         PD Disaggregation          ← Prefill/Decode 节点分离
```

---

## 笔记清单

### ✅ Step 1：Continue Batching — LLM推理服务永动机

**[[AI/LLM/Inference/Continue-Batching-手撕实操|Continue Batching 手撕实操]]**

- **问题**：Static Batching 里一条请求生成完，整个 Batch 都在等 → GPU 利用率极低
- **解法**：请求完成即离队，新请求随时插入，Batch 动态变化
- **代码**：`Request` / `RequestManager` / `ContinueBatchingEngine` 完整实现
- **面试必问**：Continue Batching 和 Static Batching 的吞吐差异？EOS 之后怎么处理？
- 深入阅读：[[Continuous Batching|Continuous Batching 原理笔记]]

---

### ✅ Step 2：PageKVCache — 解决显存碎片

**[[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM PageKVCache 手撕实操]]** ✅ 2026-02-25完成

- **问题**：Continue Batching 里 KV Cache 按最大长度预分配 → 内部碎片严重，Cache 利用率 <40%
- **解法**：借鉴 OS 虚拟内存分页：`BlockTable` 维护 `request_id → [page_id]` 映射，KV Cache 结构从 `[L, slot, seq, H, D]` 变为 `[L, page_id, page_size, H, D]`
- **代码**：`BlockTable` / `PageKVCacheEngine` / `vLLMPageCacheEngine` 完整实现，含 prefill/decoding 两路写入逻辑 + page→batch 聚合
- **关键发现**：decoding 时 `length % T == 0` 触发动态追加 page；`get_sequence_kvcache` 的 reshape 是 CPU gather，真实 PA Kernel 用 CUDA gather 避免内存拷贝
- **面试必问**：Block size 怎么选？分页前后碎片率对比公式？PageAttention 和 FlashAttention 的关系？

---

### ✅ Step 3：PageAttention — 不连续块上的 Attention

**[[AI/LLM/Inference/vLLM-PageAttention-手撕实操|vLLM PageAttention 手撕实操]]** ✅ 2026-02-25完成

- **问题**：物理 page 不连续，标准 Attention 需要先 gather 聚合（O(L) 内存拷贝）→ 专用 Kernel 直接在 pages 上计算
- **解法**：Prefill 用 request-wise 切片；Decoding 用 `repeat_interleave` 分发 q + Online Softmax combine 聚合跨 page 结果
- **代码**：`forward_prefill` / `forward_decoding` / `combine_result` + `FlashAttention` Backend 完整实现（含 Online Softmax 公式推导）
- **关键发现**：`repeat_interleave(q, num_pages)` 把 per-request loop 变成并行矩阵乘；`combine_result` 用 (O, M, L) 三元组跨 page Online Softmax 合并
- **与 FlashAttention 的关系**：FA 解决**计算效率**（IO 复杂度），PA 解决**内存管理**（不连续 pages），两者正交叠加
- 深入阅读：[[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention 手撕实操]]

---

### ✅ Step 4 & Step 6：vLLM V0 / V1 — 完整系统

**[[AI/LLM/Inference/vLLM-V0-V1-完整系统实操|vLLM V0/V1 完整系统实操]]** ✅ 2026-02-25完成

- **V0**：PageKVCache + PageAttention 的完整集成；Prefill page-level 输入转换（`prompt → [num_pages, page_size]` tensor）；page-level logits → request-level last-token 提取
- **V1**：`SchedulerInfo` 统一 PD 混合调度；`merge_prompt` 把所有 token 拼成一串统一前向；`KV.split(chunk_len, dim=2)` 精确拆分 KV 写回各 request 的 pages
- **V0→V1 本质**：调度粒度从 request 降到 token；`last_pos=-1` 标记中间 Chunked chunk（logits 无效，只更新 KV）
- **代码结构**（V1）：`config` / `request` / `scheduler` / `kvcache` / `kernel` / `model` / `wrapper` / `engine` / `server` / `user` 完整 10 文件
- **面试必问**：page-level logits 如何提取 last-token？`last_pos=-1` 的作用？`KV.split(dim=2)` 为什么是 dim=2？

---

### ✅ Step 5：Chunked Prefill — 平衡延迟与吞吐

**[[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked Prefill 手撕实操]]** ✅ 2026-02-25完成

- **问题**：长 prompt Prefill 独占 GPU 数百ms → Decode 请求 TPOT 变高（卡顿）
- **解法**：三阶段 Proj 搭便车（Decoding 免费附在 Prefill 的 GEMM 上）；Chunk-Prefill Attention 需要传历史 KV（第三种 attention 模式）
- **代码**：`ChunkPrefillLinearForward` / `ChunkedPrefillAttention` / `ChunkedPrefillMergeAttention` / `Scheduler` / `ChunkPrefillEngine` 完整实现（含 Mix-PD batch 三路调度）
- **关键发现**：错误版（不传 KV Cache）导致跨 chunk attention 断裂；正确版每次传 `KVCache[:, :, :i*page_size]`；Mix-PD batch 中 Decoding 打头（第 0 行），Prefill 后跟
- **面试必问**：Chunk-Prefill 为什么会增加 TTFT？和 Prefix Caching 的天然兼容性？

---

---

### ✅ Step 7：Speculative Decoding — 推测加速

**[[AI/LLM/Inference/Speculative-Decoding-手撕实操|Speculative Decoding 手撕实操]]** ✅ 2026-02-25完成

- **原理**：小模型（Draft）串行生成 k 个 token → 大模型（Target）一次并行处理 L+k tokens → rejection sampling 验证
- **两种变体**：Greedy（argmax 比较，简单但分布不保证）vs Sampling（rejection sampling，数学保证 = Target 分布）
- **代码**：`SPDecoding` + `SPSamplingDecoding` 完整实现，含 rejection sampling 核心逻辑 `r < min(1, q/p)` + 修正分布重采样
- **加速比公式**：接受率 α 时，期望加速 ≈ (n+1)α；α=0.8/spec_n=5 约 3.2x
- **实现注意**：Notebook 忽略了 KV Cache 更新逻辑（实际接受 k 个 token 后需截断/更新 KV），生产部署时需补充
- 深入阅读：[[Sparrow-Video-LLM-Speculative-Decoding|Sparrow: Speculative Decoding 前沿]]

---

### ✅ Step 8：PD Disaggregation — 集群级架构

**[[AI/LLM/Inference/PD-Disaggregation-手撕实操|PD Disaggregation 手撕实操]]** ✅ 2026-02-25完成

- **动机**：Prefill（compute-bound）和 Decode（memory-bound）对硬件需求截然不同 → 同节点是资源错配
- **架构**：`PrefillActor` + `DecodingActor` + `DistributedKVCacheEngine`（@ray.remote），三方通过 Ray 协调
- **代码**：基于 Ray 的完整分布式推理系统（`pd-inference/` 目录：`actor.py` / `actor_prefill.py` / `actor_decoding.py` / `engine.py` / `KVCache.py` / `scheduler.py` / `transfer.py`）
- **关键发现**：KV 传输带宽是瓶颈（Llama3-70B 8K context ≈ 2GB/request）；async `update_from_prefill` 实现计算-通信重叠；三种 KV Cache 引擎模式（D节点独有/PD各存/中心化）
- **延伸洞察**：PD 分离 = 推理侧"解耦化"；训练侧同样适用（rollout ↔ trainer，即 GRPO/AReal/slime 异步训练框架）
- **面试必问**：KV 传输带宽如何估算？P:D 节点比如何确定？

---

## 核心概念速查

| 概念 | 解决的问题 | 关键数据结构/算法 |
|------|-----------|-----------------|
| Continue Batching | GPU 利用率低 | 动态队列，EOS 触发离队 |
| PageKVCache | 显存碎片（内部碎片 >60%） | Block Table，物理/逻辑块分离 |
| PageAttention | 不连续块上的 Attention | Gather Kernel，分块计算 |
| Chunked Prefill | 长 Prefill 阻塞 Decode（TPOT 抖动） | 固定 Chunk + PD 混合 Batch |
| Speculative Decoding | Decode 是串行瓶颈 | Draft+Verify，rejection sampling |
| PD Disaggregation | 集群资源错配 | Ray Actor，KV 传输协议 |

---

## FlashAttention 在推理系统中的位置

FlashAttention 是推理系统的**底层 Kernel**，不是系统架构组件：
- Continue Batching / PageAttention / Chunked Prefill 都**依赖** FA 做实际的矩阵计算
- FA 解决的是：如何在 IO 受限的 GPU 上高效计算 Attention（在 SRAM 内完成，不落 HBM）
- 专题笔记：[[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention 系列手撕实操]]

---

## 面试高频场景题

**Q：你能描述 vLLM 的完整架构吗？**  
A：从 Continue Batching（调度）→ PageKVCache（内存管理）→ PageAttention Kernel（计算）→ Chunked Prefill（延迟平衡）→ 可选 PD 分离（集群扩展）。

**Q：如何设计一个处理 10 万 QPS 的 LLM 推理服务？**  
A：单机用 vLLM V1（CB+PageKV+ChunkedPrefill）；集群层用 PD 分离 + Ray 编排；加速用 Speculative Decoding（适合 chat 场景）。

**Q：PageAttention 和 FlashAttention 的关系？**  
A：正交关系，FA 管计算效率，PA 管内存布局，生产系统通常两者都用。
