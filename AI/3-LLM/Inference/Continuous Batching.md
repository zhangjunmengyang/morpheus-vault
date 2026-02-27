---
title: "Continuous Batching 深度解析"
brief: "LLM 推理服务核心优化：Continuous Batching（动态批处理）将传统静态批（等最长序列结束）改为 token 级别的流式调度，GPU 利用率从<30%提升至80%+。是 vLLM/TRT-LLM/SGLang 等推理引擎的基础架构，面试必考。"
date: 2026-02-13
tags:
  - ai/llm/inference
  - ai/llm/serving
  - type/concept
  - interview/hot
status: active
---

# Continuous Batching 深度解析

> 动态批处理——LLM 推理吞吐量提升 10-20x 的核心调度策略

## 1. 问题背景：为什么需要 Batching？

LLM 推理是 **memory-bound** 的：Decode 阶段每步只生成一个 token，但需要加载整个模型权重。单请求 GPU 利用率极低（通常 < 5%）。Batching 通过并行处理多个请求来分摊模型权重的加载开销。

```
单请求 Decode:
  加载 7B 参数 (14GB FP16) → 只算 1 个 token → GPU 利用率 ~1%

Batch=32 Decode:
  加载 7B 参数 (14GB FP16) → 算 32 个 token → GPU 利用率 ~32%
  
模型权重加载是固定开销，batch 越大越划算（直到 compute-bound）
```

## 2. Static Batching 的问题

### 传统方式

最直观的批处理：收集一批请求，一起送入模型，等 **所有请求都完成** 才释放：

```
Static Batching (batch_size=4):

时间 →
Req1: [████████████░░░░░░░]  生成 12 tokens，在第 12 步完成
Req2: [███████████████████]  生成 19 tokens，最慢的
Req3: [████████░░░░░░░░░░░]  生成 8 tokens，但必须等 Req2
Req4: [██████████████░░░░░]  生成 14 tokens

░ = GPU 做无效计算（padding）或空等
整个 batch 等最慢的 Req2 完成才能接受新请求
```

### Static Batching 的三大缺陷

1. **尾部浪费**：短请求完成后 GPU 空转，平均浪费 50%+ 算力
2. **延迟增加**：新请求必须等当前 batch 全部完成才能开始
3. **吞吐瓶颈**：batch size 受限于最长序列的显存需求

## 3. Continuous Batching：核心原理

### 核心思想

**Iteration-level scheduling**——在每个 decode step（而非每个 request）级别做调度：

```
Continuous Batching:

时间 →  Step1  Step2  Step3  Step4  Step5  Step6  Step7  Step8 ...
Req1:   [██████████████████████████████████████████████]  done
Req2:   [██████████████████████████]  done → Req5 立即插入
Req3:   [████████████████████████████████]  done → Req6 插入
Req4:   [██████████████████████████████████████]  done
Req5:                              [███████████████████████████]
Req6:                                        [██████████████████]

✓ 一旦某个请求完成，新请求立即补位
✓ GPU 始终满载，没有 padding 浪费
```

### 调度流程

```python
# Continuous Batching 调度器伪代码
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size, max_tokens):
        self.running = []      # 正在生成的请求
        self.waiting = deque() # 等待队列

    def step(self):
        # 1. 移除已完成的请求
        finished = [r for r in self.running if r.is_done()]
        for r in finished:
            self.running.remove(r)
            r.return_response()

        # 2. 填充空位：从等待队列取新请求
        while (len(self.running) < self.max_batch_size
               and self.waiting
               and self.has_memory_for(self.waiting[0])):
            new_req = self.waiting.popleft()
            new_req.run_prefill()  # 先做 prefill
            self.running.append(new_req)

        # 3. 所有 running 请求做一步 decode
        if self.running:
            tokens = self.model.decode_step(
                [r.get_input() for r in self.running]
            )
            for r, tok in zip(self.running, tokens):
                r.append_token(tok)
```

## 4. 关键实现细节

### Prefill vs Decode 的混合调度

新请求加入时需要 **Prefill**（处理整个 prompt），这比 Decode 重得多：

```
Prefill: 处理 1000 tokens 的 prompt → 计算量 ~1000x decode step
Decode:  生成 1 个 token → 计算量很小

两种策略:
├── Chunked Prefill: 将长 prompt 分块，穿插在 decode step 中
│   → 不阻塞正在 decode 的请求
│   → vLLM, SGLang 采用
│
└── Prefill-Decode 分离: 专用 prefill 实例 + decode 实例
    → DeepSeek、Mooncake 架构
    → 适合超大规模部署
```

### Token Budget

每个 step 有 token 预算，综合考虑 prefill 和 decode：

```
每步 token budget = max_num_batched_tokens (如 8192)

分配策略:
  decode tokens = len(running)           # 每个请求 1 token
  prefill budget = budget - decode tokens
  
  如果 budget=8192, running=100:
    prefill 可用 = 8192 - 100 = 8092 tokens
    → 可以 prefill 一个 8000 token 的新 prompt
```

## 5. 各框架实现对比

| 框架 | 批处理策略 | Prefill 处理 | 内存管理 | 特色 |
|------|-----------|-------------|---------|------|
| **[[AI/3-LLM/推理/vLLM|vLLM]]** | Continuous Batching | Chunked Prefill | [[AI/3-LLM/推理/KV Cache|PagedAttention]] | 最成熟的开源方案 |
| **TGI** (HuggingFace) | Continuous Batching | Token streaming | 连续内存 + Flash | 生产级 API |
| **[[AI/3-LLM/Inference/TensorRT-LLM|TensorRT-LLM]]** | In-flight Batching | Chunked + Pipelined | Paged KV Cache | NVIDIA 极致优化 |
| **SGLang** | Continuous Batching | Chunked Prefill | RadixAttention | Prefix Caching 强 |
| **DeepSpeed-MII** | Dynamic SplitFuse | Split-and-Fuse | — | 长短请求拆分融合 |

### vLLM 的调度器实现

```python
# vLLM 简化调度流程
class vLLMScheduler:
    def schedule(self):
        # 优先级: prefill(waiting) > decode(running) > swap-in(swapped)
        
        # Phase 1: 尝试调度 waiting 队列中的新请求
        num_batched_tokens = 0
        scheduled_prefills = []
        for seq_group in self.waiting:
            num_tokens = seq_group.num_prompt_tokens
            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq_group):
                break  # 显存不足
            self.block_manager.allocate(seq_group)
            scheduled_prefills.append(seq_group)
            num_batched_tokens += num_tokens
        
        # Phase 2: 调度所有 running 的请求做 decode
        scheduled_decodes = []
        for seq_group in self.running:
            if not self.block_manager.can_append_slots(seq_group):
                # 显存不够 → preempt (swap/recompute)
                self.preempt(seq_group)
                continue
            scheduled_decodes.append(seq_group)
        
        return SchedulerOutputs(
            prefills=scheduled_prefills,
            decodes=scheduled_decodes
        )
```

## 6. 性能数据

### Static vs Continuous Batching 吞吐对比

| 场景 | Static Batching | Continuous Batching | 提升 |
|------|----------------|-------------------|------|
| 短对话 (avg 50 tokens) | ~200 req/s | ~2000 req/s | **10x** |
| 长文档生成 (avg 500 tokens) | ~20 req/s | ~300 req/s | **15x** |
| 混合负载 | ~50 req/s | ~800 req/s | **16x** |

关键因素：**输出长度方差越大，Continuous Batching 优势越明显**。

### 延迟分布

```
Static Batching:
  P50 TTFT: 500ms | P99 TTFT: 5000ms  (等前一个 batch 完成)

Continuous Batching:
  P50 TTFT: 100ms | P99 TTFT: 800ms   (几乎立即开始 prefill)

TTFT 改善 5-10x，因为不需要等待整个 batch 结束
```

## 7. Preemption：显存不足时的处理

当 running 请求太多，显存不够分配新 KV Cache 块时：

```
策略 1: Swapping (交换到 CPU 内存)
  GPU KV Cache ──复制──→ CPU 内存
  等有空间时再 swap back
  缺点: PCIe 带宽有限，swap 慢

策略 2: Recomputation (丢弃并重算)
  直接丢弃被 preempt 的请求的 KV Cache
  重新调度时从 prompt 开始重新 prefill
  适合: 生成长度不长的请求

vLLM 默认策略: Recomputation（开销更可预测）
```

## 8. 与其他优化技术的关系

- **[[AI/3-LLM/Inference/KV Cache|PagedAttention]]**：Continuous Batching 的基础设施——分页内存管理使动态增减请求成为可能
- **[[AI/3-LLM/Architecture/FlashAttention|FlashAttention]]**：加速每一步的 Attention 计算，与 Continuous Batching 正交互补
- **[[AI/3-LLM/Inference/Speculative Decoding|Speculative Decoding]]**：在 Continuous Batching 中实现 SD 需要特殊处理（验证 token 的动态 batch 大小变化）
- **[[AI/3-LLM/Inference/推理优化|推理优化]]**：Continuous Batching 是推理优化 stack 的调度层核心
- **[[AI/3-LLM/Inference/量化综述|量化]]**：量化减小模型和 KV Cache 大小 → 相同显存下 batch size 更大 → 吞吐更高

## 面试常见问题

### Q1: Continuous Batching 和 Static Batching 的核心区别是什么？

**调度粒度不同**：Static Batching 以 request 为粒度，所有请求一起开始、等最慢的完成；Continuous Batching 以 iteration (decode step) 为粒度，每一步都可以移除已完成请求、插入新请求。核心好处是消除了短请求等待长请求的 padding 浪费，GPU 始终满载。

### Q2: 新请求加入时的 Prefill 如何处理？会影响正在 Decode 的请求吗？

会。Prefill 计算量远大于 Decode（一次处理数百上千 tokens）。两种处理方式：(1) **Chunked Prefill**——将长 prompt 分成小块，穿插在 decode step 中，每步只 prefill 一小块，不影响 decode 延迟；(2) **Prefill-Decode 分离**——用独立 GPU 分别处理 prefill 和 decode，如 DeepSeek 的 Mooncake 架构。

### Q3: Continuous Batching 的吞吐提升主要来自哪里？

两个来源：(1) **消除 padding 浪费**——短请求完成立即释放资源，不用空等长请求；(2) **提高 GPU 利用率**——始终保持接近最大 batch size 运行。在输出长度方差大的场景下（如对话系统），提升可达 10-20x。

### Q4: 显存不够时如何处理？Preemption 策略有哪些？

两种策略：(1) **Swapping**——将被抢占请求的 KV Cache 从 GPU 交换到 CPU 内存，有空间时 swap back，瓶颈是 PCIe 带宽；(2) **Recomputation**——直接丢弃 KV Cache，重新调度时从头 prefill，vLLM 默认用此策略因为开销更可预测。选择取决于场景——如果请求已生成大量 tokens，swap 更划算。

### Q5: 为什么说 PagedAttention 是 Continuous Batching 的关键基础设施？

Continuous Batching 需要 **动态分配和释放 KV Cache 内存**：请求随时加入和离开，如果用连续内存分配，会产生严重碎片化。PagedAttention 借鉴 OS 虚拟内存分页机制，将 KV Cache 分成固定大小的页（block），按需分配，完全消除碎片——这使得高效的动态调度成为可能。没有 PagedAttention，Continuous Batching 的内存管理会极其复杂。
