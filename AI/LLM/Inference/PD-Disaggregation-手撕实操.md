---
title: "PD Disaggregation 手撕实操"
brief: "从零实现 Prefill-Decode 分离架构：Prefill 集群（计算密集）与 Decode 集群（内存带宽密集）分别部署，KV Cache 通过分布式存储传输。实现 DisaggregatedInference 完整流程，理解 PD 分离为何能提升专用硬件利用率。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, inference, pd-disaggregation, lc10]
related:
  - "[[AI/LLM/MA-RLHF课程/lc10-推理系统-MOC]]"
  - "[[AI/LLM/Inference/LLM-推理优化-2026-全景]]"
  - "[[AI/LLM/Inference/vLLM]]"
---

# PD Disaggregation 手撕实操

> **来源**: MA-RLHF lc10_inference / PD-Disaggreation.ipynb + pd-inference/
> **系列**: [[AI/LLM/MA-RLHF课程/lc10-推理系统-MOC|lc10-推理系统-MOC]]
> **关联**: [[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] | [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] | [[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]]
> **日期**: 2026-02-25

---

## TL;DR

PD Disaggregation（Prefill/Decode 分离）是集群级推理优化的核心技术：把 Prefill 节点和 Decoding 节点物理分开，各自针对计算特性配置硬件，分别扩展。

**核心问题**：Prefill（compute-bound，需要大量 FLOPS）和 Decoding（memory-bound，需要高带宽 KV 访问）对硬件的需求截然不同。放在同一节点是资源错配。

**解法**：Prefill 节点生产 KV Cache，传输给 Decoding 节点消费。两边独立调度，各自优化 batch size 和硬件配置。

---

## PD 融合 vs PD 分离

| 维度 | PD 融合（Chunked Prefill）| PD 分离 |
|------|--------------------------|---------|
| 节点 | 单节点混合 | 专用节点各自 |
| 硬件配置 | 统一，折中 | P 节点高 FLOPS，D 节点高带宽 |
| Batch Size | PD 共享，互相约束 | 独立优化 |
| KV 传输 | 无（同节点内存）| 有，网络传输开销 |
| 实现复杂度 | 中 | 高（分布式调度 + KV 传输）|
| 适用场景 | 中小规模，延迟敏感 | 大规模生产，吞吐优先 |

**notebook 作者的判断**：不用过度争 PD 融合/分离 的优劣，复杂推理系统一定是**融合和分离兼并**的（Prefill 节点内部也可以用 Chunked Prefill）。系统设计 case-by-case。

---

## 三种 KV Cache 引擎模式

```python
# PD 分离有 3 种 KVCache 引擎
# 1. KVCache 仅存于 Decoding 节点 ← 本 notebook 实现（最简单）
# 2. KVCache 在 P/D 各存各的（Prefill 节点 Cache 用于 Chunked-Prefill）
# 3. KVCache 中心化/去中心化服务（P/D 节点都可以从 Cache 服务存取）
```

---

## 架构设计（基于 Ray）

```
集群架构：
┌─────────────────────────────────────────┐
│              Engine（协调器）             │
│   Scheduler + 请求路由 + KV 传输管理      │
└──────────────┬──────────────┬───────────┘
               │              │
    ┌──────────▼──────┐   ┌───▼──────────────┐
    │  PrefillActor   │   │  DecodingActor   │
    │  高 FLOPS GPU   │──▶│  高带宽 GPU       │
    │  大 Batch P     │   │  大 Batch D       │
    │  Chunked P 可选  │   │  KVCache in mem  │
    └─────────────────┘   └──────────────────┘
         ↓ KV Transfer
    ┌────────────────────────────┐
    │  DistributedKVCacheEngine  │
    │  @ray.remote               │
    │  统一管理 KV 存储           │
    └────────────────────────────┘
```

---

## 核心实现：分布式 KV Cache Engine

```python
@ray.remote
class DistributedKVCacheEngine:
    """
    分布式 KV Cache 管理器（Ray remote actor）
    负责 Prefill 节点写入 → Decoding 节点读取
    """
    def __init__(self, config):
        # KV Cache 主体：[2(K/V), num_layers, batch, seq_len, heads, head_dim]
        self.kv_cache = torch.zeros(
            2, config.num_layers, config.kv_cache_batch,
            config.kv_cache_len, config.num_heads, config.head_dim
        )
        self.request_to_batch = {}   # request_id → kv_batch_id
        self.batch_to_request = {}   # kv_batch_id → request_id

    def get_kv_cache(self, reqs):
        """Decoding 节点读取 KV"""
        kv_batch_ids = [self.request_to_batch[req_id] for req_id in reqs]
        return self.kv_cache[:, :, kv_batch_ids]

    async def update_from_prefill(self, reqs, kv):
        """Prefill 节点写入 KV（async，支持异步传输）"""
        start_ids = len(self.request_to_batch)
        for i, req_id in enumerate(reqs):
            self.request_to_batch[req_id] = start_ids + i
            self.batch_to_request[req_id] = i + start_ids
        self.kv_cache[:, :, start_ids : start_ids + len(reqs)] = kv

    def update_from_decoding(self, reqs, kv, decoding_idx):
        """Decoding 节点更新新 token 的 KV"""
        for i, req_id in enumerate(reqs):
            pos = decoding_idx[i]
            batch_id = self.request_to_batch[req_id]
            self.kv_cache[:, :, batch_id, pos] = kv[:, :, i, 0]

    def free(self, reqs):
        """请求完成后释放 KV 槽位"""
        for req_id in reqs:
            batch_id = self.request_to_batch[req_id]
            self.kv_cache[:, :, batch_id, :] = 0
```

---

## Actor 层：分布式计算单元

### BaseDistributedActor（底层通信）

```python
class BaseDistributedActor:
    """封装 NCCL/GLOO 分布式通信环境的基类"""
    def __init__(self, world_size, rank, master_addr, master_port):
        # 设置 torch.distributed 环境变量
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
```

### PrefillActor（Prefill 节点）

```python
@ray.remote(num_cpus=1)
class PrefillActor(BaseModelActor):
    def init_model_from_pretrained(self, config, model_type):
        self.actor = Actor(config, model_type)
    
    def prefill(self, x):
        """执行 Prefill 前向，返回 KV Cache"""
        logits, kv = self.actor(x=x)
        return logits, kv
```

### DecodingActor（Decoding 节点）

```python
@ray.remote(num_cpus=1)
class DecodingActor(BaseModelActor):
    def init_model_from_pretrained(self, config, model_type):
        self.actor = Actor(config, model_type)
    
    def forward(self, x, kvcaches=None, current_length=None):
        """执行 Decoding 前向（带 KV Cache）"""
        logits, kv = self.actor(x=x, kvcaches=kvcaches, current_length=current_length)
        return logits, kv
```

### RayActorGroup（Actor 集群管理）

```python
class RayActorGroup:
    """管理一组 Ray Actor，支持 1-N 个 GPU 的 Actor 组"""
    
    def __init__(self, num_nodes, num_gpus_per_node, ray_actor_type, pg=None, ...):
        world_size = num_nodes * num_gpus_per_node
        
        # 创建 master actor（rank 0）
        master_actor = ray_actor_type.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(pg=pg, bundle_index=0)
        ).remote(world_size, 0, None, None)
        
        # 创建 worker actors（rank 1 ~ world_size-1）
        master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
        for rank in range(1, world_size):
            worker = ray_actor_type.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(pg=pg, bundle_index=rank)
            ).remote(world_size, rank, master_addr, master_port)
    
    def async_run_method(self, method_name, *args, **kwargs):
        """在所有 Actor 上并行执行方法（返回 future refs）"""
        return [getattr(actor, method_name).remote(*args, **kwargs) 
                for actor in self._actor_handlers]
```

---

## KV Cache 传输机制（两种实现）

### 简单版：同步传输

```
Prefill 完成 → 直接 ray.put(kv) → Decoding 节点 ray.get(kv_ref)
```

优点：实现简单；缺点：Prefill 完成后 Decoding 节点等待，有 KV 传输延迟。

### 分布式版：异步传输（生产推荐）

```
Prefill 节点（生产者）→ 写入 KV 到传输队列（KVCacheEngine）
                           ↓ 网络传输（NVLink / InfiniBand）
Decoding 节点（消费者）← 从 KVCacheEngine 取 KV，完成后即开始 Decode
```

关键：使用 `async def update_from_prefill()`，计算-通信重叠，避免 Decoding 节点等待 Prefill 完成。

---

## 系统设计权衡

### PD 分离适合的场景

1. **超长 context**（10K+ tokens）：Prefill 需要巨量算力，专用节点更高效
2. **大规模并发**（1000+ QPS）：P/D 独立扩容，资源利用率更高
3. **异构硬件**：Prefill 用 H100（高 FLOPS），Decoding 用 A100（高带宽）
4. **成本敏感**：针对 P/D 的不同特性分别选型，避免浪费

### PD 分离的瓶颈

1. **KV 传输带宽**：KV Cache 大小 = 2 × layers × heads × head_dim × seq_len × bsz × dtype_bytes
   - Llama3-70B 处理 8K context：约 2×80×8×128×8192×1×2 ≈ 2GB/request
   - NVLink 带宽约 600 GB/s，InfiniBand 约 200 GB/s
2. **协调复杂度**：Prefill 完成后如何通知 Decoding 节点？如何处理 Decoding 节点繁忙时的 backpressure？
3. **资源分配**：P/D 节点数量比如何确定？（通常 Decoding 是瓶颈，P:D ≈ 1:2~1:4）

---

## "分离"趋势的更大意义

PD 分离是 LLM 推理"解耦化"趋势的一个实例：

```
PD 分离:   Prefill 节点 ↔ Decoding 节点
AF 分离:   Attention 节点 ↔ FFN 节点
训练分离:  Generator（rollout）↔ Trainer（gradient update）
           如：GRPO 异步训练、AReal / Slime 框架
```

共同思路：识别不同计算任务的资源需求差异，解耦后独立优化/扩展。

这与 **Agentic RL 中的异步 rollout 和 training 分离**（如 AReal、slime、verl）是同一种系统设计哲学在不同领域的体现。

---

## 面试考点

**Q1: Prefill 和 Decoding 的计算特性有什么根本差异？**
A: Prefill 处理整个 prompt，计算量 ∝ L²（attention）+ L（FFN），每次 GEMM 矩阵很大，GPU 计算单元充分利用 → **compute-bound**。Decoding 每次只处理 1 个 token，计算量 ∝ L（attention over KV）+ 1（FFN），矩阵乘极小但 KV 巨大 → 主要时间花在从 HBM 加载 KV Cache → **memory-bound**。

**Q2: PD 分离后 KV 传输的带宽需求如何估算？**
A: KV 大小 = 2（K+V）× num_layers × kv_heads × head_dim × seq_len × dtype_bytes。Llama3-70B（80层，GQA 8 heads，128 head_dim）处理 8K context：2×80×8×128×8192×2 ≈ 2.1GB/request。以 1000 QPS 计，需要 ~2TB/s 带宽——实际生产中需要 NVSwitch + InfiniBand 的组合，或限制 P:D 节点比。

**Q3: 为什么 PD 分离后 Decoding 节点 batch size 可以更大？**
A: PD 融合场景下，Prefill 的存在限制了 Decoding batch——长 Prefill 时 Decoding 必须让步（或被切成 chunks）。PD 分离后 Decoding 节点只处理 decoding 请求，可以把所有显存都给 KV Cache，支持超大 batch 的并行 Decoding，吞吐大幅提升。

**Q4: Chunked Prefill 和 PD 分离能共存吗？**
A: 完全可以，而且推荐。PD 分离后，Prefill 节点内部也可以使用 Chunked Prefill（把长 prompt 切片处理），P 节点的 KV Cache 可以用 PageKV 管理。PD 分离解决集群级资源错配，CP 解决单节点内 P/D 时序冲突，两者互补。

**Q5: 训练中的 PD 分离类比是什么？**
A: Agentic RL 中的 rollout/training 分离：rollout 进程（生成轨迹，类似 Prefill）和 training 进程（更新权重，类似 Decoding）分别运行，通过数据队列异步交换数据。verl、AReal、slime 都采用这种设计。本质上都是"识别不同计算阶段的资源需求差异，解耦后独立优化"。

---

## 延伸阅读

- [[AI/LLM/Inference/Chunked-Prefill-手撕实操|Chunked-Prefill-手撕实操]] — PD 融合方案，与 PD 分离互补
- [[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]] — Ray Actor Group 的训练侧应用
- [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] — PD 分离中 KV 的存储管理
- DistServe (OSDI 2024)：PD 分离的系统设计论文，SLO-aware P:D 比例优化
- Mooncake (2024, 月之暗面)：KV 中心化缓存 + 跨 P/D 节点共享的工程实践
- AReal / Slime / verl：训练侧的"计算分离"（rollout ↔ train）
