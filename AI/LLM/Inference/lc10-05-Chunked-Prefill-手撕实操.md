---
title: "Chunked-Prefill 手撕实操"
brief: "Chunked-Prefill（SARATHI）解决 Prefill 阻塞 Decode（TTFT/TPOT 矛盾）：把长 Prefill 切成 chunk，与 Decode token piggybacking，实现 Prefill/Decode 混合调度。vLLM V1 的核心调度机制，高频面试考点。"
date: 2026-02-26
type: code-practice
source: "MA-RLHF lc10 推理系统 / Chunked-Prefill.ipynb"
tags: ["code-practice", "inference", "vLLM", "chunked-prefill", "SARATHI", "scheduling"]
related:
  - "[[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]]"
  - "[[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]]"
  - "[[AI/LLM/Inference/vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]]"
  - "[[AI/LLM/Inference/LLM-推理优化-2026-全景|LLM推理优化2026全景]]"
---

# Chunked-Prefill 手撕实操

> **来源**：MA-RLHF lc10 推理系统 / Chunked-Prefill.ipynb（基于 SARATHI 论文）  
> **论文参考**：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills  
> **难度**：★★★★★  
> **面试频率**：★★★★☆（vLLM V1 核心机制，高频）  
> **关联**：[[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]] [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] [[AI/LLM/Inference/PD-Disaggregation-手撕实操|PD-Disaggregation-手撕实操]]

---

## 核心问题

**Prefill 阻塞 Decode（TTFT vs TPOT 矛盾）**：

| 阶段 | 特征 | 问题 |
|------|------|------|
| Prefill | Compute-bound，$X_{[L,d]} @ W_{[d,d]}$，充分利用 GPU | 长 prompt 时耗时数秒，期间 decode 请求全部等待 |
| Decode | Memory-bound，$x_{[1,d]} @ W_{[d,d]}$，大量权重读取，计算量低 | GPU 利用率低，但实时性要求高（TPOT 影响用户体验）|

**矛盾**：一批长 prompt 做 Prefill 时，正在 decoding 的其他请求被完全阻断 → 延迟峰值（stall）→ 用户感受到停顿。

**Chunked-Prefill 解法**：把长 prompt 切成 chunk，每个 step 只处理一个 chunk，与 decode 请求**混合执行**（Piggybacking）。

---

## 计算模式的三元分类

```
传统分类（二元）：
  Prefill  → 输入: [全prompt]，KVCache: None
  Decoding → 输入: [1 token]，KVCache: [全历史]

Chunked-Prefill 引入第三种：
  Chunk-Prefill → 输入: [chunk]，KVCache: [前几个chunk的KV]
  
本质：Chunk-Prefill = 带 KVCache 的局部 Prefill
```

**关键 Attention 模式**：
```
正常 Prefill（seq=20, page=8, 3个chunk）：
  chunk0 (pos 0-7):   Q[0:8]  attend K[0:8]          ← 无 KVCache
  chunk1 (pos 8-15):  Q[8:16] attend K[0:8] + K[8:16] ← 用 chunk0 的 KVCache
  chunk2 (pos 16-19): Q[16:20] attend K[0:16] + K[16:20]
  
矩阵形式：
  chunk0: Q(8×d) @ K(8×d).T  = S(8×8)
  chunk1: Q(8×d) @ K(16×d).T = S(8×16)  ← 必须 attend 到所有之前的 K
  chunk2: Q(4×d) @ K(20×d).T = S(4×20)
```

---

## 核心实现

### 1. Proj 搭便车（Piggybacking Linear Proj）

第一个关键优化：**把 Prefill 和 Decode 请求的线性层合并为一次矩阵乘法**。

```python
def ChunkPrefillLinearForward(W, XP, XD):
    """
    把 Prefill batch 和 Decode batch 的投影合并到一次 forward
    XP: [BP×LP, D] (prefill, 多个请求)
    XD: [BD×LD, D] (decode, 1 token/请求)
    """
    BP, LP, D = XP.shape
    BD, LD, D = XD.shape
    
    # 1. 展平并拼接
    XP_flat = XP.reshape(BP*LP, D)      # [BP*LP, D]
    XD_flat = XD.reshape(BD*LD, D)      # [BD*1,  D]
    XPD = torch.cat((XP_flat, XD_flat), dim=0)  # [BP*LP + BD, D]
    
    # 2. 一次矩阵乘法
    YPD = W(XPD)                         # [BP*LP + BD, D]
    
    # 3. 分割结果
    YP = YPD[:BP*LP].reshape(BP, LP, D)
    YD = YPD[BP*LP:].reshape(BD, LD, D)
    
    return YP, YD
```

**为什么叫 Piggybacking？**  
Decode 请求"搭便车"在 Prefill 的矩阵乘法上——Prefill 的 `X(1024, d) @ W` 计算本来就需要加载权重 W 到 SRAM，顺便把 decode 的 `x(1, d)` 也乘一遍，额外成本极低。

**计算效率分析**：
```
不搭便车：
  Prefill: 加载 W(d×d) → 做 (L×d)@(d×d) → 写结果 → 卸载 W
  Decode:  加载 W(d×d) → 做 (1×d)@(d×d) → 写结果 → 卸载 W
  W 被加载了 2 次

搭便车：
  Combined: 加载 W 一次 → 做 ((L+1)×d)@(d×d) → 分割结果
  W 只加载了 1 次，节省了一次显存→SRAM 传输
```

### 2. ChunkedPrefillAttention

Attention 层需要特殊处理——chunk 之间要 attend 到之前 chunk 的 KV：

```python
class ChunkedPrefillAttention(Attention):
    def forward(self, X, KVCache=None):
        """
        X: 当前 chunk 的 embedding [bsz, chunk_len, dim]
        KVCache: 前几个 chunk 的 KV [2, bsz, cached_len, dim]（可为 None）
        """
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        
        if KVCache is not None:
            K_cache, V_cache = KVCache[0], KVCache[1]
            # 把历史 KV 拼到当前 chunk KV 前面
            K_ = torch.cat([K_cache, K], dim=1)  # [bsz, cached_len + chunk_len, dim]
            V_ = torch.cat([V_cache, V], dim=1)
        else:
            K_, V_ = K, V
        
        # Q attend to 全量 KV（历史 + 当前chunk）
        Z = attention_kernel(Q, K_, V_)
        O = self.Wo(Z)
        return O, [K, V]  # 返回当前 chunk 的 KV，供下一个 chunk 使用
```

**注意**：返回的是 `[K, V]`（当前 chunk 的 KV），不是 `[K_, V_]`——下一个 chunk 会拼上全量 cache，不需要返回历史。

### 3. Chunk Prefill 主循环

```python
def chunk_prefill_method(model, x, page_size, dim):
    """
    把长序列 x 分成 chunks，逐步 prefill 同时维护 KVCache
    """
    bsz, seq_len = x.shape
    x_chunks = x.split(page_size, dim=1)  # 按 page_size 切分
    KVCache = torch.zeros(2, bsz, seq_len, dim)  # 预分配全量 KVCache
    
    for i, x_chunk in enumerate(x_chunks):
        cur_len = x_chunk.shape[1]
        
        # 获取前 i 个 chunk 的已有 KVCache
        if i == 0:
            chunk_kv_cache = None
        else:
            chunk_kv_cache = KVCache[:, :, :i*page_size]  # [2, bsz, i*page_size, dim]
        
        # forward：Q attend to 历史KV + 当前KV
        logits, cur_kv = model.forward(x_chunk, KVCache=chunk_kv_cache)
        
        # 写入当前 chunk 的 KV
        KVCache[0, :, i*page_size: i*page_size+cur_len] = cur_kv[0]
        KVCache[1, :, i*page_size: i*page_size+cur_len] = cur_kv[1]
    
    # 最后一个 chunk 的最后一个 token 的 logits = 下一个 token 的预测
    last_token_logits = logits[:, -1, :]
    return last_token_logits, KVCache
```

### 4. Prefill+Decode 混合执行（核心优化）

真正的 Chunked-Prefill 是把 Prefill chunk 和 Decode 请求**同一个 step 里执行**：

```python
class ChunkedPrefillMergeAttention(Attention):
    """
    同一个 attention 层同时处理：
    - Decode 请求：每个 [1 token]，有完整 KVCache
    - Prefill chunk 请求：[chunk_len tokens]，有部分 KVCache
    """
    def forward(self, X, KVCache_P, KVCache_D, batch_len_p, batch_len_d):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        
        # 分别处理 Decode 和 Prefill chunk 的 Attention
        if batch_len_d != 0:
            # Decode：每个请求单独 attend 自己的完整 KVCache
            Q_d = Q[:len(batch_len_d), :, :]  # decode 请求的 Q
            Z_d = self.forward_decode(Q_d, K_d, V_d, KVCache_D, batch_len_d)
        
        if batch_len_p != 0:
            # Prefill chunk：每个 chunk attend 历史 KVCache + 当前 chunk
            Q_p, K_p, V_p = Q[len_d:], K[len_d:], V[len_d:]
            Z_p = self.forward_prefill_chunk(Q_p, K_p, V_p, KVCache_P, batch_len_p)
        
        # 合并输出
        Z = torch.cat([Z_d, Z_p], dim=0)
        return self.Wo(Z), [K, V]
```

**调度策略**（Scheduler）：
```python
class Scheduler:
    def get_merge_batch(self):
        # 取出正在 decode 的请求（已有完整 KVCache）
        decoding_batch = self._get_decoding_requests()
        
        # 取出正在 prefill 的请求，获取下一个 chunk
        prefill_batch = self._get_next_prefill_chunks()
        
        # 合并成统一输入
        merged_input = self._merge_pd_inputs(prefill_batch, decoding_batch)
        return merged_input, prefill_batch, decoding_batch
```

---

## 关键公式 & 原理

### Attention 的计算量分析

```
Decode 单步（1 token × 全量 KVCache）：
  FLOPs = 2 × seq_len × d²（Q@K.T + 结果@V）
  Memory: O(seq_len × d)（读 KVCache）
  → Memory-bound

Prefill 全量（seq_len tokens）：
  FLOPs = 2 × seq_len² × d（self-attention）
  Memory: O(seq_len × d)
  → Compute-bound（相同 memory，更多计算）

Prefill chunk（chunk_size tokens，历史 cached_len）：
  FLOPs = 2 × chunk_size × (cached_len + chunk_size) × d
  介于两者之间 → 可以与 Decode 混合，平衡 GPU 利用率
```

### TTFT vs TPOT 的 tradeoff

```
chunk_size → 0：每次 Prefill 极少，Decode 不被阻断，TPOT 最优，TTFT 最差
chunk_size → max_seq_len：传统全量 Prefill，TTFT 最优，TPOT 最差
中间值（如 512 tokens）：tradeoff，vLLM V1 默认设置
```

---

## 与 Context Parallelism 的区别

```
Chunked-Prefill（时序切分）：
  单设备，按时间顺序处理 chunks
  chunk N 必须等 chunk N-1 的 KV 写完
  目的：减少 TTFT 对 TPOT 的影响

Ring Attention / Context Parallelism（空间切分）：
  多设备，按序列位置切分
  每设备处理自己的 chunk，通过 P2P 通信传递 KV
  目的：支持超长序列（单卡放不下的 context）
```

**关键区别**：Chunked-Prefill 是时域的，Context Parallelism 是空域的；前者改善服务延迟，后者扩展序列长度。

---

## 面试考点

**Q1: Chunked-Prefill 解决什么问题？**
> A: Prefill 阶段 compute-bound，处理长 prompt 时会阻塞 decode 请求（stall），导致正在生成的用户感受到明显延迟。Chunked-Prefill 把长 prompt 切成 chunk，每 step 只处理一个 chunk，与 decode 请求混合执行，平衡 TTFT 和 TPOT。

**Q2: Chunk-Prefill 的 Attention 为什么要 attend 前面 chunk 的 KV？**
> A: Transformer 的 Attention 是全局的——位置 i 的 Q 需要 attend 到所有 j≤i 的 K/V。如果 chunk 独立处理，chunk1 的 token 就看不到 chunk0 的 context，等价于把序列强行截断，产生错误的 attention 结果。正确实现是：处理 chunk i 时，把 chunk 0...i-1 的 KV 拼到 chunk i 的 KV 前面一起做 attention。

**Q3: Piggybacking 是什么，为什么有效？**
> A: 把 Prefill 和 Decode 请求的线性投影（Wq/Wk/Wv/Wo）合并成一次矩阵乘法。Prefill 本来就需要把权重矩阵 W 加载到 SRAM，顺便乘以 decode 的 next-token embedding 几乎没有额外 memory 开销，却能把 decode 的 projection 计算完成，节省一次 W 的读取。

**Q4: Chunked-Prefill 和 Ring Attention 的本质区别？**
> A: Chunked-Prefill 是时间维度的切分（单机串行处理 chunks），改善服务延迟特性；Ring Attention 是空间维度的切分（多机并行处理不同 token 段），扩展单次处理的序列长度上限。两者不冲突，可以组合使用。

**Q5: vLLM V1 的 Chunked-Prefill 和 V0 有什么区别？**
> A: V0 的 Prefill 是全量的（整个 prompt 一次 forward），独占一个 batch；V1 引入 Chunked-Prefill，把 Prefill 和 Decode 混合在同一个 batch 里，通过 chunk_size 控制 Prefill 每 step 的计算量，与 PageKVCache 配合实现更细粒度的调度。

---

## 工程要点

1. **KVCache 写入位置**：必须按 chunk 的实际位置写入，不能用 `i*page_size` 一刀切，最后一个 chunk 可能不满一个 page_size
2. **Decode 和 Prefill 的 Attention mask 不同**：Decode 用 causal mask（attend [0, cached_len)），Prefill chunk 用 cross-attend（Q attend to cache + current chunk）
3. **OOM 风险**：混合 batch 时，Prefill 的中间激活 + Decode 的 KVCache 同时在显存 → 需要精细的内存管理
4. **chunk_size 的选择**：太小 → Prefill 进度慢，影响 TTFT；太大 → 阻塞 Decode，影响 TPOT。vLLM 默认 512 tokens

---

## 延伸阅读

- [[AI/LLM/Inference/Continue-Batching-手撕实操|Continue-Batching-手撕实操]] — 基础，理解 step() 调度
- [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操|vLLM-PageKVCache-手撕实操]] — 分页 KV 管理
- [[AI/LLM/Inference/vLLM-PageAttention-手撕实操|vLLM-PageAttention-手撕实操]] — block-wise attention
- [[AI/LLM/Infra/xtrain-lc6-Context并行RingAttention手写|xtrain-lc6-Context并行RingAttention手写]] — 空间维度的 chunk（对比理解）
- [[AI/LLM/Inference/PD-Disaggregation-手撕实操|PD-Disaggregation-手撕实操]] — 把 Prefill/Decode 彻底分离到不同机器

---

*笔记来源：MA-RLHF lc10 / Chunked-Prefill.ipynb，基于 SARATHI 论文 — 2026-02-26*
