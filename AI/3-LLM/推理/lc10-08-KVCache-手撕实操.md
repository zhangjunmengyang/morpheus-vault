# KV Cache 手撕实操

> 来源：MA-RLHF notebook/Easy_generation_with_kvcache.ipynb, lecture/lc3_gpt/KVCache.ipynb

---

## 1. 为什么需要 KV Cache

### 1.1 自回归生成的冗余计算

Decoder-only 模型的推理是**逐 token 生成**的：

```
Step 0: "The cat"     → 计算 Q,K,V for [The, cat]       → 预测 "sat"
Step 1: "The cat sat" → 计算 Q,K,V for [The, cat, sat]  → 预测 "on"
Step 2: "The cat sat on" → 计算 Q,K,V for [The, cat, sat, on] → 预测 "the"
```

**问题**：每步都重新计算所有 token 的 K 和 V，但 `The` 和 `cat` 的 K/V 在之前的步骤已经算过了！

### 1.2 计算量分析

**不用 KV Cache**：每次 forward 处理完整序列

- 第 $t$ 步：Q/K/V 投影 $O(t \cdot d^2)$，attention 计算 $O(t^2 \cdot d)$
- 生成 $n$ 个 token 总计：$\sum_{t=1}^{n} O(t^2 \cdot d) = O(n^3 \cdot d)$

**用 KV Cache**：每次只计算新 token 的 Q/K/V

- 第 $t$ 步：Q/K/V 投影 $O(d^2)$（只投影 1 个 token），attention 计算 $O(t \cdot d)$（1 个 Q 与 $t$ 个 K/V）
- 生成 $n$ 个 token 总计：$\sum_{t=1}^{n} O(t \cdot d) = O(n^2 \cdot d)$

**加速比**：$O(n^3) → O(n^2)$，长序列时差异巨大。

### 1.3 核心原理

Next token prediction 的本质：第 $t$ 时刻的 $q_t$ 与 $k_{1:t}, v_{1:t}$ 做注意力计算。

由于 causal mask 的存在，$k_{1:t-1}$ 和 $v_{1:t-1}$ **不依赖** $t$ 时刻的输入，可以缓存复用。

---

## 2. 完整实现

### 2.1 不带 KV Cache 的 Decoder（基线）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x, mask):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        s = q @ k.transpose(2, 1) / math.sqrt(self.dim)
        s = s + mask.unsqueeze(0)
        p = F.softmax(s, dim=-1)
        z = p @ v
        return self.wo(z)

class SimplesDecoder(nn.Module):
    def __init__(self, dim=512, vocab_size=100, max_len=1024):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, dim)
        self.attn = Attention(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        # 预计算 causal mask
        self.mask = -(1 - torch.tril(torch.ones(max_len, max_len))) * float('inf')
        self.mask = torch.nan_to_num(self.mask, nan=0.0)

    def forward(self, x):
        bs, seq_len = x.shape
        X = self.embd(x)
        X = self.attn(X, self.mask[:seq_len, :seq_len])
        return self.lm_head(X)
```

**生成循环（无 KV Cache）**——每次传入完整 input_ids：

```python
def generation_naive(model, input_ids, max_new_token=100):
    for i in range(max_new_token):
        logits = model(input_ids)            # 每次传入完整序列！
        logits = logits[:, -1, :]            # 只取最后一个 token 的 logits
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)  # 序列不断增长
    return input_ids
```

### 2.2 带 KV Cache 的 Attention

```python
class AttentionKVCache(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.KV_cache = None                 # 初始为空

    def forward(self, x, mask):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # ===== KV Cache 核心逻辑 =====
        if self.KV_cache is None:
            self.KV_cache = [k, v]           # Prefill：首次填充
        else:
            self.KV_cache[0] = torch.cat((self.KV_cache[0], k), dim=1)  # 追加新 K
            self.KV_cache[1] = torch.cat((self.KV_cache[1], v), dim=1)  # 追加新 V

        # Q 与完整 KV Cache 做 attention
        s = q @ self.KV_cache[0].transpose(2, 1) / math.sqrt(self.dim)

        # Mask 处理：decoding 阶段只取最后一行
        if q.shape[1] == 1:  # decoding 阶段
            mask = mask[-1, :].unsqueeze(0).unsqueeze(1)
        else:                # prefill 阶段
            mask = mask.unsqueeze(0)
        s = s + mask

        p = F.softmax(s, dim=-1)
        z = p @ self.KV_cache[1]
        return self.wo(z)
```

### 2.3 带 KV Cache 的生成循环

```python
class SimplesDecoderKVCache(nn.Module):
    def __init__(self, dim=512, vocab_size=100, max_len=1024):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, dim)
        self.attn = AttentionKVCache(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.mask = -(1 - torch.tril(torch.ones(max_len, max_len))) * float('inf')
        self.mask = torch.nan_to_num(self.mask, nan=0.0)

    def forward(self, x, cur_len):
        X = self.embd(x)
        X = self.attn(X, self.mask[:cur_len, :cur_len])
        return self.lm_head(X)

def generation_kvcache(model, input_ids, max_new_token=100):
    input_len = input_ids.shape[1]
    output_ids = input_ids.clone()

    for i in range(max_new_token):
        logits = model(input_ids, cur_len=input_len + i)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        input_ids = next_token       # ← 关键：只传 1 个 token！
        output_ids = torch.cat([output_ids, next_token], dim=-1)

    return output_ids
```

**关键变化**：
1. 生成循环中 `input_ids = next_token`（1 个 token），不再传完整序列
2. Attention 内部：新 K/V 追加到 cache，Q 只有 1 个 token

### 2.4 极简版 KV Cache（直观理解）

```python
class xiaodonggua_kv_cache(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.Embedding = torch.nn.Embedding(V, D)
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.lm_head = torch.nn.Linear(D, V)
        self.cache_K = self.cache_V = None

    def forward(self, X):
        X = self.Embedding(X)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)

        # KV Cache：首次填充 or 追加
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

---

## 3. KV Cache 显存占用

### 3.1 显存公式

```
KV Cache 大小 = 2 × n_layers × n_heads × seq_len × head_dim × bytes_per_element
```

- **2**：K 和 V 各一份
- **n_layers**：每层 attention 各有独立的 KV Cache
- **n_heads × head_dim** = hidden_dim（对 MHA）
- **bytes_per_element**：FP16 = 2 bytes，FP32 = 4 bytes

### 3.2 数值举例：LLaMA-2 70B

| 参数 | 值 |
|------|-----|
| n_layers | 80 |
| n_heads | 64 |
| head_dim | 128 |
| hidden_dim | 8192 |
| 精度 | FP16 (2 bytes) |

```
每 token KV Cache = 2 × 80 × 64 × 128 × 2 bytes
                  = 2 × 80 × 8192 × 2
                  = 2,621,440 bytes ≈ 2.5 MB/token

4096 tokens: 2.5 MB × 4096 ≈ 10 GB
```

**这意味着**：batch_size=8 时仅 KV Cache 就需要 ~80 GB，比模型权重还大！

### 3.3 增长特性

- 随 `seq_len` **线性递增**
- Prefill 阶段一次性分配，Decoding 阶段逐 token 追加
- 总显存 = 模型权重 + 激活值 + KV Cache（推理时 KV Cache 常是瓶颈）

---

## 4. MHA / MQA / GQA 的 KV Cache 对比

### 4.1 公式对比

| 方法 | KV Cache 大小 | 公式 |
|------|--------------|------|
| **MHA** | 最大 | `2 × L × n_layers × n_heads × d_head × bytes` |
| **MQA** | 最小 | `2 × L × n_layers × 1 × d_head × bytes` |
| **GQA** (g 组) | 中间 | `2 × L × n_layers × n_groups × d_head × bytes` |

### 4.2 数值举例

以 LLaMA-2 70B 参数（n_heads=64, d_head=128, n_layers=80, FP16）为例，seq_len=4096：

| 方法 | n_kv_heads | 每 token KV 大小 | 4096 tokens |
|------|-----------|-----------------|-------------|
| MHA | 64 | 2.5 MB | 10.0 GB |
| GQA-8 | 8 | 320 KB | 1.25 GB |
| MQA | 1 | 40 KB | 160 MB |

**GQA-8（LLaMA-2 70B 实际配置）**：KV Cache 减少 8×，效果接近 MHA。

### 4.3 各方法 tradeoff

```
MHA: 效果最好，KV Cache 最大
 ↓ 减少 KV head → 显存下降，效果略降
GQA: 分组共享，兼顾效果和效率（主流选择）
 ↓ 极限压缩
MQA: 最省显存，但效果损失明显
 ↓ 换个思路：不减 head，而是压缩维度
MLA: 低秩压缩，效果不损失，显存极低
```

---

## 5. Prefix Caching

### 5.1 动机

很多 LLM 请求共享相同的 **system prompt**：

```
请求 A: [system prompt] + "帮我写个邮件..."
请求 B: [system prompt] + "翻译以下内容..."
请求 C: [system prompt] + "解释什么是..."
```

System prompt 的 KV Cache 完全相同，但如果每个请求独立计算，就会重复浪费。

### 5.2 核心思想

**共享 system prompt 的 KV Cache**：
1. 第一个请求计算 system prompt 的 KV Cache
2. 后续请求直接复用，只计算用户输入部分的 KV

```
┌────────────────┐
│ System Prompt   │  ← KV Cache 计算一次，多请求共享
│ KV Cache        │
└────────┬───────┘
         │
    ┌────┼────┐
    ↓    ↓    ↓
 请求A  请求B  请求C  ← 只计算各自的 user prompt KV
```

### 5.3 工程实现

- **vLLM 的 Automatic Prefix Caching**：自动检测共享前缀，用 hash 匹配
- **SGLang 的 RadixAttention**：用 Radix Tree 管理 prefix cache，支持更灵活的共享模式
- **关键约束**：共享的 KV Cache 是只读的，不同请求的后续 KV 独立管理

### 5.4 收益

- 省去 system prompt 的 prefill 计算（通常是 500-2000 tokens）
- 减少重复的 KV Cache 显存占用
- 对 RAG 场景（共享 retrieved context）特别有效

---

## 6. Prefill vs Decoding：两阶段分析

| 维度 | Prefill | Decoding |
|------|---------|----------|
| 输入 | 完整 prompt（多 token） | 单个 token |
| Q 矩阵 | [seq_len, dim] 多行 | [1, dim] 单行 |
| 计算模式 | Block-wise，并行度高 | Line-wise，需要 batch 提高并行 |
| 瓶颈 | **Compute-bound** | **Memory-bound**（频繁加载 KV Cache + 权重） |
| 与训练的关系 | 计算模式相同 | 独有模式 |

**推理优化方向**：
- Prefill：FlashAttention、Tensor Parallel（提高计算效率）
- Decoding：Continuous Batching、Speculative Decoding（提高 batch 利用率）

---

## 面试考点

### Q1：KV Cache 的原理是什么？节省了多少计算？

KV Cache 利用了 causal attention 的特性：之前 token 的 K/V 不依赖后续 token。因此只需计算新 token 的 K/V 并追加到缓存中，Q 只有新 token 的 1 行。计算量从 $O(n^3)$（每步重算完整序列）降为 $O(n^2)$（每步只算 1 个 Q 与 cache 的 attention）。代价是额外的显存开销：`2 × n_layers × hidden_dim × seq_len × bytes`，对于 70B 模型约 2.5 MB/token。

### Q2：KV Cache 的显存瓶颈如何缓解？

四个方向：
1. **减少 KV head 数**：GQA/MQA（LLaMA-2 70B 用 GQA-8 减少 8×）
2. **低秩压缩**：MLA（DeepSeek-V2 压缩约 56×）
3. **量化**：KV Cache 用 INT8/INT4 存储（精度换显存）
4. **内存管理**：PagedAttention（vLLM）避免显存碎片，Prefix Caching 共享重复前缀

### Q3：Prefill 和 Decoding 阶段的计算特性有什么区别？为什么要分开优化？

Prefill 是多 Q 多 KV 的矩阵乘，compute-bound，需要 FlashAttention 等算子优化。Decoding 是 1 个 Q 与多个 KV 的向量-矩阵乘，memory-bound（瓶颈在从显存加载 KV Cache 和权重矩阵），需要 Continuous Batching 凑 batch、Speculative Decoding 减少 forward 次数来优化。两阶段计算模式完全不同，统一处理会浪费硬件利用率。
