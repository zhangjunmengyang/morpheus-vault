---
title: GQA + KV Cache 手撕实操 · MA-RLHF Batch D
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - ma-rlhf
  - lc8
  - gqa
  - kv-cache
  - attention
  - llama3
  - inference
brief: GQA（Grouped Query Attention）从 MHA→MQA→GQA 推导：KV 头复用 repeat_kv 机制 + 参数量对比；KV Cache 增量解码完整实现：past_key_values 缓存、position 追踪、动态扩容，是 Llama3/Mistral/DeepSeek V3 推理加速的核心组件。
related:
  - "[[Projects/MA-RLHF/lc5/lc5-02-DeepSeek-MLA-手撕实操]]"
  - "[[Projects/MA-RLHF/lc4/lc4-04-TPA-YaRN-RoPE外推手撕实操]]"
  - "[[Projects/MA-RLHF/lc10/lc10-02-vLLM-PageKVCache-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Architecture/Attention 变体综述]]"
---

# GQA + KVCache 手撕实操（MA-RLHF Batch D）

> **来源**：`notebook/Llama3-GQA.ipynb` + `notebook/Easy_generation_with_kvcache.ipynb`
> **评级**：★★★★★
> **字数**：~7000

---

## TL;DR

**GQA（Grouped Query Attention）**：KV 头数 < Q 头数，多个 Q 头共享一组 KV，同时减少 KV Cache 体积和参数量，是 Llama3 / Mistral / DeepSeek V3 的标配注意力机制。

**KV Cache**：Decoder-Only 推理时每个新 token 只需做一次 Q 计算，历史 token 的 K/V 缓存复用，把 $O(n^2)$ 重复计算压缩到 $O(n)$。

两者配合：GQA 减小 KV Cache 存储体积；KV Cache 减少推理 FLOPs。

---

## 一、核心问题

### MHA 的代价

标准 MHA（Multi-Head Attention）中，每个 token 每个头都要存 K 和 V：

$$\text{KV Cache size} = 2 \times \text{seq\_len} \times n\_\text{heads} \times d\_\text{head} \times \text{bytes}$$

以 Llama2-70B 为例：128 heads × 128 dim × 2 (K+V) × 2 (fp16) × 4096 (seq) = **512MB/sequence**。生产环境难以批量并发。

### GQA 的解决思路

减少 KV 头数：让 Q 头分组，每组共享同一 K/V 头：

```
n_q_heads = 6,  n_kv_heads = 2  →  n_rep = 3
Q: Q1 Q2 Q3 Q4 Q5 Q6  (6头)
K: K1 K1 K1 K2 K2 K2  (2头，各复制3份)
V: V1 V1 V1 V2 V2 V2
```

KV Cache 体积降低 `n_rep` 倍（这里是 3x）。

---

## 二、GQA 实现（Llama3 官方版）

### 参数设置

```python
class ModelArgs:
    dim: int = 18           # 总 embedding 维度
    n_heads: int = 6        # Q 头数
    n_kv_heads: int = 2     # KV 头数（关键参数）
    rope_theta: float = 500000
    max_seq_len: int = 17
    model_parallel_size = 1
```

单头维度：`head_dim = dim // n_heads = 18 // 6 = 3`

复制倍数：`n_rep = n_heads // n_kv_heads = 6 // 2 = 3`

### 核心函数：repeat_kv

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 从 n_kv_heads 复制到 n_heads
    输入：[batch, seq_len, n_kv_heads, head_dim]
    输出：[batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # MHA 退化，直接返回
    return (
        x[:, :, :, None, :]                              # [bs, slen, n_kv, 1, d]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # [bs, slen, n_kv, n_rep, d]
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # [bs, slen, n_q, d]
    )
```

**关键 trick**：在第 3 维（n_kv_heads）后插一个维度，`expand` 复制，再 `reshape` 合并。等价于 `torch.repeat_interleave(x, dim=2, repeats=n_rep)` 但更高效。

### Attention 层实现

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        model_parallel_size = args.model_parallel_size
        
        # TP 切分后的局部头数
        self.n_local_heads = args.n_heads // model_parallel_size      # Q头/GPU
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size # KV头/GPU
        self.n_rep = self.n_local_heads // self.n_local_kv_heads        # 复制倍数
        self.head_dim = args.dim // args.n_heads                        # 单头维度

        # 注意权重矩阵维度差异！
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        #                              ↑ 6 * 3 = 18，全头输出
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        #                              ↑ 2 * 3 = 6，只有 KV 头数
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        #                   ↑ 输出是 Q 头数维度（concat 后）

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        
        # 线性变换
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        
        # KV 复制到 Q 头数量（核心步骤）
        xk = repeat_kv(xk, self.n_rep)  # [bs, seq, n_heads, head_dim]
        xv = repeat_kv(xv, self.n_rep)
        
        # 标准 MHA 计算（此时 Q K V 头数相同）
        xq = xq.transpose(1, 2)  # [bs, n_heads, seq, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

---

## 三、GQA × 模型并行（重要！）

### TP 下的 GQA 分配策略

当 `model_parallel_size=2`（2块 GPU），以 6Q + 2KV 头为例：

```
GPU0: Q1, Q2, Q3,  K1, V1
GPU1: Q4, Q5, Q6,  K2, V2
```

每块 GPU 计算：
```
GPU0: Q1,Q2,Q3 × repeat(K1,V1, 3次) → O1,O2,O3
GPU1: Q4,Q5,Q6 × repeat(K2,V2, 3次) → O4,O5,O6
AllReduce → 合并后 Wo 投影
```

**工程意义**：`n_kv_heads` 必须能被 `model_parallel_size` 整除。这是 GQA 模型并行的硬约束，Llama3-70B 设计 8KV 头也是为了支持 8 卡 TP 并行。

### 权重切分

对于 `Wk[n_kv_heads, head_dim, dim]`：
- GPU0 只加载 `Wk[0, :, :]`（第0个KV头的权重）
- GPU1 只加载 `Wk[1, :, :]`（第1个KV头的权重）

参数量也减少了 `model_parallel_size` 倍。

---

## 四、KV Cache 实现（从零手撕）

### 无 KV Cache 的推理（低效版）

```python
# 每步推理把所有历史 token 重新计算
idx = {'input_ids': X}  # X = [1, 10]，初始 10 个 token
for i in range(4):
    output = model(**idx)
    logits = output['logits'][:, -1, :]  # 只取最后一个位置的预测
    idx_next = torch.argmax(logits, dim=1)[0]
    # 把新 token 拼回去，下次要重算 11 个 token，再次 12 个...
    idx['input_ids'] = torch.cat((idx['input_ids'], idx_next.unsqueeze(0).unsqueeze(1)), dim=-1)
```

**问题**：生成第 N 个 token 时，前 N-1 个 token 的 K/V 被重复计算了 N-1 次，复杂度 $O(n^2)$。

### 有 KV Cache 的推理（手写版）

```python
class xiaodonggua_kv_cache(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.lm_head = nn.Linear(D, V)
        self.cache_K = self.cache_V = None  # 初始化为空

    def forward(self, X):
        X = self.Embedding(X)
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        # [bs, seq, D]，此时 seq=1（只有新 token）

        if self.cache_K is None:
            # 第一次：Prefill 阶段，完整计算所有 prompt token 的 KV
            self.cache_K = K
            self.cache_V = V
        else:
            # 后续：只有新 token 的 K/V，拼接到 Cache 末尾
            self.cache_K = torch.cat((self.cache_K, K), dim=1)
            self.cache_V = torch.cat((self.cache_V, V), dim=1)
            K = self.cache_K  # 完整 K（历史 + 新 token）
            V = self.cache_V  # 完整 V

        # Q 只用新 token 的，K/V 用完整序列的
        # Attention: Q[1, d] × K^T[n, d] → scores[1, n]
        scores = Q @ K.transpose(1, 2) / math.sqrt(D)
        weights = F.softmax(scores, dim=-1)
        output = weights @ V  # [bs, 1, D]
        return self.lm_head(output)
```

**推理循环**：
```python
X = torch.randint(0, 64, (1, 10))  # Prompt: 10 tokens
for i in range(3):
    output = model.forward(X)
    next_token = torch.argmax(F.softmax(output, dim=-1), -1)[:, -1]
    X = next_token.unsqueeze(0)  # 下次只输入 1 个 token！
```

**关键设计**：
1. Prefill（第一次）：输入全部 prompt，K/V 写入 Cache，输出也只取最后一个 token 的预测
2. Decode（后续）：每步只输入 1 个 token，K 和 V 实时追加，Q 始终是 `[bs, 1, D]`

---

## 五、GQA + KV Cache 协同

| 维度 | 标准 MHA | GQA | GQA + KVCache |
|------|---------|-----|----------------|
| KV 存储/token | $2 \times n_h \times d_h$ | $2 \times n_{kv} \times d_h$ | 同左（不重复计算） |
| Prefill FLOP | $O(n^2)$ | $O(n^2)$（但 K/V 计算少） | $O(n^2)$，一次性 |
| Decode FLOP/step | $O(n \cdot d)$ 重算 | 同左重算 | $O(d)$，仅新token |
| 内存访问 | 高 | 较高 | 低（cache 复用） |

**DeepSeek V3 实际配置**：
- MLA（替代 GQA）：c_kv 低秩压缩，KV Cache 存 compressed latent，比 GQA 再省 40%
- GQA 是 MLA 的"前置理解"，MLA = GQA + 低秩分解进一步压缩

---

## 六、面试考点

**Q1：GQA 和 MHA、MQA 的区别？**
- MHA：Q/K/V 头数相同（`n_h = n_kv = H`）
- MQA：只有 1 个 KV 头，所有 Q 共享（极端压缩，质量下降）
- GQA：`n_kv` 头（1 < `n_kv` < `n_h`），平衡质量与效率

**Q2：`repeat_kv` 是 inplace copy 还是 view？**
`expand` + `reshape` 的组合。`expand` 返回共享内存的 view（不复制数据），`reshape` 后如果需要连续内存则触发一次 copy。实际 forward 时在 attention 计算前通常需要连续内存，会有一次复制。

**Q3：GQA 的 KV Cache 节省多少？**
节省比例 = `n_heads / n_kv_heads`。Llama3-70B：8Q头/8KV头 = 8x；Llama3-8B：8Q/8KV = MHA（无节省，质量优先）；Mistral-7B：32Q/8KV = 4x。

**Q4：KV Cache 是在哪里存的？**
GPU HBM（显存）。每个 transformer 层都有独立的 KV Cache。KV Cache 容量决定了 max batch size × max seq len，这是显存的主要占用之一，也是 vLLM PageAttention 要解决的核心问题。

**Q5：KV Cache 的两阶段是什么？**
- **Prefill**：并行处理 prompt，$O(n^2)$ 计算，一次性填充 Cache
- **Decode**：逐 token 生成，每步只用新 token 查询，$O(n)$ 计算，IO-bound

**Q6：模型并行下 GQA 的约束是什么？**
`n_kv_heads % model_parallel_size == 0`。每块 GPU 必须有整数个 KV 头，这决定了 GQA 模型的 TP 上限。

---

## 七、与 MLA 的关系

```
MHA → GQA（减少KV头数）→ MQA（极限：1个KV头）
             ↓
           MLA（低秩分解：c_kv = W_DKV × x，再展开）
            ✓ 存 c_kv 而非 K/V，Cache 更小
            ✓ 矩阵吸收：推理时消除展开开销
```

MLA 可以看作 GQA 的深化：GQA 在"头数"维度压缩，MLA 在"低秩"维度压缩。

---

## See Also

- [[Projects/MA-RLHF/lc5/lc5-02-DeepSeek-MLA-手撕实操]] — MLA 低秩分解 + 矩阵吸收（GQA 的深化：头数压缩→低秩压缩）
- [[Projects/MA-RLHF/lc4/lc4-04-TPA-YaRN-RoPE外推手撕实操]] — TPA 的 head-specific 低秩 KV（另一维度压缩）
- [[Projects/MA-RLHF/lc10/lc10-02-vLLM-PageKVCache-手撕实操]] — KV Cache 的物理内存管理（PagedAttention），工程层扩展
- [[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操]] — TP 下 GQA 权重切分细节（分布式场景）
- [[AI/3-LLM/Architecture/Attention 变体综述]] — MHA/MQA/GQA/MLA 全谱系理论对比
