# DeepSeek MLA 手撕实操

> 来源：MA-RLHF notebook/DeepSeek-MLA.ipynb
> 论文：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

---

## 1. MHA → MQA → GQA → MLA 演进

注意力机制的核心 tradeoff：**显存（KV Cache 大小）** vs **模型效果（表达能力）**。

| 方法 | KV Head 数 | KV Cache 大小 | 效果 |
|------|-----------|---------------|------|
| **MHA** (Multi-Head Attention) | n_h | 2 × n_h × d_h × L | 最优，每头独立 KV |
| **MQA** (Multi-Query Attention) | 1 | 2 × 1 × d_h × L | 显存最省，效果损失明显 |
| **GQA** (Grouped-Query Attention) | n_g (分组数) | 2 × n_g × d_h × L | 折中，LLaMA 3 采用 |
| **MLA** (Multi-Head Latent Attention) | — | 1 × d_c × L | 压缩到极致，效果不损失 |

**演进逻辑**：
- MQA 把所有 head 共享一组 KV → 太粗暴，效果掉太多
- GQA 分组共享 → 折中方案，但压缩比有限
- MLA 另辟蹊径：不减 head 数，而是用**低秩压缩**把 KV 投影到低维空间

---

## 2. MLA 核心原理：低秩联合压缩

### 关键公式

传统 MHA：
```
Q = W_Q × h_t       # [dim] → [n_h × d_h]
K = W_K × h_t       # [dim] → [n_h × d_h]  
V = W_V × h_t       # [dim] → [n_h × d_h]
```

MLA 用 **down-up projection** 替代直接投影：

```
# KV 联合压缩（down projection）
c_KV = W_DKV × h_t        # [dim] → [d_c]，d_c << n_h × d_h

# 从压缩表示恢复（up projection）
K = W_UK × c_KV           # [d_c] → [n_h × d_h]
V = W_UV × c_KV           # [d_c] → [n_h × d_h]

# Q 也做类似压缩
c_Q = W_DQ × h_t          # [dim] → [d_c']
Q = W_UQ × c_Q            # [d_c'] → [n_h × d_h]
```

**核心思想**：K 和 V 共享一个低维压缩向量 `c_KV`，KV Cache 只需存储 `c_KV`。

---

## 3. 完整 PyTorch 实现

### 3.1 配置

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelArgs:
    dim: int = 64         # 隐藏层维度
    n_heads: int = 8      # 注意力头数
    n_kv_heads: int = 2   # GQA 的 KV 头数
    dc_kv: int = 4        # KV 压缩维度 (远小于 dim)
    dc_q: int = 4         # Q 压缩维度
```

### 3.2 标准 GQA（参考基线）

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV head 复制 n_rep 份以匹配 Q head 数"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class MultiHeadsAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)        # [bs, n_heads, seqlen, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = xq @ keys.transpose(2, 3) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = scores @ values
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

### 3.3 MLA 模型定义

```python
class MultiHeadsLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.dc_kv = args.dc_kv   # KV 压缩维度
        self.dc_q = args.dc_q     # Q 压缩维度

        # Q: down → up
        self.wq_down = nn.Linear(args.dim, args.dc_q, bias=False)
        self.wq_up = nn.Linear(args.dc_q, args.dim, bias=False)

        # KV: 共享 down → 各自 up
        self.wkv_down = nn.Linear(args.dim, args.dc_kv, bias=False)
        self.wk_up = nn.Linear(args.dc_kv, args.dim, bias=False)
        self.wv_up = nn.Linear(args.dc_kv, args.dim, bias=False)

        self.wo = nn.Linear(args.dim, args.dim, bias=False)
```

### 3.4 MLA Forward（训练阶段）

```python
# ===== Down Projection（压缩）=====
c_q = mla.wq_down(h)       # [bs, seq_len, dim] → [bs, seq_len, dc_q]
xq = mla.wq_up(c_q)        # [bs, seq_len, dc_q] → [bs, seq_len, dim]

c_kv = mla.wkv_down(h)     # [bs, seq_len, dim] → [bs, seq_len, dc_kv]  ← KV Cache 存这个！
xk = mla.wk_up(c_kv)       # [bs, seq_len, dc_kv] → [bs, seq_len, dim]
xv = mla.wv_up(c_kv)       # [bs, seq_len, dc_kv] → [bs, seq_len, dim]

# ===== 标准多头注意力（与 MHA 无差别）=====
xq = xq.view(bs, seq_len, n_heads, head_dim)
xk = xk.view(bs, seq_len, n_heads, head_dim)
xv = xv.view(bs, seq_len, n_heads, head_dim)

query = xq.transpose(1, 2)     # [bs, n_heads, seq_len, head_dim]
key = xk.transpose(1, 2)
value = xv.transpose(1, 2)

scores = query @ key.transpose(2, 3) / math.sqrt(head_dim)
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
output = scores @ value
output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
output = mla.wo(output)
```

### 3.5 RoPE 特殊处理

MLA 的位置编码不能直接加在压缩后的 KV 上（RoPE 不可低秩分解），解决方案：**分离出额外的位置编码通道**。

```python
class MultiHeadsLatentAttention_withRoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # ... 同上 MLA 参数 ...

        # 额外的 RoPE 权重
        self.wq_up_rope = nn.Linear(args.dc_q, args.dim, bias=False)       # Q: 多头
        self.wk_head_rope = nn.Linear(args.dim, self.head_dim, bias=False)  # K: 单头
```

**Forward 中 RoPE 处理**：

```python
# 位置编码通道（与内容通道分离）
r_q = mla_rope.wq_up_rope(c_q)   # [bs, seq_len, dim] → 多头
r_k = mla_rope.wk_head_rope(h)   # [bs, seq_len, head_dim] → 单头

# 对位置编码通道施加 RoPE
rope_q = apply_rope(r_q)
rope_k = apply_rope(r_k)

# reshape 为多头
rope_q_head = rope_q.view(bs, seq_len, n_heads, head_dim).transpose(1, 2)
rope_k_head = rope_k.unsqueeze(1).repeat(1, n_heads, 1, 1)  # 单头广播到多头

# 拼接：内容 + 位置
query_cat = torch.cat((query, rope_q_head), dim=-1)  # [bs, n_heads, seq_len, 2*head_dim]
key_cat = torch.cat((key, rope_k_head), dim=-1)

# 注意力计算（分母也相应调整）
scores = query_cat @ key_cat.transpose(2, 3) / math.sqrt(2 * head_dim)
```

**为什么 K 的 RoPE 是单头？**
- 从 KV Cache 角度：存单头 rope_k 最经济
- 如果 rope_k 是多头，维度与常规 KV 无差别，失去压缩意义
- Q 的 rope 保留多头是为了**不损失精度**

**KV Cache 最终需要存储**：`c_KV`（latent 向量）+ `rope_k`（位置编码）

> 论文原文：DeepSeek-V2 requires a total KV cache containing $(d_c + d_h^R) \times l$ elements.

---

## 4. KV Cache 压缩比计算

### MHA KV Cache

```
Size_MHA = 2 × n_kv_heads × d_h × L × bytes
```

以 DeepSeek-V2 为例：`n_h = 128, d_h = 128, L = seq_len`
```
Size_MHA = 2 × 128 × 128 × L = 32,768 × L
```

### MLA KV Cache

只存 `c_KV`（压缩表示）+ `rope_k`（单头位置编码）：
```
Size_MLA = (d_c + d_h^R) × L
```

以 DeepSeek-V2 为例：`d_c = 512, d_h^R = 64`
```
Size_MLA = (512 + 64) × L = 576 × L
```

### 压缩比

```
ratio = Size_MHA / Size_MLA = 32,768 / 576 ≈ 56.9×
```

Notebook 中的简化例子：
```
MHA: 2 × n_kv_heads × head_dim = 2 × 4096
MLA: dc_kv = 512
压缩比 = 8192 / 512 = 16×
```

**实际意义**：KV Cache 减少 → 推理服务（如 vLLM）能跑更大 batch size → 提高 decoding 吞吐。

---

## 5. 推理时的矩阵吸收技巧

### 5.1 Q 矩阵吸收

训练时分两步：
```python
c_q = W_DQ @ h        # down
Q = W_UQ @ c_q        # up
```

推理时合并：
```python
W_Q = W_UQ @ W_DQ     # 离线预计算
Q = W_Q @ h            # 一步到位
```

**目的**：
1. 训练时省显存（低秩分解，参数量少）
2. 推理时 Q 用满矩阵保精度（decoding 阶段逐 token 计算 Q，开销不大）

### 5.2 W_UK 吸收到 W_UQ

```
Score = Q × K^T = (W_UQ × c_q) × (W_UK × c_kv)^T
      = c_q^T × W_UQ^T × W_UK × c_kv
```

可以预计算 `W_UQ^T × W_UK`，推理时直接用 `c_kv` 计算 attention score，不需要显式恢复 K。

### 5.3 W_UV 吸收到 W_O

```python
# 原始计算
V = c_kv @ W_UV           # [seq, d_c] × [d_c, dim] → [seq, dim]
O = softmax(S) @ V        
U = O @ W_O               # [seq, dim] × [dim, dim]

# 吸收后
W_UV_absorbed = W_UV @ W_O    # 离线预计算 [d_c, dim] × [dim, dim] → [d_c, dim]
V' = c_kv @ W_UV_absorbed     # 直接得到最终投影
O = softmax(S) @ V'           # 不再需要 W_O
```

**参数节省**：原本存 `W_UV [d_c × dim]` + `W_O [dim × dim]`，吸收后只存 `W_UV_absorbed [d_c × dim]`。

> 论文原文：*"Fortunately, due to the associative law of matrix multiplication, we can absorb W^{UK} into W^{UQ}, and W^{UV} into W^{O}."*

---

## 6. MLA 的本质总结

| 维度 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| Q 计算 | low-rank: W_DQ → W_UQ | 吸收后满矩阵 W_Q = W_UQ × W_DQ |
| KV 计算 | low-rank: W_DKV → W_UK/W_UV | 只存 c_KV，按需 up-project |
| KV Cache | — | c_KV + rope_k（极度压缩）|
| 位置编码 | RoPE 分离通道 | rope_k 也需要 cache |
| 矩阵吸收 | — | W_UK → W_UQ，W_UV → W_O |

**MLA 压缩的本质**：用计算换空间。存储压缩的 `c_KV`，decoding 时实时计算 `K = W_UK @ c_KV`。

---

## 面试考点

### Q1：MLA 和 GQA 的本质区别是什么？

GQA 通过**减少 KV head 数**来压缩 KV Cache，是在 head 维度上做共享；MLA 通过**低秩联合压缩**把所有 head 的 KV 投影到一个低维向量 `c_KV`，是在特征维度上做压缩。MLA 的压缩比远高于 GQA，且由于 up-projection 恢复了完整的多头 KV，理论上不损失表达能力。

### Q2：MLA 为什么要把 RoPE 从 KV 中分离出来？

如果直接对压缩后的 KV 加 RoPE：`RoPE(W_UK @ c_KV)`，那么推理时每次需要先 up-project 再加 RoPE，**无法直接从 c_KV cache 计算**。分离后，位置编码走独立通道（`rope_k = W_K_rope @ h`），可以直接 cache `rope_k`，c_KV 的 up-projection 可以被吸收到 W_Q 中。

### Q3：推理时矩阵吸收具体怎么做？节省了什么？

两处吸收：
1. `W_UK` 吸收到 `W_UQ`：`Score = c_q^T × (W_UQ^T × W_UK) × c_kv`，不需要显式恢复 K
2. `W_UV` 吸收到 `W_O`：`W_UV_absorbed = W_UV @ W_O`，少一次矩阵乘

节省的是**运行时计算**（不需要对每个 token 做 up-projection）和**参数存储**（合并后参数矩阵更少）。

### Q4：MLA 的 KV Cache 到底存什么？大小是多少？

存两部分：
1. **c_KV**：低秩压缩向量，维度 d_c，每 token 一个
2. **rope_k**：位置编码的 key 分量，维度 d_h^R，每 token 一个

总大小 = `(d_c + d_h^R) × L × num_layers × bytes`

相比 MHA 的 `2 × n_h × d_h × L × num_layers × bytes`，DeepSeek-V2 约压缩 **56 倍**。
