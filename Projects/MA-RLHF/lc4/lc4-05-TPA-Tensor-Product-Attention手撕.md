# TPA 手撕实操（Tensor Product Attention）

> 来源：`ma-rlhf/notebook/TPA-Pytorch.ipynb`
> 核心思想：用张量积（CP 分解）重构 K/V，减少参数量同时保留多头表达能力

---

## 1. TPA 核心思想

### 标准 MHA 的问题

标准多头注意力中，Q/K/V 各需一个 `(d_model, n_heads × head_dim)` 的完整投影矩阵。KV Cache 在推理时需要存储所有层所有头的 K/V，显存压力大。

### TPA 的解法：张量积分解

TPA 将 K 和 V 的投影分解为两个低秩因子的**张量积**（CP Decomposition）：

$$K = A_k \otimes B_k, \quad V = A_v \otimes B_v$$

具体地：
- **A 因子**：`x → W_A_k(x)`，投影到 `(n_heads, rank)`，捕获**头间分配**
- **B 因子**：`x → W_B_k(x)`，投影到 `(rank, head_dim)`，捕获**头内表示**
- **K = A_k @ B_k / rank**：通过 `bmm` 重构完整 K

这样 K/V 的参数从 `d × (n_heads × head_dim)` 降为 `d × (n_heads × rank + rank × head_dim)`。当 `rank << head_dim` 时，参数量显著减少。

### 参数量分析

| 配置 | 标准 MHA (K 参数) | TPA (K 参数) | 压缩比 |
|------|-----------------|-------------|--------|
| d=512, h=8, d_h=64, r=4 | 512×512 = 262K | 512×32 + 512×256 = 147K | 1.78× |
| d=4096, h=32, d_h=128, r=8 | 4096×4096 = 16.8M | 4096×256 + 4096×1024 = 5.2M | 3.2× |

Q 仍然用完整投影（保留 Query 的完整表达能力）。

---

## 2. 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TPAConfig:
    n_embd: int = 512       # 模型维度
    n_head: int = 8         # 注意力头数
    head_dim: int = 64      # 每头维度
    rank: int = 4           # CP 分解的秩（核心超参数）


class CPLinear(nn.Module):
    """
    Tensor Product Attention 的 QKV 投影层
    Q: 标准全秩投影
    K, V: CP 分解为 A × B 两个低秩因子
    """
    def __init__(self, config: TPAConfig):
        super(CPLinear, self).__init__()
        self.in_features = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.rank = config.rank

        # Q: 标准投影，不做低秩分解（保留完整表达能力）
        self.c_q = nn.Linear(self.in_features, 
                             self.n_head * self.head_dim, bias=False)
        
        # K 的 CP 分解：K = A_k @ B_k
        # A_k: (d_model → n_heads × rank) — "哪个头分配多少权重"
        self.W_A_k = nn.Linear(self.in_features, 
                               self.n_head * self.rank, bias=False)
        # B_k: (d_model → rank × head_dim) — "每个秩分量的头内表示"
        self.W_B_k = nn.Linear(self.in_features, 
                               self.rank * self.head_dim, bias=False)
        
        # V 的 CP 分解：V = A_v @ B_v（结构同 K）
        self.W_A_v = nn.Linear(self.in_features, 
                               self.n_head * self.rank, bias=False)
        self.W_B_v = nn.Linear(self.in_features, 
                               self.rank * self.head_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        关键：A 和 B 因子需要按 (n_head, rank) 和 (rank, head_dim) 
        的 shape 做 xavier 初始化，而不是按展平后的 shape。
        否则初始化尺度不对，训练会不稳定。
        """
        # K 的 A/B 因子初始化
        W_A_k_tensor = self.W_A_k.weight.view(
            self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_k_tensor)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        
        W_B_k_tensor = self.W_B_k.weight.view(
            self.in_features, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_k_tensor)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)

        # V 的 A/B 因子初始化（同理）
        W_A_v_tensor = self.W_A_v.weight.view(
            self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_v_tensor = self.W_B_v.weight.view(
            self.in_features, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)

        nn.init.xavier_uniform_(self.c_q.weight)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            q, k, v: (B, L, n_head, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Q: 标准投影
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # K 的 CP 分解
        # A_k: (B, L, n_head, rank) — 每个头在每个秩分量上的系数
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.n_head, self.rank)
        # B_k: (B, L, rank, head_dim) — 每个秩分量的特征表示
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank, self.head_dim)
        
        # V 的 CP 分解（同理）
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.n_head, self.rank)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank, self.head_dim)

        # 重构 K = A_k @ B_k / rank
        # reshape 为 (B*L, n_head, rank) @ (B*L, rank, head_dim) → (B*L, n_head, head_dim)
        A_k = A_k.view(batch_size * seq_len, self.n_head, self.rank)
        B_k = B_k.view(batch_size * seq_len, self.rank, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank)  # 除以 rank 做缩放
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # 重构 V = A_v @ B_v / rank
        A_v = A_v.view(batch_size * seq_len, self.n_head, self.rank)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim)
        
        return q, k, v
```

### 验证

```python
config = TPAConfig(n_embd=512, n_head=8, head_dim=64, rank=4)
linear = CPLinear(config)

X = torch.randn(6, 7, 512)  # (batch=6, seq_len=7, dim=512)
q, k, v = linear(X)
print(q.shape)  # (6, 7, 8, 64)
print(k.shape)  # (6, 7, 8, 64)
print(v.shape)  # (6, 7, 8, 64)
```

---

## 3. TPA vs MLA 对比

TPA 和 MLA（Multi-head Latent Attention，DeepSeek-V2）都是 KV 压缩方法，但分解方式不同：

| 维度 | TPA（张量积分解） | MLA（低秩联合压缩） |
|------|-----------------|-------------------|
| **分解对象** | 单独分解 K 和 V 的投影矩阵 | 联合压缩 KV 为低维 latent `c_kv` |
| **分解方式** | CP Decomposition: K = A⊗B（头间×头内） | 下投影+上投影: c = W_d·x, K = W_uk·c |
| **KV Cache** | 存 A_k, B_k, A_v, B_v（4 个低秩张量） | 只存 c_kv（一个低维向量） |
| **Cache 大小** | `2(n_h × r + r × d_h) × L` | `d_c × L`（d_c << n_h × d_h） |
| **RoPE 兼容** | B 因子可直接应用 RoPE | 需要额外的 RoPE 分支（`k_rope`） |
| **参数共享** | 头间通过 B 因子共享 | 所有头通过 latent 共享 |
| **来源** | Tensor Product Attention 论文 | DeepSeek-V2 |

**核心区别**：TPA 是「头间-头内」两个维度的因式分解；MLA 是「高维-低维」投影的信息瓶颈。两者都实现了低秩压缩，但 MLA 的 KV Cache 压缩更极致（只存一个向量 vs 四个张量）。

---

## 4. 面试考点

### 考点 1：TPA 的 CP 分解相比标准 MHA 节省了什么？有什么代价？

**答**：CP 分解将 K/V 的投影矩阵从 `d × (n_h × d_h)` 分解为两个因子 `d × (n_h × r)` 和 `d × (r × d_h)`，参数量从 $O(d \cdot n_h \cdot d_h)$ 降为 $O(d \cdot r \cdot (n_h + d_h))$。当 rank $r \ll \min(n_h, d_h)$ 时压缩显著。代价是：(1) forward 多一次 `bmm` 操作重构 K/V；(2) 表达能力受限于 rank——rank 太小会损失多头间的差异性。Q 不做分解是为了保持 Query 侧的完整表达，这是一个工程权衡。

### 考点 2：TPA 初始化为什么要按分解后的 shape 做 xavier？

**答**：A 因子形状为 `(d, n_h, r)`，B 因子为 `(d, r, d_h)`。如果按展平后的 `(d, n_h×r)` 做 xavier，初始化方差会按错误的 fan_in/fan_out 计算，导致 K = A@B 的初始方差偏离预期（可能过大或过小）。按 `(d, n_h, r)` 和 `(d, r, d_h)` 分别初始化，能保证重构后的 K 和标准投影的初始分布接近一致，训练更稳定。
