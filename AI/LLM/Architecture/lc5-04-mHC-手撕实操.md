# mHC 手撕实操（Manifold Hyper-Connections）

> 来源：`ma-rlhf/notebook/mHC.ipynb`
> 论文：[Hyper-Connections](https://arxiv.org/pdf/2409.19606) | [mHC](https://arxiv.org/pdf/2512.24880)

---

## 1. 核心思想：从残差连接到 mHC

### 标准残差连接

```
X' = X + F(X)
```

恒等分支 `X` 不可学习，只有一条残差分支 `F(X)`。三个关键问题：

1. **如何破坏恒等分支？** → 让变换分支 `T(X) ≠ I(X)`
2. **如何构造可学习的残差连接？** → 引入可学习缩放因子
3. **如何扩展残差连接？** → 多条变换分支

### HC（Hyper-Connections）的做法

将单条残差连接扩展为 **多条变换分支 + 可学习缩放因子**：

$$X' = \sum_i (\beta_i h + F(h))$$

HC 引入三组缩放因子矩阵：
- **$\mathcal{A}_m$**（pre）：选择哪条分支送入子层（如 Attention）
- **$\mathcal{A}_r$**（res）：残差分支间的交叉连接
- **$\mathcal{B}$**（post）：子层输出的缩放

这些因子通过 `tanh(norm(H) @ W) * scale + bias` 动态生成，构成一个 HC 矩阵。

### mHC 的改进：超球面流形约束

**HC 的问题**：各分支权重缺乏约束，训练不稳定，权重可能发散。

**mHC 的解法**：对残差连接矩阵 $\mathcal{H}^\text{res}$ 施加 **双随机矩阵约束**（Birkhoff 多面体），使得行列和均为 1：

$$\mathcal{H}^\text{res} \in \{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1} = \mathbf{1},\ \mathbf{1}^\top M = \mathbf{1}^\top,\ M \geq 0 \}$$

实现方式：**Sinkhorn-Knopp 迭代**——交替行列归一化，约 20 次迭代即可收敛。

**为什么能改善梯度流动？**
- 双随机约束保证信息在分支间"守恒"传递，不会因某条分支权重过大而淹没其他分支
- 约束在流形空间上，等价于将更新投影到超球面上，避免梯度爆炸/消失
- 数学上，双随机矩阵的特征值谱更紧凑，梯度传播更稳定

### 在 DeepSeek V3 中的应用

DeepSeek 团队在 DeepSeek-V3 中实测了 HC，发现存在训练稳定性问题，从而提出 mHC。mHC 应用于每个 Transformer 解码块的 **Attention 和 FFN 的残差连接位置**，替换标准 `x + F(x)`。总体流程：

1. 第 0 层：对特征 `[B, L, D]` repeat `n` 次 → `[B, L, n, D]`
2. 每个解码块输入输出均为 `[B, L, n, D]`
3. 最后一层在 `n` 维度做 `sum` 压回 `[B, L, D]`

工程上可结合重计算（recomputation）减少激活显存，变换分支计算量小，主要开销在残差分支。

---

## 2. 完整 PyTorch 实现

### 2.1 基础组件

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """RMSNorm：比 LayerNorm 少一个减均值操作，计算更快"""
    def __init__(self, d_model, eps=1e-12):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x ** 2).mean(-1, keepdim=True)
        out = x / torch.sqrt(mean + self.eps)
        return self.gamma * out
```

### 2.2 Sinkhorn-Knopp 迭代（批量版本）

```python
def sinkhorn_knopp_batched(A, it=20, eps=1e-8):
    """
    将非负矩阵 A 转换为双随机矩阵
    通过交替行列归一化实现：u = 1/(Av), v = 1/(A^T u)
    
    Args:
        A: (B, n, n) 非负矩阵（通常用 exp 保证非负）
        it: 迭代次数，约 20 次即收敛
    Returns:
        P: 双随机矩阵，U, V: 对角缩放矩阵
    """
    batch_size, n, _ = A.shape
    u = torch.ones(batch_size, n, device=A.device)
    v = torch.ones(batch_size, n, device=A.device)
    
    for _ in range(it):
        v_temp = v.unsqueeze(2)                          # (B, n, 1)
        Av = torch.bmm(A, v_temp).squeeze(2)             # (B, n)
        u = 1.0 / (Av + eps)
        
        u_temp = u.unsqueeze(2)                          # (B, n, 1)
        At_u = torch.bmm(A.transpose(1, 2), u_temp).squeeze(2)
        v = 1.0 / (At_u + eps)
    
    U = torch.diag_embed(u)  # (B, n, n)
    V = torch.diag_embed(v)  # (B, n, n)
    P = torch.bmm(torch.bmm(U, A), V)
    return P, U, V
```

### 2.3 mHC Fuse 实现（论文工程优化版）

```python
class ManifoldHyperConnectionFuse(nn.Module):
    """
    mHC 融合实现：将 pre/post/res 三组投影合并为一次矩阵乘法
    
    h: (B, L, N, D) — N=expansion rate，D=feature dim
    
    关键优化：RMSNorm 的除以范数操作延迟到矩阵乘之后，
    数学等价但减少了高维 norm 计算的延迟。
    """
    def __init__(self, dim, rate, layer_id, max_sk_it=20):
        super(ManifoldHyperConnectionFuse, self).__init__()

        self.n = rate       # 扩展率（分支数）
        self.dim = dim
        self.nc = self.n * self.dim   # 展平后的维度
        self.n2 = self.n * self.n     # res 矩阵元素数

        # RMSNorm（作用于展平后的 nC 维度）
        self.norm = RMSNorm(dim * rate)

        # 融合投影：一次乘法得到 pre(n) + post(n) + res(n²) = n²+2n 个值
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2 * self.n))
        
        # 三组独立缩放因子
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        
        # 静态偏置
        self.beta = nn.Parameter(torch.zeros(self.n2 + 2 * self.n))

        self.max_sk_it = max_sk_it

    def mapping(self, h, sk_fn):
        """计算三组超连接因子"""
        B, L, N, D = h.shape

        # Step 1: 展平 + RMSNorm 的 gamma 缩放
        h_vec_flat = h.reshape(B, L, N * D)
        h_vec = self.norm.gamma * h_vec_flat    # gamma 缩放

        # Step 2: 融合投影（一次矩阵乘法）
        H = h_vec @ self.w  # (B, L, n²+2n)

        # Step 3: RMSNorm 延迟计算 — 除以 r = ||x||/√(nC)
        r = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        r_inv = 1.0 / r

        # Step 4: 拆分 + 各自缩放 + 加偏置
        n = N
        H_pre  = r_inv * H[:, :, :n]       * self.alpha[0] + self.beta[:n]
        H_post = r_inv * H[:, :, n:2*n]    * self.alpha[1] + self.beta[n:2*n]
        H_res  = r_inv * H[:, :, 2*n:]     * self.alpha[2] + self.beta[2*n:]

        # Step 5: 约束映射
        H_pre  = F.sigmoid(H_pre)           # pre ∈ (0, 1)
        H_post = 2 * F.sigmoid(H_post)      # post ∈ (0, 2)，允许放大

        # Step 6: Sinkhorn-Knopp 约束 res 为双随机矩阵
        H_res = H_res.reshape(B, L, N, N)
        H_res_exp = H_res.exp()              # exp 保证非负
        with torch.no_grad():
            _, U, V = sk_fn(H_res_exp.reshape(B * L, N, N), self.max_sk_it)
        # detach U, V 使 SK 迭代不参与梯度计算（straight-through）
        P = torch.bmm(torch.bmm(U.detach(), H_res_exp.reshape(B * L, N, N)), V.detach())
        H_res = P.reshape(B, L, N, N)

        return H_pre, H_post, H_res

    def process(self, h, H_pre, H_res):
        """width connection：用 pre 和 res 因子处理输入"""
        # H_pre: (B, L, N) → unsqueeze → (B, L, 1, N) @ (B, L, N, D) → (B, L, 1, D)
        h_pre = H_pre.unsqueeze(dim=2) @ h   # 送入子层的特征
        h_res = H_res @ h                     # 残差分支间交叉连接
        return h_pre, h_res

    def depth_connection(self, h_res, h_out, beta):
        """depth connection：子层输出 × post 因子 + 残差"""
        # beta(H_post): (B, L, N) → (B, L, N, 1) @ (B, L, 1, D) → (B, L, N, D)
        post_mapping = beta.unsqueeze(dim=-1) @ h_out
        return post_mapping + h_res
```

### 2.4 在 Transformer Block 中的调用

```python
# h: (B, L, N, D) — N 条扩展分支

# === Attention Block ===
H_pre, H_post, H_res = attn_mhc.mapping(h, sinkhorn_knopp_batched)
h_pre, h_res = attn_mhc.process(h, H_pre, H_res)
h_out = self_attention(attn_norm(h_pre))          # 子层计算
h = attn_mhc.depth_connection(h_res, dropout(h_out), H_post)

# === FFN Block ===
H_pre, H_post, H_res = ffn_mhc.mapping(h, sinkhorn_knopp_batched)
h_pre, h_res = ffn_mhc.process(h, H_pre, H_res)
h_out = ffn(ffn_norm(h_pre))                      # 子层计算
h = ffn_mhc.depth_connection(h_res, dropout(h_out), H_post)
```

---

## 3. mHC 与标准残差连接的对比

| 维度 | 标准残差连接 | HC | mHC |
|------|------------|-----|-----|
| 公式 | `x + F(x)` | `Σ βᵢx + F(αx)` | 同 HC + 双随机约束 |
| 可学习参数 | 0 | 有（投影 + 缩放） | 有（投影 + 缩放） |
| 分支数 | 1 恒等 + 1 残差 | n 变换 + 1 残差 | n 变换 + 1 残差 |
| 约束 | 无 | 无约束 | res 矩阵 ∈ Birkhoff 多面体 |
| 稳定性 | 稳定但表达力有限 | 可能发散 | 训练稳定 |
| 额外显存 | 0 | 激活值 ×n | 激活值 ×n（可重计算） |

---

## 4. 面试考点

### 考点 1：mHC 对比 HC 做了什么关键改进？为什么有效？

**答**：mHC 对 HC 的残差交叉连接矩阵 $\mathcal{H}^\text{res}$ 施加了**双随机矩阵约束**（行列和均为 1），通过 Sinkhorn-Knopp 迭代实现。这等价于将连接权重投影到 Birkhoff 多面体流形上，保证信息在分支间"守恒"传递——不会因某条分支权重过大而产生数值不稳定。工程上，SK 迭代的 U/V 矩阵用 `detach()` 做 straight-through 估计，不参与反向传播，计算开销可控。

### 考点 2：mHC 的总体数据流是怎样的？额外参数和计算量是多少？

**答**：
- **数据流**：第 0 层将 `[B, L, D]` repeat n 次到 `[B, L, n, D]`；每个解码块输入输出均为 `[B, L, n, D]`；最后一层在 n 维度 sum 回 `[B, L, D]`
- **额外参数**：每层一个融合投影矩阵 `(nD, n²+2n)` + 缩放/偏置共 `(n²+2n+3)` 标量。对 n=2, D=7168 的 DeepSeek-V3，每层约 `14336×8 ≈ 115K` 额外参数，相比层参数量（~百M）可忽略
- **额外计算**：主要是 SK 迭代（n×n 矩阵，n=2~4 极小）和一次 `(nD, n²+2n)` 投影，几乎不影响总 FLOPs
- **显存**：激活量 ×n，但变换分支可用重计算减少
