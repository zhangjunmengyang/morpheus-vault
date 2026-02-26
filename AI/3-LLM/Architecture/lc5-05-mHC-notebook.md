---
title: mHC 流形超连接从零手写 · MA-RLHF lc8
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - ma-rlhf
  - lc8
  - mhc
  - hyper-connections
  - residual
  - deepseek
  - architecture
brief: mHC（Manifold-Constrained Hyper-Connections）手撕实操：从标准残差连接出发，逐步推导多变换分支 → HC（可学习多分支融合）→ mHC（doubly stochastic 约束 + Sinkhorn-Knopp 迭代归一化），是 DeepSeek V4 预研的残差连接系统性重设计方向。
related:
  - "[[AI/LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]]"
  - "[[AI/LLM/Architecture/DeepSeek-V3-手撕实操]]"
  - "[[mHC-Manifold-Constrained-Hyper-Connections-DeepSeek]]"
  - "[[AI/LLM/Architecture/Transformer-手撕实操]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]]"
---

# mHC（Manifold-Constrained Hyper-Connections）从零手写

> MA-RLHF Batch D / Architecture notebooks
> Source: `notebook/mHC.ipynb`
> Ref: HC (arXiv:2409.19606) + mHC (arXiv:2512.24880)
> 评分: ★★★★★

---

## TL;DR

mHC 是对残差连接的深度理论扩展：标准残差 `X' = X + F(X)` → 多变换分支 → HC（可学习的多分支融合）→ mHC（约束 HC 矩阵为 doubly stochastic matrix，行列和均为 1）。核心算法是 **Sinkhorn-Knopp 迭代归一化**。这是 DeepSeek V4 预研方向之一，代表下一代 LLM 架构对残差连接的系统性重新设计。

---

## Part 1：从标准残差到多变换分支

### 标准残差
```python
X' = X + F(X)    # F = attention / FFN
```

### 问题
1. 恒等分支（X）是固定的，不可学习
2. 只有一条变换分支 F(X)
3. 各分支权重固定（1:1）

### Multi-Residual Connection（过渡步骤）

```python
class MultiResidualConnection(nn.Module):
    def __init__(self, dim, n):
        self.beta = nn.Parameter(torch.ones(n) * (1.0/n))  # 可学习权重

    def forward(self, h, fun):
        f = fun(h)
        h_ = sum(self.beta[i] * h + f for i in range(n))
        return h_
```

$$X' = \sum_i (\beta_i \cdot h + F(h))$$

但这还是 trajectory-level 的权重（同一 β 对所有 token 相同）。HC 进一步让权重 per-token 动态变化。

---

## Part 2：HC（Hyper-Connections）

### 核心设计

HC 引入一个 **HC 矩阵**，把"恒等分支"和"变换分支"的混合做成可学习的动态加权：

```
输入：h ∈ [B, L, N, D]（N = expansion rate，N 条流）

HC 矩阵结构：
┌─────────────────────────────────┐
│  0          │  B(H)             │  ← 第一行：β 缩放因子（输出给变换分支）
│─────────────│───────────────────│
│  Am(H)      │  Ar(H)            │  ← 其余行：α 混合因子（输入给残差分支）
└─────────────┴───────────────────┘
```

- `Am(H)`：变换分支的输入混合系数
- `Ar(H)`：残差分支的混合系数
- `B(H)`：变换分支输出的缩放系数

### 动态 α（width connection 输入侧）

```python
if dynamic:
    norm_h = layer_norm(h)                         # [B, L, N, D]
    wc_weight = norm_h @ dynamic_alpha_fn          # [B, L, N, N+1]，N+1 = Am + Ar
    dynamic_alpha = tanh(wc_weight) * alpha_scale  # 小缩放（0.01），保持训练稳定
    alpha = dynamic_alpha + static_alpha           # 静态初始值 + 动态调整
    mix_h = alpha.T @ h                            # [B, L, N, D]  混合后的流
```

`static_alpha` 初始化：`layer_id % rate` 位置为 1，其余为单位矩阵——确保初始时每层只激活对应的那条流（类似 Identity init）。

### 动态 β（width connection 输出侧）

```python
dc_weight = norm_h @ dynamic_beta_fn        # [B, L, N]
beta = tanh(dc_weight) * beta_scale + static_beta  # per-token 缩放因子
```

### depth connection（变换后汇合）

```python
def depth_connection(self, mix_h, h_out, beta):
    # h_out：变换分支（attn/FFN）的输出 [B, L, D]
    # beta：缩放因子 [B, L, N]
    nh = einsum("blh,bln->blnh", h_out, beta)  # 把 h_out 复制 N 份，每份用 β_i 缩放
    out = nh + mix_h[..., 1:, :]               # 加上残差分支 Ar(h)
    return out
```

### HC 矩阵示例（论文 Figure）

```
HC = [[0, 1, 1],    ← β 行：输出 β_1=1, β_2=1
      [1, 1, 0],    ← α 行 1：主要来自第 1 流，少量来自第 2 流
      [0, 0, 1]]    ← α 行 2：纯粹第 2 流的恒等传递
```

---

## Part 3：mHC（Manifold-Constrained HC）

### HC 的问题

HC 矩阵的 α（残差权重矩阵）缺乏约束——行列可以任意大，导致梯度不稳定、各流的总信号量不守恒。

### mHC 的解法：约束为 Doubly Stochastic Matrix

**Doubly Stochastic Matrix（双随机矩阵）**：非负矩阵，行和 = 列和 = 1。

$$\mathcal{H}^{res}_l \in \mathbb{R}^{N \times N},\quad \text{each row sum} = \text{each col sum} = 1$$

**物理意义**：
- 行和 = 1：每条输入流的"贡献"总量守恒（不放大不缩小）
- 列和 = 1：每条输出流接收的总信号量守恒

**与最优传输的联系**：doubly stochastic matrix 是最优传输问题的解空间。Sinkhorn-Knopp 就是求这个空间内投影的高效算法。

### Sinkhorn-Knopp 迭代（基础版）

```python
def sinkhorn_knopp_basic(A, it=100, eps=1e-8):
    n, _ = A.shape
    A = torch.clamp(A, min=eps)    # 确保非负
    u, v = torch.ones(n), torch.ones(n)
    
    for _ in range(it):
        u = 1.0 / (A @ v + eps)   # 行归一化
        v = 1.0 / (A.T @ u + eps) # 列归一化
    
    # P = diag(u) @ A @ diag(v)
    return torch.diag(u) @ A @ torch.diag(v)
```

**原理**：交替做行归一化和列归一化，每次迭代都更接近 doubly stochastic matrix。
- 10次迭代：行列和≈1（训练中够用）
- 1000次迭代：行列和≈1.0000（数值精度极高）

### 批量版（生产实现）

```python
def sinkhorn_knopp_batched(A, it=20, eps=1e-8):
    # A: [B, n, n]
    u, v = torch.ones(B, n), torch.ones(B, n)
    
    for _ in range(it):
        Av = torch.bmm(A, v.unsqueeze(2)).squeeze(2)  # [B, n]
        u = 1.0 / (Av + eps)
        At_u = torch.bmm(A.T, u.unsqueeze(2)).squeeze(2)
        v = 1.0 / (At_u + eps)
    
    U, V = torch.diag_embed(u), torch.diag_embed(v)
    P = U @ A @ V  # [B, n, n]
    return P, U, V
```

mHC Fuse 版本用 20 次迭代（约 2% 的误差，换来 10× 的速度）。

### mHC Fuse 实现

作者做了一个关键的工程优化：**把 pre/post/res 三个投影合并成一次大矩阵乘**：

```python
class ManifoldHyperConnectionFuse(nn.Module):
    def __init__(self, dim, rate, max_sk_it=20):
        self.nc = rate * dim                    # 展开后的维度
        self.n2 = rate * rate                   # res 矩阵元素数

        # 一个大投影矩阵替代 3 个独立投影
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2*self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)  # [pre, post, res] 缩放
        self.beta = nn.Parameter(...)

    def mapping(self, h, res_norm):
        h_flat = h.reshape(B, L, N*D)    # 展平
        h_normed = norm.gamma * h_flat    # RMSNorm gamma 部分

        # 一次大矩阵乘，再 split
        H = h_normed @ self.w            # [B, L, n2+2n]
        
        # RMSNorm fused trick：r = norm(h_flat) / sqrt(nc)
        r_ = 1.0 / (h_flat.norm(-1, keepdim=True) / sqrt(nc))

        H_pre = sigmoid(r_ * H[:,:,:n] * alpha[0] + beta[:n])       # [B,L,n], ∈(0,1)
        H_post = 2*sigmoid(r_ * H[:,:,n:2n] * alpha[1] + beta[n:2n]) # [B,L,n], ∈(0,2)
        H_res_raw = r_ * H[:,:,2n:] * alpha[2] + beta[2n:]

        # Sinkhorn-Knopp 约束 res 为 doubly stochastic
        H_res_exp = H_res_raw.reshape(B,L,N,N).exp()   # 先取 exp 确保非负
        with torch.no_grad():
            _, U, V = res_norm(H_res_exp, max_sk_it)    # SK 迭代（no_grad，只取 U,V）
        H_res = U.detach() @ H_res_exp @ V.detach()     # 应用归一化

        return H_pre, H_post, H_res
```

**SK 用 `torch.no_grad()` 的原因**：SK 迭代只是为了找到归一化系数 U,V，梯度通过 `H_res_exp` 直接流过矩阵乘，不需要经过 SK 迭代本身（SK 不可微，也不需要可微）。

### RMSNorm Fused Trick

论文发现：`RMSNorm` 在高维向量（`N × D` 维）上有显著延迟。优化方法：**把 norm 操作的除法移到矩阵乘之后**：

```
原始：norm(h) = gamma * h / rms(h)
     H = norm(h) @ W

等价：H = (gamma * h @ W) / rms(h)    ← 只在一次大矩阵乘后做一次标量除法
```

数学等价（线性操作），但 GPU kernel 效率更高（减少一次 N×D 的 norm 计算）。

---

## Part 4：HC vs mHC vs 标准残差对比

| 机制 | 公式 | 参数量 | 约束 | 核心优势 |
|------|------|-------|------|---------|
| 标准残差 | `X + F(X)` | 0 | 无 | 简单高效 |
| Multi-Residual | `Σ β_i(X + F(X))` | N | 无（β 无界）| 可学习权重 |
| HC | `α @ h + β ⊙ F(mix_h)` | O(N²) | 无 | 动态 per-token 混合 |
| mHC | HC + SK constraint | O(N²) | 双随机 | 信号量守恒，训练稳定 |

---

## 面试高频考点

**Q: mHC 为什么要约束为 doubly stochastic matrix？**
A: 双随机约束保证行列和均为 1，即每条流的输入/输出信号量守恒。没有约束时，HC 矩阵的某些流可能被放大或缩小，破坏梯度流和训练稳定性。类比 GroupNorm/BatchNorm，mHC 是在"信息流"维度上做归一化。

**Q: Sinkhorn-Knopp 算法的原理？**
A: 交替进行行归一化（`u = 1/(Av)`）和列归一化（`v = 1/(A^T u)`）。每次迭代都让矩阵更接近 doubly stochastic matrix。收敛速度取决于矩阵条件数，实践中 20 次迭代足够。

**Q: mHC Fuse 的工程优化是什么？**
A: 把 pre/post/res 三个独立投影合并成一次大矩阵乘（`self.w` 维度 `nc × (n²+2n)`），然后 split。减少了 kernel launch 次数，提高 GPU 利用率。另外 RMSNorm 的除法操作移到矩阵乘之后（等价变换），减少一次高维向量的归一化延迟。

**Q: SK 迭代为什么用 `torch.no_grad()`？**
A: SK 迭代是为了找归一化因子 U,V，这两个是常数（相对于当前 step 的参数）。梯度直接通过 `U @ H_res_exp @ V` 这个矩阵乘传给 `H_res_exp`（即 `H_res_raw`），再流回投影参数 `self.w`。SK 迭代本身不需要也不能有梯度（它不是参数化的操作）。

---

## See Also

- [[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]] — 同批次 Architecture notebook，MLA 低秩压缩（残差连接 vs Attention 压缩两个正交方向）
- [[mHC-Manifold-Constrained-Hyper-Connections-DeepSeek]] — mHC 论文精读（arXiv:2512.24880），理论背景与实验结果
- [[AI/LLM/Architecture/DeepSeek-V3-手撕实操]] — DeepSeek V3 完整架构手撕，mHC 所在的架构体系
- [[AI/LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]] — lc5 课程地图，DeepSeek 组件学习顺序
- [[AI/LLM/Architecture/Transformer-手撕实操]] — 标准残差连接（mHC 的起点）手撕实操
