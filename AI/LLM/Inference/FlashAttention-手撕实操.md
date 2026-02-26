---
title: "FlashAttention 手撕实操"
brief: "FlashAttention-1/2完整推导与实现：Online Softmax数值稳定性、分块前向（IO复杂度从O(N²)降至O(N)）、反向传播重计算、循环顺序优化（FlashAttention-2），含Triton GPU内核手撕，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, flashattention, inference, gpu, triton, pytorch]
related:
  - "[[AI/LLM/Architecture/Transformer-手撕实操|Transformer-手撕实操]]"
  - "[[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]]"
  - "[[AI/LLM/Architecture/基础数学组件手撕|基础数学组件手撕]]"
---

# FlashAttention 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

1. [Online Softmax 原理与推导](#1-online-softmax-原理与推导)
2. [FlashAttention 前向：分块计算与数值稳定性](#2-flashattention-前向分块计算与数值稳定性)
3. [FlashAttention-2 前向：改变循环顺序](#3-flashattention-2-前向改变循环顺序)
4. [FlashAttention-2 反向传播](#4-flashattention-2-反向传播)

---

## 1. Online Softmax 原理与推导

FlashAttention 的数学基础是 **Online Softmax**——能增量计算的 Softmax，无需一次性看到所有数据。

### 1.1 标准 Softmax → Safe Softmax

```python
# 标准 Softmax：数值不稳定（exp 溢出）
softmax(x_i) = exp(x_i) / Σ exp(x_j)

# Safe Softmax：减去最大值，数值稳定
M = max(x)
softmax(x_i) = exp(x_i - M) / Σ exp(x_j - M)
```

```python
X = torch.tensor([-0.3, 0.2, 0.5, 0.7, 0.1, 0.8])

# Safe Softmax
X_max = X.max()
X_safe_softmax = torch.exp(X - X_max) / torch.exp(X - X_max).sum()
```

### 1.2 Online Softmax：增量更新

当新数据 `x_new` 到达时，不需要重新计算整个 Softmax，只需更新两个统计量：

- **M**（running max）
- **L**（running sum of exp）

**递推公式**：

```
已有 t-1 步的统计量：M_{t-1}, L_{t-1}

新数据 x_t 到来：
  M_t = max(M_{t-1}, x_t)
  L_t = L_{t-1} * exp(M_{t-1} - M_t) + exp(x_t - M_t)

最终：softmax(x_i) = exp(x_i - M_t) / L_t
```

```python
# 已有前 5 个元素的统计量
X_pre = X[:-1]
M_pre = X_pre.max()
L_pre = torch.exp(X_pre - M_pre).sum()

# 新元素到来，增量更新
M_cur = torch.max(M_pre, X[-1])
L_cur = L_pre * torch.exp(M_pre - M_cur) + torch.exp(X[-1] - M_cur)

# 计算完整 softmax
X_online_softmax = torch.exp(X - M_cur) / L_cur
```

### 1.3 Block Online Softmax：分块并行

将数据切成多个 block，每个 block 独立计算局部 M 和 L，再合并：

```python
X_block = torch.split(X, 3, dim=0)  # 切成 2 块，每块 3 元素

# 并行计算各 block 的局部统计量
M_0 = X_block[0].max()
L_0 = torch.exp(X_block[0] - M_0).sum()

M_1 = X_block[1].max()
L_1 = torch.exp(X_block[1] - M_1).sum()

# 合并：Online Update
M_global = torch.max(M_0, M_1)
L_global = L_0 * torch.exp(M_0 - M_global) \
         + torch.exp(X_block[1] - M_global).sum()

result = torch.exp(X - M_global) / L_global
```

### 1.4 Batch Online Softmax

推广到 batch 维度，对每行独立做 Online Softmax：

```python
X_batch = torch.randn(4, 6)  # [batch=4, dim=6]
X_block_0 = X_batch[:, :3]
X_block_1 = X_batch[:, 3:]

# 各 block 沿 dim=1 计算
M_0, _ = X_block_0.max(dim=1, keepdim=True)
L_0 = torch.exp(X_block_0 - M_0).sum(dim=1, keepdim=True)

M_1, _ = X_block_1.max(dim=1, keepdim=True)
L_1 = torch.exp(X_block_1 - M_1).sum(dim=1, keepdim=True)

# 合并
M_global = torch.maximum(M_0, M_1)
L_global = L_0 * torch.exp(M_0 - M_global) \
         + torch.exp(X_block_1 - M_global).sum(dim=1, keepdim=True)

result = torch.exp(X_batch - M_global) / L_global
# 等价于 F.softmax(X_batch, dim=1)
```

**核心认知**：Online Softmax 使得 Attention 的 Softmax 可以分块增量计算，这是 FlashAttention 的数学基础。

---

## 2. FlashAttention 前向：分块计算与数值稳定性

### 2.1 标准 Attention

```python
O = softmax(Q @ K^T) @ V  # 需要 O(N²) 存储 attention matrix
```

**问题**：N² 的 attention matrix 在长序列时占满 GPU HBM。

### 2.2 FlashAttention-V1：KV 外循环，Q 内循环

**核心思想**：不显式存储完整 attention matrix，逐块计算并用 Online Softmax 增量更新输出。

```python
# 设置
Q_LEN = K_LEN = 6
BLOCK_SIZE = 3
Tr = Q_LEN // BLOCK_SIZE  # Q block 数
Tc = K_LEN // BLOCK_SIZE  # KV block 数

# 初始化：每个 Q block 维护 O, l(sum), m(max)
O_BLOCKS = [zeros] * Tr
l_BLOCKS = [zeros] * Tr   # running sum(exp)
m_BLOCKS = [-inf]  * Tr   # running max

# 外循环：KV blocks；内循环：Q blocks
for j in range(Tc):        # KV loop (外)
    Kj, Vj = K_BLOCKS[j], V_BLOCKS[j]
    for i in range(Tr):    # Q loop (内)
        Qi = Q_BLOCKS[i]
        Oi, li, mi = O_BLOCKS[i], l_BLOCKS[i], m_BLOCKS[i]

        # 1. 计算局部 attention score
        S_ij = Qi @ Kj.T

        # 2. 局部统计量
        m_block = max(S_ij, dim=-1)
        P_ij = exp(S_ij - m_block)            # 局部 softmax 分子
        l_block = sum(P_ij, dim=-1)           # 局部 softmax 分母

        # 3. Online Softmax 更新全局统计量
        m_new = maximum(m_block, mi)
        l_new = li * exp(mi - m_new) + exp(m_block - m_new) * l_block

        # 4. 更新输出（带 rescale）
        O_BLOCKS[i] = (li / l_new) * exp(mi - m_new) * Oi \
                     + (exp(m_block - m_new) / l_new) * P_ij @ Vj

        l_BLOCKS[i] = l_new
        m_BLOCKS[i] = m_new

O = cat(O_BLOCKS)
```

**关键**：V1 每步都对 O 做 `1/l_new` 的 rescale（inner scaled）。

### 2.3 完整 PyTorch 实现

```python
NEG_INF = -1e10
EPSILON = 1e-10

O = torch.zeros_like(Q, requires_grad=True)
l = torch.zeros(Q.shape[:-1])[..., None]
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

for j in range(Tc):  # KV 外循环
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]
    for i in range(Tr):  # Q 内循环
        Qi = Q_BLOCKS[i]
        Oi, li, mi = O_BLOCKS[i], l_BLOCKS[i], m_BLOCKS[i]

        S_ij = Qi @ Kj.transpose(2, 3)
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
        P_ij = torch.exp(S_ij - m_block_ij)
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

        mi_new = torch.maximum(m_block_ij, mi)
        li_new = torch.exp(mi - mi_new) * li \
               + torch.exp(m_block_ij - mi_new) * l_block_ij

        # Inner scaled: 每步 rescale
        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                     + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij @ Vj

        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
```

---

## 3. FlashAttention-2 前向：改变循环顺序

### 3.1 核心改进

FlashAttention-2 将**循环顺序反转**：Q 外循环，KV 内循环。

**好处**：
- 减少 O 的 HBM↔SRAM 交换（O 驻留 SRAM 整个内循环）
- 只在**最后**做一次 `1/l` 的 rescale（outer scaled）

### 3.2 数学推导

FlashAttention-1（inner scaled）：每步除以 `l_new`
$$O^{(t)} = \frac{l^{(t-1)}}{l^{(t)}} \cdot e^{m^{(t-1)}-m^{(t)}} \cdot O^{(t-1)} + \frac{e^{m_{block}-m^{(t)}}}{l^{(t)}} \cdot P_{ij}V_j$$

FlashAttention-2（outer scaled）：累积未归一化的 $\tilde{O}$，最后统一除
$$\tilde{O}^{(t)} = e^{m^{(t-1)}-m^{(t)}} \cdot \tilde{O}^{(t-1)} + e^{S_{ij}-m^{(t)}} \cdot V_j$$
$$O = \frac{\tilde{O}^{(N)}}{l^{(N)}}$$

### 3.3 实现

```python
for i in range(Tr):  # Q 外循环 —— O 驻留 SRAM
    Qi = Q_BLOCKS[i]
    Oi = O_BLOCKS[i]
    li = l_BLOCKS[i]
    mi = m_BLOCKS[i]

    for j in range(Tc):  # KV 内循环
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]

        S_ij = Qi @ Kj.transpose(2, 3)
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
        mi_new = torch.maximum(m_block_ij, mi)
        P_ij_hat = torch.exp(S_ij - mi_new)
        l_block_ij = torch.sum(P_ij_hat, dim=-1, keepdims=True) + EPSILON

        li_new = torch.exp(mi - mi_new) * li + l_block_ij

        # Outer scaled: 不除 l，累积未归一化的 O
        Oi = torch.exp(mi - mi_new) * Oi + P_ij_hat @ Vj

        li = li_new
        mi = mi_new

    # 最终统一 rescale
    O_BLOCKS[i] = Oi / li_new
    l_BLOCKS[i] = li_new
    m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
```

### 3.4 V1 vs V2 对比

| | FlashAttention-1 | FlashAttention-2 |
|---|---|---|
| 外循环 | KV blocks | **Q blocks** |
| 内循环 | Q blocks | **KV blocks** |
| O rescale | 每步内部 rescale | **最后统一 rescale** |
| O 在 SRAM | 内循环每步读写 | **整个内循环驻留** |
| HBM 访问 | O 频繁交换 | **O 交换减少** |

---

## 4. FlashAttention-2 反向传播

### 4.1 标准 Attention 反向（手写推导）

给定前向计算链：

```
X → (Wq,Wk,Wv) → Q,K,V → S = QK^T/√d → P = softmax(S) → O = PV → Y = Wo·O
```

逐步反向：

```python
# Loss = 0.5 * MSE
dY = (Y - Y_label) / N

# dWo = O^T @ dY
dwo = dY.T @ O

# dO = dY @ Wo
dO = dY @ model.wo.weight

# dV = P^T @ dO        (O = P @ V)
dV = S_softmax.T @ dO

# dP = dO @ V^T        (O = P @ V)
dS_softmax = dO @ V.T

# Softmax 梯度：dS = P ⊙ (dP - D)，其中 D_i = Σ_j P_ij · dP_ij
# 等价于：dS[i] = dP[i] @ (diag(P[i]) - P[i]^T P[i])
for i in range(n):
    I = torch.diag(S_softmax[i]) - torch.outer(S_softmax[i], S_softmax[i])
    dS[i] = dS_softmax[i] @ I

# dQ = dS @ K / √d     (S = QK^T/√d)
dQ = dS @ K / math.sqrt(dim)

# dK = dS^T @ Q / √d
dK = dS.T @ Q / math.sqrt(dim)
```

### 4.2 前向需要保存什么？

反向计算需要：**Q, K, V, O, L**（其中 `L = M + log(l)`）

**不需要**保存完整的 N×N attention matrix P —— 反向时 **recompute** P：

```python
# 用 Forward 保存的 L 重计算 P
P_ij = exp(S_ij - L_i)
# 展开：exp(S_ij - M - log(l)) = exp(S_ij - M) / l
# 这正是归一化后的 softmax
```

### 4.3 D 向量的计算

Softmax 反向中的关键中间量：

$$D_i = \sum_j O_{ij} \cdot dO_{ij}$$

```python
D = torch.sum(O * dO, dim=1, keepdim=True)  # [n, 1]
```

有了 D，softmax 梯度简化为：
$$dS_{ij} = P_{ij} \cdot (dP_{ij} - D_i)$$

### 4.4 FlashAttention-2 分块反向实现

**循环顺序**：KV 外循环，Q 内循环（与前向 V2 相反）。原因：反向时 dK, dV 需要累积，驻留 SRAM。

```python
def flash_attention_backward(Q, K, V, O, dO, L):
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # 预计算 D = rowsum(O ⊙ dO)
    D = torch.sum(O * dO, dim=1, keepdim=True)

    for tk in range(block):  # KV 外循环 —— dK, dV 驻留 SRAM
        k = K[tk*nb:(tk+1)*nb]
        v = V[tk*nb:(tk+1)*nb]

        for tq in range(block):  # Q 内循环
            q = Q[tq*nb:(tq+1)*nb]
            l = L[tq*nb:(tq+1)*nb]
            do = dO[tq*nb:(tq+1)*nb]
            d = D[tq*nb:(tq+1)*nb]

            # Recompute: 重算 attention score + softmax
            s = q @ k.T / math.sqrt(dim)
            p = torch.exp(s - l)  # 用保存的 L 重算 P

            # dV += P^T @ dO
            dv = p.T @ do
            dV[tk*nb:(tk+1)*nb] += dv

            # dP = dO @ V^T
            dp = do @ v.T

            # dS = P ⊙ (dP - D)
            ds = p * (dp - d)

            # dQ += dS @ K / √d
            dq = ds @ k / math.sqrt(dim)
            dQ[tq*nb:(tq+1)*nb] += dq

            # dK += dS^T @ Q / √d
            dk = ds.T @ q / math.sqrt(dim)
            dK[tk*nb:(tk+1)*nb] += dk

    return dQ, dK, dV
```

### 4.5 反向关键设计要点

| 要点 | 说明 |
|------|------|
| **Recompute P** | 不存储 N×N 的 P，反向时用 `exp(S - L)` 重算 |
| **保存 L** | 前向只存 `L = M + log(l)`，一个 [N,1] 向量 |
| **D 预计算** | `D = rowsum(O ⊙ dO)`，避免 softmax 的 diag-outer 计算 |
| **循环顺序** | KV 外 + Q 内：dK/dV 驻留 SRAM，减少 HBM 访问 |
| **无迭代** | 反向不需要 online softmax 的迭代更新，直接用全局 L |

### 4.6 计算量与通信量分析

**前向**：
- 计算：O(N²d) — 与标准 attention 相同
- HBM 访问：O(N²d²/M)，其中 M 是 SRAM 大小 — **远小于标准 attention 的 O(N² + Nd)**

**反向**：
- 计算：~2.5× 前向（需要重算 P，且 dQ/dK/dV 三路梯度）
- HBM 访问：同样 O(N²d²/M)，因为 P 不存 HBM

### 4.7 深入思考

1. **FlashAttention 与 PagedAttention 的联系**：
   - FlashAttention 解决的是 Attention 计算的 IO 效率
   - PagedAttention 解决的是 KV Cache 的内存管理
   - 两者可以结合：PagedAttention 使用 FlashAttention 作为 block 内的计算后端

2. **分布式序列并行**：
   - 长文本 KV 可分布在多 GPU 上
   - 各 GPU 计算局部 block attention，再通过 Online Softmax 聚合
   - 这正是 Ring Attention 等方法的基础

3. **GPU 并行**：
   - 不同 Q block 对应的多个 KV block 可以在 GPU 的不同 SM 上并行计算
   - 最终通过 Online Softmax reduce 合并

---

## 总结

```
标准 Softmax
    │
    ▼
Safe Softmax（减最大值，数值稳定）
    │
    ▼
Online Softmax（增量更新 M, L）
    │
    ▼
Block Online Softmax（分块并行 + 合并）
    │
    ├──────────────────────────────┐
    ▼                              ▼
FlashAttention-V1                FlashAttention-V2
(KV外循环, Q内循环)              (Q外循环, KV内循环)
(Inner Scaled)                   (Outer Scaled)
                                       │
                                       ▼
                              FlashAttention-2 Backward
                              (KV外循环, Q内循环)
                              (Recompute P, 保存 L)
                              (D = rowsum(O⊙dO))
```

**IO 复杂度对比**：

| 方法 | HBM 读写 | 额外存储 |
|------|----------|---------|
| 标准 Attention | O(N²+Nd) | O(N²) — attention matrix |
| FlashAttention | O(N²d²/M) | O(N) — 仅 L 向量 |

FlashAttention 通过**分块 + Online Softmax + Recompute** 三板斧，将 Attention 的 IO 复杂度从 O(N²) 降到 O(N²d²/M)，在长序列上实现数倍加速。
