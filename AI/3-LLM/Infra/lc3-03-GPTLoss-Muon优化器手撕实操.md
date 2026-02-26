---
title: "GPT Loss + Muon 优化器手撕实操 · MA-RLHF Batch D"
type: code-practice
date: 2026-02-26
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, ma-rlhf, lc8, gpt, loss, muon, optimizer, newton-schulz]
brief: "GPT Next-Token-Prediction Loss 正确格式（logits.transpose(1,2)→[B,V,L]）+ Muon 优化器从零实现：Newton-Schulz 5次矩阵乘法近似正交化，解决 Adam 在大 batch 语言模型训练中的动量方向偏差问题，GLM-5 已在生产中使用 Muon Split 变体。"
related:
  - "[[AI/3-LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操]]"
  - "[[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]]"
  - "[[AI/3-LLM/Architecture/GPT2-手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]]"
---

# GPT Loss + Muon 优化器手撕实操（MA-RLHF Batch D）

> **来源**：`notebook/GPT-loss.ipynb` + `notebook/muon/Muon.ipynb`
> **评级**：★★★★★
> **字数**：~8000

---

## TL;DR

**GPT Loss**：Decoder-Only 语言模型的训练目标是 Next Token Prediction，CrossEntropy Loss 的正确输入格式是 `logits.transpose(1,2)` → `[B, V, L]`，label 是 `[B, L]`。损失计算前向轮转一位：输入 `x[t]` → 预测 `y[t+1]`。

**Muon 优化器**：矩阵优化器（Matrix Optimizer），核心是用矩阵符号函数 `msign(M)` 替代 Adam 的 element-wise sign，msign 通过 Newton-Schulz 迭代高效近似 SVD 正交化，实现梯度的"最优正交近似"。相当于 2-范数约束下的梯度下降，比 Adam 少一组缓存变量，显存更低。

---

## 一、GPT Loss 全流程手撕

### 完整前向链

```python
import torch
import math

batch_size = 1
length = 4
vocab_size = 32000
embd_dim = 512

# 输入
x = torch.randn(batch_size, length, embd_dim)  # [1, 4, 512]
y = torch.randint(0, vocab_size, (batch_size, length), dtype=torch.long)  # [1, 4]
```

#### Step 1：Attention

```python
q = torch.randn(512, 512)
k = torch.randn(512, 512)
v = torch.randn(512, 512)
o = torch.randn(512, 512)

# Causal Mask（下三角），防止 token 看到未来
mask = torch.tril(torch.ones(1, 4, 4))  # [1, 4, 4]
# [[1, 0, 0, 0]
#  [1, 1, 0, 0]
#  [1, 1, 1, 0]
#  [1, 1, 1, 1]]

Q, K, V = x @ q, x @ k, x @ v            # [1, 4, 512]
scores = Q @ K.transpose(1, 2) / math.sqrt(512.0)  # [1, 4, 4]
scores = scores.masked_fill(mask == 0, float('-inf'))  # future=−∞
weight = torch.softmax(scores, dim=2)     # [1, 4, 4]
attn = weight @ V                         # [1, 4, 512]
attn = attn @ o                           # [1, 4, 512]（output projection）
```

#### Step 2：MLP

```python
mlp_up = torch.randn(512, 1024)
mlp_down = torch.randn(1024, 512)
mlp = attn @ mlp_up @ mlp_down            # [1, 4, 512]
```

#### Step 3：Output + Loss

```python
lm_head = torch.randn(512, 32000)
logits = mlp @ lm_head                    # [1, 4, 32000]

# ⚠️ 关键：CrossEntropyLoss 期望的格式
# input:  [Batch, Class, Length] = [1, 32000, 4]   ← 必须 transpose!
# target: [Batch, Length]        = [1, 4]
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits.transpose(1, 2), y)  # logits: [1, 4, 32000] → [1, 32000, 4]
```

**为什么 transpose？** PyTorch CrossEntropyLoss 对 3D 输入的规范：`input = [N, C, d1]`，C 必须是类别维度（在第 1 位）。语言模型输出是 `[B, L, V]`（vocab 在最后），所以要 `transpose(1, 2)` 变成 `[B, V, L]`。

### 两种写法等价性验证

```python
# 写法 1：不 reshape，直接 transpose
loss1 = loss_fn(logits.transpose(1, 2), y)

# 写法 2：展平到 2D
logits_flat = logits.permute(0, 2, 1).reshape(batch_size * length, vocab_size)
y_flat = y.view(batch_size * length)
loss2 = loss_fn(logits_flat, y_flat)

assert torch.allclose(loss1, loss2)  # ✓ 数学等价
```

### Next Token Prediction 的本质

```python
# 每个位置预测下一个 token
# 位置 0（我）    → 预测位置 1（很）
# 位置 1（我很）  → 预测位置 2（开）
# 位置 2（我很开）→ 预测位置 3（心）
# 位置 3（我很开心）→ 预测下一个（呀）← 这才是推理时用的

# 所以推理时取最后一个位置
pred = torch.argmax(logits, dim=2)   # [1, 4]，4 个位置的预测
next_token = pred[0, -1]             # 只取最后一个 = next token
```

**训练 vs 推理的差异**：训练时 teacher forcing（标签是真实 token 序列），同时优化所有位置；推理时只用最后一个位置的预测，自回归展开。

---

## 二、Muon 优化器深度解析

### 动机：Adam 的局限

Adam 是 element-wise 操作：逐元素维护一阶矩（动量）+ 二阶矩（方差），更新时按元素独立处理。对于矩阵参数（`Wq, Wk, Wv, MLP`），完全忽略了不同行/列之间的相关性。

Muon 的思路：梯度矩阵 G 有 total gradient 信息（行列关系），应该在矩阵层次做优化。

### Muon 算法核心

$$\boldsymbol{M}_t = \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \quad \text{（动量累积）}$$

$$\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}] \quad \text{（参数更新）}$$

区别于 Adam，更新方向不是 `sign(g)` 或 `g/sqrt(v)`，而是 `msign(M)`。

### msign：矩阵符号函数

**标量 sign**：$\text{sign}(x) = x(x^2)^{-1/2}$，将 $x$ 映射到 $\{-1, +1\}$。

**矩阵 msign**（最优正交近似）：

$$\text{msign}(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M} = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$$

**几何意义**：msign(M) 是 M 的最优正交近似，即解优化问题：

$$\text{msign}(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}}\Vert \boldsymbol{M} - \boldsymbol{O}\Vert_F^2$$

证明：令 $M = U\Sigma V^T$（SVD），则 msign(M) = UV^T。

#### SVD 证明

$$\text{msign}(\boldsymbol{M}) = (M M^\top)^{-1/2} M = (U\Sigma V^\top V \Sigma U^\top)^{-1/2} U\Sigma V^\top$$

$$= (U\Sigma^2 U^\top)^{-1/2} U\Sigma V^\top = U\Sigma^{-1} U^\top \cdot U\Sigma V^\top = UV^\top$$

**msign 丢弃了奇异值 Σ（缩放信息），只保留 U, V（旋转信息）**。等价于把梯度矩阵投影到正交矩阵流形上。

```python
def matrix_inv_sqrt_svd(M, eps=1e-10):
    """计算 M^{-1/2}，用于 msign 的基础计算（教学用）"""
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    inv_sqrt_S = torch.diag_embed(1.0 / torch.sqrt(S + eps))
    return U @ inv_sqrt_S @ V

# msign(M) = M × (M^T M)^{-1/2}，等价于 U @ V
msign_M = M @ matrix_inv_sqrt_svd(M.t() @ M)
# 也等价于：
U, S, V = torch.linalg.svd(M, full_matrices=False)
msign_M_fast = U @ V  # ← 直接 SVD 更快
```

**特殊情况**：
- 当 $M \in \mathbb{R}^{n \times 1}$（列向量）：$\text{msign}(\boldsymbol{m}) = \boldsymbol{m}/\Vert\boldsymbol{m}\Vert_2$（单位化）
- 当 $M \in \mathbb{R}^{1 \times 1}$（标量）：退化为 sign 函数

---

## 三、Newton-Schulz 迭代：工程化近似

**问题**：大矩阵的 SVD 复杂度 $O(nm^2)$，DeepSeek-V3（7168×18432）运行太慢。

**解法**：用多项式迭代近似 msign，避免 SVD。

### 泰勒展开基础

标量 $t^{-1/2}$ 在 $t=1$ 处的泰勒展开：

$$t^{-1/2} \approx 1 - \frac{1}{2}(t-1) + \frac{3}{8}(t-1)^2 - \frac{5}{16}(t-1)^3 + \cdots$$

保留到 2 阶（3 项）：$t^{-1/2} \approx (15 - 10t + 3t^2)/8$

矩阵推广（令 $t = M^\top M$）：

$$\text{msign}(\boldsymbol{M}) \approx \frac{15}{8}\boldsymbol{M} - \frac{5}{4}\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M}) + \frac{3}{8}\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^2$$

### 迭代形式（Newton-Schulz）

$$\boldsymbol{X}_{t+1} = \frac{15}{8}\boldsymbol{X}_t - \frac{5}{4}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + \frac{3}{8}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2$$

初始化：$X_0 = M/\|M\|_F$（归一化）

```python
M_old = M
for i in range(5):
    A = M_old.t() @ M_old           # [n, n] 或 [m, m]（取较小维度）
    M_new = (15/8) * M_old \
          - (5/4)  * M_old @ A \
          + (3/8)  * M_old @ torch.matrix_power(A, 2)
    print(f"Step {i}: error = {(M_new - msign_M).norm():.6f}")
    M_old = M_new
```

### Muon 魔法系数（3.4445, -4.7750, 2.0315）

官方 Muon 不用标准泰勒系数，而是通过 SGD 优化求出针对大矩阵的最优系数：

$$\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2$$

将问题转化为：$g(x) = ax + bx^3 + cx^5$ 将奇异值从任意值映射到 1。

通过采样 1000 个随机矩阵的 SVD，用 SGD 优化 $(k, x_1, x_2)$：

```python
# 结果（1024×1024 矩阵，5次迭代）：
# k=1.724, x1=0.935, x2=1.235 → a=3.297, b=-4.136, c=1.724
```

**为什么 Muon 选 3.4445 而不是 15/8=1.875？**：标准泰勒展开只在奇异值接近 1 时精确，大矩阵的奇异值分布不均匀，需要专门拟合的系数。

### Muon vs Adam 对比

| 维度 | Adam | Muon |
|------|------|------|
| 操作粒度 | Element-wise | 矩阵级 |
| 缓存变量 | 2个（m, v） | 1个（M，动量） |
| 更新方向 | `g/sqrt(v+ε)` | `msign(M)` = 最优正交近似 |
| 几何含义 | 梯度归一化 | 投影到正交矩阵流形 |
| 计算开销 | 低 | 5次迭代矩阵乘法 |
| 约束 | 无 | 2-范数约束下的梯度下降 |
| 实际效果 | 通用基线 | 矩阵参数更优（GLM-5用了Muon Split） |

---

## 四、GLM-5 的 Muon Split

GLM-5 技术报告中提到 **Muon Split** 解决了 MLA + Muon 的不兼容问题：

**问题**：MLA 中 W_UQ（up projection）的参数是 `[d_c, n_heads × head_dim]`，跨头混合；而 Muon 对矩阵整体做 msign，会把不同头的梯度混合归一化，破坏头的独立性。

**解法**：按 head 视角重排矩阵，分成 `n_heads` 个子矩阵，对每个子矩阵独立做 Newton-Schulz 正交化：

```python
# 伪代码：Muon Split for W_UQ
W_UQ = ...  # [d_c, n_heads * head_dim]
W_UQ_by_head = W_UQ.view(d_c, n_heads, head_dim)  # 按头重排
for h in range(n_heads):
    W_h = W_UQ_by_head[:, h, :]  # [d_c, head_dim]
    W_h_updated = muon_update(W_h)  # 每头独立 Newton-Schulz
```

这是实用工程创新，说明 Muon 在大规模训练中需要对架构做适配。

---

## 五、面试考点

**Q1：GPT 训练时 CrossEntropyLoss 的 label 如何构造？**
`inputs = tokens[:-1]`，`labels = tokens[1:]`。序列向右 shift 1 位，让每个位置预测下一个 token。PyTorch 中传入 `logits[B, V, L]` 和 `labels[B, L]`。

**Q2：msign 和 sign 的关系？**
sign 是 msign 在标量或列向量上的特例。sign(x) = x/|x|，msign(v) = v/||v||（列向量归一化），msign(M) = UV^T（矩阵正交化）。

**Q3：为什么 msign 比 element-wise sign 更好？**
Element-wise sign 只考虑每个元素的符号，忽略矩阵行列之间的相关性。msign 保留了矩阵的"旋转结构"（U, V），丢弃了"缩放杂质"（Σ），相当于在正交矩阵流形上做梯度下降，更能体现矩阵参数的内在几何结构。

**Q4：Newton-Schulz 为什么比直接 SVD 快？**
SVD 复杂度 $O(\min(m,n)^2 \cdot \max(m,n))$，对 7168×18432 大矩阵极慢。Newton-Schulz 只需 5 次矩阵乘法，每次 $O(mn^2)$ 或 $O(m^2n)$，可以 GPU 并行，总体快很多。

**Q5：Muon 显存比 Adam 少多少？**
Adam 维护 2 组变量（m 和 v），Muon 只维护 1 组（M，动量）。等于参数量的 50%显存节省（在优化器状态上），对大模型训练很显著。

**Q6：什么情况下用 Muon，什么情况用 Adam？**
Muon 适合矩阵参数（线性层 Wq/Wk/Wv/FFN）；embedding 层和 lm_head 通常仍用 Adam（embedding 是离散查表，没有矩阵乘法结构）。GLM-5/Kimi 等工业模型实践：混合使用，Muon 处理中间层矩阵，Adam 处理 embedding。

---

## 六、与其他笔记的关联

```
GPT Loss 前向链：
Attention（见 lc8-DeepSeek-MLA）→ MLP → LM Head → CrossEntropy Loss

KV Cache 推理：
Prefill → KV Cache 存储 → Decode（见 lc8-GQA-KVCache）

Muon 优化器：
梯度 G → 动量 M → msign(M)=UV^T → 参数更新
                          ↑
                    Newton-Schulz 迭代（5次矩阵乘法）

GLM-5 Muon Split → 处理 MLA + Muon 不兼容（见 GLM-5 Vault 笔记）
Adam → ZeRO 分片（见 xtrain-lc3-ZeRO）
```

---

## See Also

- [[AI/3-LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操]] — Batch D 同批次：KV Cache 完整推理实现（GQA + 增量解码）
- [[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — ZeRO 分片与 Adam 工程实现（Muon 的对比基准）
- [[AI/3-LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]] — MLA 架构（GLM-5 Muon Split 为解决 MLA+Muon 不兼容而设计）
- [[AI/3-LLM/Architecture/GPT2-手撕实操]] — GPT-2 完整训练流程（GPT Loss 的应用场景）
- [[AI/3-LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]] — lc5 DeepSeek V3 课程地图（Muon 在架构创新中的位置）
