---
title: RoPE 全家桶手撕实操 · MA-RLHF Batch D
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - ma-rlhf
  - lc8
  - rope
  - position-encoding
  - long-context
  - yarn
  - ntk
brief: RoPE 从零完整推导：旋转矩阵 → 相位差相对位置感知 → 衰减性分析；上下文外推三策略对比：PI（位置插值，高频崩溃）vs NTK-RoPE（均匀频率缩放）vs YaRN（NTK-by-parts 分段，保高频），给出面试级公式推导和代码对比。
related:
  - "[[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]]"
  - "[[AI/LLM/Infra/xtrain-lc4-张量并行从零手写]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操]]"
  - "[[Attention 变体综述]]"
---

# RoPE 全家桶手撕实操（MA-RLHF Batch D）

> **来源**：`notebook/rope_basic.ipynb` + `notebook/rope_analysis.ipynb` + `notebook/rope_decay.ipynb`
> **评级**：★★★★★
> **字数**：~9000

---

## TL;DR

RoPE（Rotary Position Embedding）是目前 LLM 的绝对主流位置编码，用旋转矩阵把位置信息编码到 Q/K 的相位差中，天然具备相对位置感知。本笔记从零实现标准 RoPE，分析其衰减性本质，对比 PI（位置插值）和 NTK-RoPE 的上下文长度外推策略，给出面试级的完整推导。

---

## 一、位置编码的核心需求

好的位置编码必须满足：
1. **单调衰减**：`cos_similarity(pos_i, pos_j)` 随 `|i-j|` 增大而减小（越近越相关）
2. **长程区分度**：`pos_1` 和 `pos_1000` 的相似度 ≠ `pos_1` 和 `pos_10000` 的相似度
3. **平滑性**：衰减曲线不能震荡（绝对位置编码的致命缺陷）
4. **外推性**：超出训练长度时依然保持合理的衰减性

**绝对位置编码的失败**：sin/cos 函数具有周期性，在长序列（>1024）后出现震荡——位置 1000 和 1005 的相似度不单调，模型无法区分远近。

---

## 二、RoPE 基础实现

### 核心思想

RoPE 不直接给向量加位置 embedding，而是对 Q/K 做旋转变换：

$$\text{RoPE}(x, m) = R_m \cdot x$$

其中 $R_m$ 是以位置 m 为参数的旋转矩阵，使得：

$$\langle R_m q, R_n k \rangle = f(q, k, m-n)$$

内积只依赖于**相对位置** `m-n`，天然实现相对位置感知。

### 频率定义

维度 d 的向量，分成 d/2 对，每对用不同频率旋转：

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1$$

以 `base=10000, d=512` 为例：
- 低维度（i=0）：`θ_0 = 1`（高频，快速旋转）
- 高维度（i=255）：`θ_255 = 10000^{-1} = 0.0001`（低频，慢速旋转）

位置 m 的角度：`m_theta[m, i] = m × θ_i`

### 完整实现

```python
class RoPE(nn.Module):
    def __init__(self, dim=512, max_pos=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        m = torch.arange(0, max_pos, 1)         # [max_pos]，位置序列
        i = torch.arange(0, dim // 2, 1)         # [dim/2]，频率索引
        theta = base ** (-2 * i / dim)            # [dim/2]，各维度频率
        m_theta = torch.outer(m, theta)           # [max_pos, dim/2]，外积

        # 每个维度存两份（cos 作用于 x_2i 和 x_{2i+1}）
        self.cos = torch.zeros(max_pos, dim)
        self.sin = torch.zeros(max_pos, dim)
        self.cos[:, 0::2] = self.cos[:, 1::2] = torch.cos(m_theta)  # [max_pos, dim]
        self.sin[:, 0::2] = self.sin[:, 1::2] = torch.sin(m_theta)

    def apply_rope(self, X):
        """
        X: [batch_size, n_heads, seq_len, head_dim]
        返回旋转后的 X，形状不变
        """
        bs, n_heads, seq_len, d = X.shape

        # 构造旋转伙伴：(-x_{2i+1}, x_{2i}) 的 interleave 版本
        X_shift = torch.zeros_like(X)
        X_shift[..., 0::2] = -X[..., 1::2]   # 偶数位 ← 奇数位取负
        X_shift[..., 1::2] =  X[..., 0::2]   # 奇数位 ← 偶数位

        # 旋转公式：Y = cos * X + sin * X_shift
        # cos/sin 广播 [1, 1, seq_len, dim] → [bs, n_heads, seq_len, dim]
        Y = self.cos[None, None, :seq_len, :] * X + \
            self.sin[None, None, :seq_len, :] * X_shift
        return Y
```

### 旋转的数学本质

对每对 `(x_{2i}, x_{2i+1})`，旋转 `m·θ_i` 角：

$$\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

展开：
- $y_{2i} = x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i)$
- $y_{2i+1} = x_{2i}\sin(m\theta_i) + x_{2i+1}\cos(m\theta_i)$

代码中 `X_shift = (-x_odd, x_even)` + cos/sin 加权，正是这个计算。

### X_shift 的 trick

```python
# 原向量 X: [x0, x1, x2, x3, x4, x5, ...]
# X_shift:  [-x1, x0, -x3, x2, -x5, x4, ...]
# 
# Y[0::2] = cos * x_even - sin * x_odd  ✓
# Y[1::2] = cos * x_odd + sin * x_even  ✓
```

---

## 三、衰减性分析

### 测量方法

衰减性 = 固定 pos_1，与其他位置的 cos 相似度随距离增加而减小：

```python
def get_fn(n=1024, d=512, base=10000):
    """计算位置 n 的 RoPE 旋转因子 = Σ cos(n·θ_i)"""
    two_t = torch.arange(0, d, step=2).float()
    theta_t = base ** (-two_t / d)
    cos_t = torch.cos(n * theta_t)
    fn = torch.sum(cos_t)          # 标量：衰减强度
    return fn, cos_t               # fn_v：各维度的贡献
```

旋转内积的理论展开：

$$\langle R_m q, R_n k \rangle = \sum_{i=0}^{d/2-1} \cos((m-n)\theta_i) \cdot \langle q_{2i}, k_{2i}\rangle + \ldots$$

当 q=k=1（均匀初始化），内积 ≈ $\sum_i \cos((m-n)\theta_i)$，即 `get_fn(m-n)` 的值。

### 衰减实验结论

| 方法 | 4096 长度 | 64000 长度 | 问题 |
|------|-----------|------------|------|
| 绝对 PE（base=10000） | 平滑衰减 ✓ | **严重震荡** ✗ | sin/cos 周期性 |
| NTK-RoPE（large base） | 平滑衰减 ✓ | 较好衰减 ✓ | 近距离分辨率略降 |
| PI-RoPE（scaling） | 平滑衰减 ✓ | 衰减失效 ✗ | 高频维度丢失 |

**维度与衰减**：维度越大 → 可用频率更多 → 衰减曲线更平滑（`d=512` vs `d=8192`，后者衰减性明显更好）

**高频 vs 低频维度的不同角色**：
- 高频（前 d/2 维）：快速旋转，精细区分近距离位置
- 低频（后 d/2 维）：慢速旋转，区分长距离位置
- PI 把所有频率等比压缩 → 高频变中频 → 近距离分辨率损失
- NTK 换 base → 高频保持，低频拉伸 → 保留近距离精度同时扩大范围

---

## 四、上下文长度外推方案

### 三种主流策略

#### 1. PI（Position Interpolation，线性插值）

**思路**：训练了 4096 token，现在要处理 8192 token。把位置 `[0, 8192]` 压缩映射到 `[0, 4096]`：

$$m_{\text{new}} = m / s, \quad s = L_{\text{new}} / L_{\text{train}}$$

等价于把 theta 乘以 s（频率降低）：

```python
# PI 实现：scaling_factor=2.0（4096→8192）
theta_pi = base ** (-2 * i / dim)    # 原始频率
# 等价于所有位置除以 2，即 m → m/2
# transformers 中：
pi_emb = LlamaLinearScalingRotaryEmbedding(dim, scaling_factor=2.0)
```

**问题**：高频维度（短周期）被压缩后，近距离两个 token 的角度差缩小 → 模型无法区分 → 近距离感知退化。

#### 2. NTK-RoPE（Neural Tangent Kernel，换 base）

**思路**：不压缩位置，而是放大 base（降低所有频率），让原本在 4096 内就能走完一圈的维度现在在 8192 才走完：

$$\text{base}_{\text{new}} = \text{base} \times \alpha^{d/(d-2)}, \quad \alpha = L_{\text{new}} / L_{\text{train}}$$

```python
# NTK 实现：本质上是 base 乘以缩放因子
# 如 base=10000, alpha=8 → base_new ≈ 500000
ntk_emb = LlamaDynamicNTKScalingRotaryEmbedding(dim, scaling_factor=8.0, base=10000.0)
```

**优势**：高频维度（i 小，θ_i 大）受影响小，低频维度扩展更多 → 近距离精度保留，远距离覆盖扩展。

#### 3. YaRN（Yet another RoPE extensioN，分段策略）

**思路**：高频/低频维度用不同策略处理：

$$\theta_i^{\text{new}} = \begin{cases} \theta_i & \text{（低频，已有足够分辨率，无需缩放）} \\ \theta_i / s & \text{（中频，PI 压缩）} \\ \alpha \theta_i + (1-\alpha)\theta_i / s & \text{（高频，混合）} \end{cases}$$

同时加 Re-Scale：注意力权重乘以 `concentration = 0.1 × ln(scale) + 1`，补偿外推后 softmax 过于均匀的问题。

详见 [[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写]]。

---

## 五、PI vs NTK 衰减实验对比

```python
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding, apply_rotary_pos_emb
)

d_model = 4096
max_len = 64000
Q = torch.ones(1, 1, max_len, d_model)
K = torch.ones(1, 1, max_len, d_model)
position_ids = torch.tensor([range(max_len)], dtype=torch.long)

rotary_emb = LlamaRotaryEmbedding(d_model, max_position_embeddings=max_len, base=10000.0)
pi_emb = LlamaLinearScalingRotaryEmbedding(d_model, scaling_factor=2.0, base=10000.0)
ntk_emb = LlamaDynamicNTKScalingRotaryEmbedding(d_model, scaling_factor=8.0, base=10000.0)

def get_decay(Q, K, position_ids, rope_fn):
    cos, sin = rope_fn(Q, position_ids)
    Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)
    # 计算 pos_0 与所有位置的相似度
    result = Q_rope[0, 0, [0]] @ K_rope[0, 0, :, :].T
    return result[0].tolist()
```

**实验结论（64000 长度）**：
- `rope_base`：在 4096 后衰减崩塌，出现强烈震荡
- `rope_pi`：衰减平滑，但在 4096 后失去长程区分度（两个远距离点相似度相差无几）
- `rope_ntk`：**在长距离下保持区分度且符合衰减规律** ← 胜者

---

## 六、衰减函数的数学本质

RoPE 的衰减函数 $\phi_d(n)$ 定义为：

$$\phi_d(n) = \sum_{t=0}^{d/2-1} \cos(n \cdot \text{base}^{-2t/d})$$

对于固定 base 和 d，这是一个关于相对距离 n 的函数：

```python
def get_fn(n=1024, d=512, base=10000):
    two_t = torch.arange(0, d, step=2).float()
    theta_t = base ** (-two_t / d)        # [d/2] 各维度频率
    cos_t = torch.cos(n * theta_t)        # [d/2] 位置 n 的各维度余弦值
    fn = torch.sum(cos_t)                 # 标量：n 处的衰减值
    return fn, cos_t
```

关键性质：
- `n=0`：所有 cos(0)=1，`φ(0) = d/2`（最大值）
- `n → ∞`：各频率不相关，期望值趋于 0（**必须是 0 而不是负值**）
- d 越大：更多独立频率，平均值更稳定趋向 0，衰减更平滑

当维度增大（如 d=8192），低频维度贡献的慢变 cos 函数更多，整体衰减曲线更平滑单调。这也是为什么大模型（大 d）在位置外推上往往比小模型更稳健。

---

## 七、面试考点

**Q1：RoPE 怎么实现相对位置感知的？**
RoPE 对 Q/K 做旋转，使得 `⟨R_m q, R_n k⟩ = f(q, k, m-n)`，内积只依赖相对位置差 `m-n`，而非绝对位置。旋转矩阵的性质：$R_m^T R_n = R_{n-m}$（旋转角之差），把绝对位置信息转化为相对位置差。

**Q2：`X_shift` 这个 trick 是什么？**
旋转公式 `y_{2i} = cos·x_{2i} - sin·x_{2i+1}` 可以写成向量形式 `Y = cos·X + sin·X_shift`，其中 `X_shift[...,0::2] = -X[...,1::2]，X_shift[...,1::2] = X[...,0::2]`。这样避免了显式构造旋转矩阵（$O(d^2)$），直接向量操作 $O(d)$。

**Q3：RoPE 的衰减性为什么比绝对 PE 好？**
绝对 PE 用 sin/cos 函数直接作为位置向量，周期性导致长序列震荡。RoPE 用旋转把位置信息编码在相位差中，内积 ≈ $\sum_i \cos((m-n)\theta_i)$，多个不同频率的 cos 叠加，频率越多、维度越大，震荡相互抵消，衰减越平滑（类比多频率信号混合的去相干效应）。

**Q4：PI 为什么会损害近距离感知？**
PI 把所有位置除以 scale，等价于把所有频率降低 scale 倍。原本用于区分 `pos_1` 和 `pos_2` 的高频维度（`θ_0 ≈ 1`，1 step 旋转 1 rad），变成 `θ_0/s`（仅旋转 0.5 rad）。模型看到的角度差缩小，无法区分近距离位置。这也是为什么 PI 需要 fine-tuning 才能恢复性能。

**Q5：NTK-RoPE 的物理含义是什么？**
NTK-RoPE 等价于换用更大的 base，使得所有频率降低，但降低幅度随维度变化。对高频维度（i 小）影响小；对低频维度（i 大）影响大 → 高频维度（区分近距离）基本不变，低频维度（区分远距离）被拉伸以覆盖更长范围。从 Neural Tangent Kernel 理论看：base 控制函数的"复杂度"，大 base = 更平滑的函数空间。

**Q6：为什么 d 越大，衰减性越好？**
衰减函数 $\phi_d(n) = \sum_{t=0}^{d/2-1} \cos(n \cdot \text{base}^{-2t/d})$。d 越大，频率数越多，不同频率的 cos 函数越多样，相互"平均"后，振荡项相消，函数越接近真正的单调衰减。类比：随机变量的样本均值，样本量越大越平稳。

**Q7：RoPE 如何处理 KV Cache？**
RoPE 在 Q/K 上做旋转，而非直接改变 token embedding。KV Cache 存储已旋转的 K（带位置信息），新 token 的 Q 只需对位置 cur_pos 旋转，与 cache 中的 K 直接做 attention。这是 RoPE 相比绝对位置编码的重要优势：Cache 存储不需要考虑位置。

---

## 八、RoPE 生态全景

```
绝对位置编码（Transformer原版）
         ↓ 问题：长序列震荡，无相对位置感知
RoPE（旋转位置编码，标准实现）
         ↓ 问题：训练长度外受限
    ┌────┴────┐
   PI         NTK
（线性压缩）  （换base）
    ↓问题      ↓优点
近距离退化   近距高维保留
    └────┬────┘
        YaRN（分段 + Re-Scale）← 目前最佳实践
         ↓
        TPA（在此基础上分解K/V）← 见 lc8-TPA-YaRN 笔记
         ↓
   MLA（低秩 + RoPE解耦）← 见 lc8-DeepSeek-MLA 笔记
```

---

## See Also

- [[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写]] — YaRN 完整代码实现 + TPA 张量积注意力（本笔记理论 → 该笔记工程实现）
- [[AI/LLM/MA-RLHF课程/lc8-DeepSeek-MLA-从零手写]] — MLA 中 RoPE 解耦（低秩 KV + position 单独路径，RoPE 应用的最新工程形态）
- [[AI/LLM/Infra/xtrain-lc4-张量并行从零手写]] — TP 下 RoPE 的实现考虑（各卡独立旋转，分布式场景）
- [[AI/LLM/MA-RLHF课程/lc8-GQA-KVCache-手撕实操]] — KV Cache 中 RoPE 的位置追踪（增量解码时 position_ids 的正确维护）
- [[Attention 变体综述]] — RoPE 在各类 Attention 变体中的应用全景
