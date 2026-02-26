# RoPE 全家桶手撕实操

> 来源：MA-RLHF notebook — rope_basic.ipynb, rope_analysis.ipynb, rope_decay.ipynb, YaRN-Pytorch.ipynb
> 参考论文：RoFormer (Su et al. 2021), YaRN (Peng et al. 2023), GPT-OSS (OpenAI)

---

## 1. RoPE 基础：旋转矩阵与相对位置编码

### 1.1 核心思想

**目标**：构造一种位置编码，使得两个 token 的注意力分数**只依赖它们的相对位置差**，而非绝对位置。

**旋转矩阵定义**：对位置 $m$ 的向量 $x$，RoPE 施加旋转：

$$
\text{RoPE}(x, m) = R_m \cdot x
$$

其中 $R_m$ 是分块对角旋转矩阵，每个 2×2 块为：

$$
R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

频率参数：$\theta_i = \text{base}^{-2i/d}$，其中 $d$ 是 head dimension，$i \in [0, d/2)$。

### 1.2 为什么能编码相对位置？

```
<q_m, k_n> = (R_m · q)^T · (R_n · k) = q^T · R_m^T · R_n · k = q^T · R_{n-m} · k
```

由于旋转矩阵的性质 $R_m^T \cdot R_n = R_{n-m}$，**点积只依赖位置差 $n-m$**，自然编码了相对位置。

### 1.3 频率直觉

- **低维（小 i）**：$\theta_i$ 大 → 高频振荡 → 区分近距离 token
- **高维（大 i）**：$\theta_i$ 小 → 低频振荡 → 编码远距离关系
- 类似傅里叶级数的多尺度位置编码

---

## 2. 完整实现

### 2.1 标准 RoPE 实现（旋转形式）

```python
class RoPE(nn.Module):
    def __init__(self, dim=512, max_pos=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_pos = max_pos

        m = torch.arange(0, self.max_pos, 1)           # 位置索引 [0, 1, ..., max_pos-1]
        i = torch.arange(0, self.dim // 2, 1)           # 维度索引 [0, 1, ..., d/2-1]
        theta = self.base ** (-2 * i / self.dim)         # 频率: base^(-2i/d)
        m_theta = torch.outer(m, theta)                  # [max_pos, d/2]

        # 扩展到 full dim: [cos(θ), cos(θ), cos(θ), cos(θ), ...] 交错排列
        self.cos = self.sin = torch.zeros(self.max_pos, self.dim)
        self.cos[:, 0::2] = self.cos[:, 1::2] = torch.cos(m_theta)
        self.sin[:, 0::2] = self.sin[:, 1::2] = torch.sin(m_theta)

    def apply_rope(self, X):
        """
        Input: X[bs, n_heads, seq_len, head_dim]
        """
        bs, n_heads, seq_len, d = X.shape

        # 构造 [-x2, x1, -x4, x3, ...] 的交错排列
        X_shift = torch.zeros_like(X)
        X_shift[..., 0::2] = -X[..., 1::2]
        X_shift[..., 1::2] = X[..., 0::2]

        # RoPE: x * cos(mθ) + x_shift * sin(mθ)
        Y = self.cos[None, None, :seq_len, :] * X + \
            self.sin[None, None, :seq_len, :] * X_shift
        return Y
```

### 2.2 复数形式等价写法（chunk 版本，OpenAI GPT-OSS 风格）

```python
def _apply_rotary_emb(
    x: torch.Tensor,     # [seq_len, n_heads, head_dim]
    cos: torch.Tensor,   # [max_pos, head_dim/2]
    sin: torch.Tensor,
) -> torch.Tensor:
    seq_len = x.size(0)
    cos = cos[:seq_len, :].unsqueeze(-2).to(x.dtype)  # [seq_len, 1, head_dim/2]
    sin = sin[:seq_len, :].unsqueeze(-2).to(x.dtype)

    # 把 head_dim 分成前后两半
    x1, x2 = torch.chunk(x, 2, dim=-1)   # 各 [seq_len, n_heads, head_dim/2]

    # 旋转：等价于复数乘法 (x1 + ix2) * (cos + i*sin)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)
```

**两种写法的关系**：
- 交错形式（rotary）：`[x0, x1, x2, x3, ...]` → 相邻两个配对旋转
- Chunk 形式（complex）：`[x_前半, x_后半]` → 前后两半配对旋转
- 数学上等价，实际实现取决于框架偏好

### 2.3 完整 RoPE 模块（GPT-OSS 版本）

```python
class RoPE(torch.nn.Module):
    def __init__(self, head_dim, base, num_tokens):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.cos, self.sin = self._compute_cos_sin(num_tokens)

    def _compute_concentration_and_inv_freq(self):
        freq = self.base ** (torch.arange(0, self.head_dim, 2) / self.head_dim)
        inv_freq = 1.0 / freq
        return inv_freq

    def _compute_cos_sin(self, num_tokens):
        inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [num_tokens, head_dim/2]
        return freqs.cos(), freqs.sin()

    def forward(self, query, key):
        num_tokens = query.shape[0]
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, self.cos, self.sin)
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, self.cos, self.sin)
        return query, key
```

---

## 3. RoPE 衰减分析

### 3.1 位置编码衰减性

好的位置编码应满足：
- **距离越远，相似度越低**（衰减性）
- 衰减曲线**平滑**，无剧烈震荡
- **长距离可区分**：pos 1 vs 10000 和 pos 1 vs 1000000 有明显差异

### 3.2 绝对位置编码 vs RoPE

**绝对位置编码问题**：
- 1024 长度内：前 200 位平滑下降，之后出现震荡
- 拉到 64000：衰减规律完全消失
- 原因：sin/cos 的周期性导致远处位置"绕回来"

**RoPE 优势**：
- 4096 长度内衰减曲线理想
- 不同维度提供多尺度编码
- 但超过训练长度（如 64000）也会衰减规律崩塌

### 3.3 Head Dimension 对衰减的影响

实验结论：**维度越大，衰减性越好**

| 维度 d | 1024 长度衰减表现 |
|--------|-------------------|
| 128 | 快速震荡，区分度差 |
| 512 | 中等衰减，200 位后有震荡 |
| 2048 | 良好衰减 |
| 4096 | 接近理想曲线 |
| 8192 | 平滑单调下降 |

直觉：更多维度 → 更多不同频率的 sin/cos → 叠加后相消更充分 → 更平滑的衰减。

### 3.4 高频 vs 低频分析

将位置编码向量分成前半（高频）和后半（低频）分别做衰减实验：

- **高频维度**：短距离区分能力强，但长距离快速震荡
- **低频维度**：长距离保持平滑衰减，但短距离几乎没有区分度
- 完整 RoPE 是两者叠加，兼顾近距离和远距离

---

## 4. NTK-aware RoPE：频率域插值

### 4.1 线性位置插值（PI）的问题

PI 的做法：把位置 $m$ 缩放为 $m/s$（$s$ 为扩展因子）

```
θ_PI(m, i) = sin(m/s × base^(-2i/d))
```

**问题**：所有维度统一缩放 → 高频维度被过度压缩 → 近距离 token 区分度丢失

### 4.2 NTK-aware RoPE

核心思想：**在频率域插值**，通过调整 base 而非位置 $m$：

$$
\text{base}_{\text{NTK}} = \text{base} \times s^{d/(d-2)}
$$

其中 $s$ 为上下文扩展倍数。

```
θ_NTK(m, i) = sin(m × base_NTK^(-2i/d))
```

**为什么比线性插值好？**
- NTK 保留高频维度的分辨率（低 i 的 θ 变化小），主要拉伸低频维度
- 效果：高频近距离保精度，低频远距离获得更大编码范围
- 数学上等价于在频率域做**非均匀插值**

实验验证：64000 长度下，NTK 衰减曲线保持平滑下降，而 RoPE 和 PI 的衰减规律已经崩塌。

### 4.3 NTK 的可视化理解

以 sin(m × θ_i) 在单位圆上的表示：
- 原始 RoPE：32 个 token 旋转角密集
- NTK (s=10)：相同 32 token 旋转角稀疏化（低频维度拉得更开）
- NTK 扩展到 64 token：仍保持合理间距，不会出现 "out of bound" 问题

---

## 5. YaRN：NTK + 注意力温度缩放

### 5.1 YaRN 三大技术

YaRN = **NTK by parts** + **Dynamic Scaling** + **Re-Scale (注意力温度)**

#### 技术 1：NTK by Parts（分段插值）

不同频率维度采用不同策略：

```
λ_i = L / (2π × base^(2i/d))    # 维度 i 的"波长"（旋转圈数）
```

- `λ_i > β`（高频，波长 > 阈值 β=32）：保持原样，不插值（**外推**）
- `λ_i < α`（低频，波长 < 阈值 α=1）：完全线性插值（**内插**）
- `α < λ_i < β`：线性混合（**ramp 过渡**）

```python
# ramp 过渡函数
ramp = (torch.arange(d_half) - low) / (high - low)
mask = 1 - ramp.clamp(0, 1)

inv_freq = interpolation * (1 - mask) + extrapolation * mask
```

#### 技术 2：Dynamic Scaling

```python
scaling_factor = max(1, cur_context_length / initial_context_length)
```

- 上下文未超训练长度时，`s=1`，不做任何变换
- 超过训练长度时，`s` 随上下文长度线性增长
- 推理时根据实际 `seq_len` 动态调整，无需预设固定扩展倍数

#### 技术 3：Re-Scale（注意力温度缩放）

**核心公式**：

$$
\sqrt{1/t} = 0.1 \ln(s) + 1.0
$$

$$
t = \frac{1}{(0.1 \ln(s) + 1.0)^2}
$$

实现中将 cos/sin 乘以 concentration 系数：

```python
concentration = 0.1 * math.log(scaling_factor) + 1.0
cos = freqs.cos() * concentration
sin = freqs.sin() * concentration
```

**为什么需要温度缩放？**
- NTK 插值后，旋转角度被拉伸，相邻 token 的注意力分数变小
- 温度缩放补偿这种衰减，保持注意力分布的锐度
- s 越大，concentration 越大，补偿越多

### 5.2 YaRN 完整实现

```python
class YaRN(torch.nn.Module):
    def __init__(self, head_dim, base, dtype,
                 initial_context_length=4096,
                 scaling_factor=1.0,
                 ntk_alpha=1.0, ntk_beta=32.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self):
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim
        )

        # Dynamic Scaling
        self.scaling_factor = max(1, self.cur_context_length / self.initial_context_length)

        if self.scaling_factor > 1.0:
            # === Re-Scale (温度补偿) ===
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.head_dim / 2

            # === NTK by Parts (分段边界) ===
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)

            interpolation = 1.0 / (self.scaling_factor * freq)  # 线性内插
            extrapolation = 1.0 / freq                          # 原样外推

            # Ramp 过渡
            ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration   # Re-Scale 应用于 cos/sin
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(self, query, key):
        num_tokens = query.shape[0]
        self.cur_context_length = num_tokens
        cos, sin = self._compute_cos_sin(num_tokens)

        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        return query, key
```

---

## 6. 上下文扩展效果对比

| 方法 | 修改内容 | 训练长度内表现 | 外推能力 | 是否需要微调 |
|------|---------|---------------|---------|-------------|
| **原始 RoPE** | 无 | ✅ 最优 | ❌ 4K 外崩塌 | — |
| **线性插值 (PI)** | 位置 m → m/s | ⚠️ 近距离区分度降低 | ⚠️ 中等，高频损失 | 轻微微调 |
| **NTK-aware** | base → base × s^(d/(d-2)) | ✅ 高频保持 | ✅ 64K 平滑衰减 | 少量微调 |
| **YaRN** | NTK by parts + 温度 + dynamic | ✅ 最优平衡 | ✅✅ 128K+ 稳定 | 约 400 步微调 |

**关键数据**（64000 长度衰减实验）：
- 原始 RoPE：衰减规律完全消失，震荡无规律
- PI (s=2)：衰减太快，远距离全部归零
- NTK：在长距离下保持区分度且符合衰减规律

---

## 面试考点

### Q1：RoPE 为什么外推性比绝对位置编码好？

绝对位置编码的问题：sin/cos 的周期性导致远处位置"绕回来"，产生周期性碰撞。RoPE 的核心优势是**编码相对位置**而非绝对位置——注意力分数 $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$ 只依赖位置差，天然避免了周期碰撞。同时 RoPE 的多尺度频率设计（$\theta_i = \text{base}^{-2i/d}$）提供了从高频到低频的覆盖，使得不同距离都有区分度。但 RoPE 也不能无限外推——超过训练见过的位置差后，高频维度仍会出现 OOD 问题。

### Q2：NTK-aware 和线性插值的根本区别是什么？

线性插值在**位置域**均匀缩放（所有维度的 θ 等比缩小），导致高频维度丢失近距离分辨率。NTK 在**频率域**非均匀缩放（调 base 而非 m），低维高频几乎不变，高维低频拉伸更多。直觉：NTK 保住了"放大镜"（高频近距离），同时配备了"望远镜"（低频远距离扩展）。

### Q3：YaRN 的温度参数 (Re-Scale) 起什么作用？

NTK 插值后，旋转角被拉伸，导致相邻 token 的内积变小，注意力分布变"平坦"（趋于 uniform）。Re-Scale 通过乘以 concentration = $0.1 \ln(s) + 1.0$ 来放大 cos/sin 值，等效于降低注意力温度，恢复分布的锐度。这是 YaRN 能无损外推的关键——不只调频率，还补偿了频率调整带来的注意力退化。

### Q4：RoPE 衰减性实验的关键结论是什么？

三个核心发现：
1. **维度越大，衰减越平滑**——更多频率叠加，相消更充分
2. **高频负责近距离，低频负责远距离**——完整 RoPE 是多尺度的叠加
3. **超过训练长度后，基础 RoPE 衰减崩塌**——这正是需要 NTK/YaRN 的原因
