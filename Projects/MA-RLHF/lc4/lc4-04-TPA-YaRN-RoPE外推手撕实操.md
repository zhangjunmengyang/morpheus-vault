---
title: TPA + YaRN：KV 低秩分解与 RoPE 长度外推从零手写 · MA-RLHF lc8
type: code-practice
date: 2026-02-26
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - ma-rlhf
  - lc8
  - tpa
  - yarn
  - rope
  - attention
  - kv-cache
  - long-context
brief: TPA（Tensor Product Attention）+ YaRN 联合手撕：TPA 用张量积低秩分解压缩 KV Cache 同时保持各头独立性；YaRN 用 NTK-by-parts 分段策略解决 RoPE 长度外推时高频维度崩溃问题，两者覆盖 KV 压缩和上下文扩展两个正交问题。
related:
  - "[[Projects/MA-RLHF/lc5/lc5-02-DeepSeek-MLA-手撕实操]]"
  - "lc5-DeepSeek-V3-MOC"
  - "[[AI/3-LLM/Architecture/Attention 变体综述]]"
  - "[[Projects/MA-RLHF/lc5/lc5-01-DeepSeek-V3-手撕实操]]"
  - "[[AI/3-LLM/Inference/KV Cache|KV Cache]]"
---

# TPA + YaRN：KV 低秩分解与 RoPE 长度外推从零手写

> MA-RLHF Batch D / Architecture notebooks
> Source: `notebook/TPA-Pytorch.ipynb` + `notebook/YaRN-Pytorch.ipynb`
> Ref: TPA (Tensor Product Attention) + YaRN (arXiv:2309.00071)
> 评分: ★★★★★

---

## TL;DR

两个 notebook 覆盖 attention 机制的两个正交问题：

- **TPA**：把 KV 的投影做成**张量积低秩分解**（A ⊗ B），每个 token 的 K/V 是 head-specific 矩阵 A 和 shared 矩阵 B 的乘积，压缩 KV Cache 同时保持各头的独立性
- **YaRN**：RoPE 长度外推的三合一方案，解决"训短推长"时高频维度崩溃的问题——NTK by parts（分段策略）+ Dynamic Scaling + Re-Scale（attention temperature）

---

## Part 1：TPA（Tensor Product Attention）

### 背景：KV 投影的低秩张量积

标准 GQA：`K = W_k × x`，每个位置的 K 是一个全秩投影。

TPA 把 K/V 的投影分解为**张量积（Tensor Product）形式**：

$$K_i = A_k^{(i)} \cdot B_k$$

其中：
- $A_k^{(i)} \in \mathbb{R}^{h \times r}$：第 i 个 token 的 head-specific 系数矩阵（每头独立）
- $B_k \in \mathbb{R}^{r \times d_h}$：shared basis matrix（所有 token 共享方向空间）
- $r$：rank（远小于 head_dim）

**物理意义**：K 是 r 个 basis vectors 的加权线性组合，weights 由 A 决定（token 相关），basis 由 B 决定（全局共享）。类比 LoRA：A 是 input-dependent 权重，B 是 shared low-rank basis。

### 代码实现

```python
class CPLinear(nn.Module):
    def __init__(self, config):
        self.n_head = 8
        self.head_dim = 64
        self.rank = 4        # r ≪ head_dim

        # Q 仍然是标准投影
        self.c_q = nn.Linear(in_features, n_head * head_dim)

        # K 的张量积投影：A 和 B 分离
        self.W_A_k = nn.Linear(in_features, n_head * rank)   # head-specific 系数
        self.W_B_k = nn.Linear(in_features, rank * head_dim) # shared basis

        # V 同理
        self.W_A_v = nn.Linear(in_features, n_head * rank)
        self.W_B_v = nn.Linear(in_features, rank * head_dim)

    def forward(self, x):
        B, L, _ = x.size()

        q = self.c_q(x).view(B, L, n_head, head_dim)

        # K 的张量积
        A_k = self.W_A_k(x).view(B*L, n_head, rank)    # [BL, H, r]
        B_k = self.W_B_k(x).view(B*L, rank, head_dim)  # [BL, r, d_h]
        k = torch.bmm(A_k, B_k).div_(rank)             # [BL, H, d_h] / r（归一化）
        k = k.view(B, L, n_head, head_dim)

        # V 同理
        A_v = self.W_A_v(x).view(B*L, n_head, rank)
        B_v = self.W_B_v(x).view(B*L, rank, head_dim)
        v = torch.bmm(A_v, B_v).div_(rank).view(B, L, n_head, head_dim)

        return q, k, v
```

### 参数量对比

| 方式 | K 的参数量 | 
|------|-----------|
| MHA | `D × H × d_h` |
| GQA | `D × G × d_h`（G ≪ H）|
| TPA | `D × (H×r + r×d_h)` = `D × r × (H + d_h)` |

当 `r ≪ min(H, d_h)` 时，TPA 参数量最小。且 TPA 保持各头独立（A 是 head-specific）。

### 初始化设计

```python
def reset_parameters(self):
    # A 按 [D, H, r] 的视角做 Xavier
    W_A_k_tensor = self.W_A_k.weight.view(in_features, n_head, rank)
    nn.init.xavier_uniform_(W_A_k_tensor)

    # B 按 [D, r, d_h] 的视角做 Xavier
    W_B_k_tensor = self.W_B_k.weight.view(in_features, rank, head_dim)
    nn.init.xavier_uniform_(W_B_k_tensor)
```

**关键**：把权重 reshape 到正确的维度再做 Xavier init，确保梯度方差正确（fan_in/fan_out 基于真实的逻辑维度，不是 flatten 后的维度）。

### `.div_(rank)` 的意义

$K = A \cdot B / r$：除以 rank 做归一化，防止低秩乘积的输出方差随 rank 增大而线性增长（等价于对 basis 的加权均值而非加权求和）。

---

## Part 2：RoPE 标准实现

### 旋转位置编码原理

```python
def _apply_rotary_emb(x, cos, sin):
    # x: [L, H, d_h]
    x1, x2 = torch.chunk(x, 2, dim=-1)   # 前半、后半 head_dim
    o1 = x1 * cos - x2 * sin             # 实部旋转
    o2 = x2 * cos + x1 * sin             # 虚部旋转
    return torch.cat((o1, o2), dim=-1)

class RoPE(nn.Module):
    def _compute_concentration_and_inv_freq(self):
        # θ_i = base^{-2i/d}，i = 0, 1, ..., d/2-1
        freq = base ** (torch.arange(0, head_dim, 2) / head_dim)
        inv_freq = 1.0 / freq   # [d/2]
        return inv_freq

    def _compute_cos_sin(self, num_tokens):
        t = torch.arange(num_tokens)        # 位置 [0, 1, ..., L-1]
        freqs = einsum("i,j->ij", t, inv_freq)  # [L, d/2]
        return freqs.cos(), freqs.sin()
```

**频率分布**：维度 0 频率最高（旋转最快），维度 d/2-1 频率最低（旋转最慢）。高频维度编码局部相对位置，低频维度编码全局远程依赖。

---

## Part 3：YaRN（三合一外推方案）

### 问题：标准 RoPE 在超长序列上崩溃

训练时 max_len = 4096，推理时 seq_len = 40960（10× 外推）：
- 高频维度（i 小）：旋转角度超出训练范围，模型从未见过这些角度 → OOD
- 低频维度（i 大）：波长很长，旋转角度变化慢，还在训练范围内

### 三个技术的组合

#### 1️⃣ NTK by Parts（分段策略）

```python
low  = d/2 * log(L0 / (β * 2π)) / log(base)   # 高频阈值维度索引
high = d/2 * log(L0 / (α * 2π)) / log(base)   # 低频阈值维度索引
# L0 = initial_context_length = 4096
# α = 1, β = 32（默认）
```

三段处理：
- **高频段（i < low）**：PI（Position Interpolation），直接除以 scale 压缩位置
- **过渡段（low ≤ i ≤ high）**：线性混合 PI 和 NTK（gamma 插值）
- **低频段（i > high）**：NTK（不压缩，保持原始频率，因为这些维度还在训练范围内）

```python
interpolation = 1.0 / (scale * freq)    # PI：压缩位置索引
extrapolation = 1.0 / freq              # NTK/原始：不压缩

ramp = (torch.arange(d_half) - low) / (high - low)
mask = 1 - ramp.clamp(0, 1)            # 高频=1，低频=0

inv_freq = interpolation * (1 - mask) + extrapolation * mask
# 高频：用 interpolation（PI）
# 低频：用 extrapolation（NTK）
# 过渡段：线性混合
```

```python
def ntk_part(angle, r, low, high, alpha=1, beta=32, scale=10):
    gamma = (r - alpha) / (beta - alpha)   # 当前维度在 [low, high] 中的位置
    mix_angle = (1-gamma) * (angle/scale) + gamma * angle  # 插值混合
    out_angle = angle.clone()
    out_angle[low:high] = mix_angle[low:high]  # 过渡段用混合
    out_angle[high:]    = angle[high:] / scale  # 低频用 PI
    return out_angle
```

#### 2️⃣ Dynamic Scaling

```python
def forward(self, query, key):
    num_tokens = query.shape[0]
    self.cur_context_length = num_tokens    # 实时更新当前长度
    self.scaling_factor = max(1, cur_context_length / initial_context_length)
    cos, sin = self._compute_cos_sin(num_tokens)  # 每次 forward 动态重算
```

**静态 vs 动态**：
- 静态：scaling_factor 固定（训练时设好）
- 动态：推理时根据当前序列长度自动调整 scaling_factor

动态方案不需要提前知道最大长度，适合部署时的变长推理。

#### 3️⃣ Re-Scale（Attention Temperature）

$$\sqrt{1/t} = 0.1 \ln(s) + 1.0 \implies t = \frac{1}{(0.1\ln s + 1)^2}$$

其中 $s$ = scaling_factor（外推倍数）。

```python
concentration = 0.1 * math.log(scaling_factor) + 1.0  # √(1/t)
cos = freqs.cos() * concentration    # 所有维度的 cos/sin 乘以 concentration
sin = freqs.sin() * concentration
```

**为什么需要 Re-Scale**：
- RoPE 的 attention score = Q·K 包含旋转矩阵，旋转角度随位置增大
- 长序列外推后，旋转矩阵的方向更"随机"，导致 Q·K 内积的分布变化（方差增大）
- concentration 相当于调整了 attention 的有效温度（scale），补偿分布偏移
- concentration 随 s 增长而增长（s=1 时 c=1，s=10 时 c≈1.23），外推倍数越大越需要调整

### 完整 YaRN forward 流程

```
初始化时（训练/首次推理）：
  计算 scaling_factor = max(1, L/L0)
  计算 NTK by parts 的 inv_freq（分段 PI + 混合）
  计算 concentration（Re-Scale 因子）

每次 forward：
  Dynamic Scaling：实时更新 scaling_factor（动态版）
  cos, sin = freqs.cos() * concentration, freqs.sin() * concentration
  apply_rotary_emb(Q, cos, sin)
  apply_rotary_emb(K, cos, sin)
```

---

## 对比总结

| 技术 | 问题 | 解法 | 关键超参 |
|------|------|------|---------|
| 标准 RoPE | 位置编码固定范围 | 正弦旋转 | base, head_dim |
| PI（Position Interpolation）| OOD position | 线性压缩位置 | scale |
| NTK-RoPE | PI 破坏高频 | 统一放大 base | scale |
| YaRN NTK by parts | 高低频处理不一致 | 分段：PI/混合/NTK | α, β, scale |
| YaRN Dynamic | 推理时不知道 max_len | 实时更新 scale | L0 |
| YaRN Re-Scale | attention 分布偏移 | concentration 调温 | — |

---

## 面试高频考点

**Q: TPA 和 MLA 的核心区别？**
A: MLA 是 KV 共享一个 down 矩阵（wkv_down），压缩后存 c_kv 做 cache。TPA 是把每个 token 的 K/V 分解为 A（head-specific，per-token）和 B（shared basis，per-token）的乘积，不显式缓存低维向量，而是直接生成 full-rank K/V（但参数量更少）。MLA 压缩的是 cache，TPA 压缩的是参数。

**Q: YaRN 的三个技术为什么要组合使用？**
A: NTK by parts 解决高低频差异问题（不同维度用不同策略）；Dynamic Scaling 解决部署时序列长度未知的问题（实时调整）；Re-Scale 解决外推后 attention 分布偏移的问题（温度补偿）。缺少任一都会有短板：只有 NTK by parts 没有动态则需要提前知道 max_len；只有动态没有 Re-Scale 则 attention 分布失真。

**Q: NTK by parts 中低频段为什么不做 PI？**
A: 低频维度的波长很长，旋转角度变化缓慢，即使位置到了 40960，对应的旋转角度还在训练时见过的范围内（因为波长远大于 L0）。做 PI 反而会不必要地压缩这些维度的分辨率。只有高频维度（旋转角度可能超出训练范围）才需要 PI 压缩。

**Q: TPA 的 `.div_(rank)` 为什么要除以 rank？**
A: $K = A \cdot B$，A 是 [H, r]，B 是 [r, d_h]，乘积相当于 r 个向量的求和。如果不归一化，输出方差 ∝ r，rank 越大输出越大，破坏 attention scale 的稳定性。除以 r 变成加权平均，方差与 rank 无关。

---

## See Also

- [[Projects/MA-RLHF/lc5/lc5-02-DeepSeek-MLA-手撕实操]] — MLA 低秩联合压缩（正交问题：MLA 压缩 Q+KV，TPA 用张量积分解 KV）
- [[Projects/MA-RLHF/lc5/lc5-01-DeepSeek-V3-手撕实操]] — DeepSeek V3 完整架构手撕（YaRN + MLA 组合实战）
- lc5-DeepSeek-V3-MOC — lc5 课程地图，YaRN 在 Step 5 有详细理论说明
- [[AI/3-LLM/Architecture/Attention 变体综述]] — Attention 变体全谱系，TPA 定位
- [[AI/3-LLM/Inference/KV Cache|KV Cache]] — KV Cache 压缩全景，TPA/MLA 的工程背景
