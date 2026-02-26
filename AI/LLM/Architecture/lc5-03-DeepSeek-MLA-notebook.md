---
title: "DeepSeek MLA 从零手写 · MA-RLHF lc8"
type: code-practice
date: 2026-02-26
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, ma-rlhf, lc8, deepseek, mla, attention, kv-cache, inference]
brief: "MLA（Multi-Head Latent Attention）手撕实操：从标准 MHA 出发，逐步推导低秩联合压缩（KV Cache 约压缩 16x）、矩阵吸收优化和 RoPE 分离，是 DeepSeek V2/V3 KV Cache 压缩的核心工程实现。"
related:
  - "[[AI/LLM/Architecture/DeepSeek-V3-手撕实操]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写]]"
  - "[[AI/LLM/MA-RLHF课程/lc5-DeepSeek-V3-MOC]]"
  - "[[AI/LLM/Inference/KV Cache|KV Cache]]"
  - "[[AI/LLM/Architecture/Attention 变体综述]]"
---

# DeepSeek MLA（多头潜在注意力）从零手写

> MA-RLHF Batch D / Architecture notebooks
> Source: `notebook/DeepSeek-MLA.ipynb`
> Ref: DeepSeek-V2 论文 §3.1 Multi-Head Latent Attention
> 评分: ★★★★★

---

## TL;DR

MLA（Multi-Head Latent Attention）是 DeepSeek-V2 的核心创新：用**低秩联合压缩**把 KV Cache 从 `[2, bs, L, n_kv_heads × head_dim]` 压缩到 `[1, bs, L, dc_kv]`（dc_kv ≪ dim），同时保持模型能力。本 notebook 从 MHA → MLA 基础版 → 矩阵吸收 → RoPE 分离，逐步推导完整实现。

---

## Part 1：标准 MHA（对比基线）

```python
class MultiHeadsAttention(nn.Module):
    def __init__(self, args):
        self.wq = nn.Linear(dim, n_heads * head_dim)       # [D, H*d_h]
        self.wk = nn.Linear(dim, n_kv_heads * head_dim)   # [D, G*d_h]  GQA
        self.wv = nn.Linear(dim, n_kv_heads * head_dim)
        self.wo = nn.Linear(n_heads * head_dim, dim)

    def forward(self, x):
        xq = self.wq(x)                          # [B, L, H*d_h]
        xk = repeat_kv(self.wk(x), n_rep)       # [B, L, H*d_h]  GQA expand
        xv = repeat_kv(self.wv(x), n_rep)
        # scaled dot-product attention
        scores = Q @ K.T / sqrt(head_dim)
        output = softmax(scores) @ V
        return self.wo(output)
```

**KV Cache 大小**：`[2, bs, seq_len, n_kv_heads × head_dim]`
- GQA 已经把 KV 头数从 H 降到 G（G << H），但每层仍然很大

---

## Part 2：MLA 基础版（低秩联合压缩）

### 核心思想

将 K、V 的投影矩阵做**低秩分解**：
- 不再直接存 `[K, V]`，而是存压缩后的潜在向量 `c_kv`（维度 dc_kv ≪ dim）
- 需要时通过 up 矩阵恢复

```python
class MultiHeadsLatentAttention(nn.Module):
    def __init__(self, args):
        # Q 也做低秩（训练省显存）
        self.wq_down = nn.Linear(dim, dc_q)     # [D, dc_q]  压缩
        self.wq_up   = nn.Linear(dc_q, dim)     # [dc_q, D]  恢复

        # KV 共享一个 down，各自有 up
        self.wkv_down = nn.Linear(dim, dc_kv)   # [D, dc_kv]  压缩（关键！）
        self.wk_up    = nn.Linear(dc_kv, dim)   # [dc_kv, D]
        self.wv_up    = nn.Linear(dc_kv, dim)   # [dc_kv, D]

        self.wo = nn.Linear(dim, dim)
```

### forward（训练阶段）

```python
# Q: 先降维再升维
c_q  = mla.wq_down(h)          # [B, L, dc_q]
xq   = mla.wq_up(c_q)          # [B, L, dim]

# KV: 共享 down，各自 up
c_kv = mla.wkv_down(h)         # [B, L, dc_kv]  ← 这就是存 KV Cache 的对象
xk   = mla.wk_up(c_kv)         # [B, L, dim]
xv   = mla.wv_up(c_kv)         # [B, L, dim]

# 后续 attention 与 MHA 相同
```

**KV Cache 压缩比**：
- MHA：`2 × n_kv_heads × head_dim` per token per layer
- MLA：`dc_kv` per token per layer（dc_kv = 4 in toy，生产中约 512，MHA 约 4096+）

---

## Part 3：矩阵吸收（推理加速）

### Q 矩阵吸收

训练时 Q 是两步：`c_q = wq_down(h)` → `xq = wq_up(c_q)`

等价于：`xq = (wq_up @ wq_down) @ h = wq_merged @ h`

```python
wq_merged = mla.wq_up.weight.data @ mla.wq_down.weight.data  # [dim, dim]
xq = h @ wq_merged   # 一次矩阵乘，不需要两步
```

**推理阶段**：合并 wq_up 和 wq_down，计算效率等同 MHA，但训练时可以省显存（低秩分解的 dc_q ≪ dim）。

### V 矩阵吸收（W_UV 吸收进 W_O）

原论文公式：
$$\mathbf{o}_t^C = W^{UV} \mathbf{c}_t^{KV} \quad \text{（V 从 c_kv 恢复）}$$
$$\text{output} = \text{Softmax}(QK^T) \cdot V \cdot W^O$$

**吸收**：$W^{UV}$ 可以吸收进 $W^O$：

```python
# 原始：V = c_kv @ W_UV；O = Softmax(QK^T) @ V；U = O @ W_O
V = c_kv @ W_UV              # [L, d]
O = softmax(Q @ K.T) @ V    # [L, d]
U = O @ W_O                  # [L, D]

# 吸收后：W_UV_absorbed = W_UV @ W_O
W_UV_absorbed = W_UV @ W_O  # [dc_kv, D]（存这一个矩阵代替两个）
V_absorbed = c_kv @ W_UV_absorbed   # [L, D]
O_absorbed = softmax(Q @ K.T) @ V_absorbed  # [L, D]，直接就是最终输出
```

**参数节省**：原本存 `W_UV(dc_kv × d) + W_O(d × D)` → 现在只存 `W_UV_absorbed(dc_kv × D)`

---

## Part 4：KV Cache 存什么？

**传统 MHA**：缓存 up 后的 K、V（或 GQA 的 grouped K、V）
- 大小：`2 × n_kv_heads × head_dim` per token

**MLA 推理时**：缓存压缩前的 `c_kv = wkv_down(h)`
- 大小：`dc_kv` per token（约 512 vs 2048+，压缩 4-8×）

**解压 on-demand**：
```
decoding 时：
K = wk_up(c_kv)   ← 从 cache 的 c_kv 恢复
V = wv_up(c_kv)   ← 同上
```

**为什么可以存 c_kv 而不是 K/V？**
因为 wkv_down 是线性的、可逆的（在给定 wk_up/wv_up 的情况下）。c_kv 包含了生成 K 和 V 所需的全部信息。存 c_kv 等于同时存储了 K 和 V 的"压缩编码"。

---

## Part 5：RoPE 的挑战与分离方案

### 问题

MLA 要存 `c_kv` 而非 `K`。但 RoPE 是位置相关的，要加在 K 上：
- 如果 `K = RoPE(wk_up(c_kv))`，那么 cache 里的 c_kv 每次取出都要重新做 RoPE 变换（计算浪费）
- 更深的问题：`RoPE(wk_up(c_kv))` 无法低秩分解（RoPE 的旋转操作和矩阵乘法不可交换）

### DeepSeek-V2 的解法：解耦 RoPE

**将位置编码分离出来，单独处理**：

```python
class MultiHeadsLatentAttention_withRoPE(nn.Module):
    # 主要参数（同基础版）
    self.wkv_down, self.wk_up, self.wv_up, self.wq_down, self.wq_up, self.wo

    # 额外的 RoPE 专用投影（只用于位置编码，不参与 value/output 计算）
    self.wq_up_rope = nn.Linear(dc_q, dim)       # Q 的 RoPE 部分（多头）
    self.wk_head_rope = nn.Linear(dim, head_dim) # K 的 RoPE 部分（单头，所有头共享）
```

**forward 中**：
```python
# 内容部分（正常 MLA）
c_q  = wq_down(h)
xq   = wq_up(c_q)      # content Q
c_kv = wkv_down(h)
xk   = wk_up(c_kv)     # content K（cache c_kv，解压得 K）

# 位置编码部分（额外计算，不缓存）
r_q = wq_up_rope(c_q)   # [B, L, dim]，多头，各头不同 RoPE
r_k = wk_head_rope(h)   # [B, L, head_dim]，单头，所有头共享

# 应用 RoPE
rope_q = apply_rope(r_q)   # per-head 不同旋转
rope_k = apply_rope(r_k)   # 单头旋转

# Concat：content + position
query_cat = cat(xq, rope_q, dim=-1)    # [B, H, L, 2*head_dim]
key_cat   = cat(xk, rope_k, dim=-1)   # [B, H, L, 2*head_dim]

# Attention（注意分母变化）
scores = query_cat @ key_cat.T / sqrt(2 * head_dim)  # cat 后维度翻倍
```

### Q 和 K 的 RoPE 维度差异

- **Q**：每头独立的 RoPE（多头，维度 = dim）→ `wq_up_rope: [dc_q → dim]`
- **K**：所有头共享同一 RoPE（单头，维度 = head_dim）→ `wk_head_rope: [dim → head_dim]`

原因（论文解释）：
- Q 是非缓存的，可以每次重算不同头的 RoPE
- K 缓存的是 `c_kv`，RoPE 部分 rope_k 单独存（不缓存，decoding 时从 h 重算）
- 所有头共享 rope_k 减少额外 cache 量

### KV Cache 最终组成

$$\text{Total KV Cache} = (d_c + d_h^R) \times L$$

其中：
- $d_c$：`dc_kv`，content 压缩维度（约 512）
- $d_h^R$：`head_dim`，单头 RoPE 维度（约 64）
- $L$：序列长度

**vs MHA**：`2 × n_heads × head_dim × L`（n_heads=128, head_dim=128 → 32768L）
**MLA**：约 576L → 压缩 **57×**

---

## 矩阵吸收的完整总结

| 矩阵 | 训练时 | 推理时 | 吸收操作 |
|------|-------|-------|---------|
| wq_down + wq_up | 两步投影 | 合并 wq | `wq = wq_up @ wq_down` |
| wkv_down | 存在 | 存在 | — （不吸收，这是 KV Cache） |
| wk_up | 训练存在 | 合并进 wq？ | 可吸收进 wq（论文提到） |
| wv_up + wo | 两步 | 合并 | `W_UV_absorbed = W_UV @ W_O` |

**wk_up 可以吸收进 wq_up**（论文 Eq.11）：
$$Q^T K = (W^{UQ} W^{DQ} h)^T (W^{UK} c_{kv}) = h^T (W^{DQ})^T (W^{UQ})^T W^{UK} c_{kv}$$

令 $W^{absorbed} = (W^{UQ})^T W^{UK}$，则 Q-K 内积只需 `h @ W^{DQ} @ W^{absorbed}` 和 `c_kv`，不需要显式算出 full K。

---

## 面试高频考点

**Q: MLA 为什么能压缩 KV Cache？**
A: 核心是低秩分解：K 和 V 的投影共享一个 down 矩阵 `wkv_down`（输出维度 dc_kv ≪ dim），推理时只缓存 `c_kv = wkv_down(h)`（约 512 维），需要时通过 wk_up/wv_up 恢复。相比 GQA 仍然缓存全量 KV，MLA 的 cache 量降低 ~57×。

**Q: RoPE 为什么要分离？怎么分离？**
A: 因为 `RoPE(W_k_up(c_kv))` 中的旋转操作位置相关，无法和线性变换交换顺序，导致 c_kv 不能作为有效 cache（每次取出都需重做 RoPE）。解法：把 Q 和 K 拆成 content 部分（不加 RoPE）和 position 部分（单独 RoPE），concat 后做 attention。Position 部分的 K 缓存单头 rope_k（从 h 实时计算）。

**Q: 矩阵吸收在推理阶段带来了什么好处？**
A: 减少了两次矩阵乘（wq_down + wq_up → 一次 wq_merged；wv_up + wo → 一次 W_UV_absorbed），同时减少了存储的参数量。生产部署中，wq_merged 和 W_UV_absorbed 在模型加载时预计算好。

**Q: MLA 的训练优势是什么？**
A: 低秩 Q（wq_down + wq_up）减少 Q 的激活显存（中间的 c_q 维度远小于 dim）。低秩 KV 共享 down 矩阵减少参数量。总参数量从 GQA 的约 `(H+2G)×d_h×D` 降到约 `dc_q×D + dc_kv×D + (H+1)×d_h×D`。

---

## See Also

- [[AI/LLM/Architecture/DeepSeek-V3-手撕实操]] — DeepSeek V3 完整架构手撕（MLA + MoE 组合），本笔记 MLA 部分的扩展
- [[AI/LLM/Inference/vLLM-PageKVCache-手撕实操]] — KV Cache 分配机制（PagedKVCache），与 MLA 低秩压缩正交
- [[AI/LLM/MA-RLHF课程/lc8-TPA-YaRN-RoPE外推从零手写]] — TPA（张量积低秩分解）+ YaRN，另一种 KV 压缩 + 长度外推组合
- [[AI/LLM/Architecture/Attention 变体综述]] — MLA/MQA/GQA/MHA 全谱系对比，理论定位
- [[AI/LLM/Inference/KV Cache|KV Cache]] — KV Cache 优化全景，MLA 的工程背景
