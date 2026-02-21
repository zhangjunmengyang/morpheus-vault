---
title: "Transformer 架构深度解析 — 2026 技术全景"
date: 2026-02-21
tags: [面试, Transformer, Attention, MoE, SSM, 架构, 深度解析]
status: complete
---

# Transformer 架构深度解析 — 2026 技术全景

> **定位**：AI 算法工程师面试终极武器。从数学第一性原理出发，覆盖 Transformer 从 2017 到 2026 的全部演进脉络。目标：让面试官问不倒你。

---

## 目录

1. [注意力机制本质](#1-注意力机制本质)
2. [位置编码演进](#2-位置编码演进)
3. [KV Cache 优化与推理加速](#3-kv-cache-优化与推理加速)
4. [稀疏注意力与长上下文](#4-稀疏注意力与长上下文)
5. [MoE 架构](#5-moe-架构)
6. [归一化变体](#6-归一化变体)
7. [激活函数](#7-激活函数)
8. [训练稳定性](#8-训练稳定性)
9. [架构范式对比](#9-架构范式对比decoder-only-vs-encoder-decoder-vs-prefix-lm)
10. [2026 前沿架构](#10-2026-前沿架构)
11. [面试题集（15+ 道，难度递进）](#11-面试题集)
12. [附录 A：必背公式表](#附录-a必背公式表)
13. [附录 B：架构演进时间线](#附录-b架构演进时间线)
14. [附录 C：常见误区](#附录-c常见误区)
15. [参考文献](#参考文献)

---

## 1. 注意力机制本质

### 1.1 Scaled Dot-Product Attention

**数学定义**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。

**为什么除以 $\sqrt{d_k}$？——第一性原理推导**：

假设 $q_i, k_j$ 各分量独立同分布，均值为 0、方差为 1：

$$
q^\top k = \sum_{i=1}^{d_k} q_i k_i
$$

由独立性：$\text{Var}(q^\top k) = \sum_{i=1}^{d_k} \text{Var}(q_i) \cdot \text{Var}(k_i) = d_k$

当 $d_k$ 很大时，点积方差大 → softmax 输入值的量级大 → softmax 进入饱和区 → 梯度接近 0（梯度消失）。

除以 $\sqrt{d_k}$ 后方差归一化为 1，softmax 工作在合理区间。

**直觉**：这是一种 temperature scaling。$T = \sqrt{d_k}$ 让 attention 分布既不太 sharp 也不太 uniform。

**关键论文**：Vaswani et al., "Attention Is All You Need" (NeurIPS 2017) [1]

### 1.2 Multi-Head Attention (MHA)

**数学定义**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

其中 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，$d_k = d_v = d_{\text{model}} / h$。

**为什么多头而非单头？**

- **子空间多样性**：不同 head 学到不同类型的注意力模式（位置关系、语法关系、语义关系等）
- **秩瓶颈理论**：单头 attention 矩阵的秩受限于 $d_k$，多头可以学到秩为 $h \cdot d_k = d_{\text{model}}$ 的注意力
- **鲁棒性**：单头 softmax 容易被极值 token 主导，多头分散了这种风险

**计算复杂度**：$O(n^2 \cdot d_{\text{model}})$，与单头相同（只是切分了维度）

**KV Cache 成本（推理时关键）**：

$$
\text{KV Cache per layer} = 2 \times h \times d_k \times n = 2 \times d_{\text{model}} \times n
$$

总 KV Cache = $2 \times L \times d_{\text{model}} \times n \times \text{bytes\_per\_param}$

**关键论文**：同 [1]

### 1.3 Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA)

**MQA 核心思想**（Shazeer, 2019 [2]）：所有 head 共享同一组 K、V 投影。

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)
$$

- KV Cache 从 $2hd_kn$ 降至 $2d_kn$，缩减 $h$ 倍
- 训练时各 head 仍各自学 Q，但 K、V 全局共享
- 代价：质量略降（尤其对需要多样化注意力模式的任务）

**GQA 核心思想**（Ainslie et al., 2023 [3]）：将 $h$ 个 query head 分成 $g$ 组，每组共享一套 K、V。

$$
\text{KV Cache} = 2 \times g \times d_k \times n
$$

- 当 $g = 1$ 时退化为 MQA；当 $g = h$ 时退化为 MHA
- LLaMA-2 70B、Mistral-7B 均采用 GQA（$g = 8$）
- **从 MHA 到 GQA 的迁移**：用 mean pooling 初始化 GQA 的 KV 头（论文中验证有效）

**KV Cache 对比表**：

| 方法 | KV Cache / layer | LLaMA-2 70B (n=4096) |
|------|------------------|-----------------------|
| MHA  | $2 h d_k n$     | 2×64×128×4096 = 64 MB |
| GQA-8| $2 \times 8 \times d_k \times n$ | 2×8×128×4096 = 8 MB |
| MQA  | $2 d_k n$       | 2×1×128×4096 = 1 MB |

**关键论文**：[2][3]

### 1.4 Multi-Head Latent Attention (MLA) — DeepSeek-V2/V3

**问题**：GQA 通过减少 KV head 数降低 cache，但牺牲了表达能力。能否既压缩 cache 又保持全秩？

**核心思想**：将 KV 压缩到低维 latent 空间 $c_{kv} \in \mathbb{R}^{d_c}$，推理时只缓存 $c_{kv}$。

$$
c_{kv} = x W^{DKV} \quad \in \mathbb{R}^{d_c}, \quad d_c \ll h \cdot d_k
$$

$$
k = c_{kv} W^{UK}, \quad v = c_{kv} W^{UV}
$$

**Absorption Trick（面试高频考点）**：

推理时不需要显式展开 K。将 $W^{UK}$ 吸收进 Q 的投影：

$$
q^\top k = (x W^Q)(c_{kv} W^{UK})^\top = (x W^Q {W^{UK}}^\top) c_{kv}^\top
$$

定义 $\tilde{W}^Q = W^Q {W^{UK}}^\top$，则 attention 直接在 latent 空间完成。

**RoPE 兼容问题**：RoPE 对 $q, k$ 施加位置旋转，但 absorption 要求 $W^{UK}$ 可被吸收。解决方案：**Decoupled RoPE Keys**——额外分出一组低维 keys 专门承载 RoPE，不参与 absorption。

$$
c_{kv\_rope} = x W^{KR} \quad \in \mathbb{R}^{d_r}, \quad d_r \ll d_c
$$

KV Cache = $d_c + d_r$（DeepSeek-V3 中 $d_c = 512, d_r = 64$，远小于 MHA 的 $h \times d_k = 16384$）

**关键论文**：DeepSeek-V2 [4]，DeepSeek-V3 [5]

### 1.5 FlashAttention

**问题**：标准 attention 需要 $O(n^2)$ 显存存储完整 attention 矩阵 $S = QK^\top$。

**核心洞察**：GPU 计算快、内存带宽慢（"memory-wall"）。Attention 是 memory-bound 操作。

**算法核心**——分块 + Online Softmax：

1. 将 $Q, K, V$ 切成块（block size $B_r \times B_c$），逐块在 SRAM 计算
2. **Online Softmax**（Milakov & Gimelshein, 2018 [6]）：在线更新 softmax 的分母
   - 维护 running max $m$ 和 running sum $l$
   - 每见到新块，rescale 之前累积的结果
3. 不需要实际化 $n \times n$ attention 矩阵到 HBM

**IO 复杂度分析**：

- 标准 Attention：$O(n^2 d + n^2)$ HBM 访问
- FlashAttention：$O(n^2 d^2 / M)$ HBM 访问（$M$ = SRAM 大小）
- 显存：$O(n^2) \to O(n)$

**FlashAttention-2 改进**：减少非 matmul FLOP、更好的 warp 并行、序列并行

**FlashAttention-3**（H100, 2024 [7]）：
- 利用 Tensor Core 异步性，计算-通信 overlap
- FP8 支持，block quantization 减少精度损失
- Incoherent processing 降低量化误差

**关键论文**：Dao et al. (2022) [8]，Dao (2023) [9]，Shah et al. (2024) [7]

### 1.6 Ring Attention

**问题**：单 GPU 显存放不下超长序列。

**核心思想**：将序列均分到 $P$ 个设备。每个设备持有自己的 Q 块，KV 块在设备间环形传递。

**算法**：

```
For step s = 0, 1, ..., P-1:
    device_i 持有 Q_i (固定不动)
    device_i 当前持有 KV_{(i-s) mod P}
    计算 partial attention: O_i += Attention(Q_i, KV_{(i-s) mod P})
    将 KV 块发送给 device_{(i+1) mod P}（环形传递）
    计算与通信 overlap
```

**关键性质**：
- 通信量 = $O(n \cdot d / P)$ per step × $P$ steps = $O(n \cdot d)$，与 all-to-all 相同
- 但通信与计算可完美 overlap（当 $n/P$ 足够大时）
- 理论上支持**无限序列长度**（只受设备数限制）

**Striped Ring Attention**：交替分配 token 到设备（striped pattern），解决 causal mask 导致的负载不均衡。

**关键论文**：Liu et al. (2023) [10]

---

## 2. 位置编码演进

### 2.1 为什么需要位置编码？

Self-Attention 本身是**置换等变**（permutation equivariant）的：

$$
\text{Attention}(\Pi Q, \Pi K, \Pi V) = \Pi \cdot \text{Attention}(Q, K, V)
$$

其中 $\Pi$ 是任意排列矩阵。这意味着 attention 无法区分 token 顺序。位置编码注入顺序信息。

### 2.2 Sinusoidal（绝对位置编码）

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**设计直觉**：
- 不同频率的正弦/余弦 → 每个位置有唯一编码（类似二进制编码的连续版本）
- 任意两个位置的编码内积只依赖相对距离 $|pos_1 - pos_2|$（因为 $\sin(a)\sin(b) + \cos(a)\cos(b) = \cos(a-b)$）
- 理论上可外推到训练未见的长度（但实际效果差）

**局限**：加在 embedding 上，后续层的 attention 不直接操作位置信息；外推能力差。

### 2.3 Learnable Position Embedding

$$
h_i = x_i + e_{pos_i}, \quad e \in \mathbb{R}^{L_{\max} \times d}
$$

- BERT、GPT-2 等采用
- 优点：完全数据驱动，简单
- 缺点：无法外推超过 $L_{\max}$，参数量 $O(L_{\max} \cdot d)$

### 2.4 ALiBi（Attention with Linear Biases）

**核心思想**（Press et al., 2022 [11]）：不修改 embedding，直接在 attention score 上加 bias：

$$
\text{softmax}\left(\frac{q_i^\top k_j}{\sqrt{d_k}} - m \cdot |i - j|\right)
$$

其中 $m$ 是 per-head 的斜率，几何级数递减：$m_k = 2^{-8k/h}$。

**直觉**：相距越远的 token，attention score 惩罚越大 → 天然的局部性偏置。

**优点**：
- 零额外参数
- 外推能力好（训练 1024，推理 2048 仍可用）
- 训练效率好（短序列训练可泛化到长序列）

**缺点**：
- 线性衰减假设过于简单（不同 head 可能需要不同的衰减形状）
- 被 RoPE 全面超越后，主流地位不再

### 2.5 RoPE（Rotary Position Embedding）

**核心数学**（Su et al., 2021 [12]）：

将 $d$ 维向量视为 $d/2$ 个二维子空间。在每个子空间 $i$ 上，对位置 $m$ 的向量施加旋转角 $m\theta_i$：

$$
f(x_m, m) = R_{\Theta, m} x_m
$$

$$
R_{\Theta, m} = \text{diag}(R_{m\theta_1}, R_{m\theta_2}, \dots, R_{m\theta_{d/2}})
$$

$$
R_{m\theta_i} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$。

**核心性质**：$f(q_m, m)^\top f(k_n, n) = g(q_m, k_n, m - n)$

即 attention score 自然只依赖**相对位置** $m - n$。

**证明**（2D case）：

$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k = g(q, k, m-n)
$$

因为旋转矩阵满足 $R_a^\top R_b = R_{b-a}$。

**频率设计的物理直觉**：
- 低维度 → 高频旋转 → 编码局部位置关系（相邻 token 差异大）
- 高维度 → 低频旋转 → 编码远程位置关系（远距离 token 缓慢变化）
- 类比傅里叶变换：多频率组合唯一确定位置

**关键论文**：[12]

### 2.6 YaRN（Yet another RoPE extensioN）

**问题**：RoPE 训练在 4K 长度上，推理 8K+ 时性能崩溃。为什么？

**根因分析**：低频维度的旋转角 $m\theta_i$ 对应的**有效注意力周期** $\lambda_i = 2\pi/\theta_i$ 小于训练序列长度。当推理长度超过训练长度时，低频维度的角度超出训练分布。

**Position Interpolation (PI)**（Chen et al., 2023 [13]）：缩放位置 $m' = m \cdot L_{\text{train}} / L_{\text{target}}$。问题：**高频维度被不必要地压缩**。

**NTK-aware Interpolation**：修改频率基数 $\theta' = \theta \cdot \alpha^{d/(d-2)}$，其中 $\alpha = L_{\text{target}} / L_{\text{train}}$。高频几乎不动，低频缩放多。但对所有维度施加相同缩放因子。

**YaRN（Peng et al., 2023 [14]）——分频段处理**：

将 $d/2$ 个频率维度分为三组：
1. **高频维度**（$\lambda_i < L_{\text{train}}$）：不缩放，保持局部关系精度
2. **低频维度**（$\lambda_i > L_{\text{target}}$）：线性插值缩放
3. **中间维度**：NTK-aware 渐进插值（ramp function）

加上 **attention temperature scaling**：

$$
\text{softmax}\left(\frac{1}{t} \cdot \frac{q^\top k}{\sqrt{d_k}}\right), \quad t = \sqrt{1 + \frac{1}{d}\ln\frac{L_{\text{target}}}{L_{\text{train}}}}
$$

**直觉**：长序列时 attention 分布变 flat → 需要略微锐化（降低 temperature）。

**实践**：仅需 400-600 步微调，即可从 4K 扩展到 128K+。

**关键论文**：[13][14]

### 2.7 NTK-aware Scaling（进阶）

**核心公式**：修改 RoPE 的基频：

$$
\theta'_i = \theta_i \cdot \alpha^{-2i/(d-2)}, \quad \alpha = \frac{L_{\text{target}}}{L_{\text{train}}}
$$

等价于修改基数：$b' = b \cdot \alpha^{d/(d-2)}$

**为什么叫 "NTK-aware"？** 类比神经切线核（Neural Tangent Kernel）理论：高频分量对应 NTK 的局部特征，应保持不变；低频分量对应全局特征，需要按比例缩放。

---

## 3. KV Cache 优化与推理加速

### 3.1 KV Cache 基础

**问题**：Autoregressive 解码时，每生成一个 token 都要对所有之前的 token 做 attention。朴素实现需要重新计算所有 K、V。

**KV Cache 原理**：缓存每层的 K、V 矩阵，每步只追加新 token 的 $k_t, v_t$。

$$
K_t = [K_{t-1}; k_t], \quad V_t = [V_{t-1}; v_t]
$$

**显存计算**（以 LLaMA-2 70B 为例，bf16）：

$$
\text{Cache} = 2 \times L \times 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times n_{\text{seq}} \times \text{bytes}
$$

$= 2 \times 80 \times 2 \times 8 \times 128 \times 4096 \times 2 = 2.6 \text{ GB}$ per sequence

Batch size 32 → 83 GB，可能超过整个 GPU 显存！

### 3.2 PagedAttention / vLLM

**问题**：KV Cache 的预分配方式（按最大长度分配连续内存）导致严重的内部碎片。实测 60-80% 的 KV Cache 内存被浪费。

**核心思想**（Kwon et al., 2023 [15]）：借鉴 OS 虚拟内存的分页机制：

1. KV Cache 被分成固定大小的 **block**（如 16 tokens）
2. 逻辑上连续的 KV 序列可以映射到物理上不连续的内存块
3. **Block Table** 维护逻辑→物理块的映射
4. 新 token 到来时，按需分配新 block

**收益**：
- 内存利用率从 20-40% 提升到 ~96%
- **Copy-on-Write**：多个 beam search 候选共享前缀 block，分叉时才复制
- **Prefix Caching**：相同系统 prompt 的请求共享 KV Cache block

**关键论文**：[15]

### 3.3 Continuous Batching

**问题**：Static batching（等所有请求生成完毕才释放）导致短请求等待长请求，GPU 利用率低。

**解决方案**：
- **Iteration-level scheduling**：每个解码步骤检查哪些请求已完成
- 完成的请求立即释放资源，新请求立即填入
- 吞吐量提升 2-5× 相比 static batching

**与 PagedAttention 的协同**：vLLM 结合两者，新请求可以立即获得已释放的内存块。

### 3.4 Speculative Decoding

**问题**：LLM 推理受限于 memory bandwidth（每步只生成一个 token，GPU 算力大量闲置）。

**核心思想**（Leviathan et al., 2023 [16]；Chen et al., 2023 [17]）：

1. **Draft Model**（小模型，如 68M）快速生成 $\gamma$ 个 candidate tokens
2. **Target Model**（大模型）一次 forward pass 验证所有 $\gamma$ 个 tokens
3. 接受匹配的 tokens，拒绝第一个不匹配的，从该位置重新采样

**数学保证——拒绝采样**：

对于每个 candidate token $x_t$：

$$
P(\text{accept}) = \min\left(1, \frac{p(x_t)}{q(x_t)}\right)
$$

其中 $p$ 是 target 分布，$q$ 是 draft 分布。被拒绝时，从修正分布采样：

$$
p'(x) = \text{normalize}\left(\max(0, p(x) - q(x))\right)
$$

**关键定理**：这保证了最终输出分布**严格等于** target model 的分布（无损）。

**加速比**：取决于 draft-target 对齐度 $\alpha$（acceptance rate）：

$$
\text{Speedup} \approx \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(c \cdot \gamma + 1)}
$$

其中 $c$ 是 draft/target 的计算成本比。典型加速 2-3×。

**关键论文**：[16][17]

### 3.5 Medusa

**问题**：Speculative Decoding 需要额外的 draft model，增加系统复杂度和显存。

**核心思想**（Cai et al., 2024 [18]）：在 target model 的最后一层之上，**并行添加多个预测 head**，每个 head 预测未来第 $k$ 个 token。

$$
\hat{y}_{t+k} = \text{MedusaHead}_k(\text{hidden}_{t}), \quad k = 1, 2, \dots, K
$$

**Tree Attention 验证**：多个 head 的 top-$s$ 候选组合成 **candidate tree**，一次 forward pass 验证整棵树。

**Medusa-2**：加入自蒸馏训练，不需要 ground truth 标签。

**对比 Speculative Decoding**：
- 优势：无需 draft model、显存友好、工程简单
- 劣势：需要微调 Medusa heads（或 SFT 阶段一起训练）

**关键论文**：[18]

### 3.6 推理加速方法对比

| 方法 | 加速比 | 是否无损 | 额外成本 | 工程复杂度 |
|------|--------|----------|----------|------------|
| Speculative Decoding | 2-3× | ✅ 严格无损 | Draft model | 中等 |
| Medusa | 2-3× | ❌ 需微调 | Medusa heads | 低 |
| EAGLE | 2.5-3.5× | ✅ 近似无损 | Feature-level draft | 中等 |
| Continuous Batching | 2-5× 吞吐 | ✅ | 调度器 | 中等 |
| Quantization (INT4) | 2-4× | ❌ 轻微损失 | 量化工具 | 低 |

---

## 4. 稀疏注意力与长上下文

### 4.1 全注意力的瓶颈

标准 Self-Attention 的计算和显存复杂度均为 $O(n^2)$：

- 计算：$O(n^2 d)$ FLOPs
- 显存：$O(n^2)$（attention 矩阵）+ $O(nd)$（KV Cache）

当 $n = 128K$ 时，$n^2 \approx 1.6 \times 10^{10}$，fp16 下 attention 矩阵约 30 GB。

### 4.2 稀疏 Attention 模式

**Sparse Transformer**（Child et al., 2019 [19]）：

- **Strided Pattern**：每个 token 只 attend to 固定间隔的 token
- **Fixed Pattern**：将序列分块，block 内全连接 + 块间稀疏连接
- 复杂度：$O(n\sqrt{n})$

**Longformer**（Beltagy et al., 2020 [20]）：

- **Sliding Window**：每个 token attend to 周围 $w$ 个 token（局部信息）
- **Global Tokens**：特殊 token（如 [CLS]）attend to 所有（全局信息）
- 复杂度：$O(n \cdot w)$

**BigBird**：Random + Sliding Window + Global 三种模式混合。理论证明：稀疏 + random 也是图灵完备的。

### 4.3 Sliding Window Attention（Mistral）

**Mistral 7B**（Jiang et al., 2023 [21]）的实现：

- Window size $W = 4096$
- 每个 token 只 attend to 前 $W$ 个 token
- **但信息可以跨窗口传播**：经过 $L$ 层后，理论感受野 = $L \times W$
  - Mistral 32 层 × 4096 window = 131072 tokens 理论感受野

**Rolling Buffer Cache**：
- KV Cache 只保留最近 $W$ 个位置
- 固定显存：$2 \times L \times W \times d$ 不随序列长度增长

**Pre-fill with Chunking**：长 prompt 分块处理，每块大小 = $W$。

### 4.4 Mamba 与 State Space Models (SSM)

**问题**：Attention 的 $O(n^2)$ 即使稀疏化也有局限。能否用 $O(n)$ 的架构替代？

**SSM 基础——连续时间状态空间**：

$$
\dot{h}(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)
$$

离散化后：$h_t = \bar{A}h_{t-1} + \bar{B}x_t$，$y_t = Ch_t$

- $A \in \mathbb{R}^{N \times N}$：状态转移矩阵
- $B \in \mathbb{R}^{N \times 1}$，$C \in \mathbb{R}^{1 \times N}$：输入/输出投影

**S4**（Gu et al., 2022 [22]）的突破：
- $A$ 初始化为 HiPPO 矩阵（optimal polynomial projection）
- 利用 $A$ 的结构性质做 $O(n \log n)$ 卷积加速（训练时）
- 推理时切回 RNN 模式：$O(1)$ per step

**Mamba（S6）**（Gu & Dao, 2023 [23]）的核心创新——**Selective SSM**：

$$
B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))
$$

$B, C, \Delta$ 依赖输入（input-dependent），而非固定参数。

**为什么 Selectivity 关键？**
- 固定 $A, B, C$ → 线性时不变（LTI）→ 可做卷积但无法"选择性记忆"
- 可变 $B, C, \Delta$ → 打破 LTI → 可以根据内容决定记忆什么、遗忘什么
- 类比 LSTM 的门控，但更高效

**计算复杂度**：

| | 训练 | 推理（per step） | 显存 |
|---|------|-----------------|------|
| Attention | $O(n^2 d)$ | $O(nd)$ | $O(n^2)$ or $O(n)$ (Flash) |
| Mamba | $O(n \cdot d \cdot N)$ | $O(d \cdot N)$ | $O(d \cdot N)$ 固定 |

其中 $N$ 是 state dimension（如 16），远小于 $n$。

**Mamba 的硬件感知实现**：
- 在 SRAM 中做 scan 操作（并行 scan）
- 不展开完整状态历史
- kernel fusion 减少 HBM 访问

### 4.5 Jamba

**核心思想**（AI21, 2024 [24]）：**混合架构** = Mamba + Attention，比例约 3:1。

- 大部分层是 Mamba 层（高效处理长序列）
- 每 4 层插入 1 个 Attention 层（精确检索兜底）
- 加入 MoE（16 experts, top-2）进一步扩容

**为什么混合？**
- 纯 Mamba 在 "needle in a haystack" 任务上表现差（需要精确检索过去某个具体 token）
- 纯 Attention 在超长序列上 KV Cache 爆炸
- 混合架构取两者之长

**关键数据**：Jamba 52B 参数（12B 活跃），256K 上下文，80GB GPU 可装

### 4.6 DeepSeek Dynamic Sparse Attention

**核心思想**（DeepSeek-V3 [5]）：根据注意力分数的分布动态决定稀疏模式，而非预设固定模式。

- **训练时**全注意力
- **推理时**对每个 query，先用少量 token（锚点）估计注意力分布
- 根据估计结果选择 top-$k$ 重要的 KV 子集
- 只对选中的子集做精确 attention

**与 Sparse Attention 的区别**：稀疏模式是 content-dependent 的，不是 position-dependent 的。

---

## 5. MoE 架构

### 5.1 MoE 基础

**核心公式**：

$$
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

$$
g(x) = \text{TopK}(\text{softmax}(x W_g))
$$

其中 $E_i$ 是第 $i$ 个 expert（通常是 FFN），$g_i$ 是 gating score，TopK 选择 $K$ 个 expert 激活。

**为什么 MoE？**
- 总参数量大（更强的容量），但每次推理只激活一部分（计算成本可控）
- 10B 活跃参数可以达到 70B dense 模型的性能，推理成本与 10B 相当

### 5.2 演进脉络

**Switch Transformer**（Fedus et al., 2022 [25]）：
- $K = 1$：每个 token 只路由到 1 个 expert
- 简化了 load balancing，减少通信
- 但 top-1 选择容易导致 expert 坍塌（少数 expert 承载大部分流量）

**GShard**（Lepikhin et al., 2021 [26]）：
- $K = 2$，top-2 gating
- 引入 **Auxiliary Load Balancing Loss**：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 是 expert $i$ 实际处理的 token 比例，$P_i$ 是 expert $i$ 的平均 gating probability。均匀时 $\mathcal{L}_{\text{aux}}$ 最小。

**Expert Choice**（Zhou et al., 2022 [27]）：
- 反转路由方向：**expert 选 token**（而非 token 选 expert）
- 每个 expert 固定处理 $n \cdot K / N$ 个 token
- 天然负载均衡，无需 auxiliary loss
- 缺点：token 可能被 0 个或多个 expert 选中

### 5.3 Mixtral

**Mixtral 8x7B**（Jiang et al., 2024 [28]）：
- 8 个 expert，top-2 routing
- 总参数 46.7B，活跃参数 12.9B
- 每层 FFN 替换为 MoE（attention 层不变）
- Sliding Window Attention ($W = 4096$)
- 性能持平或超越 LLaMA-2 70B

### 5.4 DeepSeek-MoE

**Fine-Grained Expert Segmentation**（DeepSeek [29]）：

将传统 FFN expert 拆分为更细粒度的小 expert：$N = 64$（甚至 256），$K = 6$。

**动机**：粗粒度 expert 内部存在知识冗余和知识混合。细粒度 expert 允许更灵活的知识组合。

**Shared Expert**：$K_s$ 个 expert 始终被激活（不经过 routing），保证通用知识的基线。

$$
y = \sum_{i=1}^{K_s} E_i^{\text{shared}}(x) + \sum_{j \in \text{TopK}} g_j(x) \cdot E_j^{\text{routed}}(x)
$$

**DeepSeek-V3 的负载均衡改进——Bias-based（无 Auxiliary Loss）**：

$$
g'_i(x) = g_i(x) + b_i
$$

$b_i$ 是可学习的 bias，倾向于将 token 分配给负载不足的 expert。好处：不像 auxiliary loss 那样干扰主要训练目标。

**关键数据**：DeepSeek-V3 = 671B 总参数，37B 活跃参数，MoE + MLA + 多种工程优化。

### 5.5 MoE 核心挑战

| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| 负载不均衡 | 少数 expert 过载 | Auxiliary loss / Expert Choice / Bias-based |
| Expert 坍塌 | Expert 趋于同质化 | Fine-grained experts + Shared experts |
| 通信开销 | All-to-All 通信（EP） | Expert Parallelism + 通信优化 |
| 训练不稳定 | Router 产生的梯度噪声 | Router z-loss、Jitter noise |
| 推理效率 | Expert 权重需频繁加载 | Expert offloading、EP + TP 组合 |

---

## 6. 归一化变体

### 6.1 Layer Normalization

**公式**：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

**为什么用 LayerNorm 而非 BatchNorm？**
- BatchNorm 沿 batch 维度归一化 → 依赖 batch 统计量 → 推理时需要 running statistics
- LayerNorm 沿 feature 维度归一化 → 每个样本独立 → 对 batch size 不敏感
- 序列模型中 batch 内不同样本长度不同，BatchNorm 的统计量有偏

### 6.2 RMSNorm

**公式**（Zhang & Sennrich, 2019 [30]）：

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x) + \epsilon}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}
$$

**与 LayerNorm 的区别**：
- 去掉了 **mean centering**（$x - \mu$ 步骤）和 **bias** $\beta$
- 只做 scale 归一化，不做 shift
- 计算量减少约 7-10%（省去均值计算和减法）

**为什么可以去掉 mean centering？**
- 实验表明 mean centering 的贡献远小于 variance normalization
- Re-centering 可以由后续的线性层补偿
- LLaMA、Mistral、DeepSeek 等 2023+ 模型几乎全部采用 RMSNorm

### 6.3 Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：

$$
x' = \text{LayerNorm}(x + \text{Attention}(x))
$$

**Pre-Norm**（GPT-2 及之后）：

$$
x' = x + \text{Attention}(\text{LayerNorm}(x))
$$

**Pre-Norm 的优势——梯度分析**：

Post-Norm 中，梯度需要穿过 LayerNorm，而 LayerNorm 的 Jacobian 行列式依赖输入：

$$
\frac{\partial \text{LN}(x)}{\partial x} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d\sigma^2}\hat{x}\hat{x}^\top\right)
$$

这可能引入梯度爆炸/消失。

Pre-Norm 中，残差路径上**无非线性变换**：

$$
\frac{\partial x_L}{\partial x_l} = I + \sum_{\text{intermediate}} \frac{\partial F}{\partial x}
$$

恒等项 $I$ 保证梯度直通，类似 ResNet 的 skip connection。

**Post-Norm 的优势**：
- 表示质量稍好（因为归一化在子层输出之后，约束更强）
- 深层训练稳定了可以恢复

### 6.4 DeepNorm

**问题**：训练 >100 层的 Transformer 时，即使 Pre-Norm 也可能不稳定。

**DeepNorm**（Wang et al., 2022 [31]）：

$$
x' = \text{LayerNorm}(\alpha \cdot x + \text{Sublayer}(x))
$$

其中 $\alpha = (2N)^{1/4}$（$N$ 是总层数），同时初始化权重时缩放 $\beta = (8N)^{-1/4}$。

**直觉**：通过放大残差路径（$\alpha > 1$）和缩小子层初始输出（$\beta < 1$），让深层网络训练初期的梯度更稳定。

**效果**：1000 层 Transformer 稳定训练（而 Pre-Norm 在 ~200 层开始不稳定）。

---

## 7. 激活函数

### 7.1 从 ReLU 到 GELU

**ReLU**：$f(x) = \max(0, x)$
- 优点：计算简单、解决 sigmoid 饱和
- 缺点：**dying ReLU**（负区间梯度为 0 → 部分神经元永久死亡）

**GELU**（Hendrycks & Gimpel, 2016 [32]）：

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

其中 $\Phi(x)$ 是标准正态 CDF。

**直觉**：**随机正则化的确定性版本**。ReLU 在 $x < 0$ 时完全置零，GELU 则是平滑地按概率 $\Phi(x)$ 保留。负值区域也有小梯度。

### 7.2 SwiGLU

**GLU（Gated Linear Unit）**（Dauphin et al., 2017 [33]）：

$$
\text{GLU}(x) = (xW_1) \odot \sigma(xW_2)
$$

**SwiGLU**（Shazeer, 2020 [34]）：

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1)) \odot (xW_2)
$$

$$
\text{Swish}(x) = x \cdot \sigma(\beta x) \approx x \cdot \sigma(x) \quad (\beta=1)
$$

**FFN 完整公式**：

$$
\text{FFN}_{\text{SwiGLU}}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2
$$

注意有**三个权重矩阵**（$W_1, W_2, W_3$），比标准 FFN 多一个。

**参数补偿**：为保持参数量不变，将 $d_{\text{ff}}$ 从 $4d$ 调整为 $\frac{8}{3}d$。

$$
3 \times d \times \frac{8}{3}d = 8d^2 \approx 2 \times d \times 4d = 8d^2 \quad \checkmark
$$

**为什么 SwiGLU 更好？**
- 门控机制：第二路投影 $xW_3$ 作为信息选择器，选择性放大或抑制
- Swish 平滑非单调（在 $x \approx -1.28$ 处有一个 "valley"），梯度传播更好
- PaLM 论文 [35] 系统比较：SwiGLU > GELU > ReLU（在相同参数量下，PPL 一致更低）

### 7.3 GeGLU

$$
\text{GeGLU}(x) = \text{GELU}(xW_1) \odot (xW_2)
$$

SwiGLU 和 GeGLU 性能接近，SwiGLU 因 PaLM/LLaMA 的采用而成为事实标准。

---

## 8. 训练稳定性

### 8.1 梯度消失/爆炸

**深层网络的链式法则**：

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^{L-1} \frac{\partial h_{l+1}}{\partial h_l} \cdot \frac{\partial h_1}{\partial W_1}
$$

- 如果 $\|\frac{\partial h_{l+1}}{\partial h_l}\| > 1$：梯度指数爆炸
- 如果 $\|\frac{\partial h_{l+1}}{\partial h_l}\| < 1$：梯度指数消失

### 8.2 残差连接——为什么是核心

$$
x_{l+1} = x_l + F_l(x_l)
$$

**梯度传播**：

$$
\frac{\partial x_L}{\partial x_l} = I + \frac{\partial}{\partial x_l}\sum_{k=l}^{L-1}F_k(x_k)
$$

**关键**：无论 $F_k$ 的梯度多小，恒等项 $I$ 保证梯度至少为 1。

**隐式集成解释**（Veit et al., 2016 [36]）：$L$ 层 ResNet 隐式包含 $2^L$ 条从输入到输出的路径（每层选择走 skip 或 sublayer）。等效于 $2^L$ 个浅层网络的集成。

### 8.3 Learning Rate Warmup

**为什么需要 warmup？**

训练初期，模型参数随机 → attention pattern 混乱 → Adam 的二阶矩估计不准确（因为样本太少）→ 大 LR 会导致 loss spike。

**标准策略**：

$$
\text{lr}(t) = \begin{cases}
\text{lr}_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\
\text{lr}_{\max} \cdot \text{decay}(t) & t > T_{\text{warmup}}
\end{cases}
$$

- 常见 $T_{\text{warmup}} = 2000$ steps（GPT-3）
- Cosine decay 是主流：$\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})(1 + \cos(\pi t / T))$

**WSD Schedule**（Warmup-Stable-Decay）：DeepSeek-V3 采用。训练大部分时间保持恒定 LR，最后 20% 快速 decay。好处：中间可以随时 checkpoint + 续训。

### 8.4 混合精度训练

**FP32 → BF16 → FP8 演进**：

| 格式 | 指数位 | 尾数位 | 范围 | 精度 |
|------|--------|--------|------|------|
| FP32 | 8 | 23 | ±3.4e38 | 7 位有效数字 |
| FP16 | 5 | 10 | ±65504 | 3 位有效数字 |
| BF16 | 8 | 7 | ±3.4e38 | 2 位有效数字 |
| FP8 E4M3 | 4 | 3 | ±448 | 1-2 位有效数字 |
| FP8 E5M2 | 5 | 2 | ±57344 | 1 位有效数字 |

**BF16 为什么优于 FP16？**
- FP16 范围小（max 65504），attention logits 容易溢出 → 需要 loss scaling
- BF16 范围与 FP32 相同，无需 loss scaling
- BF16 精度低于 FP16，但实验表明对 LLM 训练影响极小

**混合精度策略**：
- 权重 master copy：FP32
- 前向/反向计算：BF16
- 权重更新（optimizer state）：FP32
- 通信（AllReduce）：BF16 或 FP32

**FP8 训练**（DeepSeek-V3）：
- Forward pass 中 GEMM 用 FP8
- 动态 per-tensor scaling：每个 tensor 维护一个 scale factor
- Block-wise quantization：将矩阵分块，每块独立量化（减少异常值影响）
- Backward 仍用 BF16（梯度对精度更敏感）

### 8.5 训练稳定性 Checklist（工程实践）

| 技术 | 作用 | 典型配置 |
|------|------|----------|
| Pre-RMSNorm | 梯度直通 | 所有 2024+ 模型 |
| 残差连接 | 恒等梯度路径 | 必选 |
| Warmup | 避免初期 loss spike | 2000 steps |
| Gradient Clipping | 防止梯度爆炸 | max norm = 1.0 |
| BF16 混合精度 | 性能 + 稳定性 | 主流 |
| μP (maximal update) | 超参数可迁移 | DeepSeek-V3 采用 |
| Weight Decay | 正则化 + 防止权重爆炸 | 0.1 |
| QK-Norm | 防止 attention logits 过大 | 部分模型采用 |

---

## 9. 架构范式对比：Decoder-only vs Encoder-Decoder vs Prefix LM

### 9.1 三种架构的注意力模式

**Encoder-Decoder**（T5, BART）：

```
Encoder:  [x1, x2, x3] — 双向 attention（全可见）
Decoder:  [y1, y2, y3] — causal attention + cross-attention to encoder
```

$$
\text{Cross-Attention}: Q = h_{\text{dec}}, \quad K = V = h_{\text{enc}}
$$

**Decoder-only**（GPT, LLaMA, DeepSeek）：

```
[x1, x2, x3, y1, y2, y3] — causal attention（下三角 mask）
```

**Prefix LM**（GLM, U-PaLM）：

```
[x1, x2, x3 | y1, y2, y3]
 prefix: 双向   suffix: causal
```

### 9.2 为什么 Decoder-only 成为主流？

| 维度 | Decoder-only 优势 | Encoder-Decoder 优势 |
|------|-------------------|---------------------|
| 预训练效率 | 每个 token 都参与 loss | 只有 decoder 侧参与 loss |
| 工程简洁性 | 单一 forward path | 需要两个 forward path |
| Scaling law | 参数全用于一个模型 | 参数分散在 encoder + decoder |
| In-Context Learning | 天然支持 few-shot | 需要特殊设计 |
| KV Cache | 只有一套 | Encoder 的 KV 也要缓存 |

**深层原因**：

1. **统一性**：Decoder-only 将所有 NLP 任务（分类、生成、翻译、摘要）统一为自回归生成，省去任务特定设计
2. **Scaling 友好**：Chinchilla [37] 等 scaling law 研究基于 decoder-only，经验更丰富
3. **Emergence**：大规模 decoder-only 模型表现出涌现能力（ICL、CoT），encoder-decoder 在相同参数量下表现弱

**Encoder-Decoder 仍有优势的场景**：
- 明确的输入-输出结构（翻译、摘要）
- 输入远长于输出（文档 QA）：encoder 的双向 attention 对输入理解更好
- T5-style span corruption 预训练在 NLU 任务上仍有优势

### 9.3 Prefix LM 的定位

- 介于两者之间：prefix 部分双向（类似 encoder），suffix 部分自回归（类似 decoder）
- **优势**：单一模型但保留了对输入的双向理解
- **劣势**：prefix 部分无法参与 next-token prediction loss → 预训练效率低于纯 causal
- **代表**：GLM-130B、U-PaLM

---

## 10. 2026 前沿架构

### 10.1 Mixture of Depths (MoD)

**核心论文**：Raposo et al. (2024) [38]

**问题**：标准 Transformer 中每个 token 经过完全相同的计算量。但不同 token 的难度不同——一些 token（如标点、常见词）可能不需要那么多计算。

**核心思想**：在每层设置一个 **capacity** $C < n$，只有最 "重要" 的 $C$ 个 token 经过该层的子层计算，其余 token 直接 skip（只走残差路径）。

$$
x'_i = \begin{cases}
x_i + F_l(x_i) & \text{if } r_i \in \text{TopK}(r, C) \\
x_i & \text{otherwise}
\end{cases}
$$

$$
r_i = x_i^\top w_r \quad \text{(learned routing score)}
$$

**与 MoE 的区别**：MoE 选择 "用哪个 expert"，MoD 选择 "是否计算"。

**收益**：
- 相同性能下 FLOPs 减少 12.5%-50%
- 或相同 FLOPs 下性能提升（将节省的计算分配给更多参数）

**工程挑战**：
- Top-K routing 导致 token 顺序被打乱 → 需要排序恢复
- 不同层跳过不同的 token → 并行化困难

### 10.2 Hyper-Connections

**核心论文**：Zhu et al., "Hyper-Connections" (2024) [39]

**问题**：标准残差连接是 $x' = x + F(x)$，只有一条恒等路径。当网络很深时，浅层信息经过层层残差加法后可能被稀释。

**核心思想**：将隐状态扩展为 $n_E$ 个 "分身"（expansion），每层的连接不再是简单加法，而是一个可学习的线性变换：

$$
\begin{pmatrix} \tilde{x}^{(1)} \\ \vdots \\ \tilde{x}^{(n_E)} \end{pmatrix} = A \begin{pmatrix} x^{(1)} \\ \vdots \\ x^{(n_E)} \end{pmatrix}, \quad x_{\text{input}} = \sum_{i} \alpha_i \tilde{x}^{(i)}
$$

$$
x^{(j)}_{\text{output}} = \beta_j F(x_{\text{input}}) + \tilde{x}^{(j)}
$$

- $A$ 是 $n_E \times n_E$ 的可学习连接矩阵
- 浅层信息可以通过专门的通道直达深层（无需累积残差加法）

**直觉**：类比 DenseNet 的 dense connection，但参数化更轻量。

**实验结果**：在 LLM 预训练中优于标准残差、DenseNet connection、SubLN。

**Manifold-Constrained Hyper-Connections (mHC)**：在 hyper-connections 基础上加入流形约束（正交约束等），进一步提升稳定性和表达能力。

### 10.3 Dynamic Token Merging (ToMe)

**核心论文**：Bolya et al. (2023) [40]；2025-2026 系列改进

**问题**：视觉 Transformer（ViT）和 LLM 中，许多 token 在经过若干层后变得高度相似——它们携带的信息冗余。

**核心思想**：在推理/训练过程中，动态合并相似的 token：

1. 计算 token 之间的相似度（通常用 attention key 的余弦相似度）
2. 配对最相似的 $r$ 对 token
3. 合并为一个 token（加权平均 or concatenation）

$$
x_{\text{merged}} = \frac{n_a x_a + n_b x_b}{n_a + n_b}
$$

其中 $n_a, n_b$ 是被合并 token 各自代表的原始 token 数。

**2026 进展——LLM 中的 Token Merging**：
- 在 KV Cache 层面合并相似的 key-value 对 → 减少 cache 大小
- 与 attention sink（StreamingLLM）结合：保留头部 sink tokens + 最近窗口 + 合并中间
- 用于 VLM（视觉语言模型）中压缩图像 token 数量

### 10.4 其他前沿方向（2025-2026）

**Mixture of Agents**：多个 LLM 作为 "experts"，协作生成响应（不同于 MoE 的 FFN experts）。

**Linear Attention 复兴**：
- TransNormerLLM：$O(n)$ attention + 改进的归一化
- Based：Linear attention + sliding window quadratic attention 混合
- RetNet：retention 机制（衰减注意力 + chunk-wise recurrence）

**Test-Time Compute Scaling**：
- 不增加模型参数，增加推理计算（如 o1 的思维链）
- 与架构的关系：需要 Decoder-only 的自回归能力

---

## 11. 面试题集

### 基础级（1-5）

#### Q1: Self-Attention 为什么除以 $\sqrt{d_k}$？如果不除会怎样？

**答**：假设 $q, k$ 的各分量独立同分布、均值 0、方差 1。点积 $q^\top k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$（独立随机变量乘积和）。当 $d_k$ 大（如 128）时，点积的绝对值可能达到 $\sqrt{d_k} \approx 11$，softmax 输入值过大 → 输出趋于 one-hot → 梯度趋于 0。

除以 $\sqrt{d_k}$ 将方差归一化到 1。不除的后果：(1) softmax 饱和、梯度消失 (2) attention 变 hard（只关注一个 token）(3) 训练早期发散或收敛极慢。

本质是 temperature scaling：$T = \sqrt{d_k}$。

#### Q2: Multi-Head Attention 相比 Single-Head 的优势？计算量变了吗？

**答**：计算量不变（$O(n^2 d)$）。优势：(1) **子空间多样性**——不同 head 学到不同注意力模式（位置、语法、语义等），Voita et al. 通过剪枝分析发现有 positional head、syntactic head 和 rare word head (2) **秩提升**——单头 attention 矩阵秩 ≤ $d_k$，$h$ 个 head 可表示秩 ≤ $hd_k = d_{\text{model}}$ 的注意力 (3) **鲁棒性**——分散了被极端 token 主导的风险。

计算量不变是因为 $d_k = d_{\text{model}} / h$，总矩阵乘尺寸相同。KV Cache 是 $2hd_kn = 2dn$，也不变。

#### Q3: RoPE 的核心原理是什么？为什么优于 learned position embedding？

**答**：将 $d$ 维向量视为 $d/2$ 个二维子空间，对位置 $m$ 的向量在每个子空间旋转角度 $m\theta_i$。核心性质：$\langle R_m q, R_n k \rangle = g(q, k, m-n)$，即 attention score 自然编码相对位置。

优于 learned PE 的原因：(1) 无额外参数 (2) 理论上可外推（配合 YaRN）(3) 编码相对位置而非绝对位置 (4) 不占 embedding 维度。

频率设计 $\theta_i = 10000^{-2i/d}$：低维度高频（局部关系）、高维度低频（远程关系），类似傅里叶变换。

#### Q4: Pre-Norm 和 Post-Norm 的区别？为什么 2026 全是 Pre-Norm？

**答**：Post-Norm: $\text{LN}(x + F(x))$，Pre-Norm: $x + F(\text{LN}(x))$。

Pre-Norm 的优势：残差路径上无非线性变换，梯度直通。Post-Norm 的 LayerNorm 的 Jacobian 可能导致梯度不稳定。Pre-Norm 不需要 warmup 或特殊初始化即可训练深层网络。

2026 共识：Pre-RMSNorm（Pre-Norm + RMSNorm 替代 LayerNorm）。RMSNorm 省去 mean centering，计算快 ~10%，性能无损。

Post-Norm 唯一优势：表示质量略好（归一化约束更强），但训练难度大。DeepNorm 通过放大残差 + 缩小初始化解决了超深 Post-Norm 的训练问题。

#### Q5: SwiGLU 的公式是什么？为什么 $d_{\text{ff}} = \frac{8}{3}d$？

**答**：$\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$，三个权重矩阵。Swish$(x) = x \cdot \sigma(x)$。

$d_{\text{ff}} = 8d/3$ 是参数补偿：标准 FFN 有 2 个矩阵 $d \times 4d$ → 参数 $8d^2$。SwiGLU 有 3 个矩阵 $d \times d_{\text{ff}}$ → 参数 $3d \cdot d_{\text{ff}}$。令 $3d \cdot d_{\text{ff}} = 8d^2$ → $d_{\text{ff}} = 8d/3$。

为什么比 ReLU 好：(1) 门控机制选择性过滤信息 (2) Swish 平滑非单调，梯度传播好 (3) PaLM 系统实验证实在相同参数量下 PPL 一致更优。

---

### 进阶级（6-10）

#### Q6: FlashAttention 降低了计算复杂度吗？它到底优化了什么？

**答**：**没有降低计算复杂度**——仍是 $O(n^2 d)$ FLOPs。优化的是 **IO 复杂度**。

标准 attention 需要将 $n \times n$ 的 attention 矩阵从 HBM 读写多次。FlashAttention 通过分块（tiling）在 SRAM 中完成计算：

1. 将 Q、K、V 分成块，逐块在 SRAM 计算
2. 用 online softmax 在不知道全局最大值的情况下递增地计算精确 softmax
3. 不需要将 $n \times n$ 矩阵实际化到 HBM

IO 复杂度从 $O(n^2 d + n^2)$ 降到 $O(n^2 d^2 / M)$（$M$ = SRAM 大小）。显存从 $O(n^2)$ 降到 $O(n)$。实际速度提升 2-4×，因为 attention 是 memory-bound 操作。

FlashAttention-3 的 H100 优化：利用 Tensor Core 异步性做计算-通信 overlap + FP8 block quantization。

#### Q7: GQA、MQA、MLA 各自的 trade-off？DeepSeek-V3 的 absorption trick 怎么工作？

**答**：

- **MHA**：$h$ 组独立 K、V，表达力最强，KV Cache 最大（$2hd_k n$ per layer）
- **MQA**：所有 head 共享一组 K、V，Cache 缩 $h$ 倍，质量有损
- **GQA**：$g$ 组 K、V，每组服务 $h/g$ 个 Q head。$g=1$ 即 MQA，$g=h$ 即 MHA。LLaMA-2 70B 用 $g=8$
- **MLA**：将 K、V 压缩到低维 latent $c_{kv} \in \mathbb{R}^{d_c}$，Cache 只存 $c_{kv}$

Absorption trick：$q^\top k = (xW^Q)(c_{kv}W^{UK})^\top = x(W^Q W^{UK\top})c_{kv}^\top$。将 $W^{UK}$ 吸收进 Q 投影，直接在 latent 空间做 attention，不需要展开 K。

RoPE 兼容问题：RoPE 施加位置旋转后 $W^{UK}$ 无法被吸收。解决方案：Decoupled RoPE keys——额外分出低维 keys ($d_r=64$) 专门承载 RoPE。

DeepSeek-V3 的 MLA：Cache = $d_c + d_r = 512 + 64 = 576$，MHA 需要 $h \times d_k = 128 \times 128 = 16384$。**压缩 28.5 倍**。

#### Q8: Speculative Decoding 的数学保证是什么？为什么输出分布严格等于 target model？

**答**：核心是**修正拒绝采样**。Draft model 采样 $x \sim q(x)$，acceptance probability：

$$
P(\text{accept}) = \min\left(1, \frac{p(x)}{q(x)}\right)
$$

被拒绝时从修正分布采样：$p'(x) = \text{normalize}(\max(0, p(x) - q(x)))$。

**证明输出等价于 $p(x)$**：对任意 token $x$：

$$
P(\text{output} = x) = q(x) \cdot \min(1, p(x)/q(x)) + (1 - \alpha) \cdot p'(x)
$$

其中 $\alpha = \sum_x q(x) \min(1, p(x)/q(x))$ 是总接受率。展开可验证 $P(\text{output} = x) = p(x)$。

加速比取决于 draft-target 对齐度。草稿长度 $\gamma$ 的最优选择需要平衡：$\gamma$ 大 → 更多 token 一次验证但接受率低；$\gamma$ 小 → 接受率高但每次验证的收益少。

#### Q9: MoE 的负载均衡问题怎么解决？DeepSeek-V3 为什么不用 auxiliary loss？

**答**：负载不均衡的危害：少数 expert 过载（成为瓶颈）→ 训练速度下降 + expert 坍塌（只有少数 expert 被训练）。

**方案对比**：

1. **Auxiliary Load Balancing Loss**（GShard/Switch）：$\mathcal{L}_{\text{aux}} = \alpha N \sum f_i P_i$。$f_i$ 是实际 token 比例，$P_i$ 是 gating probability。均匀时最小。**问题**：干扰主训练目标，$\alpha$ 需要调参，过大会牺牲模型质量。

2. **Expert Choice**：expert 主动选 token，天然均衡但 token 可能被 0 个 expert 选中。

3. **DeepSeek-V3 Bias-based**：$g'_i = g_i + b_i$，bias 按实际负载动态调整。负载不足的 expert 的 bias 增大，吸引更多 token。好处：**完全不干扰 gating 的梯度信号**，不像 auxiliary loss 那样在 loss 中加噪声。

DeepSeek-V3 另外还有 Shared Expert（始终激活），保证基线知识不依赖路由。

#### Q10: Mamba 的 Selective SSM 和标准 SSM 的关键区别？为什么纯 Mamba 不能替代 Attention？

**答**：标准 SSM（S4）：$A, B, C$ 是固定参数 → 线性时不变（LTI）→ 可以用卷积加速训练但无法根据内容决定记忆/遗忘。

Mamba（Selective SSM）：$B_t = f_B(x_t), C_t = f_C(x_t), \Delta_t = f_\Delta(x_t)$ → input-dependent → 可以"选择性"地记忆重要信息、遗忘无关信息。类比 LSTM 门控但更高效（并行 scan）。

纯 Mamba 不能替代 Attention 的原因：

1. **精确检索瓶颈**：SSM 的隐状态是压缩表示（固定维度 $N$），无法精确回忆任意历史 token。Needle-in-a-haystack 测试中 Mamba 明显弱于 Attention。
2. **ICL 依赖精确记忆**：Few-shot learning 需要精确记住示例的输入-输出映射，压缩表示不够。
3. **FlashAttention 已大幅缩小差距**：Attention 的实际 wall-clock time 比理论 $O(n^2)$ 好很多。

最优解：混合架构（Jamba 3:1 Mamba:Attention，DeepSeek 的 SSM-Attention hybrid）。

---

### 专家级（11-18）

#### Q11: 如何从零设计一个 2026 年的 7B LLM 架构？逐层分析每个设计选择。

**答**：

**整体架构**：Decoder-only，Pre-RMSNorm，SwiGLU FFN，GQA。

详细配置：

| 组件 | 选择 | 理由 |
|------|------|------|
| 架构范式 | Decoder-only | 统一生成范式、scaling law 成熟 |
| 层数 × 宽度 | 32L × 4096d | 7B 参数的典型配置 |
| 注意力 | GQA ($h=32, g=8$) | KV Cache 4× 压缩，质量接近 MHA |
| 位置编码 | RoPE ($\theta=500000$) | 原生支持长上下文 |
| 长度扩展 | YaRN | 从 4K 扩展到 128K+ |
| FFN | SwiGLU, $d_{\text{ff}} = 11008$ | 约 $8/3 \times 4096 \approx 10923$，对齐到 128 |
| 归一化 | Pre-RMSNorm | 训练稳定、计算快 |
| 注意力 head dim | 128 | $d_k = d/h = 4096/32 = 128$ |
| Vocab size | 128K | 多语言 + 代码 |
| Tie embedding | 否 | 7B+ 模型 untied 更好 |
| Context length | 4K 训练 → 128K 扩展 | 长上下文渐进训练 |
| 精度 | BF16 训练 | 无需 loss scaling |

**如果要 MoE 版本**（类似 Mixtral）：FFN 替换为 8 experts × SwiGLU + top-2 routing + 1 shared expert。总参数 ~46B，活跃 ~13B。

**进阶选择**：MLA（如果追求极致 KV Cache 压缩）、Mixture of Depths（如果追求推理效率）。

#### Q12: YaRN 的 NTK-by-parts 具体怎么分频段？为什么高频不动、低频大幅缩放？

**答**：

每个 RoPE 维度 $i$ 有波长 $\lambda_i = 2\pi / \theta_i = 2\pi \cdot b^{2i/d}$（$b = 10000$）。

**分频段规则**：

- **高频**（$\lambda_i < L_{\text{train}}$）：这些维度在训练长度内已经完成多个完整周期，模型已充分学习。外推时角度仍在训练分布内。→ **不缩放**。
- **低频**（$\lambda_i > L_{\text{target}}$）：这些维度即使在目标长度内也不到一个周期。→ **线性插值**（Position Interpolation）。
- **中间**：用 ramp function 在 "不缩放" 和 "线性插值" 之间平滑过渡。

**物理直觉**：高频 → 编码局部关系（如相邻 token 的语法关系），这在任何长度下都是一样的 → 不应改变。低频 → 编码远程关系，超出训练长度后确实需要重新校准。

attention temperature $t = \sqrt{1 + \frac{1}{d}\ln(s)}$（$s$ = 缩放比）：长序列时 attention 分布因维度增加变 flat → 需要小幅锐化。

#### Q13: PagedAttention 的 block table 机制具体怎么工作？和 OS 的虚拟内存有什么异同？

**答**：

**相似之处**：
- 逻辑地址 → 物理地址的映射（Block Table ≈ Page Table）
- 按需分配（不预分配最大长度）
- 内存碎片大幅减少

**Block Table 机制**：
- KV Cache 按 block size（如 16 tokens）分配
- 每个序列维护一个 block table：`logical_block_id → physical_block_ptr`
- 新 token 填满当前 block 后，分配新 physical block
- Attention kernel 通过 block table 间接寻址

**与 OS 虚拟内存的区别**：
- 无 TLB（Translation Lookaside Buffer）：GPU 上直接查表
- 无 page fault handler：显存不足时直接 reject 请求（或 swap to CPU）
- **Copy-on-Write** 的具体应用：beam search 中多个 beam 共享前缀 block，分叉时才复制
- **Prefix Caching**：多个请求共享相同系统 prompt 的 block

**碎片分析**：传统预分配浪费 60-80%，PagedAttention 仅在最后一个 block 有内部碎片（平均浪费 $B/2$ tokens），利用率 >96%。

#### Q14: Ring Attention 如何处理 Causal Mask 导致的负载不均衡？

**答**：标准 Ring Attention 按顺序分配 token 到设备。在 causal LM 中，前面的设备（持有序列开头的 Q）需要 attend to 的 KV 少，后面的设备需要 attend to 的 KV 多 → 负载不均衡。

**Striped Ring Attention**：交替分配 token。设备 $p$ 持有 token $\{p, p+P, p+2P, \dots\}$。这样每个设备持有的 token 在序列中均匀分布，每个设备需要处理的 attention 计算量大致相同。

**通信 overlap 的条件**：当 $n/P$ 足够大时（每个设备的 block size > 某个阈值），计算时间 > 通信时间，可以完美 overlap。threshold 取决于 interconnect bandwidth 和计算速度。

**实践中**：Ring Attention 主要用于极长上下文训练（>1M tokens），常规推理用 Tensor Parallelism + PagedAttention 更高效。

#### Q15: DeepSeek-V3 的 FP8 训练是怎么做到的？和 BF16 训练的关键区别？

**答**：

**FP8 的挑战**：E4M3 只有 1-2 位有效数字（精度极低），E5M2 范围大但精度更低。直接替换 BF16 → loss divergence。

**DeepSeek-V3 的方案**：

1. **只在 forward GEMM 用 FP8**：$Y = X_{fp8} \cdot W_{fp8}$，计算结果回到 BF16
2. **Backward 仍用 BF16**：梯度对精度更敏感
3. **Per-tensor dynamic scaling**：每个 tensor 维护一个 scale factor $s$，$X_{fp8} = \text{round}(X_{bf16} / s)$，$s$ 根据 tensor 的 max 值动态计算
4. **Block-wise quantization**：将矩阵分成小块（如 128×128），每块独立计算 scale → 减少异常值对整体精度的影响
5. **精度敏感层保持 BF16**：embedding、最后的 output projection、LayerNorm

**收益**：训练速度提升 ~40%（H100 的 FP8 Tensor Core throughput 是 BF16 的 2×），显存减少 ~25%。

**与 BF16 的关键区别**：BF16 有 8 位指数（范围大），7 位尾数（精度尚可）。FP8 E4M3 只有 3 位尾数，必须配合 per-tensor scaling 才能避免溢出/下溢。

#### Q16: Mixture of Depths 和 Early Exit 有什么区别？各自的优劣势？

**答**：

**Early Exit**：token 在某层 "提前退出"，之后所有层都跳过。

$$
y_i = F_l(x_i) \quad \text{if confidence > threshold at layer } l
$$

**Mixture of Depths**：每层独立决定哪些 token 计算、哪些跳过。token 可能在第 3 层被跳过但在第 7 层被计算。

**关键区别**：

| 维度 | Early Exit | Mixture of Depths |
|------|-----------|-------------------|
| 粒度 | 整个剩余网络 | 逐层 |
| 灵活性 | 低（一旦退出不回来） | 高（每层独立决策） |
| 实现 | 每层加 classifier head | 每层加 routing score |
| 训练 | 需要多头训练 | End-to-end 训练 |
| 批处理 | 困难（不同 token 不同深度） | 相对容易（TopK 选择） |

MoD 的优势：更细粒度的计算分配（有的层需要 attention 但不需要 FFN），对 batched inference 更友好。

MoD 的挑战：TopK routing 引入不确定性（训练时需要 straight-through estimator），token 顺序被打乱。

#### Q17: 从信息论角度分析：为什么 Attention 是 $O(n^2)$ 而不能更低？SSM 的 $O(n)$ 牺牲了什么？

**答**：

**Attention 的 $O(n^2)$ 本质**：Self-Attention 计算所有 token 对之间的交互 → $\binom{n}{2}$ 对 → $O(n^2)$。这是实现**完全交互**（任意两个 token 可以直接通信）的代价。

**信息论视角**：长度 $n$ 的序列的互信息矩阵 $I(x_i; x_j)$ 有 $O(n^2)$ 个条目。要精确建模所有 pairwise 依赖，$O(n^2)$ 是信息论下界。

**SSM 的 $O(n)$ 牺牲了什么**：
- 隐状态 $h \in \mathbb{R}^N$ 是序列历史的**有损压缩**（$N$ 固定，如 16 维）
- 信息瓶颈：$I(y_t; x_1, \dots, x_t) \leq \log |\mathcal{H}|$，其中 $\mathcal{H}$ 是隐状态空间
- 无法精确存储/检索任意历史 token（lossy memory）
- 对于需要精确 copy/retrieval 的任务（如 needle-in-a-haystack），SSM 理论上就有局限

**稀疏 Attention 的中间地带**：$O(n \cdot w)$ 或 $O(n \sqrt{n})$，保留部分 pairwise 交互。BigBird 证明了 random + local + global 的稀疏模式在理论上仍是图灵完备的。

#### Q18: 如果让你设计一个 1T 参数的 MoE 模型，你会怎么设计架构和并行策略？

**答**：

**架构设计**：

| 组件 | 选择 |
|------|------|
| 总参数 | 1T（~100B 活跃） |
| 层数 | 96 层 |
| 隐藏维度 | 8192 |
| 注意力 | MLA（d_c=1024, d_r=128）或 GQA-16 |
| FFN | MoE: 128 fine-grained experts + 2 shared experts, top-4 routing |
| 每层 MoE | FFN 层替换，Attention 保持 dense |
| 位置编码 | RoPE (base=1M) + YaRN |
| 归一化 | Pre-RMSNorm + QK-Norm |
| 激活 | SwiGLU |
| 精度 | FP8 forward + BF16 backward |

**并行策略**（假设 2048 × H100）：

| 策略 | 维度 | 作用 |
|------|------|------|
| Expert Parallelism (EP) | 16-way | MoE experts 分布到不同 GPU |
| Tensor Parallelism (TP) | 8-way | Attention + Shared Expert 切分 |
| Pipeline Parallelism (PP) | 16-way | 层间切分 |
| Data Parallelism (DP) | $2048/(16 \times 8 \times 16) = 1$ | ZeRO-1 |

**通信优化**：
- EP 的 All-to-All 使用 NVLink（节点内）+ IB（节点间）
- Overlap All-to-All with computation（dispatch 和 combine 阶段）
- 梯度通信用 BF16，AllReduce 和 DP 之间用 ZeRO-1 减少内存

**训练稳定性**：
- μP 做超参数搜索（在 1B proxy model 上搜索，迁移到 1T）
- WSD learning rate schedule
- Gradient clipping = 1.0
- 监控 expert load balance，动态调整 bias

---

## 附录 A：必背公式表

### 注意力核心

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

$$
\text{MultiHead} = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 位置编码

$$
\text{RoPE: } f(x,m) = R_{\Theta,m} x, \quad R_{m\theta} = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}
$$

$$
\text{ALiBi: } \text{bias}_{ij} = -m|i-j|, \quad m_k = 2^{-8k/h}
$$

### 归一化

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}
$$

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

### 激活

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2, \quad \text{Swish}(x) = x \cdot \sigma(x)
$$

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

### MoE

$$
y = \sum_{i=1}^{K_s} E_i^{\text{shared}}(x) + \sum_{j \in \text{TopK}} g_j \cdot E_j(x), \quad g = \text{TopK}(\text{softmax}(xW_g + b))
$$

$$
\mathcal{L}_{\text{aux}} = \alpha N \sum_{i=1}^N f_i P_i
$$

### SSM

$$
h_t = \bar{A}h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t
$$

### KV Cache

$$
\text{Cache (MHA)} = 2 L d n \cdot \text{bytes}, \quad \text{Cache (GQA-}g\text{)} = 2 L g d_k n \cdot \text{bytes}
$$

### Speculative Decoding

$$
P(\text{accept}) = \min\left(1, \frac{p(x)}{q(x)}\right), \quad p'(x) = \text{norm}\left(\max(0, p(x)-q(x))\right)
$$

---

## 附录 B：架构演进时间线

| 年份 | 里程碑 | 关键创新 |
|------|--------|----------|
| 2017 | Transformer (Vaswani et al.) | Self-Attention, Scaled Dot-Product |
| 2018 | GPT-1, BERT | Decoder-only / Encoder-only pretrain |
| 2019 | GPT-2, Sparse Transformer | 1.5B scale, 稀疏 attention |
| 2020 | GPT-3, T5 | 175B, few-shot ICL, text-to-text |
| 2020 | Longformer, BigBird | 高效长文档 attention |
| 2021 | RoPE, Switch Transformer | 旋转位置编码, top-1 MoE |
| 2021 | GShard | top-2 MoE + auxiliary loss |
| 2022 | Chinchilla | Scaling law: 数据同等重要 |
| 2022 | FlashAttention v1 | IO-aware attention, $O(n)$ 显存 |
| 2022 | ALiBi | 线性位置 bias, 外推能力 |
| 2022 | S4, H3 | 结构化状态空间模型 |
| 2023 | LLaMA-1/2 | 开源 7B-70B, GQA, RoPE |
| 2023 | Mistral-7B | Sliding Window + GQA |
| 2023 | FlashAttention-2 | 更好的并行化 |
| 2023 | Mamba (S6) | Selective SSM, input-dependent |
| 2023 | Position Interpolation | RoPE 长度外推 |
| 2023 | YaRN | NTK-by-parts + temperature |
| 2023 | vLLM / PagedAttention | 虚拟内存式 KV Cache |
| 2023 | Speculative Decoding | 无损推理加速 |
| 2024 | Mixtral 8x7B | 开源 MoE, 46B 总 / 13B 活跃 |
| 2024 | Jamba | Mamba + Attention + MoE 混合 |
| 2024 | DeepSeek-V2/V3 | MLA + Fine-grained MoE + FP8 |
| 2024 | FlashAttention-3 | H100 优化, FP8 support |
| 2024 | Mixture of Depths | 动态计算分配 |
| 2024 | Medusa / EAGLE | 并行 draft heads |
| 2024 | Hyper-Connections | 可学习残差通道 |
| 2025 | LLaMA-4 | 10M+ context, MoE + MLA |
| 2025 | Mamba-2 + Hybrid | SSM-Attention 混合成熟 |
| 2025 | Token Merging for LLM | 动态 token 合并减少计算 |
| 2026 | DeepSeek-V4 (推测) | Dynamic Sparse + Conditional Memory |

---

## 附录 C：常见误区

### 误区 1：FlashAttention 降低了计算复杂度
**纠正**：计算复杂度不变（仍 $O(n^2d)$），优化的是 **IO 复杂度**和显存。速度提升来自减少 HBM 访问。

### 误区 2：MoE 的推理速度和总参数成正比
**纠正**：MoE 推理只激活 top-K experts，FLOPs 与**活跃参数**成正比。671B 总参的 DeepSeek-V3 推理成本接近 37B dense 模型。但 expert 权重需要加载到显存，对**显存**有要求。

### 误区 3：RoPE 自动支持无限长度
**纠正**：RoPE 理论上对任意位置有定义，但实际效果受限于训练长度。超出训练长度后，低频维度的角度超出训练分布 → 性能下降。需要 YaRN / NTK-aware scaling 等方法微调。

### 误区 4：Pre-Norm 一定比 Post-Norm 好
**纠正**：Pre-Norm 训练更稳定，但多项研究表明 Post-Norm 的表示质量略好（尤其在 NLU 任务上）。DeepNorm 证明了深层 Post-Norm 也可以稳定训练。选择取决于训练 budget 和任务。

### 误区 5：Mamba 可以完全替代 Attention
**纠正**：SSM 的固定维度隐状态是信息瓶颈，精确检索（needle-in-a-haystack）和 ICL 能力弱于 Attention。2026 共识是混合架构（少量 Attention 层 + 大量 SSM 层）。

### 误区 6：GQA/MQA 只是 MHA 的 "降级版"
**纠正**：GQA 从 MHA 迁移时可用 mean pooling 初始化 KV heads，性能损失很小（<1% perplexity）。对于推理密集场景，KV Cache 的节省远超质量损失。MLA 更进一步：压缩 + absorption 可以在更小 cache 下保持甚至超越 MHA 质量。

### 误区 7：Speculative Decoding 的加速来自并行生成
**纠正**：加速来自 **verification 的并行性**（一次 forward pass 验证多个 token）。Draft 仍是串行生成。核心收益：target model 的单次 forward pass 成本几乎不随验证 token 数增加（compute-bound 而非 memory-bound）。

### 误区 8：Transformer 的 $O(n^2)$ 无法避免
**纠正**：理论上精确全注意力确实 $O(n^2)$，但实践中：(1) FlashAttention 让 $O(n^2)$ 跑得很快 (2) 稀疏 attention 降到 $O(n\sqrt{n})$ 或 $O(nw)$ (3) SSM 降到 $O(n)$。关键是大多数 NLP 任务不需要所有 token 对之间的精确交互。

### 误区 9：MoE 模型更难训练是因为架构复杂
**纠正**：主要挑战是 **routing 的离散性**（top-K 操作不可微，需要 straight-through estimator）和 **负载不均衡**。架构本身的前向/反向计算和 dense 模型一样。DeepSeek-V3 的 bias-based 方法大幅简化了训练。

---

## 参考文献

[1] Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.

[2] Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." arXiv 2019.

[3] Ainslie, J. et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.

[4] DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv 2024.

[5] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv 2024.

[6] Milakov, M. & Gimelshein, N. "Online normalizer calculation for softmax." arXiv 2018.

[7] Shah, J. et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv 2024.

[8] Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

[9] Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.

[10] Liu, H. et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024.

[11] Press, O. et al. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization." ICLR 2022.

[12] Su, J. et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024.

[13] Chen, S. et al. "Extending Context Window of Large Language Models via Positional Interpolation." arXiv 2023.

[14] Peng, B. et al. "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024.

[15] Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

[16] Leviathan, Y. et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.

[17] Chen, C. et al. "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv 2023.

[18] Cai, T. et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024.

[19] Child, R. et al. "Generating Long Sequences with Sparse Transformers." arXiv 2019.

[20] Beltagy, I. et al. "Longformer: The Long-Document Transformer." arXiv 2020.

[21] Jiang, A.Q. et al. "Mistral 7B." arXiv 2023.

[22] Gu, A. et al. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022.

[23] Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2023.

[24] Lieber, O. et al. "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv 2024.

[25] Fedus, W. et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022.

[26] Lepikhin, D. et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." ICLR 2021.

[27] Zhou, Y. et al. "Mixture-of-Experts with Expert Choice Routing." NeurIPS 2022.

[28] Jiang, A.Q. et al. "Mixtral of Experts." arXiv 2024.

[29] Dai, D. et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." arXiv 2024.

[30] Zhang, B. & Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019.

[31] Wang, H. et al. "DeepNet: Scaling Transformers to 1,000 Layers." arXiv 2022.

[32] Hendrycks, D. & Gimpel, K. "Gaussian Error Linear Units (GELUs)." arXiv 2016.

[33] Dauphin, Y. et al. "Language Modeling with Gated Convolutional Networks." ICML 2017.

[34] Shazeer, N. "GLU Variants Improve Transformer." arXiv 2020.

[35] Chowdhery, A. et al. "PaLM: Scaling Language Modeling with Pathways." JMLR 2023.

[36] Veit, A. et al. "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." NeurIPS 2016.

[37] Hoffmann, J. et al. "Training Compute-Optimal Large Language Models." NeurIPS 2022.

[38] Raposo, D. et al. "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models." arXiv 2024.

[39] Zhu, D. et al. "Hyper-Connections." arXiv 2024.

[40] Bolya, D. et al. "Token Merging: Your ViT But Faster." ICLR 2023.

[41] Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.

[42] Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." arXiv 2022.

---

> **Last Updated**: 2026-02-21
> **See Also**: [[AI/LLM/Architecture/_MOC|Architecture MOC]] · [[Career/AI面试速查手册|AI 面试速查手册]] · [[AI/LLM/Architecture/MoE 深度解析|MoE 深度解析]] · [[AI/LLM/Architecture/Transformer 位置编码|位置编码详解]] · [[AI/LLM/Architecture/FlashAttention|FlashAttention]] · [[AI/LLM/Training/LLM微调实战-2026技术全景|LLM 微调实战 2026]] — 配套：先懂架构再学微调，两文合读=面试全覆盖
