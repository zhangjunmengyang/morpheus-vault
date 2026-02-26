---
brief: "Transformer 架构演进 2026（面试武器版）——从原始 Attention is All You Need 到现代变体（MQA/GQA/RoPE/SSM/MoE）的完整演化图谱；面试 rating 5 标注，Transformer 设计决策的深度参考。"
title: "Transformer 架构演进 2026（面试武器版）"
date: 2026-02-20
domain: AI/Foundations/Architecture
tags: [transformer, architecture, attention, ssm, interview-prep]
rating: 5
status: active
---

# Transformer 架构演进 2026（面试武器版）

> 本文以面试场景为主线，梳理从 Vanilla Transformer 到 2026 年前沿架构的完整演进脉络。每节以"面试官会问"引入，附带数学推导与工程直觉。

---

## 1. Transformer 核心机制回顾

### 面试官会问

> "请从零推导 Self-Attention 的计算过程，解释为什么要除以 $\sqrt{d_k}$，以及 FFN 在 Transformer 中扮演什么角色。"

### 1.1 Self-Attention 机制

给定输入序列 $X \in \mathbb{R}^{n \times d}$，通过三组可学习的投影矩阵得到 Query、Key、Value：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$。

注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**为什么除以 $\sqrt{d_k}$？**

当 $d_k$ 较大时，$Q$ 和 $K$ 的点积结果的方差约为 $d_k$（假设 $Q, K$ 的每个分量独立、均值为 0、方差为 1）。点积值过大会将 softmax 推入梯度极小的饱和区。除以 $\sqrt{d_k}$ 将方差归一化到 1，使 softmax 的梯度保持在有效区间。

> **面试加分点**：这本质上是一个 temperature scaling，$T = \sqrt{d_k}$。有些工作（如 Yarn）会在推理阶段动态调整这个 temperature 来适配超长上下文。

### 1.2 Multi-Head Attention (MHA)

将 $d$ 维空间拆分为 $h$ 个子空间，每个 head 独立计算注意力再拼接：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**关键直觉**：不同 head 可以学到不同类型的依赖关系——有的关注局部语法，有的捕捉远距离语义依赖，有的追踪共指关系。

### 1.3 Position-wise Feed-Forward Network (FFN)

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$，通常 $d_{ff} = 4d$。

**FFN 的角色**：
- Attention 负责 **token 间的信息混合**（mixing across positions）
- FFN 负责 **逐位置的特征变换**（per-position transformation）
- 有研究（Geva et al., 2021）将 FFN 解释为 key-value memory：$W_1$ 的行是 key pattern，$W_2$ 的列是 value memory

### 1.4 LayerNorm

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu, \sigma^2$ 在最后一维（hidden dimension）上计算。与 BatchNorm 不同，LayerNorm 不依赖 batch 统计量，天然适用于变长序列和自回归生成。

### 1.5 完整 Transformer Block

```
x → LayerNorm → Multi-Head Attention → + (residual) → LayerNorm → FFN → + (residual) → output
```

> **面试高频陷阱**：原始论文用 Post-Norm（先 attention 再 norm），但现代大模型几乎全部用 Pre-Norm（先 norm 再 attention）。原因见第 4 节。

---

## 2. 位置编码演进

### 面试官会问

> "为什么 Transformer 需要位置编码？RoPE 和 ALiBi 的原理是什么？如何做到长度外推（length extrapolation）？"

### 2.1 为什么需要位置编码

Self-Attention 的计算是置换等变的（permutation equivariant）——打乱输入 token 顺序，输出只是相应打乱，不会有任何"位置感知"。必须显式注入位置信息。

### 2.2 演进路线

#### Sinusoidal（Vaswani et al., 2017）

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- **优势**：无需学习参数，理论上可外推到训练时未见的位置
- **劣势**：实际外推效果有限；不同位置的相对关系是隐式编码的

#### Learned Positional Embedding（GPT 系列）

- 直接为每个位置学习一个 $d$ 维向量
- **优势**：实现简单，表达能力强
- **劣势**：无法外推到训练最大长度之外；参数量 $O(L \times d)$

#### RoPE — Rotary Position Embedding（Su et al., 2021）

**核心思想**：通过旋转矩阵编码相对位置，使得 $q_m^\top k_n$ 只依赖相对位置 $m - n$。

对 $d$ 维向量的每一对相邻维度 $(x_{2i}, x_{2i+1})$，应用旋转矩阵：

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$，$m$ 为位置索引。

**关键性质**：

$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = g(q, k, m - n)$$

点积只依赖相对位置差 $m - n$，这正是我们想要的。

- **优势**：自然编码相对位置，与 attention 的点积运算完美兼容，内存开销极小
- **被采用**：LLaMA 全系列、Qwen、DeepSeek、Mistral 等

#### ALiBi — Attention with Linear Biases（Press et al., 2022）

不修改 embedding，直接在 attention score 上加线性偏置：

$$\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} - m \cdot |i - j|\right)$$

其中 $m$ 是每个 head 预设的斜率（geometric sequence），$|i - j|$ 是位置距离。

- **优势**：零外推代价，训练短推理长效果好
- **劣势**：表达能力不如 RoPE；对某些需要精确远程定位的任务稍弱
- **被采用**：BLOOM、MPT 系列

#### YaRN — Yet another RoPE extensioN（Peng et al., 2023）

**问题**：RoPE 在超出训练长度时，高频分量的外推会崩坏。

**方案**：
1. **NTK-aware interpolation**：不是简单地线性拉伸所有频率，而是根据频率高低采用不同插值策略
2. 高频维度（局部位置信息）基本不动
3. 低频维度（全局位置信息）做插值
4. 配合 attention temperature 调整

$$\theta_i' = \theta_i \cdot \alpha^{-2i/d}$$

其中 $\alpha$ 是扩展比例。实际实现中还有分段处理（high-freq/low-freq boundary）。

- **效果**：4K 训练 → 128K 推理，perplexity 保持稳定
- **被采用**：多数需要长上下文的 LLaMA 微调版本

### 2.3 长度外推总结

| 方法 | 训练改动 | 推理外推 | 主要思路 |
|------|---------|---------|---------|
| Sinusoidal | 无 | 理论可以，实际差 | 固定频率函数 |
| Learned | 无 | 不行 | 查表 |
| RoPE | 无 | 有限 | 旋转编码相对位置 |
| ALiBi | 无 | 好 | 线性距离惩罚 |
| RoPE + NTK | 少量微调 | 好 | 频率缩放 |
| YaRN | 少量微调 | 很好 | 分频段插值 + temperature |

---

## 3. 注意力变体

### 面试官会问

> "MHA、MQA、GQA 有什么区别？DeepSeek 的 MLA 是怎么做到又省 KV cache 又不损性能的？"

### 3.1 从 MHA 到 GQA：KV Cache 优化之路

在自回归推理中，每一步需要缓存所有已生成 token 的 Key 和 Value。对于长序列、大 batch 推理，KV cache 是显存瓶颈。

#### Multi-Head Attention (MHA)

- $h$ 个 head，每个 head 有独立的 $W^Q_i, W^K_i, W^V_i$
- KV cache 大小：$O(n \times h \times d_k)$ per layer
- 标准配置，表达能力最强

#### Multi-Query Attention (MQA)（Shazeer, 2019）

- $h$ 个 Query head，但所有 head **共享一组** Key 和 Value
- KV cache 缩小到 $1/h$
- **问题**：共享过激，质量有下降

#### Grouped-Query Attention (GQA)（Ainslie et al., 2023）

- 将 $h$ 个 head 分成 $g$ 组，每组共享一组 KV
- MHA 是 $g = h$ 的特例，MQA 是 $g = 1$ 的特例
- 典型配置：$h = 32, g = 8$，KV cache 缩小 4×
- **被采用**：LLaMA 2/3、Mistral、Qwen2

| 方法 | Q heads | KV heads | KV cache 比例 | 质量 |
|------|---------|----------|--------------|------|
| MHA | $h$ | $h$ | 1× | 最高 |
| GQA | $h$ | $g$ | $g/h$ | 接近 MHA |
| MQA | $h$ | 1 | $1/h$ | 有损 |

### 3.2 Multi-Head Latent Attention (MLA) — DeepSeek V2/V3

**核心创新**：用低秩压缩代替分组共享。

传统 KV cache 存每层每 head 的 $k_i, v_i$ 向量。MLA 的做法：

**1. 联合压缩 KV**

$$c_{kv} = W^{DKV} \cdot x \in \mathbb{R}^{d_c}$$

其中 $d_c \ll h \cdot d_k$（例如 DeepSeek V3 中 $d_c = 512$，而 $h \cdot d_k = 128 \times 128 = 16384$）。

然后从 $c_{kv}$ 还原出所有 head 的 K 和 V：

$$k_i = W^{UK}_i \cdot c_{kv}, \quad v_i = W^{UV}_i \cdot c_{kv}$$

**2. KV Cache 只存 $c_{kv}$**

- 缓存大小从 $O(n \times h \times d_k \times 2)$ 降到 $O(n \times d_c)$
- DeepSeek V3: 压缩率 > 30×

**3. 吸收技巧（Absorption Trick）**

在推理时，将 $W^{UK}_i$ "吸收" 进 $W^Q_i$：

$$q_i^\top (W^{UK}_i \cdot c_{kv}) = (W^{UK\top}_i q_i)^\top c_{kv}$$

这样可以直接用 $c_{kv}$ 做 attention，不需要显式展开 K。

**4. 对 RoPE 的适配**

RoPE 需要作用在 K 上，但 K 被压缩了。解决方案：为每个 head 额外维护一个小的 RoPE 专用 key（decoupled RoPE keys），维度很小（如 64 维），与 $c_{kv}$ 解耦。

$$k_i^{rope} = \text{RoPE}(W^{KR} x), \quad k_i^{final} = [W^{UK}_i c_{kv}; k_i^{rope}]$$

> **面试加分点**：MLA 的精髓在于——GQA 通过减少 KV head 数来省 cache，MLA 则通过低秩投影把所有 head 的 KV 信息压缩到一个潜在向量中，理论上信息保留更完整。

### 3.3 Sliding Window Attention

- 每个 token 只关注前后 $w$ 个 token 的窗口
- 复杂度从 $O(n^2)$ 降到 $O(n \cdot w)$
- 通过多层堆叠，实际感受野 = $w \times L$（$L$ 为层数）
- **被采用**：Mistral 7B（$w = 4096$）、Longformer（混合 global + sliding）

### 3.4 Dilated Attention

- 类似 dilated convolution 的思路，以固定间隔采样 attention 位置
- Dilation rate $r$：关注位置 $\{i, i \pm r, i \pm 2r, \ldots\}$
- 不同 head 用不同 dilation rate → 兼顾局部和全局
- **代表**：LongNet（Microsoft, 2023）

---

## 4. 归一化与激活函数

### 面试官会问

> "Pre-Norm 和 Post-Norm 有什么区别？为什么现代大模型都用 Pre-Norm + RMSNorm？SwiGLU 是什么，为什么它比 ReLU 好？"

### 4.1 Post-Norm vs Pre-Norm

#### Post-Norm（原始 Transformer）

```
x → Attention → + (residual) → LayerNorm → FFN → + (residual) → LayerNorm
```

- 梯度必须"穿过" LayerNorm 才能到达残差路径
- 深层网络训练不稳定，需要 warmup

#### Pre-Norm（GPT-2 以后的主流）

```
x → LayerNorm → Attention → + (residual) → LayerNorm → FFN → + (residual)
```

- 残差路径上没有任何非线性操作，梯度可以无阻碍地回传
- 训练更稳定，不需要 warmup
- **代价**：有理论分析指出 Pre-Norm 的表达能力上界略低于 Post-Norm（因为残差路径"太通畅"，深层表示趋同）

> **2025 新进展**：有些工作尝试 "Post-Norm + 改进初始化"（如 DeepNorm: $\alpha \cdot x + \text{sublayer}(x)$, 其中 $\alpha > 1$），声称兼得两者之长。但工程上 Pre-Norm 仍是绝对主流。

### 4.2 RMSNorm（Zhang & Sennrich, 2019）

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

- 去掉了 LayerNorm 中的 **减均值** 和 **bias** 操作
- 计算量减少约 20%~30%
- 实验表明效果与 LayerNorm 持平或更优
- **被采用**：LLaMA 全系列、Qwen、DeepSeek、Gemma

### 4.3 激活函数演进

#### ReLU → GELU → SwiGLU

**ReLU**：$\text{ReLU}(x) = \max(0, x)$
- 简单高效，但"死神经元"问题，且不平滑

**GELU**（Hendrycks & Gimpel, 2016）：
$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$$
- 平滑版 ReLU，BERT/GPT 时代标准

**SwiGLU**（Shazeer, 2020）：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$$

其中 $\text{Swish}(x) = x \cdot \sigma(\beta x)$（通常 $\beta = 1$）。

- 引入了 **Gated Linear Unit (GLU)** 机制：两路投影，一路做门控
- 因为有三个矩阵 $W_1, W_2, W_3$，通常将 $d_{ff}$ 调小（如 $\frac{8}{3}d$ 而非 $4d$）保持参数量不变
- **实验结果**：在相同参数量下，SwiGLU 的 PPL 一致优于 ReLU/GELU

**GeGLU**：用 GELU 替代 Swish 做门控，效果类似。

$$\text{GeGLU}(x) = (\text{GELU}(xW_1) \odot xW_3) W_2$$

| 激活函数 | 表达式 | 特点 | 代表模型 |
|---------|--------|------|---------|
| ReLU | $\max(0,x)$ | 简单但有死区 | 早期 Transformer |
| GELU | $x\Phi(x)$ | 平滑 | BERT, GPT-2 |
| SwiGLU | $\text{Swish}(xW_1) \odot xW_3$ | 门控 + Swish | LLaMA, Qwen, DeepSeek |
| GeGLU | $\text{GELU}(xW_1) \odot xW_3$ | 门控 + GELU | PaLM, Gemini |

---

## 5. Mixture of Experts (MoE) 架构

### 面试官会问

> "MoE 的核心思想是什么？路由策略有哪些？DeepSeek MoE 有什么独特设计？为什么 MoE 能用更少的计算达到 dense model 的效果？"

### 5.1 核心思想

将 FFN 层替换为多个 "Expert"（每个 expert 就是一个 FFN），通过 Router 为每个 token 选择 top-$k$ 个 expert 激活。

$$y = \sum_{i \in \text{Top-k}} g_i \cdot E_i(x)$$

其中 $g_i$ 是 router 输出的门控权重，$E_i$ 是第 $i$ 个 expert。

**核心优势**：总参数量大（表达能力强），但每个 token 只激活一小部分（计算高效）。

### 5.2 演进路线

#### GShard（Lepikhin et al., 2020）

- 首次将 MoE 扩展到 600B 参数
- Top-2 routing：每个 token 选 2 个 expert
- **Expert capacity** 机制：每个 expert 处理的 token 数有上限，超出的被丢弃
- 辅助负载均衡损失（auxiliary load balancing loss）

#### Switch Transformer（Fedus et al., 2022）

- Top-1 routing：每个 token 只选 1 个 expert
- 更激进的稀疏化，训练效率更高
- 引入 **capacity factor** $C$：允许 expert 容量有 $C$ 倍冗余
- 简化的负载均衡损失：

$$\mathcal{L}_{balance} = N \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是 expert $i$ 实际接收的 token 比例，$P_i$ 是 router 分配给 expert $i$ 的平均概率。

#### DeepSeek MoE（2024-2025）

DeepSeek MoE 引入了两个关键创新：

**1. Fine-Grained Expert Segmentation**

传统 MoE：8 个大 expert，选 2 个 → 2/8 = 25% 激活

DeepSeek MoE：将每个大 expert 拆成多个小 expert。例如 256 个小 expert，选 8 个 → 激活量相同，但**组合灵活度** $\binom{256}{8} \gg \binom{8}{2}$。

更细粒度的 expert 意味着更灵活的 token-expert 分配。

**2. Shared Expert + Routed Expert**

- **Shared Experts**（$K_s$ 个）：所有 token 必经，学习通用知识
- **Routed Experts**（$N_r$ 个）：通过 router 选择 top-$K_r$ 个，学习专门知识

$$y = \sum_{j=1}^{K_s} E_j^{shared}(x) + \sum_{i \in \text{Top-}K_r} g_i \cdot E_i^{routed}(x)$$

> **面试加分点**：Shared expert 的引入解决了 expert 之间 "knowledge redundancy" 问题——否则每个 expert 都会学一份通用知识的拷贝，浪费容量。

**3. DeepSeek V3 的具体配置**

- 总参数：671B，每 token 激活 37B
- 1 个 Shared Expert + 256 个 Routed Experts，每 token 选 8 个
- 使用 **Auxiliary-Loss-Free Load Balancing**：通过给每个 expert 加一个可学习的 bias term 到 router logits 中，动态调节负载，避免传统 auxiliary loss 对模型质量的干扰

### 5.3 路由策略对比

| 策略 | 做法 | 优劣 |
|------|------|------|
| Top-k + Softmax | softmax 后取 top-k | 标准做法 |
| Expert Choice | 由 expert 选 token（而非 token 选 expert） | 负载天然均衡 |
| Hash Routing | 用 hash 函数分配 | 无需学习，但不灵活 |
| Auxiliary Loss | 额外损失项推动均衡 | 有效但影响主损失 |
| Bias-based (DeepSeek) | 学习 bias 调整 router | 无 aux loss，更优雅 |

---

## 6. 线性注意力与 State Space Models (SSM)

### 面试官会问

> "标准 Attention 的 $O(n^2)$ 问题有哪些解决方案？Mamba 的核心思想是什么？SSM 和 Attention 各有什么优劣？"

### 6.1 标准 Attention 的瓶颈

标准 Self-Attention 的时间和空间复杂度均为 $O(n^2)$，这在长序列（128K+ tokens）场景下成为严重瓶颈。

### 6.2 线性注意力

**核心思想**：用核函数将 softmax attention 分解为线性运算。

标准 attention：$O_i = \frac{\sum_j \exp(q_i^\top k_j) v_j}{\sum_j \exp(q_i^\top k_j)}$

线性 attention（Katharopoulos et al., 2020）：

$$O_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}$$

令 $S = \sum_j \phi(k_j)v_j^\top$（一个 $d \times d$ 矩阵），则可以用 $O(nd^2)$ 完成计算。当 $d \ll n$ 时，这是线性复杂度。

### 6.3 State Space Models (SSM)

SSM 源自连续时间的线性系统理论：

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)$$

离散化后：

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = Ch_t + Dx_t$$

**关键优势**：
- 推理时为 RNN 形式，$O(1)$ per step，KV cache 为固定大小的隐状态
- 训练时可展开为卷积形式，可并行

#### S4（Gu et al., 2022）

- 对 $A$ 矩阵施加 HiPPO 初始化，使隐状态能有效编码历史信息
- 通过对角化加速计算

#### Mamba（Gu & Dao, 2023）

**核心创新 — Selective State Space**：

S4 的 $A, B, C$ 是与输入无关的常数 → 无法做内容感知的选择性处理。

Mamba 让 $B, C, \Delta$（离散化步长）成为输入的函数：

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

- $\Delta_t$ 控制 "遗忘速率"：$\Delta$ 大 → 遗忘历史、关注当前；$\Delta$ 小 → 保留历史
- 这赋予了模型 **选择性记忆** 能力，类似 attention 的内容感知

**硬件优化**：
- 利用 GPU SRAM 的 scan 算法（hardware-aware selective scan）
- 避免 $O(n \times d_{state})$ 的中间状态材化

> **面试关键点**：Mamba 的本质突破是让 SSM 从"线性时不变系统"变成"线性时变系统"，获得了内容选择能力。

#### Mamba-2（Dao & Gu, 2024）

- 揭示 SSM 和 Attention 之间的数学联系：**State Space Duality (SSD)**
- 结构化状态空间模型可以表示为一种受限的注意力矩阵（半可分矩阵, semiseparable matrix）
- 利用矩阵分块算法（chunk-wise parallel scan）进一步加速
- 比 Mamba 快 2-8 倍

### 6.4 RWKV（Peng et al., 2023-2024）

**"线性 Transformer + RNN 训练"**

$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

- $w$ 是每个 channel 的固定衰减率（time decay）
- $u$ 是当前时刻的 bonus
- 可以递推计算 → RNN 推理，$O(1)$ per step
- 训练时可并行

RWKV 的演进：
- RWKV-4: 基础线性注意力 + time decay
- RWKV-5 (Eagle): 引入 multi-head 和矩阵值状态
- RWKV-6 (Finch): 数据依赖的 time decay（类似 Mamba 的选择性）
- RWKV-7: 进一步优化的状态更新机制

### 6.5 RetNet（Sun et al., 2023）

**"Retentive Network"**：三种等价计算范式：
1. **并行模式**：类似 attention，用于训练
2. **递推模式**：类似 RNN，用于推理
3. **分块递推**：折中，用于长序列训练

核心机制 — Multi-Scale Retention：

$$\text{Retention}(Q, K, V) = (QK^\top \odot D) V$$

其中 $D$ 是因果衰减矩阵：$D_{ij} = \gamma^{i-j}$（$i \geq j$ 时），不同 head 用不同的 $\gamma$。

### 6.6 Jamba（AI21 Labs, 2024）

**混合架构的代表作**：在同一模型中交替使用 Attention 层和 Mamba 层。

典型配置（Jamba 1.5）：
- 每 4 层中：3 层 Mamba + 1 层 Attention
- Mamba 层处理大部分序列建模
- Attention 层提供全局信息检索能力
- 还结合了 MoE

**为什么混合？**
- 纯 SSM 在精确检索任务（如 "copying"、需要回溯的推理）上弱于 Attention
- 纯 Attention 在长序列上的 KV cache 开销太大
- 混合架构用少量 Attention 层保底，大部分用 SSM 层省内存

### 6.7 SSM vs Attention 对比

| 维度 | Standard Attention | SSM (Mamba) | 混合架构 |
|------|-------------------|-------------|---------|
| 训练复杂度 | $O(n^2d)$ | $O(nd)$ | 取决于比例 |
| 推理复杂度 | $O(n)$ per step (KV cache) | $O(1)$ per step | 取决于比例 |
| 推理显存 | KV cache $O(n)$ 增长 | 固定隐状态 | 低 |
| 长程依赖 | 强（直接访问所有位置） | 理论可达，实际衰减 | 强 |
| 精确检索 | 强 | 弱 | 中等偏强 |
| 硬件效率 | 成熟优化 (FlashAttention) | 需专用 kernel | 复杂 |

---

## 7. 长上下文技术

### 面试官会问

> "如果要把模型的上下文窗口从 8K 扩展到 1M token，需要解决哪些工程和算法问题？Ring Attention 是怎么工作的？"

### 7.1 挑战

1. **计算**：Self-attention $O(n^2)$，128K 时 attention 计算量巨大
2. **显存**：KV cache 随序列长度线性增长，百万 token 时 TB 级
3. **位置编码**：超出训练长度后性能急剧下降
4. **训练数据**：缺乏超长上下文的高质量训练数据

### 7.2 FlashAttention（Dao et al., 2022-2023）

不是降低复杂度，而是优化内存访问模式：

- **分块计算**：将 $Q, K, V$ 分成小块，在 GPU SRAM 中完成 attention 计算
- **在线 softmax**：增量更新 softmax 的分母，避免两次扫描
- **不存储 attention 矩阵**：$O(n)$ 额外内存而非 $O(n^2)$
- FlashAttention-2: 优化 warp-level 并行
- FlashAttention-3: 利用 Hopper 架构的 TMA 和 FP8

### 7.3 Ring Attention（Liu et al., 2023）

**问题**：单个 GPU 放不下完整的 KV sequence。

**方案**：
1. 将序列均匀切分到 $P$ 个设备
2. 每个设备持有自己的 $Q$ 块（不动）
3. $K, V$ 块在设备之间**环形传递**
4. 每个设备每轮用当前的 $K, V$ 块计算部分 attention，然后将 $K, V$ 传给下一个设备
5. $P$ 轮后，所有设备完成完整的 attention

**核心优势**：
- 通信和计算可以 **重叠**（overlap）
- 序列长度可以线性扩展到设备数 × 单设备容量
- 理论上可支持无限长度

### 7.4 Sequence Parallelism

#### Megatron-style Sequence Parallelism

- 将 LayerNorm 和 Dropout 的计算按序列维度分割
- 与 Tensor Parallelism 互补，减少通信量

#### Ulysses（DeepSpeed, 2023）

- 在 attention 计算前对 $Q, K, V$ 做 all-to-all 通信
- 每个设备负责所有 token 的一部分 head
- 比 Ring Attention 通信效率更高（all-to-all vs ring）

### 7.5 Memory Layers / Memory-Augmented Approaches

- **Memorizing Transformer**（Wu et al., 2022）：用 kNN 检索历史 KV
- **Landmark Attention**：在关键位置放"地标" token，稀疏检索
- **Infini-Attention**（Munkhdalai et al., 2024）：将历史 KV 压缩到固定大小的 compressive memory 中
  - 每个 attention 层维护一个固定大小的记忆矩阵 $M$
  - 新的 KV 通过 delta rule 更新 $M$
  - 查询时从 $M$ 和局部 KV cache 联合检索

### 7.6 长上下文技术栈总结

```
位置编码外推：RoPE scaling / YaRN / ABF
        ↓
计算优化：FlashAttention-3
        ↓
单机序列扩展：Sliding Window + Global tokens
        ↓
多机序列并行：Ring Attention / Ulysses
        ↓
KV Cache 压缩：GQA / MLA / Quantized KV Cache
        ↓
超长记忆：Infini-Attention / Memory Layers
```

---

## 8. 2026 架构趋势

### 面试官会问

> "你认为 2026 年大模型架构的主要趋势是什么？Transformer 会被替代吗？DeepSeek V3 的架构有什么值得关注的设计？"

### 8.1 Hybrid SSM-Attention：混合架构成为主流

2025-2026 年最明确的趋势是 **SSM 和 Attention 的融合**，而非非此即彼：

**代表工作**：
- **Jamba**（AI21）：3:1 的 Mamba:Attention 比例 + MoE
- **Zamba**（Zyphra）：Mamba 为主，共享 Attention 层
- **NVIDIA Hymba**：在每一层中同时运行 SSM 和 Attention 分支

**设计直觉**：
- SSM 擅长 **流式处理**（streaming）、**长序列压缩**、**推理高效**
- Attention 擅长 **精确信息检索**（retrieval）、**in-context learning**
- 最优方案是让两者各司其职

### 8.2 DeepSeek V3 架构深度解析

DeepSeek V3 是 2024-2025 年最具影响力的开源大模型之一，架构上有多项值得深入研究的设计：

**架构全景**：
- 61 层 Transformer
- MLA（Multi-Head Latent Attention）
- DeepSeekMoE：1 Shared Expert + 256 Routed Experts, top-8
- 总参数 671B，激活参数 37B
- Pre-Norm (RMSNorm) + SwiGLU
- RoPE (with decoupled RoPE keys in MLA)

**训练创新**：
- **FP8 Mixed Precision Training**：首次在如此大规模上成功使用 FP8 训练（包括前向和反向传播中的 GEMM 操作），将训练成本降低约 40%
- **Multi-Token Prediction (MTP)**：在每个位置同时预测未来多个 token（而非仅下一个），作为辅助训练目标
  - 提供更密集的训练信号
  - 推理时可用于 speculative decoding 加速
- **Auxiliary-Loss-Free Load Balancing**：通过 per-expert bias 而非辅助损失来平衡负载
- **DualPipe**：优化的流水线并行策略，减少 pipeline bubble

**成本震撼**：
- 训练总成本约 557 万美元（2048 × H800 GPU）
- 训练 token 数：14.8T
- 性能与 GPT-4o、Claude 3.5 Sonnet 可比

### 8.3 Post-Transformer 探索

#### 纯 SSM 路线

- **Mamba 系列**：从语言模型扩展到视觉（Vision Mamba）、音频、多模态
- **RWKV 系列**：持续迭代到第 7 代，证明线性复杂度模型可以达到 Transformer 级别
- **挑战**：在 in-context learning 和精确检索上仍有差距

#### Test-Time Compute / Inference Scaling

- **不只是架构创新**，更是 paradigm shift
- DeepSeek R1, OpenAI o1/o3: 通过在推理时消耗更多计算（chain-of-thought、search）来提升性能
- 对架构的影响：需要高效的长上下文推理、需要动态计算分配

#### 架构搜索与自动化设计

- 传统的 NAS 在 LLM 尺度上不可行
- 新方向：基于 scaling law 预测的架构优化
- 例如：最优的层数、$d_{model}$、$d_{ff}$、head 数等配比

### 8.4 2026 年架构设计共识

**已成定论的最佳实践**：

| 组件 | 2026 标准配置 | 替代方案 |
|------|--------------|---------|
| 归一化 | Pre-RMSNorm | Post-Norm (with DeepNorm) |
| 激活 | SwiGLU ($d_{ff} = \frac{8}{3}d$) | GeGLU |
| 位置编码 | RoPE (+ YaRN/ABF for extension) | ALiBi |
| KV 效率 | GQA 或 MLA | MQA |
| 稀疏化 | MoE (细粒度 + shared expert) | Dense |
| 长上下文 | FlashAttention + Sequence Parallel | Ring Attention |
| 混合架构 | SSM-Attention hybrid | 纯 Attention |

**仍在探索的前沿**：
- 最优的 SSM:Attention 层比例
- MoE 的最优 expert 数量和粒度
- 真正高效的百万级上下文方案
- 动态计算分配（early exit, adaptive compute）
- Sub-quadratic attention 的实用化

---

## 9. 面试高频题 12 道 + 参考答案

### Q1: Transformer 中 Self-Attention 的时间复杂度是多少？为什么是瓶颈？

**答**：时间复杂度 $O(n^2 d)$，其中 $n$ 是序列长度，$d$ 是隐藏维度。空间复杂度 $O(n^2)$（需要存 attention 矩阵）。当 $n > d$ 时（长序列场景），$n^2$ 项主导。对于 128K token 的序列，attention 矩阵有 $128K \times 128K \approx 16.4B$ 个元素，单精度需要约 64GB。FlashAttention 通过分块计算将空间降到 $O(n)$，但时间复杂度不变。

### Q2: RoPE 为什么优于 Learned Positional Embedding？

**答**：三个优势：(1) 自然编码相对位置——旋转操作使得 $q_m^\top k_n$ 只依赖 $m-n$；(2) 外推能力——配合 NTK interpolation 或 YaRN，可以从 4K 训练扩展到 128K+ 推理；(3) 无额外参数——不增加模型参数量和 embedding 表大小。而 Learned PE 无法外推到训练长度之外，且参数与最大长度耦合。

### Q3: GQA 和 MLA 各是怎么压缩 KV cache 的？哪个更好？

**答**：GQA 通过减少 KV head 数来压缩——32 个 Q head 共享 8 组 KV，压缩 4×。MLA 通过低秩投影将所有 head 的 KV 压缩到一个潜在向量 $c_{kv}$ 中，DeepSeek V3 中压缩 30×+。MLA 理论上更优：GQA 是 "粗暴减少信息源"，MLA 是 "保留完整信息但压缩表示"。但 MLA 实现更复杂，需要额外的 decoupled RoPE keys 和 absorption trick。

### Q4: Pre-Norm 为什么比 Post-Norm 训练更稳定？

**答**：Pre-Norm 中残差路径是 $x_{l+1} = x_l + F(LN(x_l))$，梯度可以通过残差路径直接回传（类似 ResNet 的 identity shortcut），不需要穿过 LayerNorm。Post-Norm 中 $x_{l+1} = LN(x_l + F(x_l))$，梯度必须穿过 LayerNorm 的非线性变换，深层网络中容易梯度消失或爆炸。工程上 Pre-Norm 不需要 learning rate warmup 即可稳定训练。

### Q5: SwiGLU 为什么比普通 ReLU FFN 好？

**答**：SwiGLU 的两个优势：(1) 门控机制——两路投影 $xW_1$ 和 $xW_3$，一路通过 Swish 激活做门控，实现了信息的选择性过滤，比 ReLU 的简单截断更精细；(2) Swish 函数是平滑的，在零点附近有非单调区域，梯度传播更好。在相同参数预算下（调小 $d_{ff}$ 到 $\frac{8}{3}d$ 来补偿第三个矩阵），SwiGLU 的 PPL 一致优于 ReLU/GELU FFN。

### Q6: MoE 中如何解决负载不均衡（load imbalance）问题？

**答**：几种策略：
- **Auxiliary loss**（Switch Transformer）：添加辅助损失 $\mathcal{L} = N\sum f_i P_i$，鼓励 token 均匀分配
- **Expert capacity + 溢出丢弃**（GShard）：限制每个 expert 的最大 token 数
- **Expert choice routing**：让 expert 选 token 而非 token 选 expert，天然均衡
- **Bias-based（DeepSeek V3）**：给每个 expert 的 router logit 加可学习 bias，动态调节，不引入辅助损失对主训练目标的干扰

DeepSeek V3 的方案最优雅——完全消除了 auxiliary loss 和主损失之间的 trade-off。

### Q7: Mamba 和标准 Transformer 相比，最大的优势和劣势分别是什么？

**答**：

**优势**：(1) 推理时 $O(1)$ per step（固定大小隐状态 vs 线性增长的 KV cache）；(2) 训练时 $O(n)$ 而非 $O(n^2)$；(3) 长序列的推理显存恒定。

**劣势**：(1) 信息压缩到固定大小隐状态，精确检索能力弱——"在第 5 段中提到的数字是什么"这类任务表现差；(2) in-context learning 能力稍弱——few-shot 需要模型"记住"所有示例的精确内容；(3) 生态和工具链不如 Transformer 成熟（FlashAttention 已极度优化，Mamba 的 selective scan kernel 仍在发展中）。

### Q8: FlashAttention 的核心思想是什么？它改变了算法的计算复杂度吗？

**答**：FlashAttention 不改变计算复杂度（仍是 $O(n^2d)$），它优化的是 **IO 复杂度**。核心思想：(1) 将 $Q, K, V$ 分成小块（tile），在 GPU SRAM（快但小，~20MB）中完成 attention 计算，避免反复读写 HBM（慢但大，~80GB）；(2) 使用 online softmax 算法——逐块更新 softmax 的分母（running max + running sum），不需要先算完所有 $QK^\top$ 再做 softmax；(3) 不需要材化 $n \times n$ 的 attention 矩阵到 HBM，额外内存 $O(n)$。速度提升 2-4×，显存减少到 $O(n)$。

### Q9: DeepSeek V3 中 MLA 的 "absorption trick" 是什么？

**答**：在 MLA 中，KV cache 只存压缩后的 $c_{kv}$，推理时需要计算 $q_i^\top k_i = q_i^\top W^{UK}_i c_{kv}$。如果显式展开 $k_i = W^{UK}_i c_{kv}$，会增加计算和显存。Absorption trick 是将 $W^{UK}_i$ "吸收" 进 Q 的投影中：预计算 $\tilde{q}_i = W^{UK\top}_i q_i$（或等价地，合并 $W^Q$ 和 $W^{UK}$ 为一个矩阵），然后直接计算 $\tilde{q}_i^\top c_{kv}$。这样推理时完全不需要展开 K，直接在压缩空间中做 attention。V 侧也有类似处理（将 $W^{UV}_i$ 吸收进 output projection）。

### Q10: Ring Attention 如何实现通信和计算的重叠？

**答**：Ring Attention 将 $P$ 个设备排成环。在第 $t$ 轮：(1) 每个设备用当前持有的 $K^{(t)}, V^{(t)}$ 块和自己的 $Q$ 块做 partial attention（计算）；(2) 同时将 $K^{(t)}, V^{(t)}$ 发送给环上的下一个设备（通信）。由于 attention 的计算量（$O(n^2/P^2 \cdot d)$）远大于 KV 块的传输量（$O(n/P \cdot d)$），通信时间可以被计算时间完全掩盖。$P$ 轮后每个设备完成对完整 KV 序列的 attention。总显存需求均摊到 $P$ 个设备，理论上可处理 $P$ 倍长度的序列。

### Q11: 为什么 2025-2026 年的趋势是混合架构而非纯 SSM 取代 Transformer？

**答**：三个原因：(1) **精确检索瓶颈**——SSM 将信息压缩到固定大小隐状态，面对 "needle in a haystack" 类任务，表现明显不如可以直接访问所有位置的 Attention；(2) **In-Context Learning**——Transformer 的 ICL 能力（few-shot learning）依赖于对所有示例的精确记忆和模式匹配，SSM 在这方面有先天不足；(3) **工程成熟度**——FlashAttention 已将 Attention 的效率推到很高，纯 SSM 的加速比优势在缩小。混合架构（如 Jamba 的 3:1 Mamba:Attention）用少量 Attention 层"兜底"检索能力，其余用 SSM 层享受线性复杂度和低推理成本，是当前的工程最优解。

### Q12: 如果让你设计一个 2026 年的 100B+ 参数模型，你会怎么选择架构？

**答**（示例回答，展示系统性思维）：

**基座选择**：混合架构——约 75% SSM 层（Mamba-2）+ 25% Attention 层（GQA 或 MLA），Attention 层均匀分布在模型中确保每隔几层就有全局信息检索能力。

**具体配置**：
- 归一化：Pre-RMSNorm
- 激活：SwiGLU，$d_{ff} = \frac{8}{3}d$
- 位置编码：RoPE（仅在 Attention 层），SSM 层天然有序列位置感知
- 稀疏化：MoE，采用 DeepSeek 的细粒度 expert + shared expert 方案
- KV 效率：MLA（如果 Attention 层较少，复杂度可控）或 GQA-8
- 长上下文：YaRN + Ring Attention + FlashAttention-3
- 训练精度：BF16 主体 + FP8 GEMM
- 辅助目标：Multi-Token Prediction

**理由**：
- MoE 让总参数量做大（知识容量）而激活量可控（推理成本）
- 混合架构平衡长序列效率和检索能力
- MLA/GQA 控制推理显存
- RoPE + YaRN 支持长上下文外推
- FP8 降低训练成本

---

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
3. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — MQA
4. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
5. Press et al. "Train Short, Test Long: Attention with Linear Biases" (2022) — ALiBi
6. Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models" (2023)
7. Shazeer. "GLU Variants Improve Transformer" (2020) — SwiGLU
8. Zhang & Sennrich. "Root Mean Square Layer Normalization" (2019) — RMSNorm
9. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models" (2022)
10. DeepSeek-AI. "DeepSeek-V3 Technical Report" (2024)
11. Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
12. Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024) — Mamba-2
13. Peng et al. "RWKV: Reinventing RNNs for the Transformer Era" (2023)
14. Sun et al. "Retentive Network: A Successor to Transformer for Large Language Models" (2023)
15. Lieber et al. "Jamba: A Hybrid Transformer-Mamba Language Model" (2024)
16. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
17. Liu et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
18. Munkhdalai et al. "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" (2024)

---

> **最后提醒**：面试中展示 "工程直觉" 比 "背公式" 更重要。能说清楚 "为什么这样设计"、"解决什么问题"、"有什么 trade-off" 比默写公式更有价值。

---

## See Also

- [[AI/LLM/Architecture/Attention 变体综述|Attention 变体综述]] — 同主题深度版，覆盖 MQA/GQA/FlashAttention 等
- [[AI/Foundations/DL-Basics/Transformer|Transformer 通识]] — 基础原理版
- [[AI/Foundations/Architecture/SSM 与 Mamba|SSM 与 Mamba]] — 2026 架构演进的另一条线
- [[AI/Foundations/目录|Foundations MOC]] — 架构基础全图谱
