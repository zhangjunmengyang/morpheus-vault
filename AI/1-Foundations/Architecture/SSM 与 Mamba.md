---
brief: "SSM/Mamba——状态空间模型作为 Transformer 替代架构；线性时间复杂度 vs Attention 二次方复杂度；Mamba 的选择性 SSM 机制（S6）使其在长序列任务上超越同量级 Transformer；面试被问非 Transformer 架构的首要参考。"
title: "SSM/Mamba: 非 Transformer 架构"
date: 2026-02-14
tags: [architecture, ssm, mamba, state-space, interview]
type: note
---

# SSM/Mamba: 非 Transformer 架构

## 1. 状态空间模型 (SSM) 基础

### 1.1 连续时间 SSM

状态空间模型源自控制理论，描述一个线性时不变（LTI）系统：

$$
\begin{aligned}
h'(t) &= \mathbf{A} h(t) + \mathbf{B} x(t) \\
y(t) &= \mathbf{C} h(t) + \mathbf{D} x(t)
\end{aligned}
$$

- $x(t) \in \mathbb{R}$：输入信号
- $h(t) \in \mathbb{R}^N$：隐状态（N 维）
- $y(t) \in \mathbb{R}$：输出信号
- $\mathbf{A} \in \mathbb{R}^{N \times N}$：状态转移矩阵（核心）
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$：输入投影矩阵
- $\mathbf{C} \in \mathbb{R}^{1 \times N}$：输出投影矩阵
- $\mathbf{D}$：跳跃连接（通常忽略）

**直觉**：系统维护一个"记忆状态" $h(t)$，不断被输入更新，输出是记忆的线性读出。

### 1.2 离散化

实际处理离散序列需要将连续系统离散化。常用**零阶保持 (ZOH)** 方法：

$$
\begin{aligned}
\bar{\mathbf{A}} &= \exp(\Delta \mathbf{A}) \\
\bar{\mathbf{B}} &= (\Delta \mathbf{A})^{-1} (\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}
\end{aligned}
$$

其中 $\Delta$ 是步长参数（可学习），离散化后得到线性递推：

$$
\begin{aligned}
h_k &= \bar{\mathbf{A}} h_{k-1} + \bar{\mathbf{B}} x_k \\
y_k &= \mathbf{C} h_k
\end{aligned}
$$

### 1.3 双重计算模式

SSM 的关键优势：**同一个模型可以用两种方式计算**：

| 模式 | 公式 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| **递推模式** | $h_k = \bar{A} h_{k-1} + \bar{B} x_k$ | O(1) 每步 | 推理（自回归生成） |
| **卷积模式** | $y = \bar{K} * x$，其中 $\bar{K} = (\mathbf{C} \bar{\mathbf{B}}, \mathbf{C} \bar{\mathbf{A}} \bar{\mathbf{B}}, \ldots)$ | O(n log n) FFT | 训练（并行化） |

这是 SSM 相比 RNN 的核心架构优势：**训练时像 CNN（并行），推理时像 RNN（常数内存）**。

---

## 2. S4：结构化状态空间

**S4 (Structured State Spaces for Sequence Modeling)** 是 Albert Gu 等人在 2021 年提出的突破性工作。

### 2.1 HiPPO 初始化

S4 的关键创新之一是 **A 矩阵的初始化**。HiPPO（High-order Polynomial Projection Operator）矩阵能让隐状态最优地压缩历史信息：

$$
\mathbf{A}_{nk} = -\begin{cases}
(2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
n + 1 & \text{if } n = k \\
0 & \text{if } n < k
\end{cases}
$$

**直觉**：HiPPO 让隐状态的每个维度记住输入的一个 Legendre 多项式系数，从而以最小误差压缩任意长的历史。

### 2.2 对角化加速

原始 HiPPO 矩阵是 $N \times N$ 的稠密矩阵，矩阵-向量乘法是 $O(N^2)$。

S4 的解法：
1. **NPLR 分解**：$\mathbf{A} = \mathbf{V} \Lambda \mathbf{V}^{-1} + \mathbf{p}\mathbf{q}^T$（Normal Plus Low-Rank）
2. 在对角基下计算卷积核，复杂度从 $O(N^2 L)$ 降到 $O(N + L)$
3. **S4D**：后续简化版直接用对角矩阵（Diagonal State Spaces），更简洁

### 2.3 S4 的成就与局限

- ✅ **Long Range Arena (LRA)**：首次在全部 6 个长序列任务上超过 Transformer
- ✅ 在 Path-X（16384 长度序列分类）上从随机水平提升到 >90%
- ❌ 语言建模效果仍不如 Transformer
- ❌ LTI 特性限制了**内容感知**能力（A、B、C 与输入无关）

---

## 3. Mamba (S6)：选择性状态空间

### 3.1 核心创新：输入依赖参数

Mamba（2023, Albert Gu & Tri Dao）的核心洞察：**LTI 是 SSM 的瓶颈**。

在经典 SSM 中，$\mathbf{A}$、$\mathbf{B}$、$\mathbf{C}$、$\Delta$ 对所有输入 token 相同。Mamba 让它们**依赖于输入**：

$$
\begin{aligned}
\mathbf{B}_k &= \text{Linear}_B(x_k) \\
\mathbf{C}_k &= \text{Linear}_C(x_k) \\
\Delta_k &= \text{softplus}(\text{Linear}_\Delta(x_k))
\end{aligned}
$$

**与 Attention 的类比：**

| | Transformer Attention | Mamba 选择性 SSM |
|--|----------------------|----------------|
| 信息选择 | Q·K 相似度决定关注哪些 token | $\Delta_k$ 决定"遗忘/记忆"门控 |
| 信息聚合 | 加权求和 V | 隐状态 $h_k$ 累积 |
| 复杂度 | $O(n^2)$ | $O(n)$ |
| 全局 vs 压缩 | 保留所有 KV（全局） | 压缩到固定大小状态（有损） |

**$\Delta$ 的直觉**：
- $\Delta_k$ 大 → $\bar{A}_k \approx I$（保留旧状态）→ 忽略当前输入
- $\Delta_k$ 小 → $\bar{A}_k \approx 0$（遗忘旧状态）→ 关注当前输入
- 这本质上是一个**输入依赖的遗忘门**，类似 LSTM 的 forget gate

### 3.2 硬件感知算法

输入依赖打破了 LTI，无法用 FFT 卷积。Mamba 用**硬件感知的并行扫描 (parallel scan)** 替代：

1. **Parallel Scan**：前缀和的推广，可在 GPU 上 $O(\log n)$ 深度并行计算递推
2. **Kernel Fusion**：将离散化、递推、输出投影融合成一个 CUDA kernel
3. **Recomputation**：前向时不保存中间状态，反向时重新计算（节省显存）
4. 使用 **SRAM**（片上内存）而非 HBM，减少内存带宽瓶颈

### 3.3 架构设计

Mamba Block（替代 Transformer Block）：

```
Input → Linear↑ (expand D→2D) → 分两路
  路径1: Conv1D → SiLU → SSM → 
  路径2: SiLU →                  ← 门控乘法
→ Linear↓ (2D→D) → Output
```

- 没有 Attention、没有 MLP，单一 block 融合了序列建模和非线性变换
- Conv1D 提供局部上下文（类似 positional 信息）
- 门控机制提供非线性

### 3.4 Mamba 的成就

- 语言建模：首次在 perplexity 上匹配同等规模 Transformer
- 推理速度：**5x** throughput 提升（1M tokens 时）
- 显存：无 KV Cache，显存与序列长度无关
- 在 DNA/Audio 等连续信号任务上显著优于 Transformer

---

## 4. Mamba-2：SSD 框架

### 4.1 核心思想：SSM = 结构化矩阵上的线性注意力

Mamba-2 (2024) 的理论贡献是揭示了 SSM 与线性注意力的统一视角：

**State Space Duality (SSD)**：选择性 SSM 等价于一种带**半可分 (semiseparable) 掩码**的线性注意力。

具体来说，SSM 的输入-输出关系可以写成：

$$
y = \mathbf{M} \cdot (x)，\quad M_{ij} = \begin{cases} C_i^T A_{i:j} B_j & i \geq j \\ 0 & i < j \end{cases}
$$

矩阵 $\mathbf{M}$ 是一个**半可分矩阵** —— 这正是线性注意力的掩码矩阵的结构化形式。

### 4.2 实际改进

| 特性 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| 状态维度 N | 16 | 64-256 |
| 头数 | 无 | 多头（类似 MHA） |
| 算法 | Parallel scan | Chunk-wise SSD |
| 速度 | 快 | 更快（2-8x on A100） |
| 理论 | 工程驱动 | 统一理论框架 |

**Chunk-wise 算法**：将序列分成固定大小的 chunk（如 256），chunk 内用矩阵乘法（利用 Tensor Core），chunk 间用递推。兼顾了并行效率和长程依赖。

---

## 5. Jamba：混合架构

### 5.1 设计思路

Jamba（AI21 Labs，2024）是第一个大规模的 **Mamba + Transformer 混合模型**（52B 参数，12B 活跃 MoE）。

架构：
```
Mamba Layer × 5 → Attention Layer × 1 → Mamba Layer × 5 → Attention Layer × 1 → ...
```

- 每 6 层中只有 1 层是全 Attention
- Mamba 层处理长程依赖和序列压缩
- Attention 层负责精确的信息检索（recall）
- 部分层使用 MoE（Mixture of Experts）增加容量

### 5.2 为什么需要混合

| 能力 | 纯 Mamba | 纯 Transformer | 混合 |
|------|---------|---------------|------|
| 长序列效率 | ✅ O(n) | ❌ O(n²) | ✅ |
| 精确回忆 | ❌ 压缩有损 | ✅ KV Cache 完整 | ✅ |
| In-context learning | ❌ 较弱 | ✅ 强 | ✅ |
| 推理吞吐 | ✅ 无 KV Cache | ❌ KV Cache 瓶颈 | ✅ 少量 KV |
| 训练效率 | ✅ | ✅ | ✅ |

---

## 6. 优势与局限

### 6.1 SSM/Mamba 的优势

1. **线性复杂度 O(n)**：训练和推理都与序列长度线性关系
2. **无 KV Cache**：推理时隐状态固定大小（~数 KB vs Transformer 的 GB 级 KV Cache）
3. **长序列高效**：在 1M+ token 序列上仍可高效处理
4. **推理吞吐高**：constant memory per token，适合高并发部署
5. **连续信号建模**：在 Audio、DNA、时间序列上天然适配

### 6.2 局限

1. **In-context learning (ICL) 能力弱**：
   - Transformer 通过 KV Cache 精确"记住"所有 context
   - SSM 将 context 压缩到固定维度的隐状态，信息有损
   - 表现：few-shot 性能不如同等规模 Transformer

2. **Recall 能力不足**：
   - "大海捞针" 测试中 Mamba 表现明显差于 Transformer
   - 原因：压缩状态无法完美恢复任意位置的信息
   - 解决：混合架构中用 Attention 层弥补

3. **生态不成熟**：
   - 推理引擎支持有限（vLLM/TensorRT-LLM 支持不如 Transformer）
   - 缺少像 Flash Attention 一样成熟的基础设施
   - 预训练规模和数据量远不及 Transformer 模型

4. **理论理解不够深**：
   - 为什么某些任务 SSM 比 Transformer 差？信息论解释不充分
   - 最优的混合比例（多少 Mamba + 多少 Attention）缺乏理论指导

---

## 7. 现状与趋势

### 7.1 混合架构是主流方向

当前共识：**纯 Mamba 不够，纯 Transformer 太贵，混合是最优解**。

代表工作：
- **Jamba / Jamba 1.5**（AI21）：Mamba + Attention + MoE
- **Zamba**（Zyphra）：Mamba + 共享 Attention 层
- **NVIDIA Hybrid Models**：Mamba-2 + Sliding Window Attention
- **Griffin**（Google DeepMind）：Gated Linear Recurrence + Local Attention
- **RWKV-6/7**：另一种线性 RNN 路线（Eagle/Finch 架构）

### 7.2 技术演进路线

```
S4 (2021) → S5 → H3 → Hyena
                          ↓
               Mamba/S6 (2023.12)
                          ↓
               Mamba-2/SSD (2024.05)
                          ↓
               混合架构 Jamba/Zamba/Griffin (2024-)
                          ↓
               下一步：硬件原生支持 + 更大规模验证
```

---

## 8. 面试常见问题及回答要点

### Q1: Mamba 和 Transformer 的核心区别是什么？

**回答要点：**
- **信息处理方式**：Transformer 用 Attention 全局交互（$O(n^2)$），Mamba 用状态递推压缩式建模（$O(n)$）
- **记忆机制**：Transformer 的 KV Cache 保存所有历史 token 的精确表示；Mamba 将历史压缩到固定大小隐状态
- **推理特性**：Transformer 每生成一个 token 需要读取整个 KV Cache；Mamba 只需更新固定大小状态
- **Mamba 的创新**：通过让 B、C、Δ 依赖输入，实现了内容感知的选择性记忆（而非 LTI）
- **权衡**：Mamba 用信息压缩换取计算效率，Transformer 用计算量换取信息完整性

### Q2: 什么是 HiPPO？为什么它对 SSM 重要？

**回答要点：**
- HiPPO 是 A 矩阵的一种特殊初始化方法
- 数学本质：让隐状态的每个维度对应输入历史的一个**正交多项式系数**（通常是 Legendre 多项式）
- 这意味着 N 维隐状态可以**最优地逼近**任意长的输入历史（在 L2 意义下）
- 没有 HiPPO 的 SSM 很容易"遗忘"早期输入（类似 vanilla RNN 的梯度消失）
- HiPPO 解决了 SSM 的**长程依赖**问题，是 S4 能在 LRA benchmark 上突破的关键

### Q3: Mamba 的选择性机制和 Attention 有什么本质联系？

**回答要点：**
- Mamba-2 的 SSD 框架证明：选择性 SSM ≈ 带半可分掩码的线性注意力
- 两者都在做**输入依赖的信息选择**：Attention 用 QK 相似度，Mamba 用 Δ 门控
- 关键区别：Attention 的选择矩阵是**满秩的**（每对 token 独立计算），而 SSM 的矩阵是**低秩的**（受状态维度 N 限制）
- 这解释了为什么 SSM 的 recall 能力弱于 Attention：低秩掩码不能表达任意的 token-token 关系
- 增大状态维度 N 可以缩小差距（Mamba-2 将 N 从 16 提升到 256）

### Q4: 为什么纯 Mamba 做不到和 Transformer 一样好的 in-context learning？

**回答要点：**
- ICL 的核心需求：**精确记住 context 中的 few-shot 示例**，在生成时精确回忆
- Transformer 的 KV Cache 是**无损存储**：每个 context token 都有独立的 K/V 向量
- Mamba 的隐状态是**有损压缩**：所有历史信息被压缩到固定维度（如 16×D）
- 当 context 中有关键的少量信息（如 few-shot 的标签）时，压缩可能丢失这些细节
- **实验证据**：Mamba 在 MQAR（Multi-Query Associative Recall）任务上显著弱于 Transformer
- **解决思路**：混合架构中加入少量 Attention 层专门负责 recall

### Q5: 如果你要部署一个 1M context 的模型，你会选择什么架构？为什么？

**回答要点：**
- 选择 **Mamba + Attention 混合架构**（如 Jamba 风格）
- 理由：
  - 纯 Transformer：1M context 的 KV Cache 需要 ~100GB+ 显存（以 70B 模型为例），不实际
  - 纯 Mamba：长序列效率好但 recall 不足，用户可能需要精确引用文档中的细节
  - 混合架构：大部分层用 Mamba（O(n)，无 KV Cache），每隔几层插入一层 Sliding Window Attention 或 Full Attention
- 具体配比：约 **7:1 的 Mamba:Attention 比例**（参考 Jamba）
- 工程考量：
  - 推理引擎需要同时支持两种算子（目前 vLLM 已有初步支持）
  - Mamba 层的 parallel scan kernel 需要针对目标 GPU 优化
  - 可以用 chunked prefill 优化长 prompt 的首 token 延迟

---

## 参考资源

- S4 论文: "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al., 2021)
- Mamba 论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- Mamba-2 论文: "Transformers are SSMs" (Dao & Gu, 2024)
- Jamba 论文: "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21, 2024)
- Albert Gu 的博士论文: 系统性介绍 SSM 发展脉络

---

## See Also

- [[Decoder-Only vs Encoder-Decoder|Decoder-Only vs Encoder-Decoder]] — Mamba 作为 Transformer 替代的背景
- [[Transformer|Transformer 通识]] — SSM 希望替代的架构
- [[Attention 变体综述|Attention 变体综述]] — Linear Attention 与 SSM 的收敛趋势
- [[AI/1-Foundations/目录|Foundations MOC]] — 架构基础全图谱
