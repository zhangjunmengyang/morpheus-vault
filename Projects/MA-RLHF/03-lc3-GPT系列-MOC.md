---
title: "lc3 · GPT 系列专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc3_gpt"
tags: [moc, ma-rlhf, gpt, decoder-only, kv-cache, perplexity, lc3]
---

# lc3 · GPT 系列专题地图

> **目标**：理解 Decoder-Only 范式如何统一 NLP，掌握 GPT-2 完整实现和关键优化组件。  
> **核心问题链**：为什么预训练比监督学习更有效？GPT 为什么选择 Decoder-Only？KV Cache 如何让推理提速 10x？

---

## 带着这三个问题学

1. **GPT-1 → GPT-2 → GPT-3 的关键创新分别是什么？** 预训练范式、规模效应、In-Context Learning 各自在哪一代引入？
2. **KV Cache 到底缓存了什么，节省了多少计算？** 推理时不用 KV Cache 的计算复杂度 vs 用了之后差多少？
3. **Perplexity 和 Loss 是什么关系？** 为什么 PPL 是评估语言模型的标准指标？

---

## 学习顺序

```
Step 1  GPT 架构演进               ← 从 Encoder-Decoder 到 Decoder-Only 的动机
   ↓
Step 2  GELU 激活函数              ← 为什么 GPT 用 GELU 而不是 ReLU
   ↓
Step 3  Pre-Normalization          ← 稳定深层训练的关键改进
   ↓
Step 4  BPE Tokenizer 通用实现     ← GPT 标配分词器，从字节级开始
   ↓
Step 5  GPT-2 完整实现             ← Model + Dataset + Train + Inference（🌟核心）
   ↓
Step 6  KV Cache                   ← 推理加速的核心机制
   ↓
Step 7  Perplexity 计算            ← 评估语言模型的标准指标
   ↓
Step 8  In-Context Learning        ← GPT-3 的「涌现」能力
```

---

## 笔记清单

### Step 1：GPT 系列演进

**[[AI/LLM/Architecture/GPT2-手撕实操|GPT2 手撕实操]]**

GPT 三代关键创新：

| 版本 | 关键创新 | 参数量 | 核心贡献 |
|------|---------|--------|---------|
| GPT-1 | 预训练+微调范式 | 117M | 证明无监督预训练 → 下游微调有效 |
| GPT-2 | 纯预训练，zero-shot | 1.5B | 证明足够大的语言模型不需要微调 |
| GPT-3 | In-Context Learning | 175B | Few-shot prompting，Scaling Law 验证 |

**为什么 Decoder-Only 胜出**：Encoder-Decoder 需要分离的编码/解码步骤，Decoder-Only 统一为自回归 next-token prediction → 任务统一 → 更容易 scale up → Scaling Law 奏效。

---

### Step 2-3：GELU & Pre-Normalization

⏳ 待入库：**GPT 组件详解笔记**

- **GELU**：`GELU(x) = x · Φ(x)`（Φ 是标准正态 CDF）。与 ReLU 不同，GELU 是平滑的，在 x≈0 处有软门控 → 比 ReLU 的硬截断更有利于梯度流动。SiLU（Swish）是 GELU 的近似。
- **Pre-Normalization**：将 LayerNorm 放在 Attention/FFN **之前**（而非之后）。数学上：残差路径 `x + Sublayer(LN(x))` 使梯度可以绕过 LN 直接回传 → 训练更稳定 → GPT-2 开始采用 Pre-LN。

课程代码：`GELU.ipynb`（推导 + 实现） · `Pre-Normalization.ipynb`（分析 Pre-Norm 有效性）

---

### Step 4：BPE Tokenizer 通用实现

⏳ 待入库：**GPT BPE Tokenizer 完整实现笔记**

- **字节级 BPE**：从 256 个字节开始（而非 Unicode 字符），避免 OOV 问题
- **GPT-2 词表**：50257 个 token = 256 字节 + 50000 merges + 1 special token
- **regex 预分词**：先用正则拆分为 word-level 片段，再对每个片段做 BPE merge

课程代码：`BPE-Tokenizer.ipynb` — 一步步实现通用分词器

深入阅读：[[AI/LLM/Architecture/Tokenizer-Embedding-手撕实操|Tokenizer-Embedding 手撕实操]] · [[Tokenizer 深度理解|Tokenizer 深度理解]]

---

### Step 5：GPT-2 完整实现（🌟 核心）

**[[AI/LLM/Architecture/GPT2-手撕实操|GPT2 手撕实操]]**

模型架构：
```
Token Embedding + Position Embedding
  → N × [Pre-LN → Masked Self-Attention → Add → Pre-LN → FFN(GELU) → Add]
  → Final LN → LM Head (Linear → Softmax)
```

关键实现细节：
- Causal Mask：下三角矩阵，确保自回归特性
- Weight Tying：Input Embedding 和 LM Head 共享权重（节省参数）
- 训练数据：GPT-2 WebText 格式，`LM_dataset.ipynb` 封装 DataLoader

课程代码：`GPT-2.ipynb`（🌟 Model + 数据封装 + 训练 + 推理，全套手撕）

---

### Step 6：KV Cache（🌟 核心）

⏳ 待入库：**KV Cache 原理与实现笔记**

- **问题**：自回归推理第 t 步，需要计算所有 t 个 token 的 Attention → 前 t-1 个 token 的 K/V 在上一步已经算过，重复计算浪费
- **解法**：缓存历史步骤的 K 和 V 向量，新步骤只计算新 token 的 Q/K/V → K/V 拼接缓存 → Attention 计算
- **节省量**：不用 cache 时，生成 T 个 token 的 Attention 计算量 O(T²·d)；用 cache 后每步只需 O(T·d) → 总计 O(T²·d) vs O(T²·d) 理论相同，但实际减少了 Proj 的重复计算（不需要对历史 token 重新过 W_K / W_V 线性层）
- **显存代价**：缓存大小 = `2 × n_layers × seq_len × n_heads × d_head × batch_size`

课程代码：`KVCache.ipynb`（🌟 完整实现 KV Cache 推理加速）

---

### Step 7：Perplexity

⏳ 待入库：**Perplexity 计算笔记**

- **定义**：`PPL = exp(H)`，H 是模型在测试集上的平均交叉熵 loss
- **直觉**：PPL = 模型在每个位置的「平均困惑选项数」。PPL=10 意味着模型平均在 10 个候选中犹豫
- **计算**：`PPL = exp(-1/N · Σ log P(x_i | x_{<i}))`

课程代码：`Perplexity.ipynb` — NTP 任务的标准性能指标

---

### Step 8：In-Context Learning

⏳ 待入库：**ICL 推理实现笔记**

- **核心思想**：不更新参数，通过在 prompt 中给出示例（few-shot），让模型学会任务模式
- **实现**：构造模板 prompt → 拼接 few-shot 示例 → 模型根据模式继续生成
- **为什么有效**：大模型在预训练中隐式学会了「模式匹配」能力（meta-learning 假说）

课程代码：`in_context_learning_inference.ipynb` — 模板提示词技巧实战

---

## 面试高频场景题

**Q：KV Cache 节省了哪些计算？**  
A：节省了对历史 token 的 K/V 线性投影计算（W_K·x, W_V·x）。不用 KV Cache 时，每生成一个新 token 都要对所有历史 token 重新做 Projection + Attention；用了 KV Cache 后只对新 token 做 Projection，历史 K/V 直接从缓存读取。实际加速比在长序列上非常显著（从 O(n) 次 Projection 降到 O(1) 次）。

**Q：GPT 系列的 Scaling Law 体现在哪里？**  
A：Kaplan et al. (2020) 发现：模型性能（loss）与模型参数量 N、数据量 D、计算量 C 呈幂律关系 `L ∝ N^{-α}`。增大任一因素都能平滑降低 loss，且三者之间存在最优分配比例（Chinchilla 比例：数据量 ≈ 20× 参数量 token 数）。这意味着：更大模型 + 更多数据 = 可预测的性能提升。

**Q：GELU 和 ReLU 的区别？为什么 GPT 选 GELU？**  
A：ReLU 在 x<0 时梯度为 0（dying ReLU 问题），x=0 处不可导。GELU 是平滑近似，`GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`，在 x≈0 处有软门控而非硬截断 → 梯度更平滑 → 深层网络训练更稳定。

**Q：Perplexity 越低越好吗？有什么局限？**  
A：PPL 越低说明模型对测试集的预测越准确，通常越好。但局限：1）PPL 只衡量「预测下一个 token 的准确度」，不直接反映生成质量（流畅性、事实性）；2）不同 tokenizer 的 PPL 不可比（token 粒度不同）；3）PPL 对重复文本可能很低但生成质量很差。
