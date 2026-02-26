---
title: "lc2 · Transformer 专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/lecture/lc2_transformer"
tags: [moc, ma-rlhf, transformer, attention, encoder-decoder, lc2]
---

# lc2 · Transformer 专题地图

> **目标**：从零手写完整 Transformer 并跑通训练推理，掌握 Encoder-Decoder 架构的每一个细节。  
> **核心挑战**：不是「读懂」Transformer，而是能独立写出 `model.py` + `train.py` + `inference.py` 并跑通一个中英翻译任务。

---

## 带着这三个问题学

1. **Attention 为什么要除以 √d？** 不除会怎样？跟 softmax 的梯度有什么关系？
2. **Encoder 和 Decoder 的 Attention mask 有什么区别？** 为什么 Decoder 需要 causal mask？
3. **Teacher Forcing 训练和自回归推理的差异在哪？** 训练时 Decoder 看到了什么？推理时呢？

---

## 学习顺序

```
Step 1  Attention 机制            ← Q/K/V 计算，Scaled Dot-Product
   ↓
Step 2  位置编码实现               ← Sinusoidal PE 代码与可视化
   ↓
Step 3  LayerNorm                 ← 算法原理 + 反向传播
   ↓
Step 4  完整 Transformer 模型      ← Encoder + Decoder + Cross-Attention
   ↓
Step 5  数据集 & Tokenizer         ← WMT19 数据加载，中英分词器训练
   ↓
Step 6  训练流程                   ← Teacher Forcing，交叉熵 Loss，断点恢复
   ↓
Step 7  推理流程                   ← 自回归生成，Greedy/Beam Search
```

---

## 笔记清单

### Step 1：Attention 机制

**[[AI/3-LLM/Architecture/Transformer-手撕实操|Transformer 手撕实操]]**（Attention 部分）

- **Scaled Dot-Product Attention**：`Attn(Q,K,V) = softmax(QK^T / √d_k) · V`
- **为什么 /√d_k**：Q·K 的内积方差随 d_k 线性增长，值过大 → softmax 进入饱和区 → 梯度消失。缩放后方差为 1，softmax 输出更平滑
- **Multi-Head Attention**：将 d_model 拆成 h 个头，每个头独立做 Attention → 捕获不同子空间的模式
- **Self-Attention vs Cross-Attention**：Self-Attn 的 Q/K/V 来自同一序列；Cross-Attn 的 Q 来自 Decoder，K/V 来自 Encoder 输出

课程代码：`Transformer_Attention.ipynb` — 注意力前向 + 手撕反向传播（🌟 必须能写）

---

### Step 2-3：位置编码 & LayerNorm

**[[AI/3-LLM/Architecture/基础数学组件手撕|基础数学组件手撕]]**

- **Sinusoidal PE**：`PE(pos,2i) = sin(pos/10000^{2i/d})`，不同频率编码不同维度
- **LayerNorm**：对每个样本的特征维度做归一化 `y = (x-μ)/σ * γ + β`
- **Pre-LN vs Post-LN**：Post-LN（原版）梯度在浅层爆炸，深层消失；Pre-LN 把 LN 放在 Attention/FFN 之前 → 梯度路径更平滑 → 训练更稳定，现代模型几乎都用 Pre-LN

课程代码：`Position_Encoding.ipynb`（实现 + 可视化） · `LayerNorm.ipynb`（原理 + 反向推导）

---

### Step 4：完整 Transformer 模型

**[[AI/3-LLM/Architecture/Transformer-手撕实操|Transformer 手撕实操]]**（完整模型部分 🌟）

Encoder-Decoder 架构：
```
Encoder:
  Input Embedding + PE
  → N × [Self-Attention → Add&Norm → FFN → Add&Norm]
  → Encoder Output

Decoder:
  Output Embedding + PE
  → N × [Masked Self-Attention → Add&Norm → Cross-Attention(Q=dec, K/V=enc) → Add&Norm → FFN → Add&Norm]
  → Linear → Softmax
```

关键实现细节：
- Encoder Self-Attention：padding mask（忽略 `<pad>` token）
- Decoder Masked Self-Attention：causal mask + padding mask（不能看到未来 token）
- Cross-Attention：K/V 用 Encoder 输出，mask 是 Encoder 的 padding mask

课程代码：`Transformer.ipynb`（🌟 核心，必须手撕） · `model.py`（工程版）

---

### Step 5-6：数据集 & 训练

⏳ 待入库：**Transformer 训练全流程笔记**

- **数据集**：WMT19 中英翻译 → `data.json`
- **Tokenizer**：分别训练中/英 BPE 分词器，存储 merges + vocab
- **Dataset 封装**：padding / mask / label 处理，DataLoader + DataCollate
- **训练**：Teacher Forcing（Decoder 输入是 ground truth shifted right），CrossEntropy Loss，AdamW
- **断点恢复**：保存 model + optimizer 状态，checkpoint 机制

课程代码：
- `Load_Dataset.ipynb` / `Dataset.ipynb` — 数据加载与封装
- `tokenizer.py` → `train.py --learning_rate 1e-4 --epochs 1` → `inference.py` — 完整训练流水线
- `Model_IO.ipynb` — 模型保存/加载/断点恢复

---

### Step 7：推理流程

⏳ 待入库：**Transformer 推理流程笔记**

- **自回归推理**：从 `<bos>` 开始，每步预测一个 token，将预测结果拼接回输入，直到 `<eos>`
- **Teacher Forcing（训练）vs 自回归（推理）**：训练时 Decoder 看到完整 ground truth（并行），推理时只能看到自己之前的预测（串行）→ Exposure Bias 问题
- **Greedy vs Beam Search**：Greedy 每步取 argmax；Beam 保留 top-k 候选，最终选全局最优序列

课程代码：`inference.py`（🌟 核心，加载训练好的模型做中英翻译）

---

## 面试高频场景题

**Q：为什么 Attention 要缩放 √d？**  
A：假设 Q 和 K 的每个分量独立同分布、均值 0 方差 1，则 Q·K 的内积（d 个乘积之和）的方差为 d。d 较大时内积值极大 → softmax 输出趋近 one-hot → 梯度接近零。除以 √d 使方差回到 1，保证 softmax 输出分布合理、梯度可流动。

**Q：Encoder 和 Decoder 的 Attention mask 有何不同？**  
A：Encoder Self-Attention 只用 **padding mask**（忽略 `<pad>` token，其他位置全可见）。Decoder Masked Self-Attention 使用 **causal mask + padding mask**（下三角矩阵，每个位置只能 attend 到自己和之前的位置），防止信息泄露未来 token。

**Q：Multi-Head Attention 的多头有什么好处？**  
A：每个头在 d_k = d_model/h 的低维子空间做 Attention，不同头可以学到不同的 attention pattern（如一个头关注局部语法，另一个关注长距离依赖）。总参数量与单头相同（Q/K/V 权重拆分），但表达力更强。

**Q：Pre-LN 和 Post-LN 的区别？为什么现代模型都用 Pre-LN？**  
A：Post-LN 的残差路径 `x + Sublayer(LN(x))` 在深层会导致梯度不稳定（残差连接的 scale 逐层累积）。Pre-LN 把 LN 放到 Sublayer 内部，残差路径直接传梯度 → 梯度范数更稳定 → 深层网络（50+ 层）也能稳定训练。代价是最终输出可能需要额外 LN。
