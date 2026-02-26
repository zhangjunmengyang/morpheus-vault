---
title: Transformer 手撕实操
brief: 从零实现 Transformer Encoder-Decoder 架构（Vaswani 2017）：Self-Attention/Cross-Attention/FFN/LayerNorm/Positional Encoding 完整 PyTorch 代码，含推理串行与训练并行对比，来源 MA-RLHF 教学项目。
date: 2026-02-25
type: code-practice
source: MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
tags:
  - code-practice
  - transformer
  - architecture
  - attention
  - pytorch
related:
  - "[[Transformer架构深度解析-2026技术全景|Transformer架构深度解析]]"
  - "[[AI/3-LLM/Architecture/GPT2-手撕实操|GPT2-手撕实操]]"
  - "[[AI/3-LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]]"
  - "[[AI/3-LLM/Architecture/基础数学组件手撕|基础数学组件手撕]]"
---

# Transformer 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

Transformer（Vaswani et al., 2017）是为机器翻译设计的 Encoder-Decoder 架构。核心创新：用**自注意力机制**替代 RNN 的递归计算，实现序列建模的完全并行化。

**架构全景**：
- **Encoder**：N 层 `[Self-Attention → LayerNorm → FFN → LayerNorm]`（带残差连接）
- **Decoder**：N 层 `[Masked Self-Attention → LayerNorm → Cross-Attention → LayerNorm → FFN → LayerNorm]`
- **输入层**：Token Embedding + Positional Encoding
- **输出层**：Linear → Softmax → 词表概率分布

**训练范式**：自回归 next-token prediction。训练时并行（teacher forcing），推理时串行（逐 token 生成）。

---

## 二、核心实现

### 2.1 注意力机制：从直觉到公式

**原理**：同一个词在不同上下文中有不同语义（如"刷子"在"有两把刷子"vs"用刷子画画"中）。注意力机制通过对序列中所有 token 做加权聚合，让每个 token 获得 context-aware 的表示：

$$S_i = \sum_j w_{ij} \cdot V_j, \quad w_{ij} = \text{softmax}(Q_i K_j^T / \sqrt{d})$$

其中 $Q = XW_Q$, $K = XW_K$, $V = XW_V$——投影变换让模型能学习"什么该注意"。

**为什么需要投影？** 直接用原始 embedding 做内积 $X_i X_j^T$ 也能算注意力，但投影参数 $W_Q, W_K, W_V$ 提供了可优化的自由度——反向传播可以调整"谁注意谁"，让权重分配对下游任务最优。

**为什么除以 $\sqrt{d}$？** 当维度 d 很大时，内积值方差随 d 线性增长，softmax 会饱和（梯度消失）。除以 $\sqrt{d}$ 稳定数值范围。

**代码**：

```python
# 来自 lecture/lc2_transformer/Transformer_Attention.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.WO = nn.Linear(dim_in, dim_out)

    def forward(self, X, mask=None):
        batch_size, seq_len, dim = X.shape
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        S = Q @ K.transpose(1, 2) / math.sqrt(dim)

        if mask is not None:
            idx = torch.where(mask == 0)
            S[idx[0], idx[1], idx[2]] = -10000.0  # mask 位置置为负无穷

        P = torch.softmax(S, dim=-1)  # 行归一化
        Z = P @ V
        output = self.WO(Z)
        return output
```

### 2.2 Mask 机制详解

**原理**：三种 mask 各有用途：

| Mask 类型 | 用途 | 形状 | 规则 |
|-----------|------|------|------|
| `src_mask` | Encoder self-attention | `[bs, src_len, src_len]` | PAD 位置的行列都置 0 |
| `trg_mask` | Decoder masked self-attention | `[bs, trg_len, trg_len]` | 下三角 + PAD mask（防止看到未来 token） |
| `src_trg_mask` | Decoder cross-attention | `[bs, trg_len, src_len]` | src 和 trg 的 PAD 位置都置 0 |

**代码**：

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
def get_src_mask(input_ids, pad_token_id=0):
    bs, seq_len = input_ids.shape
    mask = torch.ones(bs, seq_len, seq_len)
    for i in range(bs):
        pad_idx = torch.where(input_ids[i, :] == pad_token_id)[0]
        mask[i, pad_idx, :] = 0  # PAD token 不作为 query
        mask[i, :, pad_idx] = 0  # PAD token 不作为 key
    return mask

def get_trg_mask(input_ids, pad_token_id=0):
    bs, seq_len = input_ids.shape
    mask = torch.tril(torch.ones(bs, seq_len, seq_len))  # 因果下三角
    for i in range(bs):
        pad_idx = torch.where(input_ids[i, :] == pad_token_id)[0]
        mask[i, pad_idx, :] = 0
        mask[i, :, pad_idx] = 0
    return mask

def get_src_trg_mask(src_ids, trg_ids, pad_token_id=0):
    bs, src_seq_len = src_ids.shape
    bs, trg_seq_len = trg_ids.shape
    mask = torch.ones(bs, trg_seq_len, src_seq_len)
    for i in range(bs):
        src_pad_idx = torch.where(src_ids[i, :] == pad_token_id)[0]
        trg_pad_idx = torch.where(trg_ids[i, :] == pad_token_id)[0]
        mask[i, trg_pad_idx, :] = 0
        mask[i, :, src_pad_idx] = 0
    return mask
```

**关键洞察**：Mask 实现时不是直接乘 0（softmax(0) ≠ 0），而是将 score 置为 -10000（近似负无穷），使 softmax 输出趋近于 0。这是一个常见的实现陷阱。

### 2.3 多头注意力（Multi-Head Attention）

**原理**：单头注意力的权重可能稀疏，只关注少数 token。多头将 Q/K/V 沿特征维度拆分为 H 份，各头独立计算注意力后拼接，捕获不同类型的依赖关系（语法、语义、位置等）。

$$\text{head}_h = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)}), \quad Q^{(h)} \in \mathbb{R}^{N \times d/H}$$

**代码**：

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
class MultiHeadScaleDotProductAttention(nn.Module):
    def __init__(self, dim_in, dim_out, heads=8):
        super().__init__()
        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.WO = nn.Linear(dim_in, dim_out)
        self.heads = heads
        self.head_dim = dim_out // self.heads

    def forward(self, X_Q, X_K, X_V, mask=None):
        bs, seq_len, dim = X_Q.shape
        bs, seq_K_len, dim = X_K.shape
        Q = self.WQ(X_Q)
        K = self.WK(X_K)
        V = self.WV(X_V)

        # 拆分多头: [bs, seq, dim] → [bs, heads, seq, head_dim]
        Q_h = Q.view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K_h = K.view(bs, seq_K_len, self.heads, self.head_dim).transpose(1, 2)
        V_h = V.view(bs, seq_K_len, self.heads, self.head_dim).transpose(1, 2)

        S = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)

        if mask is not None:
            idx = torch.where(mask == 0)
            S[idx[0], :, idx[1], idx[2]] = -10000.0  # 头维度广播 mask

        P = torch.softmax(S, dim=-1)
        Z = P @ V_h

        # 恢复维度: [bs, heads, seq, head_dim] → [bs, seq, dim]
        Z = Z.transpose(1, 2).reshape(bs, seq_len, dim)
        output = self.WO(Z)
        return output
```

**关键洞察**：多头注意力的参数量与单头相同（$W_Q$ 仍是 $d \times d$），区别仅在于 reshape + transpose 的操作。头并行是"免费"的，不增加参数。

### 2.4 LayerNorm

**原理**：对每个 token 的特征维度做归一化（均值0、方差1），再用可学习的 $\gamma, \beta$ 缩放和偏移。与 BatchNorm 不同，LayerNorm 在特征维度而非 batch 维度操作，天然适配变长序列。

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad \tilde{x}_i = \gamma \odot \hat{x}_i + \beta$$

**代码**：

```python
# 来自 lecture/lc2_transformer/LayerNorm.ipynb
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, X):
        mu = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, unbiased=False, keepdim=True)
        X_hat = (X - mu) / torch.sqrt(var + self.eps)
        Y = X_hat * self.gamma + self.beta
        return Y
```

**LayerNorm 反向传播手撕**：

```python
# 来自 lecture/lc2_transformer/LayerNorm.ipynb
with torch.no_grad():
    dy = 2 * (y - label) / y.numel()
    dgamma = (out_mean_var * dy).sum(dim=(0, 1))
    dbeta = dy.sum(dim=(0, 1))

    x_mean = x - mean
    x_std_var = torch.sqrt(var + eps)

    diag_I = torch.ones(d).diag()
    I = torch.ones(d, d)
    left = diag_I - 1/d * I

    x_grad = torch.zeros_like(x)
    for i in range(bs):
        for j in range(seq_len):
            d_ln = (left / x_std_var[i,j,0]
                    - x_mean[i,j,:].outer(x_mean[i,j,:]) / (x_std_var[i,j,0]**3 * d)
                   ) * gamma
            x_grad[i,j,:] = dy[i,j,:] @ d_ln
```

### 2.5 位置编码（Positional Encoding）

**原理**：Attention 是置换等变的（permutation equivariant）——交换输入顺序，输出也交换，无法感知位置。需要显式注入位置信息。

Transformer 使用 sin-cos 绝对位置编码：

$$PE(n) = [\sin(n\theta_0), \cos(n\theta_0), \ldots, \sin(n\theta_{d/2-1}), \cos(n\theta_{d/2-1})]$$

其中 $\theta_i = 1/10000^{2i/d}$。

**关键性质**：$PE(m) \cdot PE(n)^T = f(m-n)$，即位置编码的内积只依赖相对距离。具体地：

$$PE'_m \cdot PE'^T_n = \cos((m-n)\theta_i)$$

**代码**：

```python
# 来自 lecture/lc2_transformer/Positional_Encoding.ipynb
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100, d_model=8, base=10000.0):
        super().__init__()
        d = d_model // 2
        i = torch.arange(d)
        theta = base ** (-i * 2 / d_model)
        L = torch.arange(max_len)

        m_theta = torch.outer(L, theta)  # [max_len, d/2]
        self.PE = torch.zeros(max_len, d_model)
        self.PE[:, 0::2] = m_theta.sin()
        self.PE[:, 1::2] = m_theta.cos()

    def forward(self, x):
        seq_len = x.shape[0]
        return self.PE[:seq_len, :]
```

**关键洞察**：
- 低维 $\theta$ 大 → 变化快（高频，捕捉局部位置差异）
- 高维 $\theta$ 小 → 变化慢（低频，捕捉全局位置关系）
- 这种多频率设计类似傅里叶变换，用不同频率的 sin/cos 组合表示任意位置

### 2.6 FFN（前馈网络）

**原理**：每个 token 独立过两层全连接，中间升维 4 倍再降回。作用：在注意力捕获的 token 关系基础上，做非线性特征变换。

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_up = nn.Linear(dim, 4 * dim)
        self.ReLU = nn.ReLU()
        self.W_down = nn.Linear(4 * dim, dim)

    def forward(self, X):
        return self.W_down(self.ReLU(self.W_up(X)))
```

### 2.7 Encoder & Decoder Block

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, X, src_mask=None):
        # Post-Norm: Attention → LN → Residual → FFN → LN → Residual
        X_attn = self.attn(X, X, X, mask=src_mask)
        X = X + self.ln1(X_attn)
        X_ffn = self.ffn(X)
        X = X + self.ln2(X_ffn)
        return X

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.masked_attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln1 = LayerNorm(dim)
        self.cross_attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim)
        self.ln3 = LayerNorm(dim)

    def forward(self, X, X_src, trg_mask=None, src_trg_mask=None):
        # Masked Self-Attention（因果）
        X_attn = self.masked_attn(X, X, X, trg_mask)
        X = X + self.ln1(X_attn)
        # Cross-Attention（Q 来自 decoder，K/V 来自 encoder）
        X_attn = self.cross_attn(X, X_src, X_src, src_trg_mask)
        X = X + self.ln2(X_attn)
        # FFN
        X_ffn = self.ffn(X)
        X = X + self.ln3(X_ffn)
        return X
```

---

## 三、完整架构代码

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
class TransformerInputLayer(nn.Module):
    def __init__(self, vocab_size=100, dim=512, max_len=1024, base=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        # 构建 sin-cos 位置编码
        theta_ids = torch.arange(0, dim, 2)
        theta = 1 / (base ** (theta_ids / dim))
        pe = torch.zeros(dim)
        pe[theta_ids] = theta
        pe[theta_ids + 1] = theta
        position_ids = torch.arange(0, max_len)
        self.PE = torch.outer(position_ids, pe)
        self.PE[:, theta_ids] = torch.sin(self.PE[:, theta_ids])
        self.PE[:, theta_ids + 1] = torch.cos(self.PE[:, theta_ids + 1])

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        X = self.embedding(input_ids)
        return X + self.PE[:seq_len, :]

class TransformerOutputLayer(nn.Module):
    def __init__(self, vocab_size=100, dim=512):
        super().__init__()
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, X):
        return self.lm_head(X)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size=100, trg_vocab_size=200,
                 dim=512, num_layers=6, heads=8, max_len=512):
        super().__init__()
        self.encoder_input = TransformerInputLayer(src_vocab_size, dim, max_len)
        self.encoder = TransformerEncoder(dim, num_layers, heads)
        self.decoder_input = TransformerInputLayer(trg_vocab_size, dim, max_len)
        self.decoder = TransformerDecoder(dim, num_layers, heads)
        self.output_layer = TransformerOutputLayer(trg_vocab_size, dim)

    def forward(self, src_ids, trg_ids, src_mask=None, trg_mask=None, src_trg_mask=None):
        X = self.encoder_input(src_ids)
        X_src = self.encoder(X, src_mask)
        Y = self.decoder_input(trg_ids)
        Y = self.decoder(Y, X_src, trg_mask, src_trg_mask)
        logits = self.output_layer(Y)
        return logits, F.softmax(logits, dim=-1)
```

---

## 四、训练与推理

### 4.1 训练：Teacher Forcing 并行

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
# Decoder 输入: <SOS>, y1, y2, ..., yn-1（去掉最后一个）
# Label:         y1, y2, ..., yn（去掉第一个）
# 这样每个位置同时预测下一个 token，实现并行训练

for batch in train_dataloader:
    X = batch[0]
    Y = batch[1][:, :-1]  # decoder input
    label = batch[1][:, 1:]  # target

    src_mask = get_src_mask(X)
    trg_mask = get_trg_mask(Y)
    src_trg_mask = get_src_trg_mask(X, Y)

    logits, _ = model(X, Y, src_mask, trg_mask, src_trg_mask)
    loss = loss_fn(logits.reshape(-1, trg_vocab_size), label.reshape(-1))
    loss.backward()
    optimizer.step()
```

### 4.2 推理：自回归生成

```python
# 来自 lecture/lc2_transformer/Transformer.ipynb
# Stage 1: Encode（仅执行一次）
with torch.no_grad():
    X = model.encoder_input(src_ids)
    X_src = model.encoder(X, src_mask)

# Stage 2: Decode（逐 token 生成）
trg_ids = torch.tensor([[SOS_TOKEN_ID]])
for i in range(max_new_tokens):
    Y = model.decoder_input(trg_ids)
    Y = model.decoder(Y, X_src)
    logits = model.output_layer(Y)
    next_token_logits = logits[:, -1, :]  # 只取最后一个位置的预测
    next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)
    trg_ids = torch.cat((trg_ids, next_token), dim=1)
```

**关键洞察**：训练时 decoder 看到完整 target（用 causal mask 防止作弊），一次前向同时预测所有位置——**O(1) 步完成**。推理时必须逐 token 生成——**O(n) 步**。这是 Transformer 训练效率远高于推理效率的根本原因。

---

## 五、Dataset 与 DataCollate

### 5.1 变长序列 + 动态 Padding

```python
# 来自 lecture/lc2_transformer/Dataset.ipynb
def paddding_collate_fn(batch_data):
    bs = len(batch_data)
    input_lens = [data['input_ids'].shape[1] for data in batch_data]
    label_lens = [data['labels'].shape[1] for data in batch_data]

    max_input_len = max(input_lens)
    max_label_len = max(label_lens)

    input_ids = torch.ones(bs, max_input_len, dtype=torch.long) * PAD_TOKEN_ID
    input_attention_masks = torch.zeros(bs, max_input_len, dtype=torch.long)
    label_ids = torch.ones(bs, max_label_len, dtype=torch.long) * PAD_TOKEN_ID
    label_attention_masks = torch.zeros(bs, max_label_len, dtype=torch.long)

    for i in range(bs):
        input_ids[i, :input_lens[i]] = batch_data[i]['input_ids'][0]
        input_attention_masks[i, :input_lens[i]] = 1
        label_ids[i, :label_lens[i]] = batch_data[i]['labels'][0]
        label_attention_masks[i, :label_lens[i]] = 1

    return {
        'input_ids': input_ids,
        'input_attention_mask': input_attention_masks,
        'label_ids': label_ids,
        'label_attention_mask': label_attention_masks,
    }
```

### 5.2 训练 Label 处理

```python
# 来自 lecture/lc2_transformer/Dataset.ipynb
# Decoder 输入去掉最后一个 token，label 去掉第一个 token
decoder_input_ids = batch['label_ids'][:, :-1]
label_for_loss = batch['label_ids'][:, 1:]
# PAD 位置不参与 loss 计算
label_for_loss[torch.where(label_for_loss == PAD_TOKEN_ID)] = IGNORE_INDEX  # -100
```

---

## 六、配套实操

> 完整代码见：`/tmp/ma-rlhf/lecture/lc2_transformer/`
> - `Transformer.ipynb` — 完整架构逐步构建
> - `Transformer_Attention.ipynb` — 注意力机制深入讲解 + backward 手撕
> - `LayerNorm.ipynb` — LayerNorm 原理、可视化、反向传播
> - `Positional_Encoding.ipynb` — 位置编码分析
> - `Dataset.ipynb` — 数据流水线
> - `transformer_framework.ipynb` — Encoder-Decoder 框架设计
> - `model.py` — 工程级完整实现

---

## 七、关键洞察与总结

1. **注意力的本质**：加权组合 $S_i = \sum_j w_{ij} V_j$，权重由 Q-K 内积决定。投影矩阵 $W_Q, W_K, W_V$ 让模型能学习对任务最有用的"注意"模式。

2. **多头 ≠ 多参数**：多头只是 reshape，总参数量不变。但允许不同头捕获不同类型的依赖，实测效果远优于单头。

3. **三种 Mask 的设计逻辑**：
   - `src_mask`：PAD token 既不做 query 也不做 key
   - `trg_mask`：因果 + PAD（下三角矩阵叠加 PAD mask）
   - `src_trg_mask`：cross-attention 中同时屏蔽两侧的 PAD

4. **Post-Norm vs Pre-Norm**：原版 Transformer 是 Post-Norm（`X + LN(Attn(X))`），但后续研究发现 Pre-Norm（`X + Attn(LN(X))`）训练更稳定，GPT-2 开始采用 Pre-Norm。

5. **位置编码的权衡**：
   - Sin-cos：不增加参数，天然支持任意长度外推
   - 可学习：表达能力更强，但无法泛化到训练未见的长度
   - 无位置编码：理论可行（decoder 的因果性隐含位置信息），但收敛慢

6. **训练 vs 推理的不对称性**：Transformer 训练并行（teacher forcing + causal mask），推理串行（自回归），这是 LLM 推理优化（KV Cache、投机解码等）的根本动机。
