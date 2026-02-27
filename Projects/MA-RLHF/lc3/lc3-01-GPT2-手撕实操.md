---
title: "GPT-2 手撕实操"
brief: "GPT-1/2/3 系列演进梳理 + GPT-2 Decoder-only 架构完整实现：RMSNorm/SwiGLU/RoPE/GQA 等现代变体对比，面试重点：GPT-1→2→3 关键创新递进，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, gpt, decoder-only, architecture, pytorch]
related:
  - "[[Projects/MA-RLHF/lc2/lc2-01-Transformer-手撕实操|Transformer-手撕实操]]"
  - "[[Projects/MA-RLHF/lc4/lc4-01-Llama-手撕实操|Llama-手撕实操]]"
  - "[[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|基础数学组件手撕]]"
---

# GPT-2 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

GPT 系列模型的核心贡献：

| 模型 | 关键创新 |
|------|----------|
| GPT-1 | Decoder-only 预训练 + SFT 微调范式 |
| GPT-2 | 验证无监督预训练可 zero-shot 泛化多种 NLP 任务 |
| GPT-3 | 175B 参数，提出 In-Context Learning（ICL），few-shot 超越 SOTA |

**核心思想**：**Causal Language Modeling（CLM）** —— 用 Decoder-Transformer 的 masked self-attention 做自回归 next-token prediction。本质是将外在知识转化为内在参数的无监督（自监督）学习。

**GPT-2 相比原版 Transformer Decoder 的改动**：
1. **Decoder-only**：去掉 Encoder 和 Cross-Attention
2. **Pre-Normalization**：LayerNorm 前置（先 LN 再 Attention/FFN）
3. **GELU 激活**：替代 ReLU，0 点处可微
4. **去掉 bias**：Linear 层不使用偏置项
5. **最后加 LayerNorm**：decoder 输出后加一层 LN

---

## 二、核心实现

### 2.1 Causal Language Modeling

**原理**：给定序列 $(x_1, x_2, \ldots, x_n)$，每个位置 $t$ 预测下一个 token $x_{t+1}$：

$$\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N-1} \log p_\theta(x_{t+1} | x_1, \ldots, x_t)$$

Label 直接由输入左移一位得到——自监督，无需人工标注。

**代码**：

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
x = torch.randint(vocab_size, (batch_size, seq_len))

# Label = 输入左移一位
label = torch.zeros_like(x)
label[:, :-1] = x[:, 1:]
label[:, -1] = x[:, 0]  # 最后位置用首 token 填充（实际训练中用 pad/ignore）

logits, prob = model(x)
loss = loss_fn(logits.reshape(b * s, v), label.reshape(b * s))
```

### 2.2 GELU 激活函数

**原理**：ReLU 在 0 点不可微（分段函数），GELU 用高斯 CDF 平滑过渡：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$$

实际使用 tanh 近似（计算更快）：

$$\text{GELU}(x) \approx x \cdot 0.5[1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3))]$$

**代码**：

```python
# 来自 lecture/lc3_gpt/GELU.ipynb
def GELU(x):
    cdf = 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))
    return x * cdf

# 精确版（用 erf）
def GELU_exact(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf

# sigmoid 近似
def GELU_approx(x):
    return x * torch.sigmoid(1.702 * x)
```

**关键洞察**：GELU 的几何意义——以 $x$ 乘以"$x$ 落在正态分布左侧的概率"。当 $x \gg 0$ 时 $\Phi(x) \to 1$，退化为恒等映射；$x \ll 0$ 时 $\Phi(x) \to 0$，输出趋零。比 ReLU 更平滑，梯度不会突变。

### 2.3 Pre-Normalization

**原理**：原版 Transformer 的 Post-Norm 是 $y = \text{LN}(F(x) + x)$，Pre-Norm 改为 $y = x + F(\text{LN}(x))$。

Post-Norm 问题：随层数增加，原始输入分量逐渐衰减。
Pre-Norm 优势：每层残差直接累加到恒等分支，训练更稳定，不需要 warmup。

```
     |
     |------->|
  LayerNorm   |  (Pre-Norm: LN 在 Attention 之前)
     |        |
  Attention   |
     |<-------|  (残差连接)
     |------->|
  LayerNorm   |
     |        |
  FFN         |
     |<-------|
     |
```

**代码**：

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
class GPT2DecoderBlock(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = MultiHeadScaleDotProductAttention(dim, dim, heads)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, X, mask=None):
        # Pre-Norm: LN → Attention → Residual
        X_ln = self.ln1(X)
        X_attn = self.attn(X_ln, X_ln, X_ln, mask=mask)
        X = X + X_attn
        # Pre-Norm: LN → FFN → Residual
        X_ln = self.ln2(X)
        X_ffn = self.ffn(X_ln)
        X = X + X_ffn
        return X
```

**关键洞察**：Pre-Norm 下深层网络的方差会逐层增大（因为残差不断累加），所以 GPT-2 在最终输出前加一个 LayerNorm。同时，GPT-2 使用特殊初始化：残差层权重缩放 $1/\sqrt{2N}$（N 为层数），控制方差增长。

### 2.4 KV Cache

**原理**：自回归推理时，第 $t$ 步只需要 $q_t$ 与 $k_{1:t}, v_{1:t}$ 做注意力。但朴素实现每步都重新计算所有 Q/K/V——$k_{1:t-1}$ 和 $v_{1:t-1}$ 是重复计算。KV Cache 缓存历史 K/V，每步只计算新 token 的 k_t, v_t 并追加。

**推理两阶段**：
- **Prefill**：首次前向，处理完整 prompt，填充 KV Cache（compute-bound，与训练类似）
- **Decoding**：逐 token 生成，每步只输入 1 个 token（memory-bound，频繁加载 KV Cache）

**代码**：

```python
# 来自 lecture/lc3_gpt/KVCache.ipynb
class AttentionKVCache(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.dim = dim
        self.KV_cache = None

    def forward(self, x, mask):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # 更新 KV Cache
        if self.KV_cache is None:
            self.KV_cache = [k, v]
        else:
            self.KV_cache[0] = torch.cat((self.KV_cache[0], k), dim=1)
            self.KV_cache[1] = torch.cat((self.KV_cache[1], v), dim=1)

        # q 只有当前 token，但 K/V 是完整历史
        s = q @ self.KV_cache[0].transpose(2, 1) / math.sqrt(self.dim)

        # Mask 只取最后一行（当前 token 对所有历史的 mask）
        if self.KV_cache is not None:
            mask = mask[-1, :].unsqueeze(0).unsqueeze(1)
        s = s + mask

        p = F.softmax(s, dim=-1)
        z = p @ self.KV_cache[1]
        return self.wo(z)
```

```python
# 来自 lecture/lc3_gpt/KVCache.ipynb
# 推理循环：input_ids 不再累增，只传最新 token
def generation_kvcache(model, input_ids, max_new_token=100):
    input_len = input_ids.shape[1]
    output_ids = input_ids.clone()
    for i in range(max_new_token):
        with torch.no_grad():
            logits = model(input_ids, cur_len=input_len + i)
        next_token = torch.argmax(F.softmax(logits[:, -1, :], dim=-1), dim=-1, keepdim=True)
        input_ids = next_token  # 关键：只传新 token
        output_ids = torch.cat([output_ids, next_token], dim=-1)
    return output_ids
```

**关键洞察**：KV Cache 存储量 = `batch × seq_len × dim × num_layers × 2（K+V）× bytes`，随序列长度线性增长。这是 LLM 推理的核心内存瓶颈，催生了 GQA、MQA、MLA 等 KV 压缩技术。

### 2.5 Perplexity（困惑度）

**原理**：PPL 是评价语言模型拟合质量的标准指标：

$$\text{PPL} = \exp\left(-\frac{1}{L}\sum_{t=1}^{L} \log p_\theta(x_t | x_{<t})\right) = \exp(\text{CrossEntropyLoss})$$

PPL 越低，模型对测试集的预测越准确。直觉：PPL = k 意味着模型在每个位置平均"犹豫"于 k 个等概率候选词之间。

**代码**：

```python
# 来自 lecture/lc3_gpt/Perplexity.ipynb
IGNORE_INDEX = -100
labels = torch.zeros_like(input_ids, dtype=torch.long)
labels[:, :-1] = input_ids[:, 1:]
labels[:, -1] = IGNORE_INDEX

loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
loss = loss_fn(logits.view(batch_size * seq_len, vocab_size),
               labels.view(batch_size * seq_len))
ppl = loss.exp()

# 手动计算：取每个位置的预测概率 → 取 log → 平均 → exp
p = F.softmax(logits, dim=-1)
p_gather = p[0, :-1, :].gather(index=input_ids[0, 1:, None], dim=1)
ppl_manual = -(p_gather.log()).mean()  # 等价于 cross-entropy
```

### 2.6 完整 GPT-2 模型

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
@dataclass
class GPT2Config:
    vocab_size: int = 64
    max_len: int = 512
    dim: int = 512
    heads: int = 8
    num_layers: int = 6
    pad_token_id: int = 0
    initializer_range: float = 0.02

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embd = GPT2InputLayer(config.vocab_size, config.dim, config.max_len)
        self.decoder = nn.ModuleList(
            [GPT2DecoderBlock(config.dim, config.heads) for _ in range(config.num_layers)]
        )
        self.ln = LayerNorm(config.dim)  # 最终 LayerNorm
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.cache_mask = torch.tril(torch.ones(config.max_len, config.max_len))
        self._init_weights(self)

    def forward(self, x):
        bs, seq_len = x.shape
        add_mask = get_add_mask(x, self.cache_mask[:seq_len, :seq_len])
        X = self.embd(x)
        for block in self.decoder:
            X = block(X, mask=add_mask)
        X = self.ln(X)  # GPT-2 特有：最终 LN
        logits = self.lm_head(X)
        return logits

    def _init_weights(self, module):
        """GPT-2 特殊初始化：残差层权重缩放 1/√(2N)"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        for name, p in module.named_parameters():
            if name == "WO.weight":
                p.data.normal_(mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.num_layers))
```

### 2.7 Additive Mask（加法 Mask）

**原理**：GPT-2 实现中 mask 改用加法形式：非法位置加 $-\infty$（而非乘法）。好处：直接加到 score 上，无需 where 操作，更适合 multi-head 广播。

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
def get_add_mask(mask, tril_mask, neg_inf=-100000.0):
    # mask: [bs, seq_len] 的 attention mask（1=有效, 0=pad）
    batch_mask = mask.unsqueeze(2) * mask.unsqueeze(1) * tril_mask  # [bs, seq, seq]
    add_mask = (1 - batch_mask) * neg_inf
    return add_mask

# 使用：S = S + add_mask[:, None, :, :]  (广播到 head 维度)
```

---

## 三、推理采样策略

### 3.1 Greedy Search

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
next_token_idx = torch.argmax(probs, dim=-1, keepdim=True)
```

### 3.2 Temperature Sampling

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
logits = logits[:, -1, :] / temperature  # T < 1 更确定, T > 1 更随机
probs = F.softmax(logits, dim=-1)
next_token_idx = torch.multinomial(probs, num_samples=1)
```

### 3.3 Top-K Sampling

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
value, idx = torch.topk(probs, k=top_k, dim=-1)
new_logits = torch.ones_like(probs) * -100000.0
new_logits[:, idx] = logits[:, idx]  # 只保留 top-k
probs = F.softmax(new_logits, dim=-1)
next_token_idx = torch.multinomial(probs, num_samples=1)
```

### 3.4 Top-P (Nucleus) Sampling

```python
# 来自 lecture/lc3_gpt/GPT-2.ipynb
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

for i in range(bs):
    idx = torch.where(cumsum_probs[i, :] > top_p)
    sorted_probs[i, idx[0][0]:] = 0  # 截断累积概率超过 p 的部分

# 映射回原词表位置 + 归一化 + 采样
topp_probs[i, sorted_indices[i, :]] = sorted_probs[i, :]
probs = topp_probs / topp_probs.sum(dim=-1, keepdim=True)
next_token_idx = torch.multinomial(probs, num_samples=1)
```

**关键洞察**：Top-K 的 K 是固定的，但不同 token 位置的概率分布差异很大——有些位置只有 2-3 个合理候选，有些有 50+。Top-P 自适应调整候选集大小。实际使用中常 Top-K 粗筛 + Top-P 精筛。

---

## 四、In-Context Learning（ICL）

**原理**：改变输入（加入任务描述 + 示例），不改变参数，利用预训练模型的泛化能力完成新任务。

- **Zero-shot**：只有任务描述，无示例
- **One-shot / Few-shot**：加入 1 个或多个示例

$$p_\theta(\text{output} | \text{input}, \text{prompt})$$

**代码**：

```python
# 来自 lecture/lc3_gpt/in_context_learning_inference.ipynb
# Zero-shot
prompt = "中文{我有一个梦想} 中译英> "

# One-shot
example = "中文{小模型} 中译英> small model\n"
prompt = example + "中文{我有一个梦想} 中译英> "

# Few-shot
examples = [
    "中文{小模型} 中译英> small model\n",
    "中文{大语言模型} 中译英> Large Language Model\n",
]
prompt = ''.join(examples) + "中文{我有一个梦想} 中译英> "
```

**关键洞察**：ICL 之所以 work，是因为预训练语料中天然包含"翻译模式"（如网页中的中英对照）。Few-shot 示例在 latent space 中激活了特定的推理模式特征——这不是学习新知识，而是"提示"模型调用已有知识。

---

## 五、预训练数据构建

```python
# 来自 lecture/lc3_gpt/LM_dataset.ipynb
# 数据分块：连续文本按固定长度切分
data_block_size = 512
data_block = [my_data[i:i+data_block_size]
              for i in range(0, len(my_data), data_block_size)]

# Dataset + Collate：自动构建 input_ids, labels, attention_mask
def paddding_collate_fn(batch_data, pad_token_id):
    # ... padding to longest ...
    # Label = input_ids 左移一位
    labels[:, 0:max_input_len-1] = input_ids[:, 1:max_input_len]
    return {
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'labels': labels,
    }
```

---

## 六、配套实操

> 完整代码见：`/tmp/ma-rlhf/lecture/lc3_gpt/`
> - `GPT-2.ipynb` — 完整模型 + 训练 + 推理
> - `GELU.ipynb` — GELU 推导与可视化
> - `KVCache.ipynb` — KV Cache 实现与分析
> - `Pre-Normalization.ipynb` — Pre-Norm vs Post-Norm 分析
> - `Perplexity.ipynb` — PPL 计算
> - `LM_dataset.ipynb` — 预训练数据封装
> - `in_context_learning_inference.ipynb` — ICL 实操
> - `/tmp/ma-rlhf/notebook/GPT-loss.ipynb` — Decoder-only Loss 手撕

---

## 七、关键洞察与总结

1. **GPT 的本质**：Decoder-only Transformer + Causal LM = 通用文本生成模型。训练任务极简（next-token prediction），但通过海量数据和参数 scaling，涌现出惊人的泛化能力。

2. **Pre-Norm 胜出的原因**：残差连接 $y = x + F(\text{LN}(x))$ 中，原始输入 $x$ 的信号不经过归一化，直接传递到深层。Post-Norm 中原始信号每过一层都被 LN "压缩"，信息逐层衰减。

3. **KV Cache 的本质**：利用因果 mask 的性质——历史 token 的 K/V 不依赖未来 token，可以缓存复用。这将推理复杂度从 $O(n^2)$ 降到 $O(n)$（每步只算 1 个 q 与 n 个 k/v 的注意力）。

4. **采样策略的哲学**：LLM 是概率模型，生成是采样过程。Greedy 确定但单调，Temperature 控制"创造力"，Top-K/P 截断"胡说八道"的长尾。好的采样策略在创造力和可靠性之间取平衡。

5. **ICL 不是训练**：参数不变，只是输入变了。预训练语料的多样性决定了 ICL 的上限——模型见过的"模式"越多，zero-shot 能力越强。

6. **监督 → 无监督的范式转变**：
   - 监督学习：数据昂贵、任务特化、泛化差
   - 自监督预训练：数据免费（纯文本）、通用表征、泛化强
   - CLM 让数据既是输入也是标签，真正的"数据 scaling 无上限"
