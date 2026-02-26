# GPT Loss + GQA 手撕实操

> 来源：`ma-rlhf/notebook/GPT-loss.ipynb` + `Llama3-GQA.ipynb`
> 涵盖：Decoder-Only Loss 计算、Perplexity、Grouped Query Attention 完整实现

---

## 1. GPT Language Model Loss

### Next-Token Prediction 的 input/target 对齐

Decoder-Only 模型的训练目标是 **预测下一个 token**，核心是 input 和 target 之间错开一位：

```
输入序列: [我,  很,  开,  心]
目标序列: [很,  开,  心,  <EOS>]
```

每个位置的 logits 对应预测**下一个位置**的 token：

```python
# 数据准备：target = input 左移一位
label = torch.roll(input_ids, shifts=-1)
label[:, -1] = -100  # 最后一位无 target，用 ignore_index 标记
```

### CrossEntropyLoss 的输入格式

PyTorch `CrossEntropyLoss` 对 3D 输入的要求：

```python
# logits: (B, vocab_size, L) — 注意 class 维度在中间！
# label:  (B, L)
loss = nn.CrossEntropyLoss(ignore_index=-100)
loss_val = loss(logits.transpose(1, 2), label)  # 或 view 展平

# 等价的展平写法（更常用）：
loss_val = loss(logits.view(B * L, vocab_size), label.view(B * L))
```

**`ignore_index=-100`**：标记不参与 loss 计算的位置（padding、最后一个 token 等），梯度为 0。

### 完整 loss 计算流程

```python
import torch
import torch.nn as nn
import math

# 1. 输入
x = torch.randn(1, 4, 512)           # (B, L, d_model)
y = torch.randint(0, 32000, (1, 4))   # (B, L) — 目标 token id

# 2. Attention（简化）
q = torch.randn(512, 512)
k = torch.randn(512, 512)
v = torch.randn(512, 512)
o = torch.randn(512, 512)
mask = torch.tril(torch.ones(1, 4, 4))  # 因果 mask

Q, K, V = x @ q, x @ k, x @ v
scores = Q @ K.transpose(1, 2) / math.sqrt(512.0)
scores = scores.masked_fill(mask == 0, float('-inf'))  # mask 未来 token
weight = torch.softmax(scores, dim=2)
attn = weight @ V @ o

# 3. FFN（简化）
mlp_up = torch.randn(512, 1024)
mlp_down = torch.randn(1024, 512)
mlp = attn @ mlp_up @ mlp_down

# 4. LM Head → logits
lm_head = torch.randn(512, 32000)
logits = mlp @ lm_head  # (1, 4, 32000)

# 5. Loss
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits.transpose(1, 2), y)  # (B, C, L) vs (B, L)
print(loss)

# 6. 推理时取最后一个位置的 argmax
pred = torch.argmax(logits, dim=2)  # (1, 4)
next_token = pred[0, -1]            # 最后一个位置 = next token
```

---

## 2. Perplexity 与 Loss 的关系

$$\text{PPL} = e^{\text{loss}} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i | x_{<i})\right)$$

| 指标 | 含义 | 直觉 |
|------|------|------|
| **Loss** | CrossEntropy，越低越好 | 模型对正确 token 的 log 概率均值 |
| **PPL** | Perplexity = exp(loss) | "模型认为有多少个等概率候选"——PPL=100 意味着模型"困惑"程度等价于 100 选 1 |

```python
def evaluate(data_loader, model, vocab_size):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch['input_ids'])
            B, L = batch['input_ids'].shape
            loss = loss_fn(logits.view(B * L, vocab_size), 
                          batch['label'].view(B * L))
            total_loss += loss.item()
            steps += 1
    avg_loss = total_loss / steps
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()
```

**注意**：PPL 只在同一 tokenizer/vocab 下可比。不同模型（GPT vs Llama）的 PPL 不能直接比较。

---

## 3. GQA 完整实现

### 核心思想

GQA（Grouped Query Attention）让多个 Q 头共享一组 KV 头：

- **MHA**：每个 Q 头有独立的 KV → n_kv_heads = n_heads
- **MQA**：所有 Q 头共享一组 KV → n_kv_heads = 1
- **GQA**：每 G 个 Q 头共享一组 KV → n_kv_heads = n_heads / G

### repeat_kv() 函数

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头复制 n_rep 次，使其与 Q 头数量对齐
    
    Args:
        x: (B, L, n_kv_heads, head_dim)
        n_rep: 每组 KV 复制的次数 = n_heads // n_kv_heads
    Returns:
        (B, L, n_heads, head_dim) — 复制后与 Q 头数一致
    
    Example: n_heads=6, n_kv_heads=2 → n_rep=3
        K1 → K1, K1, K1
        K2 → K2, K2, K2
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:  # MHA 情况，无需复制
        return x
    return (
        x[:, :, :, None, :]                               # (B, L, n_kv, 1, d_h)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)    # (B, L, n_kv, rep, d_h)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # (B, L, n_heads, d_h)
    )
```

### 完整 GQA Attention（Llama3 风格）

```python
class GQAttention(nn.Module):
    def __init__(self, dim=18, n_heads=6, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # 每组 KV 复制次数
        self.head_dim = dim // n_heads

        # Q: 完整 n_heads 个头
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # K, V: 只有 n_kv_heads 个头（参数量节省的关键！）
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # Output
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # 投影
        xq = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        # 关键：复制 KV 头以匹配 Q 头数
        keys = repeat_kv(xk, self.n_rep)     # (B, L, n_heads, d_h)
        values = repeat_kv(xv, self.n_rep)   # (B, L, n_heads, d_h)

        # transpose 为 (B, n_heads, L, d_h) 方便计算
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = (xq @ keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = scores @ values  # (B, n_heads, L, d_h)

        # Concat + Output projection
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(output)
```

### 显存节省计算

以 Llama3-8B 为例：`dim=4096, n_heads=32, n_kv_heads=8, head_dim=128`

**KV Cache 每层每 token**：
- MHA: `2 × 32 × 128 = 8192` 元素
- GQA: `2 × 8 × 128 = 2048` 元素 → **节省 75%**

**W_k + W_v 参数量**：
- MHA: `2 × 4096 × 4096 = 33.6M`
- GQA: `2 × 4096 × 1024 = 8.4M` → **节省 75%**

---

## 4. MHA → MQA → GQA 参数量对比表

以 `dim=4096, n_heads=32, head_dim=128` 为例：

| 机制 | n_kv_heads | W_k 大小 | W_v 大小 | KV Cache/token/层 | 相对 MHA |
|------|-----------|---------|---------|-------------------|---------|
| **MHA** | 32 | 4096×4096 | 4096×4096 | 2×32×128 = 8192 | 1.0× |
| **GQA-8** | 8 | 4096×1024 | 4096×1024 | 2×8×128 = 2048 | 0.25× |
| **GQA-4** | 4 | 4096×512 | 4096×512 | 2×4×128 = 1024 | 0.125× |
| **MQA** | 1 | 4096×128 | 4096×128 | 2×1×128 = 256 | 0.03× |

> GQA 是 MHA 和 MQA 的插值：n_kv_heads = n_heads 退化为 MHA，n_kv_heads = 1 退化为 MQA。

### 模型并行下的 GQA 分配

以 2 GPU、n_heads=6、n_kv_heads=2 为例：
- **GPU0**: Q1, Q2, Q3 + K1, V1 → repeat → K1K1K1, V1V1V1
- **GPU1**: Q4, Q5, Q6 + K2, V2 → repeat → K2K2K2, V2V2V2
- **输出**: GPU1 的 o4,o5,o6 发送到 GPU0，concat 后过 W_o

GQA 的优势：每个 GPU 只需存储 `n_kv_heads / n_gpu` 组 KV 权重和 Cache，天然适配张量并行。

---

## 5. 面试考点

### 考点 1：CrossEntropyLoss 的 input 和 target 怎么对齐？ignore_index 的作用？

**答**：GPT 训练时 target = input 左移一位（`torch.roll(input, -1)`），最后一位的 target 设为 -100（`ignore_index`）。`ignore_index=-100` 告诉 loss 函数跳过该位置：不计算 loss、不产生梯度。常用场景包括：padding token、序列末尾、以及 SFT 中不需要监督的 prompt 部分。PyTorch `CrossEntropyLoss` 对 3D 输入要求 shape 为 `(B, C, L)`（class 维度在 dim=1），因此通常需要 `logits.transpose(1, 2)` 或展平为 `(B*L, C)`。

### 考点 2：GQA 中 repeat_kv 的实现原理？为什么不直接用 torch.repeat_interleave？

**答**：`repeat_kv` 通过 `expand + reshape` 实现零拷贝的 KV 头复制：`expand` 不分配新内存（共享底层存储），只在 `reshape` 时才可能触发拷贝。而 `torch.repeat_interleave` 总是创建新 tensor。在 KV Cache 场景下，用 expand 可以延迟/避免内存分配。实现上：先在 `n_kv_heads` 和 `head_dim` 之间插入一个新维度 `(B, L, n_kv, 1, d_h)`，expand 到 `(B, L, n_kv, n_rep, d_h)`，再 reshape 合并为 `(B, L, n_heads, d_h)`。

### 考点 3：Perplexity 的计算和直觉含义？什么时候 PPL 不可比？

**答**：PPL = exp(avg_cross_entropy_loss)。直觉上，PPL=100 意味着模型在每个位置"平均认为有 100 个等概率候选 token"。PPL 越低模型越好。**不可比的情况**：(1) 不同 tokenizer/vocab_size——vocab 大的模型天然 PPL 更高；(2) 不同 token 粒度——BPE vs char-level 不可比；(3) 不同评测集——训练集 PPL 和测试集 PPL 混着比没有意义。正确做法是在相同 tokenizer、相同测试集下比较不同模型或训练阶段的 PPL。
