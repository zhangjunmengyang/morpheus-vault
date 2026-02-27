---
title: "Llama 手撕实操"
brief: "Llama/Llama-2/Llama-3 架构演进 + 完整 PyTorch 实现：RoPE/RMSNorm/SwiGLU/GQA/KV Cache 核心组件，从 MHA→GQA 的参数效率提升路径，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, llama, gqa, rope, rmsnorm, pytorch]
related:
  - "[[Projects/MA-RLHF/lc3/lc3-01-GPT2-手撕实操|GPT2-手撕实操]]"
  - "[[Projects/MA-RLHF/lc5/lc5-01-DeepSeek-V3-手撕实操|DeepSeek-V3-手撕实操]]"
  - "[[Projects/MA-RLHF/lc1/lc1-01-Tokenizer-Embedding-手撕实操|Tokenizer-Embedding-手撕实操]]"
  - "[[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|基础数学组件手撕]]"
---

# Llama 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

Llama（Meta, 2023.02）是 ChatGPT 发布后首个具有对标能力的开源模型，加速了整个开源 LLM 生态。架构上是 Decoder-only Transformer 的进化版，相比 GPT-2/3 的关键改进：

| 组件 | GPT-2 | Llama | 改进动机 |
|------|-------|-------|----------|
| 归一化 | LayerNorm | **RMSNorm** | 去掉 re-center，计算更简、梯度更稳 |
| 注意力 | MHA | **GQA** | KV Cache 减少 n_heads/n_kv_heads 倍 |
| 位置编码 | 绝对 sin-cos | **RoPE** | 严格相对位置表示，更好的长距离衰减 |
| FFN | GELU + 2层 | **SwiGLU** + 3层 | 门控特征选择，更精细的特征学习 |
| 长文本扩展 | 无 | **NTK-RoPE / YaRN** | 高频外推、低频内插 |

Llama 架构稳定，是各种工具链的标准适配模型（微调、推理、量化、部署）。

---

## 二、核心实现

### 2.1 RMSNorm

**原理**：LayerNorm 做 $\frac{x-\mu}{\sigma}$（re-center + re-scale），但 re-center 与 ReLU-like 激活（均值 > 0）冲突，且需计算均值和方差。RMSNorm 只做 re-scale：

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

**关键性质——尺度不变性**：$\frac{sx}{\text{RMS}(sx)} = \frac{x}{\text{RMS}(x)}$（$s > 0$）。输入缩放不影响输出，反向时梯度更稳定。

**代码**：

```python
# 来自 lecture/lc4_llama/RMSNorm.ipynb
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))  # 只有 gamma，没有 beta
        self.eps = eps

    def forward(self, x):
        mean_sq = (x ** 2).mean(-1, keepdim=True)
        x_normed = x / torch.sqrt(mean_sq + self.eps)
        return self.gamma * x_normed
```

**关键洞察**：RMSNorm vs LayerNorm 的几何差异——LayerNorm 将数据投影到过原点的超平面上（re-center），RMSNorm 将数据投影到超球面上（只缩放模长）。后者保留了特征分布的偏移信息，更易保留预训练学到的特征模式。

### 2.2 Grouped Query Attention（GQA）

**原理**：标准 MHA 每个头都有独立的 K/V 投影，KV Cache = `batch × seq × n_heads × head_dim`。GQA 让多个 Q 头共享一组 K/V 头，减少 KV Cache 存储：

| 方案 | Q 头数 | KV 头数 | KV Cache 压缩比 |
|------|--------|---------|----------------|
| MHA | H | H | 1× |
| GQA | H | H/G | G× |
| MQA | H | 1 | H× |

**代码**：

```python
# 来自 lecture/lc4_llama/GroupedQueryAttention.ipynb
class GroupQueryAttention(nn.Module):
    def __init__(self, dim=512, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.share_heads = n_heads // n_kv_heads  # 每组共享的 Q 头数

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, self.head_dim * n_kv_heads)  # KV 投影维度更小
        self.wv = nn.Linear(dim, self.head_dim * n_kv_heads)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        bsz, seq_len, dim = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Q: [bsz, seq, n_heads, head_dim]
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # K/V: [bsz, seq, n_kv_heads, head_dim]
        k = k.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # KV 头 repeat 到与 Q 头数对齐
        k = torch.repeat_interleave(k, self.share_heads, dim=1)
        v = torch.repeat_interleave(v, self.share_heads, dim=1)

        # 标准 attention 计算
        s = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        if mask is not None:
            s = s + mask[:, None, :, :]
        p = F.softmax(s, dim=-1)
        z = p @ v
        z = z.transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.wo(z)
```

**关键洞察**：GQA 为什么能 work？多头注意力存在冗余——不同头学到的 K/V 模式高度相似。GQA 从参数角度没有减少计算量（repeat 后矩阵乘法不变），但从存储角度大幅压缩 KV Cache，这是推理阶段的关键瓶颈。在模型并行场景下，GQA 天然适配——每个 GPU 分配一组 KV 头 + 对应的 Q 头。

### 2.3 RoPE（旋转位置编码）

**原理**：绝对位置编码 $PE(m)$ 加到 embedding 上，内积展开有 $P_m P_n^T$ 项但也有混合项 $E_m P_n^T$，受绝对位置干扰。RoPE 直接对 Q/K 做旋转变换：

$$\tilde{q}_m = R(m\theta) \cdot q_m, \quad \tilde{k}_n = R(n\theta) \cdot k_n$$

关键性质：$R(m\theta) R^T(n\theta) = R((m-n)\theta)$，注意力分数 $\tilde{q}_m \cdot \tilde{k}_n^T$ 只依赖相对位置 $m-n$。

每 2 个维度构成一组，独立做 2D 旋转，角度为 $m\theta_i$，其中 $\theta_i = 10000^{-2i/d}$。

**代码**：

```python
# 来自 lecture/lc4_llama/RoPE.ipynb
class RoPE(nn.Module):
    def __init__(self, dim=512, max_pos=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        m = torch.arange(0, max_pos, 1)
        i = torch.arange(0, dim // 2, 1)
        theta = base ** (-2 * i / dim)
        m_theta = torch.outer(m, theta)

        self.cos = torch.zeros(max_pos, dim)
        self.sin = torch.zeros(max_pos, dim)
        self.cos[:, 0::2] = self.cos[:, 1::2] = torch.cos(m_theta)
        self.sin[:, 0::2] = self.sin[:, 1::2] = torch.sin(m_theta)

    def apply_rope(self, X):
        """X: [bs, n_heads, seq_len, dim]"""
        bs, n_heads, seq_len, d = X.shape
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        # 旋转的线性化实现
        X_shift = torch.zeros_like(X)
        X_shift[..., 0::2] = -X[..., 1::2]
        X_shift[..., 1::2] = X[..., 0::2]

        return X * cos[None, None, :, :] + X_shift * sin[None, None, :, :]
```

**使用位置**：RoPE 作用在 Q 和 K 上（不作用在 V 上），在 multi-head split 之后、attention 计算之前。KV Cache 存储的是 apply_rope 之后的 K/V。

```python
# 伪代码 - Attention with RoPE
q, k, v = wq(X), wk(X), wv(X)
q, k, v = split_head(q), split_head(k), split_head(v)
q = apply_rope(q, sin, cos)  # RoPE 在 head split 之后
k = apply_rope(k, sin, cos)
# kv_cache 存储 rope 后的 k, v
# ... 正常 attention 计算
```

### 2.4 SwiGLU

**原理**：标准 FFN 是 `ReLU(xW_up) · W_down`。SwiGLU 引入门控机制——用独立的参数学习"哪些特征维度该通过"：

$$h = \text{Swish}(xW_{gate}) \odot (xW_{up}), \quad y = hW_{down}$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$ 是 sigmoid 加权的恒等映射，平滑版 ReLU。

3 个参数矩阵（$W_{gate}, W_{up}, W_{down}$）代替 2 个，为保持参数量不变，hidden_dim 从 $4d$ 改为 $\frac{8d}{3}$。

**代码**：

```python
# 来自 lecture/lc4_llama/SwiGLU.ipynb
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 8 // 3  # 保持总参数量与标准 FFN 相当
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)      # up projection
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)       # down projection

    def forward(self, x):
        gate = F.silu(self.w_gate(x))   # Swish 激活 = SiLU
        x_up = self.w1(x)               # 特征投影
        h = gate * x_up                  # 门控特征选择
        return self.w2(h)
```

**关键洞察**：门控的本质是 token-feature-aware 的特征选择。不同于 LayerNorm 对所有 token 施加相同的 $\gamma$，SwiGLU 的 gate 对每个 token 的每个特征维度产生不同的权重——真正的"因材施教"。

### 2.5 NTK-aware RoPE（长文本扩展）

**原理**：原始 RoPE 在超过训练长度时衰减性崩溃（远距离 attention score 震荡不衰减），PPL 爆炸。

**Position Interpolation（PI）**：将位置 $m$ 缩放为 $m/s$，等价于所有频率等比降低。问题：高频信息被大幅削弱，近距离位置区分度下降。

**NTK-RoPE**：缩放 base 而非位置——$\theta'_i = (s \cdot b)^{-2i/d}$，实现"高频少插值、低频多插值"：

$$\text{插值强度}_i = s^{2i/d} \quad \text{（随维度递增，低频维度插值更强）}$$

**代码**：

```python
# 来自 lecture/lc4_llama/NTK-aware-RoPE.ipynb
class NTKRoPE(nn.Module):
    def __init__(self, dim=512, scale=10.0, max_pos=4096, base=10000.0):
        super().__init__()
        self.base = base * scale  # 核心：缩放 base
        m = torch.arange(0, max_pos, 1)
        i = torch.arange(0, dim // 2, 1)
        theta = self.base ** (-2 * i / dim)  # 高频 theta 几乎不变，低频 theta 显著缩小
        m_theta = torch.outer(m, theta)
        # ... 同 RoPE 构建 sin/cos
```

**YaRN**：在 NTK-RoPE 基础上进一步分段处理——将频率分为三个区域：
- 高频（旋转圈数 < α）：完全外推，不插值
- 中频（α ~ β 之间）：线性混合 NTK 和 PI
- 低频（旋转圈数 > β）：完全内插（PI）

```python
# 来自 notebook/YaRN-Pytorch.ipynb (概念代码)
def ntk_part(angle, r, low, high, alpha=1, beta=32, scale=10):
    ntk_angle = angle
    pi_angle = angle / scale
    gamma = (r - alpha) / (beta - alpha)
    mix_angle = (1 - gamma) * pi_angle + gamma * ntk_angle
    out_angle = ntk_angle.clone()
    out_angle[low:high] = mix_angle[low:high]
    out_angle[high:] = pi_angle[high:]
    return out_angle
```

**关键洞察**：长文本扩展的核心矛盾——近距离位置区分度 vs 远距离衰减性。PI 牺牲近距离，NTK-RoPE 权衡两者。实践中先短后长训练（如 Llama3: 4K→8K→...，DeepSeek-V3: 32K→128K），逐步适配新的位置编码分布。

---

## 三、完整 Llama 模型

```python
# 来自 lecture/lc4_llama/Llama.ipynb
@dataclass
class LlamaConfig:
    vocab_size: int = 200
    max_len: int = 512
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4      # GQA
    num_layers: int = 6
    position_encoding_base: float = 10000.0
    pad_token_id: int = 0
    attention_bias: bool = False  # 无 bias

class LlamaDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = GroupQueryAttention(config)
        self.norm2 = RMSNorm(config.dim)
        self.ffn = SwiGLU(config.dim)

    def forward(self, X, mask=None, sin=None, cos=None):
        # Pre-Norm + GQA with RoPE
        X_norm = self.norm1(X)
        X_attn = self.attn(X_norm, mask=mask, sin=sin, cos=cos)
        X = X + X_attn
        # Pre-Norm + SwiGLU
        X_norm = self.norm2(X)
        X_ffn = self.ffn(X_norm)
        X = X + X_ffn
        return X

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd = nn.Embedding(config.vocab_size, config.dim)  # 无位置编码
        self.decoder = nn.ModuleList(
            [LlamaDecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.ln = RMSNorm(config.dim)       # 最终 RMSNorm
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # RoPE sin/cos 预计算，所有层共享
        sin, cos = create_rope(config.max_len, config.dim // config.n_heads,
                               config.position_encoding_base)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)
        self.cache_mask = torch.tril(torch.ones(config.max_len, config.max_len))

    def forward(self, x):
        bs, seq_len = x.shape
        add_mask = get_add_mask(x, self.cache_mask[:seq_len, :seq_len])
        X = self.embd(x)  # 纯 embedding，无位置编码
        for block in self.decoder:
            X = block(X, mask=add_mask, sin=self.sin, cos=self.cos)
        X = self.ln(X)
        return self.lm_head(X)
```

**关键设计**：输入层只有 Embedding，没有加位置编码。位置信息完全由 RoPE 在 attention 内部注入——这是 Llama 相比 GPT-2 的根本架构差异。

---

## 四、Benchmark 测评

```python
# 来自 lecture/lc4_llama/Benchmark.ipynb
# MMLU 三种测评方式：

# 1. 生成式判别：生成 1 个 token，直接匹配 A/B/C/D
next_token = torch.argmax(logits[:, -1, :], dim=-1)

# 2. 概率判别：比较 A/B/C/D 四个 token 的预测概率
p = F.softmax(logits[0, :], dim=0)
answer = max(['A','B','C','D'], key=lambda c: p[tokenizer.encode(c)[0]])

# 3. PPL 判别：拼接每个选项后计算 PPL，选最低的
def mmlu_ppl(prompt, tokenizer, model):
    input_ids = tokenizer.encode(prompt, return_pt=True)
    logits = model(input_ids)
    p = F.softmax(logits, dim=-1)
    p_next = p[0, :-1, :].gather(index=input_ids[0, 1:, None], dim=1)
    return -p_next.log().mean().item()
```

---

## 五、配套实操

> 完整代码见：
> - `/tmp/ma-rlhf/lecture/lc4_llama/` — Llama.ipynb, GQA, RoPE, RMSNorm, SwiGLU, NTK-RoPE, Benchmark
> - `/tmp/ma-rlhf/notebook/Llama3-GQA.ipynb` — Llama3 官方 GQA 实现解析
> - `/tmp/ma-rlhf/notebook/rope_basic.ipynb` — RoPE 基础实现
> - `/tmp/ma-rlhf/notebook/rope_analysis.ipynb` — RoPE 衰减性深度实验
> - `/tmp/ma-rlhf/notebook/rope_decay.ipynb` — PE 衰减分析
> - `/tmp/ma-rlhf/notebook/YaRN-Pytorch.ipynb` — YaRN 实现与分析

---

## 六、关键洞察与总结

1. **Llama 的四大改进统一哲学**：去冗余、提效率
   - RMSNorm：去掉 re-center → 计算减半
   - GQA：KV 头共享 → KV Cache 压缩
   - RoPE：旋转代替加法 → 严格相对位置
   - SwiGLU：门控特征选择 → 更精细的学习

2. **RoPE 是 Transformer 位置编码的终极进化**：
   - 绝对 PE：加法注入，混合项破坏纯相对性
   - RoPE：旋转注入，$R(m)R^T(n) = R(m-n)$ 严格相对
   - 维度是 head_dim（不是 dim），多头共享一份 sin/cos

3. **长文本扩展的渐进路径**：RoPE → PI（等比内插）→ NTK-RoPE（频率自适应）→ YaRN（分段混合）→ 实际训练中"先短后长"逐步拉伸

4. **GQA 与模型并行的天然配合**：如 Llama3 有 32 Q 头、8 KV 头，在 4 GPU 上部署时，每 GPU 分配 8 Q 头 + 2 KV 头，本地 repeat 后独立计算，无需跨卡通信 KV。

5. **模型架构趋势**：
   - Attention 侧：降低时空复杂度（GQA → Sparse → Linear → MLA）
   - FFN 侧：提升学习容量（增加 hidden dim → sMoE）
   - Llama 是 dense 模型的代表，DeepSeek-V3 等 MoE 模型是 sparse 的代表
