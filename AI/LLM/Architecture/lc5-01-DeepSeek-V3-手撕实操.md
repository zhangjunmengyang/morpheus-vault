---
title: "DeepSeek-V3 手撕实操"
brief: "DeepSeek-V3 核心创新完整实现：MLA（Multi-Head Latent Attention，KV cache降16x）+ MoE（Expert并行/共享专家/辅助无损负载均衡/多Token预测）+ mHC，是目前开源效率最高的MoE架构代码精读，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, deepseek, mla, moe, architecture, pytorch]
related:
  - "[[AI/LLM/Architecture/Llama-手撕实操|Llama-手撕实操]]"
  - "[[AI/LLM/Architecture/MoE 深度解析|MoE深度解析]]"
  - "[[AI/LLM/Inference/FlashAttention-手撕实操|FlashAttention-手撕实操]]"
  - "[[AI/LLM/Architecture/基础数学组件手撕|基础数学组件手撕]]"
---

# DeepSeek-V3 架构手撕实操

> 来源：MA-RLHF (<https://github.com/dhcode-cpp/MA-RLHF>)
> 入库日期：2026-02-25

---

## 一、架构全景

DeepSeek-V3（671B）是一个从推理效率导向设计的大型 MoE 模型。核心问题：**低成本训练和低成本推理哪个更重要？** V3 的回答是二者兼顾——从硬件角度（存储、计算、通信）优化每个组件。

**五大核心组件：**

| 组件 | 解决的问题 | 核心思路 |
|------|-----------|---------|
| **DeepSeek-MoE** | 扩展参数量同时控制计算量 | 共享专家 + 细粒度路由专家 + sigmoid 门控 |
| **MLA** | KV-Cache 爆炸导致推理 batch 受限 | 低秩压缩 KV 到 ~10% cache 量 + 矩阵吸收 |
| **MTP** | 提升预训练特征质量 | 递归式 next-token-prediction，引入时序 latent 特征 |
| **YaRN** | 上下文长度扩展 | 分段 RoPE（保高频+插低频）+ 动态 scaling + 注意力缩放 |
| **序列级负载均衡** | 专家并行下各 GPU 负载不均 | sequence-wise balance loss + 可学习 bias |

**设计哲学**：Attention 做序列加权组合，FFN 做特征表示，MoE 做集成学习，LM_head 做 NTP，MTP-LM_head 做时序特征预测。

**V3 完整前向计算流程**（集成版）：

```python
@dataclass
class DeepSeekV3Config:
    vocab_size: int = 200
    dim: int = 512
    n_heads: int = 8
    head_dim: int = dim // n_heads
    num_layers: int = 12
    # MoE
    expert_nums: int = 20
    shared_expert_nums: int = 4
    top_k: int = 4
    # MLA
    dc_kv: int = 32
    dc_q: int = 32
    # YaRN
    position_encoding_base: float = 10000.0
    base_scale: float = 10.0
    yarn_alpha: int = 1
    yarn_beta: int = 32
    max_len: int = 512
    # MTP
    num_mtp: int = 5
```

单个 Decode Block 结构：

```python
class DeepSeekV3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Norm1 = RMSNorm(config.dim)
        self.Norm2 = RMSNorm(config.dim)
        self.MLA = MLA(config)
        self.MoE = DeepSeekV3MoE(config)

    def forward(self, X, mask=None, sin=None, cos=None):
        X = X + self.MLA(self.Norm1(X), sin, cos)
        X_moe, weight, idx = self.MoE(self.Norm2(X))
        X = X + X_moe
        return X, weight, idx
```

---

## 二、MoE：从 SMoE 到 DeepSeek-MoE

### 2.1 标准 SMoE（Mixtral 风格）

**核心思想**：MoE 是一种集成学习方法——多个 FFN 作为"专家"，门控网络决定每个 token 路由给哪些专家，最终加权组合输出。

**Dense MoE**：所有专家都参与计算（冗余）。

**Sparse MoE**：只激活 top-k 专家，扩展参数量的同时控制计算量。

$$
G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))
$$

$$
\text{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in top } k \\ -\infty & \text{otherwise} \end{cases}
$$

**Mixtral 风格实现**（dispatch-combine 三段式）：

```python
class SparseMixtreOfExpert(MoEBasic):
    def forward(self, x):
        bsz, seq_len, dim = x.shape
        N = bsz * seq_len
        x = x.view(N, dim)

        # 0. gate
        gates = self.gate(x)
        weight = F.softmax(gates, dim=-1)
        v, idx = torch.topk(weight, dim=-1, k=self.topk)
        v /= v.sum(dim=-1, keepdim=True)

        # 1. dispatch — 按专家视角收集 token
        token_to_expert = [None] * self.num_experts
        for i in range(self.num_experts):
            token_id = torch.where(idx == i)
            if not is_empty_expert(token_id[0]):
                token_to_expert[i] = token_id

        # 2. compute — 每个专家一次性 forward 所有分配的 token
        dispatch_y = [None] * self.num_experts
        for i in range(self.num_experts):
            if token_to_expert[i] is not None:
                dispatch_x = x[token_to_expert[i][0], :]
                dispatch_y[i] = self.experts[i](dispatch_x)

        # 3. combine — 加权回填
        y = torch.zeros_like(x)
        for i in range(self.num_experts):
            if dispatch_y[i] is not None:
                cur_weight = v[token_to_expert[i][0], token_to_expert[i][1]]
                y[token_to_expert[i][0], :] += cur_weight.unsqueeze(-1) * dispatch_y[i]

        return y.reshape(bsz, seq_len, dim)
```

**Mixtral 的 Expert = SwiGLU FFN**：

```python
class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        y = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        return self.w2(y)
```

**设计动机**：FFN 对 Transformer 特征表示至关重要，在 FFN 中扩展参数量（而非 Attention）是 MoE 的核心思路。每个 token 根据门控选择不同专家，SMoE 是扩展模型容量同时减少计算量的行之有效做法。

### 2.2 DeepSeek-MoE 改进（共享专家 + 细粒度）

**三大改进**：

1. **共享专家（Shared Experts）**：不经过门控，所有 token 都通过共享专家——保证基础特征表示。
2. **Sigmoid 代替 Softmax**：门控采用 sigmoid，避免 softmax 的竞争性归一化。
3. **可学习偏置（Auxiliary-Loss-Free Load Balancing）**：在 top-k 选择时加入可学习 bias 辅助负载均衡，但计算权重时用未修正的分数。

**V3 MoE 公式**：

$$
h'_t = u_t + \sum_{i=1}^{N_s} \text{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}^{(r)}_i(u_t)
$$

$$
s_{i,t} = \text{Sigmoid}(u_t^T e_i)
$$

**修正门控权重**（bias 仅参与选择，不参与权重计算）：

$$
g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j\}, K_r) \\ 0, & \text{otherwise} \end{cases}
$$

```python
class DeepSeekV3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim) for _ in range(expert_nums)])
        self.w_gate = nn.Linear(dim, expert_nums)
        self.bias = nn.Parameter(torch.zeros(expert_nums))  # 可学习负载偏置
        self.shared_experts = nn.ModuleList([Expert(dim) for _ in range(shared_expert_nums)])

    def forward_route(self, x):
        g = F.sigmoid(self.w_gate(x))  # sigmoid 代替 softmax
        weight, idx = torch.topk(g, k=self.k, dim=-1)
        weight_norm = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
        # dispatch-compute-combine ...
        return y_result, g, idx

    def forward_shared(self, x):
        y = torch.zeros_like(x)
        for expert in self.shared_experts:
            y += expert(x)  # 无门控，全部 token 通过
        return y

    def forward(self, x):
        y_route, weight, idx = self.forward_route(x)
        y_shared = self.forward_shared(x)
        return x + y_route + y_shared, weight, idx
```

**设计动机**：
- 共享专家保证所有 token 有基础特征表示，路由专家提供专业化学习——二者是不同形式的集成学习。
- V3 有 256 个路由专家，分布式专家并行时不一定需要 256 个 GPU——可以多个专家共存于一个 GPU。

### 2.3 负载均衡（辅助 loss + top-k 梯度）

**问题**：稀疏门控导致非 top-k 专家梯度为 0，训练中专家分布不均衡。

**Switch Transformer 负载均衡 loss**（当前主流）：

$$
\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 是专家 $i$ 处理的 token 比例，$P_i$ 是专家 $i$ 的平均路由概率。

```python
def load_balance_loss_switch(idx, weight, n_experts):
    idx_mat, weight_mat = sparse_to_matrix(idx, weight, n_experts)
    fi = idx_mat.mean(dim=0)   # 每个专家处理的 token 比例
    pi = weight_mat.mean(dim=0) # 每个专家的平均权重
    return n_experts * (fi * pi).sum()
```

**V3 序列级负载均衡**（针对推理 prefill 优化）：

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i
$$

```python
def load_balance_sequence_wise(model, s, idx):
    """sequence-wise: prefill 时各 GPU 均匀分散，bsz=1 时效益显著"""
    Nr = model.expert_nums
    bs, seq_len, dim = s.shape
    l_lab = torch.zeros(1)
    for k in range(bs):
        seq_expert_count = torch.zeros(Nr)
        for i in range(Nr):
            seq_expert_count[i] = torch.where(idx[k,:,:] == i)[1].numel()
        fi = Nr / (model.k * seq_len) * seq_expert_count
        si_ = s / s.sum(dim=-1, keepdim=True)
        pi = si_.sum(dim=0) / seq_len
        l_lab += (fi * pi).sum() / seq_len
    return model.alpha * l_lab
```

**top-k 梯度问题**：

- top-k 数学上不可导，但 PyTorch 实现的 top-k 可以反传梯度（类似 embedding 层的梯度传递）。
- 只有被选中的 top-k 元素接收梯度，未选中的梯度为 0。

```python
# 手撕 top-k backward：只对被选中的元素回传梯度
dp = torch.zeros(bs, experts_num)
for i in range(bs):
    dp[i, idx[i]] = dv[i,:]  # 仅 top-k 位置有梯度
# 然后经过 softmax 的 jacobian 矩阵反传
```

---

## 三、MLA（Multi-head Latent Attention）

### 3.1 KV 压缩原理

**问题**：MHA/GQA 推理时产生大量 KV-Cache，占用越多 → 减少 decoding batching 数量 → 降低推理吞吐。

**核心思想**：MLA 是 MQA 的升级版——将 KV 压缩到一个低维 latent 空间 $c$，再通过升维投影恢复。

**MQA vs MLA**：

| | MQA | MLA |
|--|-----|-----|
| 压缩方式 | 40 头→1 头（直接共享） | 低秩投影（down→up） |
| 还原方式 | 单头 repeat 到多头 | 升维投影矩阵（learnable） |
| 表达力 | 丢失多头差异性 | 低秩分解近似满秩 $W_q = W_A W_B$ |

$$
c = W_{\text{down}}(X), \quad K = W^K_{\text{up}}(c), \quad V = W^V_{\text{up}}(c)
$$

**KV-Cache 压缩量**：

- 传统 MHA：`[2, bs, seq_len, n_kv_head * head_dim]`
- MLA：`[1, bs, seq_len, dc_kv]`（仅存 latent $c$）
- 若 `2 * n_kv_head * head_dim = 8192`，`dc_kv = 512`，**压缩到 1/16**。

**为什么 MLA 效果比 MQA 好？**

1. 预训练任务学习了有效的低秩压缩表示——高维特征本身有冗余和稀疏性
2. LoRA 证明了低秩表示有效性，MLA 的升维投影是 MQA 不具备的
3. MLA 等价于对投影矩阵做了分解 $W_q = W_A W_B$ 去近似满秩

### 3.2 手撕实现

**基础 MLA（不含 RoPE）**：

```python
class MultiHeadsLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Q: down → up
        self.wq_down = nn.Linear(dim, dc_q, bias=False)
        self.wq_up = nn.Linear(dc_q, dim, bias=False)
        # KV: 共享一个 down，分别 up
        self.wkv_down = nn.Linear(dim, dc_kv, bias=False)
        self.wk_up = nn.Linear(dc_kv, dim, bias=False)
        self.wv_up = nn.Linear(dc_kv, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
```

Forward（关键——先降维再升维）：

```python
C_Q = self.wq_down(X)   # [bs, seq, dc_q]  — latent
Q = self.wq_up(C_Q)     # [bs, seq, dim]   — 恢复

C = self.wkv_down(X)    # [bs, seq, dc_kv] — KV 共享 latent（这就是 cache 的内容！）
K = self.wk_up(C)       # [bs, seq, dim]
V = self.wv_up(C)       # [bs, seq, dim]
# 后续与标准 MHA 相同
```

**矩阵吸收**（推理优化）：

训练时用 down/up 省显存；推理时合并为满矩阵保精度：

```python
# Q 矩阵吸收：wq_up @ wq_down → 单个满矩阵
WQ = (mla.wq_up.weight.data @ mla.wq_down.weight.data).t()
Q = X @ WQ  # 一次矩阵乘替代两次

# V-O 吸收：W_UV @ W_O → 减少参数存储
w_uv_absord = W_UV @ W_O
# 原本存 dxd + cxd，现在只存 cxd
```

**位置编码分离（RoPE 适配）**：

RoPE 位置敏感——如果不分离，推理阶段要对所有历史 token 重复做 RoPE 变换。解决方案：额外增加 RoPE 专用投影：

```python
# RoPE 专用权重
self.wq_up_rope = nn.Linear(dc_q, dim, bias=False)    # Q RoPE: 多头
self.wk_head_rope = nn.Linear(dim, head_dim, bias=False)  # K RoPE: 单头（省 cache）

# Forward
R_Q = self.wq_up_rope(C_Q)  # 多头 RoPE Q
R_K = self.wk_head_rope(X)  # 单头 RoPE K（广播到多头）

# cat 操作：nope 部分 + rope 部分
Q = torch.cat((Q_nope, RoPE_Q), dim=-1)
K = torch.cat((K_nope, RoPE_K), dim=-1)

# Attention 时 scaling 也要调整
S = Q @ K.transpose(2, 3) / math.sqrt(2 * head_dim)  # cat 后维度变化
```

**KV-Cache 存什么？** 存两样东西：
1. latent $C$（`dc_kv` 维）
2. RoPE K（`head_dim` 维，单头）

总 cache 量：$(d_c + d^R_h) \times l$，远小于传统 MHA。

**完整 MLA Forward（含 RoPE、矩阵吸收）**：

```python
class MLA(nn.Module):
    def forward(self, X, sin, cos, mask=None):
        bsz, seq_len, _ = X.shape
        # latent 压缩
        C_Q = self.wq_down(X)
        Q = self.wq_up(C_Q)
        C_KV = self.wkv_down(X)
        K = self.wk_up(C_KV)
        V = self.wv_up(C_KV)
        # RoPE 分离
        R_Q = self.wq_up_rope(C_Q).view(bsz, seq_len, n_heads, head_dim).transpose(1,2)
        R_K = self.wk_head_rope(X).unsqueeze(1).repeat(1, n_heads, 1, 1)
        RoPE_Q = _apply_rotary_emb(R_Q, sin, cos)
        RoPE_K = _apply_rotary_emb(R_K, sin, cos)
        # reshape 为多头
        Q = Q.view(bsz, seq_len, n_heads, head_dim).transpose(1,2)
        K = K.view(bsz, seq_len, n_heads, head_dim).transpose(1,2)
        V = V.view(bsz, seq_len, n_heads, head_dim).transpose(1,2)
        # cat nope + rope
        Q = torch.cat((Q, RoPE_Q), dim=-1)
        K = torch.cat((K, RoPE_K), dim=-1)
        # attention
        S = Q @ K.transpose(2,3) / math.sqrt(2 * head_dim)
        P = F.softmax(S.float(), dim=-1)
        Z = (P @ V).transpose(1,2).contiguous().view(bsz, seq_len, -1)
        return self.wo(Z)
```

---

## 四、MTP（Multi-Token Prediction）

### 4.1 原理与训练目标

**传统 NTP**：给定 `[t1,t2,...,tn]`，预测 `t_{n+1}`。

**V3 MTP**：引入递归式 next-token-prediction——主体模型做 NTP，每个 MTP 头接收上一个头的 hidden state，递归预测后续 token。

**关键区别**：
- **Basic MTP**：对同一个 hidden state 用多个独立 head 预测 t+1, t+2, ...（隔空预测）
- **V3 MTP**：递归串行，每个头仍然做 NTP，但输入包含前一个头的时序 latent 特征

```
主体模型: [t1,t2,t3,t4] → 预测 [t2,t3,t4,t5]
MTP-1:    [t2,t3,t4,t5] → 预测 [t3,t4,t5,t6]   (接收主体 hidden)
MTP-2:    [t3,t4,t5,t6] → 预测 [t4,t5,t6,t7]   (接收 MTP-1 hidden)
```

**训练 Loss**：

$$
\text{loss} = \text{loss}_{\text{LM}} + \lambda \cdot \text{mean}(\text{loss}_{\text{MTP}_1}, ..., \text{loss}_{\text{MTP}_N})
$$

```python
def deepseek_v3_mtp_loss(lm_logits, mtp_logits, y, lam=0.1):
    N, bsz, seq_len, vocab_size = mtp_logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss_lm = loss_fn(lm_logits.view(-1, vocab_size), y[:,:-N].reshape(-1))
    loss_mtp = torch.zeros(N)
    for i in range(N):
        loss_mtp[i] = loss_fn(mtp_logits[i].view(-1, vocab_size),
                               y[:,i:-N+i].reshape(-1))
    return loss_lm + lam * loss_mtp.mean()
```

### 4.2 参数共享策略

**MTP 模块**共享主体模型的 embedding 和 lm_head：

```python
class MTPModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.RMSNorm_pre = RMSNorm(dim)   # 对上一个 hidden 归一化
        self.RMSNorm_cur = RMSNorm(dim)   # 对当前 embedding 归一化
        self.Proj = nn.Linear(dim*2, dim)  # cat 后投影
        self.Transformer_block = nn.Linear(dim, dim)  # 轻量解码块

    def forward(self, X_embd, H_pre):
        X_embd = self.RMSNorm_cur(X_embd)
        H_pre = self.RMSNorm_pre(H_pre)
        X = torch.cat((X_embd, H_pre), dim=-1)  # 类 RNN：cat 而非加法
        X = self.Proj(X)
        return self.Transformer_block(X)
```

**递归调用**（关键：detach 断梯度）：

```python
# MTP 递归
hidden_states = main_hidden.clone()
for i in range(num_mtp):
    X_cur = embd(x[:, i+1: i+1+seq_len-num_mtp, :])
    hidden_states_i = mtp_heads[i](X_cur, hidden_states.detach())  # detach!
    mtp_logits_i = lm_head(hidden_states_i)  # 共享 lm_head
    hidden_states = hidden_states_i
```

**Inference 并行解码**：

```python
# 主体模型预测 t5
hidden_states = model.w(model.embd(x))
next_token = torch.argmax(model.lm_head(hidden_states)[:, -1, :], dim=-1)
x = torch.cat((x, next_token.unsqueeze(1)), dim=1)

# MTP 递归预测 t6, t7, ...
for i in range(model.num_mtp):
    X = model.embd(x[:, i+1:])
    hidden_states_i = model.mtp_heads[i](X, hidden_states)
    next_token = torch.argmax(model.lm_head(hidden_states_i)[:, -1, :], dim=-1)
    x = torch.cat((x, next_token.unsqueeze(1)), dim=1)
```

**设计动机**：
- MTP 本质仍是 NTP，但在时序上引入 latent 特征——加强模型主体的 lm_head 预测能力
- MTP 模块的计算成本（单层解码块）远低于网络主体（多层解码块）
- 训练完毕后去除 MTP 头，推理时可选用于并行解码

---

## 五、YaRN 上下文扩展

### 5.1 NTK-aware + 插值

**问题**：NTK-Aware RoPE 虽然实现了维度动态插值，但高频仍有轻微内插——破坏近距离位置建模。

**反直觉事实**：**保留 short-context 位置表示能力，才是保证拓展 long-context 的前提。**

**YaRN 三大特性**：

1. **100% 保高频**：分段处理——高频不内插，低频内插，中间插值过渡
2. **外推 2x**：动态 scaling 实现训短推长
3. **注意力缩放**：长序列 softmax 丧失焦点时，用温度参数 sharp 注意力分布

**分段策略**：根据波长找高低频分界点

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{2(i-1)/d}, \quad r_i = \frac{L}{\lambda_i}
$$

- 旋转圈数 $r > \beta$ (默认 32)：**高频** → 外推（保持原样）
- 旋转圈数 $r < \alpha$ (默认 1)：**低频** → 内插（scale 因子缩放）
- 中间频段：**平滑插值过渡**

```python
# YaRN 分段：高频外推、低频内插、中频混合
low = d_half * math.log(L / (beta * 2 * math.pi)) / math.log(base)
high = d_half * math.log(L / (alpha * 2 * math.pi)) / math.log(base)

interpolation = 1.0 / (scaling_factor * freq)  # 内插
extrapolation = 1.0 / freq                      # 外推

ramp = (torch.arange(d_half) - low) / (high - low)
mask = 1 - ramp.clamp(0, 1)

inv_freq = interpolation * (1 - mask) + extrapolation * mask
```

### 5.2 实现

**Dynamic Scaling**：

```python
class YaRN(nn.Module):
    def forward(self, query, key):
        num_tokens = query.shape[0]
        self.cur_context_length = num_tokens
        # scaling 随推理长度动态变化！
        # 静态: s = max(1, L'/L)
        # 动态: s = max(1, l/L)  ← YaRN 采用
        cos, sin = self._compute_cos_sin(num_tokens)
        query = _apply_rotary_emb(query, cos, sin)
        key = _apply_rotary_emb(key, cos, sin)
        return query, key
```

**注意力分数缩放**（长序列 sharp attention）：

$$
\text{softmax}\left(\frac{q^T_m k_n}{t\sqrt{|D|}}\right), \quad \sqrt{1/t} = 0.1\ln(s) + 1
$$

```python
concentration = 0.1 * math.log(scaling_factor) + 1.0
cos = freqs.cos() * concentration  # 缩放 cos/sin
sin = freqs.sin() * concentration
```

**Dynamic scaling 能外推的原因**：
- 固定 scaling 对短距离推理不友好（位置编码突变）
- Dynamic scaling 短距离保持原有推理，随长度增加平滑缩放
- in-context learning 能力逐渐适应外推范围

---

## 六、mHC（Manifold Hyper-Connection）

mHC 是 DeepSeek 对残差连接的根本性改造——从"恒等分支 + 单残差分支"扩展为**可学习的多分支连接**。

**演进路线**：残差连接 → 多残差连接 → Hyper-Connection (HC) → mHC（流形约束）

**核心思想**：
- 传统残差：$X' = X + F(X)$（恒等分支无参数）
- HC/mHC：$X' = \sum_i \beta_i X + F(\alpha X)$（多条变换分支 + 可学习缩放因子）

**mHC 的三个缩放因子**：

$$
H^{\text{pre}}_l = \sigma(\tilde{H}^{\text{pre}}_l), \quad H^{\text{post}}_l = 2\sigma(\tilde{H}^{\text{post}}_l), \quad H^{\text{res}}_l = \text{Sinkhorn-Knopp}(\tilde{H}^{\text{res}}_l)
$$

- $H^{\text{pre}}$（对应 HC 的 $A_m$）：残差分支的输入缩放
- $H^{\text{post}}$（对应 HC 的 $B$）：残差分支的输出缩放
- $H^{\text{res}}$（对应 HC 的 $A_r$）：变换分支的缩放，**用 Sinkhorn-Knopp 约束为双随机矩阵**

**Sinkhorn-Knopp 算法**（交替行列归一化 → 双随机矩阵）：

```python
def sinkhorn_knopp_batched(A, it=20, eps=1e-8):
    """将非负矩阵转换为双随机矩阵（行列和均为 1）"""
    batch_size, n, _ = A.shape
    u = torch.ones(batch_size, n)
    v = torch.ones(batch_size, n)
    for _ in range(it):
        Av = torch.bmm(A, v.unsqueeze(2)).squeeze(2)
        u = 1.0 / (Av + eps)
        At_u = torch.bmm(A.transpose(1,2), u.unsqueeze(2)).squeeze(2)
        v = 1.0 / (At_u + eps)
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return torch.bmm(torch.bmm(U, A), V), U, V
```

**mHC Fuse 实现**（合并投影 + RMSNorm 重排序优化）：

```python
class ManifoldHyperConnectionFuse(nn.Module):
    def __init__(self, dim, rate, layer_id, max_sk_it):
        super().__init__()
        self.n = rate
        self.nc = rate * dim
        self.n2 = rate * rate
        self.norm = RMSNorm(dim * rate)
        # 统一投影矩阵
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2*self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        self.beta = nn.Parameter(torch.zeros(self.n2 + 2*self.n))

    def mapping(self, h, res_norm):
        B, L, N, D = h.shape
        h_vec = self.norm.gamma * h.reshape(B, L, N*D)
        H = h_vec @ self.w
        r_ = 1.0 / (h.reshape(B,L,-1).norm(dim=-1,keepdim=True) / math.sqrt(self.nc))
        # 分割 pre/post/res
        H_pre = F.sigmoid(r_ * H[:,:,:N] * self.alpha[0] + self.beta[:N])
        H_post = 2 * F.sigmoid(r_ * H[:,:,N:2*N] * self.alpha[1] + self.beta[N:2*N])
        H_res = sinkhorn_knopp(exp(r_ * H[:,:,2*N:] * self.alpha[2] + self.beta[2*N:]))
        return H_pre, H_post, H_res
```

**Block 内使用**：

```python
# Attention Block
h_pre, h_res = mHC_attn.process(h, H_pre, H_res)
h_out = self_attention(attn_norm(h_pre))
h = mHC_attn.depth_connection(h_res, dropout(h_out), H_post)
# FFN Block
h_pre, h_res = mHC_ffn.process(h, H_pre, H_res)
h_out = ffn(ffn_norm(h_pre))
h = mHC_ffn.depth_connection(h_res, dropout(h_out), H_post)
```

**设计动机**：DeepSeek 测试了 HC 发现稳定性/工程性问题，mHC 通过双随机矩阵约束解决——行列和均为 1 保证各分支权重守恒，避免梯度爆炸/消失。

---

## 七、关键洞察与设计哲学

### 推理优先设计

V3 的每个组件都从推理效率出发：
- **MLA**：减少 KV-Cache → 增大推理 batch → 提升吞吐
- **MoE**：稀疏激活 → 控制推理计算量（256 个专家只激活 top-k）
- **序列级负载均衡**：prefill 阶段各 GPU 负载均匀 → 减少通信等待
- **矩阵吸收**：训练时省显存，推理时合并为满矩阵保精度

### 低秩与压缩哲学

高维特征本身有冗余——MLA 的低秩投影、MoE 的稀疏激活、MTP 的参数共享，都是基于这一观察。

### 集成学习视角

- **Attention**：序列维度的加权组合
- **MoE**：特征空间的集成学习
- **MTP**：时序维度的递归预测

三者在不同维度提升模型的表达能力。

### 训练稳定性工程

- **mHC 的 Sinkhorn-Knopp 约束**：保证多分支权重守恒
- **MoE 的 sigmoid 代替 softmax**：降低门控竞争导致的不稳定
- **MTP 的 detach**：每个 MTP 头独立更新，避免深层梯度链
- **YaRN 的 dynamic scaling**：平滑过渡避免位置编码突变

### 完整训练 Loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{MTP}} \cdot \mathcal{L}_{\text{MTP}} + \lambda_{\text{bal}} \cdot \frac{1}{L}\sum_{l=1}^{L} \mathcal{L}_{\text{bal}}^{(l)}
$$

```python
def DeepSeekV3Loss(config, lm_logits, mtp_logits, y, weight_list, idx_list):
    loss_lm, loss_mtp = DeepSeekV3LMLoss(lm_logits, mtp_logits, y, lam=0.1)
    loss_load = torch.stack([
        load_balance_sequence_wise(config, s, idx)
        for s, idx in zip(weight_list, idx_list)
    ])
    return loss_lm + 0.1 * loss_load.mean()
```
