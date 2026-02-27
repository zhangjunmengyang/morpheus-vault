---
title: "Tensor Parallel 手撕实操"
brief: "Megatron-LM张量并行完整实现：列并行/行并行Linear层（MLP/Attention分割策略）、all-reduce通信位置、序列并行（LayerNorm/Dropout通信消除）、Flash-Attn CP集成，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, tensor-parallel, megatron, distributed-training, pytorch]
related:
  - "[[Projects/MA-RLHF/xtrain/xtrain-03b-ZeRO-手撕实操|ZeRO-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-05b-Pipeline-Parallel-手撕实操|Pipeline-Parallel-手撕实操]]"
  - "[[Projects/MA-RLHF/lc-comm/lc-comm-01-分布式训练通信原语-手撕实操|分布式训练通信原语-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-07b-MoE-Context-Parallel-手撕实操|MoE-Context-Parallel-手撕实操]]"
---

# 张量并行（Tensor Parallelism）手撕实操

> 来源：MA-RLHF xtrain (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

1. [前置：DDP 数据并行](#1-前置ddp-数据并行)
2. [列并行 Linear（Column Parallel）](#2-列并行-linearcolumn-parallel)
3. [行并行 Linear（Row Parallel）](#3-行并行-linearrow-parallel)
4. [MLP 张量并行](#4-mlp-张量并行)
5. [Attention 张量并行（含 GQA）](#5-attention-张量并行含-gqa)
6. [Embedding 并行](#6-embedding-并行)
7. [完整模型 TP（XtrainModel）](#7-完整模型-tpxtrainmodel)

---

## 1. 前置：DDP 数据并行

张量并行的基础是理解分布式训练中梯度同步的本质。DDP（DistributedDataParallel）是最基础的并行策略：

**核心机制**：
- 各 rank 上参数**同步**，数据**不同**
- 每个 rank 独立 forward/backward，得到不同 loss 但**相同梯度**（通过 all-reduce）
- optimizer 更新后参数保持一致

**手写 DDP 的关键洞察**：

```python
# 方案1（实际采用）：每个 rank 独立反传所有层，最终对每层梯度做聚合
# 方案2：逐层 backward → all-reduce → backward... （不采用）

def train_my_ddp(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    lr = optimizer.param_groups[0]['lr']
    bs, _ = input.shape
    _, dim_out = labels.shape
    with torch.no_grad():
        for i in range(epochs):
            output, hidden = model(input)
            loss = loss_fn(output, labels)

            # 反向传播：每层独立计算，梯度 all-reduce
            do = 2 * (output - labels) / (bs * dim_out)
            grad_w2 = do.t() @ hidden
            grad_hidden = do @ model.w2.weight        # hidden 梯度不需要聚合
            dist.all_reduce(grad_w2, dist.ReduceOp.SUM)
            grad_w2 = grad_w2 / world_size
            model.w2.weight -= lr * grad_w2

            grad_w1 = grad_hidden.t() @ input
            dist.all_reduce(grad_w1, dist.ReduceOp.SUM)
            grad_w1 = grad_w1 / world_size
            model.w1.weight -= lr * grad_w1
```

**DDP 完整训练流程**（含 DataLoader + DistributedSampler）：

```python
dataset = MyDataset(N, dim_in, dim_out)
sampler = DistributedSampler(dataset)    # sampler 定义数据分配行为
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

model = ToyModel(dim_in, dim_hidden, dim_out)
ddp_model = DDP(model)  # PyTorch 官方 DDP 封装
```

---

## 2. 列并行 Linear（Column Parallel）

### 切分方式

权重矩阵 `W[dim_in, dim_out]` 沿**列（输出维度）**切分：

```
W = [W₁ | W₂ | ... | Wₚ]    # 每个 GPU 持有 W[dim_in, dim_out/P]
```

- **输入**：每个 GPU 持有完整输入 `x[N, dim_in]`（broadcast）
- **输出**：每个 GPU 得到部分输出 `y_i = x @ Wᵢ`，shape `[N, dim_out/P]`
- **前向无通信**，各 GPU 独立计算

### 通信操作

- **Forward**：无通信（输入已 broadcast）
- **Backward**：输入梯度 `dx` 需要 **reduce**（因为每个 GPU 只有部分 W，算出的 dx 是不完整的）

### 数学推导（Notebook）

```
y = x @ W，W 按列切分为 W₁, W₂
y₁ = x @ W₁,  y₂ = x @ W₂
y = [y₁ | y₂]（按列拼接）

∂L/∂Wᵢ = xᵀ @ ∂L/∂yᵢ          # 每个 GPU 独立计算自己分片的参数梯度
∂L/∂x  = ∂L/∂y₁ @ W₁ᵀ + ∂L/∂y₂ @ W₂ᵀ  # 需要 reduce 才能得到完整的 dx
```

### 代码实现

```python
class ColFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w
        return output           # 前向无通信

    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()
        dist.reduce(grad_input, dist.ReduceOp.SUM)   # dx 需要 reduce
        grad_w = input.transpose(2, 1) @ grad_output  # dw 各 GPU 独立
        return grad_input, grad_w


class ColParallelLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out // world_size, bias=False)
        # 注意：不同 shape 对应的初始化系数不一致

    def forward(self, x):
        return ColFunction.apply(x, self.w.weight.t())
```

---

## 3. 行并行 Linear（Row Parallel）

### 切分方式

权重矩阵 `W[dim_in, dim_out]` 沿**行（输入维度）**切分：

```
W = [ W₁ ]    # W₁[dim_in/P, dim_out]
    [ W₂ ]    # 每个 GPU 持有 W[dim_in/P, dim_out]
    [ .. ]
    [ Wₚ ]
```

- **输入**：需要对应切分 `x_i = x[:, i*k:(i+1)*k]`（scatter 到各 GPU）
- **输出**：每个 GPU 得到 `y_i = x_i @ Wᵢ`，shape `[N, dim_out]`，需要 **all-reduce 求和**

### 通信操作

- **Forward**：输出需要 **all-reduce**（`y = Σ y_i`）
- **Backward**：梯度无需额外通信（每个 GPU 只维护自身参数梯度）

### 代码实现

```python
class RowFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w
        dist.all_reduce(output, dist.ReduceOp.SUM)   # 前向 all-reduce
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()              # dx 无需通信
        grad_w = input.transpose(2, 1) @ grad_output  # dw 各 GPU 独立
        return grad_input, grad_w


class RowParallelLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w = nn.Linear(dim_in // world_size, dim_out, bias=False)

    def forward(self, x):
        return RowFunction.apply(x, self.w.weight.t())
```

### 列并行 vs 行并行 通信对比

| | 列并行 (Column) | 行并行 (Row) |
|---|---|---|
| 权重切分 | 按输出维度 `dim_out/P` | 按输入维度 `dim_in/P` |
| 输入 | 完整 x（broadcast） | 切分 x（scatter） |
| 前向通信 | **无** | **all-reduce** |
| 反向通信 | dx 需要 **reduce** | **无** |
| 输出形状 | `[N, dim_out/P]`（部分） | `[N, dim_out]`（完整） |

---

## 4. MLP 张量并行

### 核心设计：列并行 → 行并行

MLP 的经典结构 `x → W₁ → act → W₂ → y` 中：

- **W₁ 用列并行**：输入完整 x，输出切分的隐状态 h_i
- **W₂ 用行并行**：输入切分的 h_i，输出完整的 y

**关键优势**：中间激活值不需要额外通信！列并行的输出恰好是行并行需要的切分输入。

### 通信分析

```
整个 MLP 只需要 1 次 all-reduce（在 W₂ 的 forward 中）
X → [ColLinear: 无通信] → act → [RowLinear: all-reduce] → Y
```

### 代码实现

```python
class mlp(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w1 = ColParallelLinear(dim_in, dim_hidden, rank, world_size)
        self.w2 = RowParallelLinear(dim_hidden, dim_out, rank, world_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())  # 列并行，无通信
        h_act = self.ReLU(h)
        output = RowFunction.apply(h_act, self.w2.w.weight.t())  # 行并行，all-reduce
        return output
```

### SwiGLU 变体（LLaMA 风格）

SwiGLU 需要两个列并行投影（gate + up），再接一个行并行：

```python
class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w1 = ColParallelLinear(dim_in, dim_hidden, rank, world_size)      # up
        self.w_act = ColParallelLinear(dim_in, dim_hidden, rank, world_size)    # gate
        self.w2 = RowParallelLinear(dim_hidden, dim_out, rank, world_size)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())
        h_act = ColFunction.apply(x, self.w_act.w.weight.t())
        h_act = self.SiLU(h_act) * h                            # gate 机制
        output = RowFunction.apply(h_act, self.w2.w.weight.t())  # all-reduce
        return output
```

---

## 5. Attention 张量并行（含 GQA）

### 切分方式

Attention 中不同 head 在特征维度上是**独立**的，天然适合张量并行：

- **WQ, WK, WV**：用**列并行**（每个 GPU 负责一部分 head）
- **WO**：用**行并行**（收集所有 head 的输出）

### GQA（Grouped Query Attention）分布式实现

GQA 中 KV head 数量少于 Q head，需要**组内共享** KV：

```
假设 8 卡, 8 头 Q, 2 头 KV:
- GPU 0,1,2,3 共享 K₁V₁
- GPU 4,5,6,7 共享 K₂V₂
```

实现要点：
1. 创建 GQA 通信组（`dist.new_group`）
2. 组内 broadcast WK/WV 参数
3. **反向时 dK/dV 需要组内 all-reduce**

### 通信操作

- **Forward**：WQ/WK/WV 列并行无通信 → attention 计算 → WO 行并行 all-reduce
- **Backward**：dK, dV 需要在 GQA 组内 **all-reduce**（因为组内多个 Q head 共享同一份 KV）

### 代码实现

```python
class AttentionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, KV_src, group):
        S = Q @ K.transpose(1, 2)
        P = torch.nn.functional.softmax(S, dim=-1)
        Z = P @ V
        ctx.save_for_backward(Q, K, V, P, S)
        ctx.custom_obj = [KV_src, group]
        return Z

    @staticmethod
    def backward(ctx, dZ):
        Q, K, V, P, S = ctx.saved_tensors
        KV_src, group = ctx.custom_obj

        dP = dZ @ V.transpose(1, 2)
        dV = P.transpose(1, 2) @ dZ

        # softmax 反向：dS = dP ⊙ diag(P) - P·Pᵀ
        dS = torch.zeros_like(P)
        for b in range(Q.shape[0]):
            for i in range(Q.shape[1]):
                I = torch.diag(P[b, i, :]) - torch.outer(P[b, i, :], P[b, i, :])
                dS[b, i, :] = dP[b, i, :] @ I

        dQ = dS.transpose(1, 2) @ K
        dK = dS.transpose(1, 2) @ Q

        # GQA 组内 dK/dV all-reduce
        # 各 GPU 各自有 dk，需要 all-reduce 聚合
        #   dk  |  dk  |  dk  |  dk
        #     \    |      |     /
        #         sum(dk)
        #           |
        #       wk ← wk + dk
        dist.all_reduce(dK, dist.ReduceOp.SUM, group)
        dist.all_reduce(dV, dist.ReduceOp.SUM, group)
        return dQ, dK, dV, None, None


class Attention(nn.Module):
    def __init__(self, dim, n_kv_heads, heads, rank=0, world_size=1):
        super().__init__()
        self.head_dim = dim // heads

        # Q: 列并行，每个 GPU 负责 heads/P 个头
        self.WQ = ColParallelLinear(dim, self.head_dim * heads, rank, world_size)
        # K, V: 列并行，GQA 组内参数一致
        self.WK = ColParallelLinear(dim, self.head_dim * heads, rank, world_size)
        self.WV = ColParallelLinear(dim, self.head_dim * heads, rank, world_size)

        # 创建 GQA 通信组
        shared_heads = heads // n_kv_heads
        gpus = list(range(world_size))
        ranks_groups = [gpus[i:i+shared_heads] for i in range(0, len(gpus), shared_heads)]
        cur_group = rank // shared_heads
        self.kv_rank_src = cur_group * shared_heads
        self.gqa_groups = [dist.new_group(ranks=r) for r in ranks_groups]
        self.cur_group = cur_group

        # 组内 broadcast KV 权重
        dist.broadcast(self.WK.w.weight.data, src=self.kv_rank_src,
                       group=self.gqa_groups[cur_group])
        dist.broadcast(self.WV.w.weight.data, src=self.kv_rank_src,
                       group=self.gqa_groups[cur_group])

        # WO: 行并行
        self.WO = RowParallelLinear(dim, self.head_dim * heads, rank, world_size)

    def forward(self, x):
        Q, K, V = self.WQ(x), self.WK(x), self.WV(x)
        Z = AttentionFunction.apply(Q, K, V, self.kv_rank_src,
                                    self.gqa_groups[self.cur_group])
        O = RowFunction.apply(Z, self.WO.w.weight.t())  # 行并行 all-reduce
        return O
```

### 分布式 GQA 下的 KV-Cache 存储思考

```
a. 组内各卡存储重复的 KV-Cache
b. 将一份 KV-Cache 分散在组内多个卡中
c. 将组 GPU 显存统一管理，page 式管理 KV-Cache
```

---

## 6. Embedding 并行

Embedding 矩阵 `E[V, H]` 有两种切分方式：

### 方式一：词表并行（Vocab Parallel）

切分词表维度：`E[V/P, H]`，每个 GPU 负责一段 token ID 范围。

**难点**：
- 需要计算 `idx_offset`（token ID 偏移到本地索引）
- 通信不均衡：可能 99% token 都在 GPU-0 上
- backward 时同一 token 多次出现需要**梯度累加**

```python
class VocabParallelEmbeddingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w, local_embedding_idx, local_idx, rank, world_size):
        local_v, dim = w.shape
        bs, seq_len = input.shape
        # 每个 GPU 只查自己负责的 token
        local_embedding = torch.zeros(bs, seq_len, dim)
        local_embedding[local_idx[0], local_idx[1], :] = \
            w[input[local_idx[0], local_idx[1]] - rank * local_v, :]
        # all_gather 收集所有 GPU 的结果
        local_embedding_list = [torch.zeros_like(local_embedding) for _ in range(world_size)]
        dist.all_gather(local_embedding_list, local_embedding)
        # 合并
        for i in range(world_size):
            if i != rank:
                idx_x, idx_y = torch.where(local_embedding_idx == rank)
                local_embedding[idx_x, idx_y, :] = local_embedding_list[i][idx_x, idx_y, :]
        return local_embedding

    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        local_v, dim = w.shape
        local_embedding_idx, local_idx, rank, world_size = ctx.custom_obj
        grad_w = torch.zeros_like(w)
        # 关键：同一 token 多次出现时梯度累加
        # 例如序列 [1,3,4,3,5]，de3 和 de3' 梯度不同，需要 de3 = de3 + de3'
        for i, j in zip(local_idx[0], local_idx[1]):
            grad_w[input[i, j] - local_v * rank] += grad_output[i, j, :]
        return input, grad_w, None, None, None, None
```

### 方式二：维度并行（Dimension Parallel）

切分嵌入维度：`E[V, H/P]`，每个 GPU 存全部 token 的一部分维度。

**优势**：实现简单，通信均衡。

```python
class ParallelEmbeddingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w, rank, world_size):
        y = w[input, :]  # 每个 GPU 取自己的维度切片
        y_list = [torch.zeros_like(y) for _ in range(world_size)]
        dist.all_gather(y_list, y)
        y = torch.cat(y_list, dim=-1)  # 拼接维度
        return y

    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        world_size, rank = ctx.custom_obj
        vocab_size, dim = w.shape
        grad_w = torch.zeros_like(w)
        bs, seq_len = input.shape
        for i in range(bs):
            for j in range(seq_len):
                grad_w[input[i, j], :] += grad_output[i, j, dim * rank: dim * (rank + 1)]
        return input, grad_w, None, None
```

### 两种方式对比

| | 词表并行 | 维度并行 |
|---|---|---|
| 切分 | `E[V/P, H]` | `E[V, H/P]` |
| 通信 | all_gather（可能不均衡） | all_gather（均衡） |
| 复杂度 | 高（idx-offset, 梯度累加） | 低 |
| 使用场景 | Megatron, LLaMA3 等 | 通用 |

---

## 7. 完整模型 TP（XtrainModel）

将所有 TP 组件组装为完整的 Transformer 模型：

```
Input IDs → Embedding(维度并行) → N × Decoder Block → RMSNorm → LM Head → Loss
```

每个 Decoder Block 内部：
```
x → RMSNorm → Attention(列并行QKV + GQA + 行并行O) → residual
  → RMSNorm → MLP(列并行W1 + 行并行W2) → residual
```

### 代码实现

```python
class XtrainModel(nn.Module):
    def __init__(self, dim, n_kv_heads, heads, num_blocks, vocab_size,
                 rank=0, world_size=1):
        super().__init__()
        self.embedding = ParallelEmbedding(dim, vocab_size, rank, world_size)
        self.decoder = Decoder(dim, n_kv_heads, heads, num_blocks, rank, world_size)
        self.rms_norm = RMSNorm(dim, rank, world_size)
        self.lm_head = LanguageModelHead(dim, vocab_size, rank, world_size)

    def forward(self, x, y):
        x = self.embedding(x)        # 维度并行 Embedding
        x = self.decoder(x)          # N × (Attention + MLP) with TP
        x = self.rms_norm(x)         # 各 GPU 参数一致，backward 梯度 reduce
        loss, logits, log_prob = self.lm_head(x, y)  # 词表并行 + fused CE
        return loss, logits, log_prob
```

### 运行配置

```python
# 8 卡 TP 运行
bs, seq_len, dim = 16, 32, 64
num_blocks, heads, n_kv_heads = 4, 8, 2
vocab_size = 512

mp.spawn(run, args=("127.0.0.1", "12801", 8), nprocs=8)
```

---

## 总结：TP 全链路通信开销

| 组件 | 切分策略 | Forward 通信 | Backward 通信 |
|---|---|---|---|
| Embedding | 维度/词表并行 | all_gather | 无/局部 |
| Attention QKV | 列并行 | 无 | dx reduce |
| Attention O | 行并行 | all-reduce | 无 |
| GQA KV | 组内共享 | 无 | dK/dV 组内 all-reduce |
| MLP W₁ | 列并行 | 无 | dx reduce |
| MLP W₂ | 行并行 | all-reduce | 无 |
| RMSNorm | 不切分 | 无 | dw reduce |
| LM Head | 词表并行 | all_gather | 局部 |

**TP vs ZeRO 的设计哲学差异**：TP 需要针对每个层设计切分策略（切分更合理），ZeRO 对任意尺寸/功能参数都可以 flatten 切分（更通用）。
