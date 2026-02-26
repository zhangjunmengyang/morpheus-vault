---
title: xtrain lc4 — 张量并行从零手写
brief: 从零手写 ColParallelLinear 和 RowParallelLinear，用 autograd.Function 实现 forward/backward 内的 AllReduce 通信。掌握 TP 中为什么 MLP 只需 2 次 AllReduce、SwiGLU 列切分对齐规则，以及 TP vs ZeRO-3 的根本差异（计算并行 vs 存储分摊）。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - distributed-training
  - tensor-parallel
  - autograd-function
  - xtrain
related:
  - "[[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]]"
  - "[[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]]"
  - "[[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]"
  - "[[分布式训练]]"
---

# xtrain lc4 — 张量并行从零手写

> 来源：`/Users/peterzhang/project/ma-rlhf/xtrain/lecture/lc4_tensor_parallelism/`
> 系列：[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]
> 难度：★★★★★（面试必考，TP 行列并行 + autograd.Function 手写 backward 通信）
> 更新：2026-02-25

---

## TL;DR

张量并行（Tensor Parallelism, TP）把**同一层权重矩阵按行或列切分**到 N 张 GPU 上，每张卡只持有参数的 1/N，且**永远不 gather 完整参数**。与 ZeRO-3（按需 gather，同一卡可以完整恢复参数）的根本区别在于：TP 的通信嵌入在前向/反向计算图内，通过 `autograd.Function` 实现，是**计算并行**而非**存储分摊**。

**两种切分方式**：

| 类型 | 权重切分 | 输入处理 | 前向通信 | 反向通信 |
|------|---------|---------|---------|---------|
| 列并行（Col-TP） | W 按列切，`[d_in, d_out/N]` | 完整广播 | 无（各卡输出不同列） | AllReduce `grad_input` |
| 行并行（Row-TP） | W 按行切，`[d_in/N, d_out]` | 按列 Scatter | AllReduce 输出 | 无（`grad_input` 各卡独立） |

**MLP 经典串联**：Col-TP → 激活函数 → Row-TP，整个 MLP 只需一次 AllReduce（Row-TP 的前向 AllReduce）。

---

## 一、核心问题：TP 解决什么？

一个 Transformer MLP 层：`W1: [d_model, 4*d_model]`，`W2: [4*d_model, d_model]`

7B 模型的 W1 单层：4096 × 16384 × 2 bytes = **128 MB**，40 层 = **5 GB**，加上激活值超出单卡。

**ZeRO-3 的局限**：每次前向需要 AllGather 完整参数（通信 = 参数大小），在超大模型 + 大 batch 下通信墙明显。

**TP 的思路**：把矩阵乘法本身并行化，每卡只做部分的矩阵乘法，结果用最少的通信聚合。不需要 gather 完整参数。

> 关键洞察：矩阵乘法的线性性允许分块计算——`X @ [W1, W2] = [X@W1, X@W2]`（列切）或 `[X1, X2] @ [W1; W2] = X1@W1 + X2@W2`（行切），两种分块都是精确等价，不是近似。

---

## 二、手撕 autograd.Function

在 TP 中，反向传播需要在 `backward()` 内做分布式通信——这无法用 PyTorch 自动微分表达，必须手写 `autograd.Function`。

### 基础写法

```python
import torch.autograd as autograd

class MyGradFuntion(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)  # 保存反向需要的张量
        output = input @ w               # 前向计算
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()   # ∂L/∂x = ∂L/∂y · W^T
        grad_w = input.t() @ grad_output   # ∂L/∂W = X^T · ∂L/∂y
        return grad_input, grad_w           # 返回值数量 = forward 的参数数量

class MyGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 4, bias=False)
    
    def forward(self, x):
        return MyGradFuntion.apply(x, self.w.weight.t())
        #                              ↑ 注意：传 weight.t()，因为 forward 写的是 input @ w
```

**规则**：
- `ctx.save_for_backward(...)` 保存 forward 中间值；`ctx.saved_tensors` 取回
- `backward` 的返回数量 = `forward` 的输入参数数量（含 `ctx`）
- 返回 `None` 表示对应输入不需要梯度

---

## 三、列并行线性层（ColParallelLinear）

**切分策略**：`W [d_in, d_out]` → 按**列**切为 N 份，每卡持有 `W_i [d_in, d_out/N]`

```
输入 X [N, d_in]  ——广播——→  每卡都有完整 X
                              ↓
卡0: X @ W0 → Y0 [N, d_out/N]    Y0 = 输出的前 1/N 列
卡1: X @ W1 → Y1 [N, d_out/N]    Y1 = 输出的中间列
...                               ...
结果 Y = cat([Y0, Y1, ...], dim=1) [N, d_out]   ← 逻辑上，但不显式 cat
```

**为什么反向需要 AllReduce `grad_input`？**

```
∂L/∂x = ∂L/∂y · W^T

∂L/∂y0 [N, d_out/N]  ·  W0^T [d_out/N, d_in]  = grad_x_0 [N, d_in]
∂L/∂y1 [N, d_out/N]  ·  W1^T [d_out/N, d_in]  = grad_x_1 [N, d_in]
...

真实的 grad_x = grad_x_0 + grad_x_1 + ...  ← 需要 AllReduce！
```

```python
class ColFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w                      # [N, d_in] @ [d_in, d_out/N] = [N, d_out/N]
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()        # [N, d_out/N] @ [d_out/N, d_in] = [N, d_in]
        dist.reduce(grad_input, dist.ReduceOp.SUM)  # ← 关键：AllReduce 汇总各卡的 grad_x
        grad_w = input.transpose(2, 1) @ grad_output  # ∂L/∂W_i（各卡独立，不需要通信）
        return grad_input, grad_w

class ColParallelLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out // world_size, bias=False)
        # 每卡参数形状：[dim_in, dim_out/N]

    def forward(self, x):
        return ColFunction.apply(x, self.w.weight.t())
    
    def all_gather_w(self):
        """收集完整权重（用于初始化/保存）"""
        w_gather = [torch.zeros(self.dim_in, self.dim_out // self.world_size)
                    for _ in range(self.world_size)]
        dist.all_gather(w_gather, self.w.weight.t())
        return torch.cat(w_gather, dim=1)
```

---

## 四、行并行线性层（RowParallelLinear）

**切分策略**：`W [d_in, d_out]` → 按**行**切为 N 份，每卡持有 `W_i [d_in/N, d_out]`  
同时输入 X 也需要按列切：`X [N, d_in]` → Scatter → 每卡 `X_i [N, d_in/N]`

```
X_i [N, d_in/N]  ×  W_i [d_in/N, d_out]  →  partial_Y_i [N, d_out]

Y = partial_Y_0 + partial_Y_1 + ...  ←  AllReduce 输出！
```

**反向时 `grad_input` 不需要 AllReduce**：

```
∂L/∂x_i = ∂L/∂y · W_i^T   [N, d_out] × [d_out, d_in/N] = [N, d_in/N]

每卡的 grad_x_i 只对应自己那份 x_i（已 scatter），所以各卡独立，不通信。
```

```python
class RowFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, w):
        ctx.save_for_backward(w, input)
        output = input @ w                         # [N, d_in/N] @ [d_in/N, d_out] = [N, d_out]
        dist.all_reduce(output, dist.ReduceOp.SUM) # ← 关键：AllReduce 汇总各卡的部分乘积
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        w, input = ctx.saved_tensors
        grad_input = grad_output @ w.t()           # [N, d_out] @ [d_out, d_in/N] = [N, d_in/N]
        # ← 不需要 AllReduce，各卡独立维护自己的 grad_input
        grad_w = input.transpose(2, 1) @ grad_output
        return grad_input, grad_w

class RowParallelLinear(nn.Module):
    def __init__(self, dim_in, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w = nn.Linear(dim_in // world_size, dim_out, bias=False)
        # 每卡参数形状：[d_in/N, d_out]

    def forward(self, x):
        return RowFunction.apply(x, self.w.weight.t())
    
    def row_partition_input(self, x):
        """把完整输入按列 Scatter 到各卡"""
        N, dim_in = x.shape
        x_shard = torch.zeros(N, dim_in // self.world_size, requires_grad=True)
        if self.rank == 0:
            scatter_list = list(x.chunk(self.world_size, dim=-1))
        else:
            scatter_list = []
        dist.scatter(x_shard, scatter_list, src=0)
        return x_shard
```

---

## 五、行列并行对比

| 维度 | 列并行（Col-TP） | 行并行（Row-TP） |
|------|----------------|----------------|
| W 切分方向 | 按**列**，`W[:,i/N:(i+1)/N]` | 按**行**，`W[i/N:(i+1)/N,:]` |
| 输入 | 完整广播（每卡都有 X） | 按**列** Scatter（每卡 X 的 1/N） |
| 输出 | 部分输出，**逻辑 cat**（不需要聚合，各卡各管自己的列） | **AllReduce** 聚合各卡部分乘积 |
| forward 通信 | 无（输入已广播） | AllReduce 输出 |
| backward 通信 | AllReduce `grad_input` | 无（各卡 `grad_input` 独立） |
| 适合位置 | 扩展层（d_model → 4*d_model） | 收缩层（4*d_model → d_model） |

---

## 六、MLP 串联：Col → Row 只需一次 AllReduce

MLP 结构：`X → W1(扩展) → 激活 → W2(收缩) → Y`

```python
class mlp(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank=0, world_size=1):
        super().__init__()
        self.w1 = ColParallelLinear(dim_in, dim_hidden, rank, world_size)   # Col-TP
        self.w2 = RowParallelLinear(dim_hidden, dim_out, rank, world_size)  # Row-TP
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())     # X[N,d] → h[N, 4d/N]，无 AllReduce
        h_act = self.ReLU(h)                                # 激活（无通信）
        output = RowFunction.apply(h_act, self.w2.w.weight.t())  # → AllReduce 一次
        return output
```

**通信分析**：
- 整个 MLP：**1 次 AllReduce**（在 Row-TP 的 forward 里）
- Col-TP 的反向会有 1 次 AllReduce（`grad_input`）
- 总计：**2 次 AllReduce per MLP 层**，通信量 = 2 × (batch × seq × d_model × 2 bytes)

对比 ZeRO-3（每层需要 AllGather W + AllReduce 梯度，通信量 ∝ 参数量），TP 的通信量 ∝ **激活值大小**，在大模型 + 小 batch 的 inference 场景更优。

---

## 七、SwiGLU 的 TP 实现

LLaMA/Qwen 使用 SwiGLU，有三个权重矩阵：W1（gate）、W_act（value）、W2（输出）

```
Y = W2 · (SiLU(W1·X) ⊙ W_act·X)
```

```python
class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank=0, world_size=1):
        # W1 和 W_act 都是 Col-TP（扩展）
        self.w1 = ColParallelLinear(dim_in, dim_hidden, rank, world_size)
        self.w_act = ColParallelLinear(dim_in, dim_hidden, rank, world_size)
        # W2 是 Row-TP（收缩）
        self.w2 = RowParallelLinear(dim_hidden, dim_out, rank, world_size)
        self.SiLU = nn.SiLU()
    
    def forward(self, x):
        h = ColFunction.apply(x, self.w1.w.weight.t())       # gate：[N, 4d/N]
        h_act = ColFunction.apply(x, self.w_act.w.weight.t())# value：[N, 4d/N]
        h_act = self.SiLU(h_act) * h                         # element-wise gate（无通信）
        output = RowFunction.apply(h_act, self.w2.w.weight.t())  # AllReduce
        return output
```

**关键**：W1 和 W_act 的列切分必须**对齐**——卡 i 的 W1_i 和 W_act_i 对应同一组 d_hidden 列，这样 SiLU(h_act_i) * h_i 才是正确的 element-wise 乘。

---

## 八、参数初始化问题

TP 中一个常见的初始化陷阱：**不同 shape 的 Linear 需要不同的初始化系数**。

PyTorch 默认用 Kaiming Uniform：`std = 1/sqrt(fan_in)`

- 完整 W：`fan_in = d_in`，`std = 1/sqrt(d_in)`
- 列并行 W：`fan_in = d_in`（输入维度不变），`std = 1/sqrt(d_in)` ✅
- 行并行 W：`fan_in = d_in/N`（每卡只看到 1/N 的输入），`std = 1/sqrt(d_in/N)` ❌（偏大）

**正确做法**：行并行权重的 fan_in 应该按完整 `d_in` 初始化，然后把初始化好的完整权重 Scatter 到各卡：

```python
# 初始化完整权重
w_full = nn.Linear(dim_in, dim_out, bias=False)  # fan_in = dim_in
# Scatter 到各卡
w_shards = torch.split(w_full.weight.data, dim_in // world_size, dim=1)
model.w.weight.data = w_shards[rank].t()
```

---

## 九、TP vs ZeRO-3 深度对比

| 维度 | TP（张量并行） | ZeRO-3（参数切分） |
|------|-------------|-----------------|
| **参数切分** | 按矩阵行/列永久切分，永不 gather | 按 flatten 位置切分，前向时 AllGather 重组 |
| **计算方式** | 每卡做不同的矩阵乘块 | 每卡做完整矩阵乘（前向 AllGather 后） |
| **通信时机** | **嵌入计算图内**（autograd.Function 里） | 计算前 AllGather，计算后 Scatter |
| **通信量** | ∝ 激活值大小（与模型大小无关） | ∝ 参数大小（与批次大小无关） |
| **扩展性** | 层内并行，需要设计切分策略 | 任意参数 flatten 切，无需设计 |
| **适合场景** | 超大模型 inference，小 batch | 大 batch 训练，参数超出单卡 |
| **实现复杂度** | 高（需要手写 autograd.Function） | 低（优化器层面接管，代码侵入性小） |

---

## 十、面试考点

**Q1：行并行和列并行各在哪里做 AllReduce？为什么不能交换？**

A：列并行在**反向（backward）** AllReduce `grad_input`；行并行在**前向（forward）** AllReduce 输出。不能交换——列并行前向各卡输出不同列，不需要通信；行并行前向各卡输出是部分乘积之和，必须 AllReduce。

**Q2：整个 MLP 需要几次 AllReduce？通信量是多少？**

A：前向 1 次（Row-TP 的输出），反向 1 次（Col-TP 的 `grad_input`），共 **2 次**。每次通信量 = `batch_size × seq_len × d_model × 2 bytes`。

**Q3：TP 为什么需要 autograd.Function，DDP/ZeRO 不需要？**

A：DDP/ZeRO 的通信（AllReduce 梯度/AllGather 参数）发生在计算图**之外**——前向结束后或 optimizer.step 前后，PyTorch 原生梯度计算不受影响。TP 的通信**嵌入在前向/反向计算流程内**（前向 AllReduce 或反向 AllReduce `grad_input`），必须在 `backward()` 钩子里显式调用，无法依赖 autograd 自动推导。

**Q4：SwiGLU TP 中 W1 和 W_act 的列切分为什么必须对齐？**

A：`SiLU(W1·X) ⊙ W_act·X` 是 element-wise 乘，要求两者的输出列对应同一组 hidden units。如果列切分不对齐（卡 0 的 W1 管前 1/2 列，卡 0 的 W_act 管后 1/2 列），element-wise 乘就会错误配对。实现时应用同一个分片索引初始化两个权重。

**Q5：为什么 TP 不适合小模型或大 batch 场景？**

A：TP 的通信量 ∝ 激活值大小（∝ batch_size × seq_len），batch 越大通信越多；而参数大小固定，batch 小时 TP 性价比更高。大 batch 训练场景下，ZeRO-2/3（通信量 ∝ 参数大小）更合算。另外 TP 要求同一机器内 NVLink 高带宽，跨机节点 TP 通信延迟过高。

**Q6：TP 中行并行权重的初始化为什么容易出错？**

A：行并行每卡 Linear 的 `fan_in = d_in/N`，PyTorch 默认 Kaiming 会用 `1/sqrt(d_in/N)` 初始化，比正确值 `1/sqrt(d_in)` 大 `sqrt(N)` 倍，导致训练初期激活爆炸。正确做法：先按完整维度初始化，再 Scatter。

---

## 十一、知识关联

- **前置**：[[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — ZeRO 切存储 vs TP 切计算
- **前置**：`autograd.Function` 基础（`custom_gradient.py`）
- **后置**：[[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]] — PP 在层间切分，TP + PP = 混合并行
- **横向**：[[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]] — 更早的 TP 原理笔记
- **深化**：Megatron-LM 的 Sequence Parallelism — 把 LayerNorm/Dropout 的激活也分到不同卡，TP 通信从 AllReduce → AllGather + ReduceScatter，显存进一步降低
- **生产对照**：FSDP + TP（2D并行）= Meta 的 PyTorch 原生混合并行方案

## See Also

- [[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — 前置：ZeRO 切存储 vs TP 切计算，根本差异对比
- [[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]] — 后置：PP 层间切分，TP+PP=混合并行
- [[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]] — 横向：更早的 TP 原理版，算法视角
- [[分布式训练]] — 分布式训练理论全景
- [[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]] — xtrain 系列课程地图
