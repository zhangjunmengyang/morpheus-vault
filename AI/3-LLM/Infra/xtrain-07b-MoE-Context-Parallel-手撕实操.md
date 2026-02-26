---
title: MoE & Context Parallel 手撕实操
brief: MoE Expert并行（all-to-all路由/token dispatch/EP通信）+ Context并行（Ring Attention：序列在N卡上分割，FlashAttention分块计算）完整实现，DeepSeek-V3 MoE专家分配代码精读，来源 MA-RLHF 教学项目。
date: 2026-02-25
type: code-practice
source: MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
tags:
  - code-practice
  - moe
  - context-parallel
  - ring-attention
  - distributed-training
  - pytorch
related:
  - "[[AI/LLM/Infra/Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]]"
  - "[[AI/LLM/Architecture/DeepSeek-V3-手撕实操|DeepSeek-V3-手撕实操]]"
  - "[[MoE 深度解析|MoE深度解析]]"
  - "[[AI/LLM/Infra/分布式训练通信原语-手撕实操|分布式训练通信原语-手撕实操]]"
---

# MoE / Context Parallel 手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

**Part A — Context Parallelism（Ring Attention）**
- [A1. Online Softmax](#a1-online-softmax)
- [A2. Ring Online Softmax（分布式）](#a2-ring-online-softmax分布式)
- [A3. Ring Attention（Ring Flash Attention V2）](#a3-ring-attentionring-flash-attention-v2)

**Part B — MoE 分布式训练**
- [B1. SMoE 前向实现](#b1-smoe-前向实现)
- [B2. SMoE 反向实现](#b2-smoe-反向实现)
- [B3. GShard 专家并行](#b3-gshard-专家并行)
- [B4. DeepSeek-V3 MoE](#b4-deepseek-v3-moe)
- [B5. 1F1B 计算通信重叠](#b5-1f1b-计算通信重叠)

---

# Part A — Context Parallelism（Ring Attention）

## 背景

上下文并行的核心挑战是 **Attention 的并行化**——token 之间不独立（需要看全序列）。其他模块（MLP、RMSNorm、Embedding）的 token 独立，可以视为数据并行。

Ring Attention 的思路：将序列切分到不同 GPU 上，每个 GPU 持有自己的 Q 块，**KV 块在设备间环形传递**，逐块计算 attention 并用 online softmax 在线融合。

---

## A1. Online Softmax

### 问题

标准 softmax 需要**两遍扫描**：第一遍求 max（或 sum），第二遍算 exp/normalize。如果数据按块到达，能否一遍完成？

### 核心推导

维护全局 `m_global`（当前最大值）和 `l_global`（归一化因子），每来一个新块：

$$
m_{new} = \max(m_{global}, m_{block})
$$
$$
l_{new} = l_{global} \cdot e^{m_{global} - m_{new}} + \sum_j e^{x_j - m_{new}}
$$

最终 softmax:
$$
p_i = \frac{e^{x_i - m_{global}}}{l_{global}}
$$

> **关键洞察**：`l_global * exp(m_global - m_new)` 这一项修正了历史累积的归一化因子——当发现新的更大值时，需要把旧的 exp 值"缩小"到新的基准。

### 实现

```python
import torch
import torch.nn.functional as F

N = 2
seq_len = 9
block_size = 3
x = torch.randn(N, seq_len)

# ---- 标准 softmax ----
p_std = F.softmax(x, dim=-1)

# ---- Online softmax ----
x_blocks = x.chunk(block_size, dim=1)
m_global = torch.ones(N, 1) * -10000.0
l_global = torch.zeros(N, 1)

for x_block in x_blocks:
    m, _ = torch.max(x_block, dim=-1, keepdim=True)
    m_new = torch.maximum(m, m_global)
    l = torch.sum(torch.exp(x_block - m_new), dim=-1, keepdim=True)
    l_global = l_global * torch.exp(m_global - m_new) + l
    m_global = m_new

p_online = (x - m_global).exp() / l_global
# p_online == p_std ✅
```

---

## A2. Ring Online Softmax（分布式）

将 Online Softmax 分布到多个 GPU：

1. **Scatter**：rank 0 将数据按列分块分发到各 rank
2. **逐 rank 流水线计算**：rank 0 → rank 1 → ... → rank N-1，传递 `(m_global, l_global)`
3. **Broadcast**：最终的 `m_global, l_global` 广播给所有 rank
4. **本地计算**：各 rank 用全局统计量计算自己那块的 softmax

> **关键洞察**：这是一个 **流水线模式**——每个 rank 依赖前一个 rank 的累积统计量。通信是 O(1) 标量传递。

```python
def run(rank, world_size):
    # Scatter 数据
    dist.scatter(x_local, x_list, src=0)

    # 流水线：从前一个 rank 接收 (m, l)，计算后传给下一个
    if rank != 0:
        dist.recv(m_global, src=rank - 1)
        dist.recv(l_global, src=rank - 1)
    _, m_global, l_global = safe_softmax_incremental(x_local, m_global, l_global)
    if rank != world_size - 1:
        dist.send(m_global, dst=rank + 1)
        dist.send(l_global, dst=rank + 1)

    # 广播最终全局统计量
    dist.broadcast(m_global, src=world_size - 1)
    dist.broadcast(l_global, src=world_size - 1)

    # 本地计算 softmax
    local_softmax = (x_local - m_global).exp() / l_global
```

---

## A3. Ring Attention（Ring Flash Attention V2）

### 架构

- 每个 rank 持有固定的 **Q 块**
- **KV 块在设备间环形传递**——每轮计算一个 (Q_local, K_remote, V_remote) 的注意力块
- 使用 online softmax 的思想融合多轮结果

```
Ring topology:  rank0 → rank1 → rank2 → ... → rank(N-1) → rank0
每轮：
  1. 用当前 KV 块计算 block attention
  2. 用 online softmax 更新全局 O, L, M
  3. 将 KV 块发送给下一个 rank
```

### Block Attention Forward

```python
def block_attention_forward(self, Q, K, V, L, M, O):
    """Flash Attention V2 的块计算核心"""
    head_dim = Q.shape[-1]
    S = Q @ K.transpose(3, 2) / math.sqrt(head_dim)
    M_local = torch.max(S, dim=-1, keepdim=True).values
    M_new = torch.maximum(M, M_local)
    L_local = torch.sum(torch.exp(S - M_new), dim=-1, keepdim=True)
    L_new = L * torch.exp(M - M_new) + L_local
    O_new = O * torch.exp(M - M_new) + torch.exp(S - M_new) @ V
    return L_new, M_new, O_new
```

### Ring Forward

```python
def step_forward(self, Q, K, V):
    L = torch.zeros(bs, heads, q_len, 1)
    M = torch.ones(bs, heads, q_len, 1) * -10000.0
    O = torch.zeros(bs, heads, q_len, head_dim)

    for i in range(self.world_size):
        L, M, O = self.block_attention_forward(Q, K, V, L, M, O)
        K, V = self.ring_comm_KV(K, V)  # 环形传递 KV

    O = O / L  # 最终归一化
    L_b = M + L.log()  # 保存 logsumexp 用于反向
    return O, L_b
```

### 环形通信（避免死锁）

> **关键洞察**：偶数 rank 先 send 后 recv，奇数 rank 先 recv 后 send，交替进行避免死锁。

```python
def ring_comm_KV(self, K, V):
    next_rank = (self.rank + 1) % self.world_size
    pre_rank = (self.rank - 1) % self.world_size
    tmp_K = torch.zeros_like(K)

    if self.rank % 2 == 0:
        dist.send(K, dst=next_rank)
        dist.recv(tmp_K, src=pre_rank)
    else:
        dist.recv(tmp_K, src=pre_rank)
        dist.send(K, dst=next_rank)
    # 同理处理 V
    return tmp_K, tmp_V
```

### Block Attention Backward

```python
def block_attention_backward(self, Q, K, V, L_b, O, dO, D):
    S = Q @ K.transpose(3, 2) / math.sqrt(head_dim)
    P = S - L_b  # 重计算 softmax（利用保存的 logsumexp）
    dV = P.transpose(3, 2) @ dO
    dP = dO @ V.transpose(3, 2)
    dS = P * (dP - D)
    dQ = dS @ K / math.sqrt(head_dim)
    dK = dS.transpose(3, 2) @ Q / math.sqrt(head_dim)
    return dQ, dK, dV
```

### Ring Backward

```python
def step_backward(self, Q, K, V, L_b, O, dO):
    dQ = torch.zeros_like(Q)
    dK, dV = torch.zeros_like(K), torch.zeros_like(V)
    D = torch.sum(O * dO, dim=-1, keepdim=True)

    for i in range(self.world_size):
        dQ_block, dK_block, dV_block = self.block_attention_backward(Q, K, V, L_b, O, dO, D)
        dQ += dQ_block  # dQ 就地累加
        dK += dK_block  # dK, dV 需要回传到原始 rank
        dV += dV_block
        K, V = self.ring_comm_KV(K, V)
    return dQ, dK, dV
```

### Striped Attention（计算均衡）

对于 Decoder-Only 模型，下三角 causal mask 导致 rank 0 只需计算 1 个块，rank N-1 需计算 N 个块。

解决方案：**调整序列块顺序**，利用"块注意力的无序性"使各 rank 计算量均衡。

```
标准：rank0 计算 1 块，rank3 计算 4 块
Striped：交换序列块 1↔3
  rank0: X00, X30, X31, X32, X33  (5 块)
  rank1: X20, X21, X22, X10, X11  (5 块)
→ 各 rank 计算量相等
```

### 其他模块的序列并行

| 模块 | 并行方式 |
|------|---------|
| MLP | 数据并行（token 独立），梯度 All-Reduce |
| RMSNorm | 反向 All-Reduce |
| Embedding | 梯度聚合 |
| **Attention** | Ring Attention（最复杂） |

---

# Part B — MoE 分布式训练

## B1. SMoE 前向实现

### 架构

```
Input x [B, S, D]
  → Gate: w_gate(x) → softmax → topk → weight, idx
  → Dispatch: 根据 idx 将 token 分发到对应专家
  → Expert Compute: 各专家独立 SwiGLU(x_select)
  → Combine: 将专家输出按 weight 加权合并回原位
```

### Gate + TopK

```python
class SMoE(nn.Module):
    def __init__(self, dim, expert_nums=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(dim) for _ in range(expert_nums)])
        self.w_gate = nn.Linear(dim, expert_nums)

    def forward(self, x):
        g = F.softmax(self.w_gate(x), dim=-1)
        weight, idx = torch.topk(g, k=self.k, dim=-1)
        ...
```

### Dispatch → Compute → Combine

> **关键洞察**：`torch.where(idx == i)` 找出被分配给专家 i 的所有 token 位置。Dispatch 将这些 token 抽出送给专家，Combine 将结果按 weight 加权放回。

```python
# Dispatch
expert_results = []
for i in range(self.expert_nums):
    cur_pos = torch.where(idx == i)
    x_select = x[cur_pos[0], cur_pos[1], :]
    if x_select.shape[0] > 0:
        expert_results.append(self.experts[i](x_select))

# Combine
y_result = torch.zeros_like(x)
for i in range(self.expert_nums):
    cur_pos = torch.where(idx == i)
    if expert_results[i] is not None:
        y_result[cur_pos[0], cur_pos[1], :] += \
            expert_results[i] * weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
```

### 负载均衡 Loss（Mixtral/Switch Transformer 风格）

```python
def load_balance_loss(self, weight, idx):
    """L_balance = N * Σ(fi * pi), fi=频率, pi=概率"""
    N = self.expert_nums
    total = bs * seq_len
    count = torch.zeros(N)
    pi_sum = torch.zeros(N)
    for i in range(N):
        cur_pos = torch.where(idx == i)
        count[i] = len(cur_pos[0]) / total
        pi_sum[i] = weight[cur_pos].sum()
    loss = N * (count / N * pi_sum / N).sum()
    return loss
```

---

## B2. SMoE 反向实现

### 反向核心挑战

1. **Top-K 不可导**：torch 实现中用 straight-through estimator
2. **y = E_i × g_i** → 两条梯度分支：`dy/dx = E_i' × g_i + E_i × g_i'`
3. **Load Balance Loss** 增加额外的 `w_gate` 梯度分支

### 专家分支反向（Dispatch → Backward → Combine）

```python
def backward(self, dy, x, expert_tmp, expert_results, weight, idx):
    # Dispatch dy to experts (same routing as forward)
    for i in range(self.expert_nums):
        cur_pos = torch.where(idx == i)
        dy_select = dy[cur_pos[0], cur_pos[1], :] * weight_scale
        dx = self.experts[i].backward(x_select, dy_select, ...)
        d_x_list.append(dx)

    # Combine expert gradients
    dexpert_x = torch.zeros_like(x)
    for i in range(self.expert_nums):
        cur_pos = torch.where(idx == i)
        dexpert_x[cur_pos] += d_x_list[i] * weight[cur_pos].unsqueeze(-1)
```

### Gate 分支反向

```python
# dgate 需要经过 softmax 的雅可比矩阵
gate_p = F.softmax(self.w_gate(x), dim=-1)
for i in range(bs):
    for j in range(seq_len):
        ds = torch.diag(gate_p[i,j,:]) - torch.outer(gate_p[i,j,:], gate_p[i,j,:])
        dgate[i,j,:] = ds @ dg[i,j,:]

self.w_gate.weight.grad = dgate.t() @ x
dx = dexpert_x + dgate_x  # 合并两条分支
```

### Load Balance 反向

> **关键洞察**：Load Balance Loss 通过 gate 概率反传到 `w_gate`，增加了一条梯度路径：`L_balance → fi → gi → gate → w_gate`。实现时在 LM 梯度基础上**累加**。

```python
def backward_load_balance(self, x, d_gi, pi):
    # 累加到 w_gate.weight.grad
    self.w_gate.weight.grad += d_w_gate
```

---

## B3. GShard 专家并行

### 核心设计

EP 的本质：**数据并行系统 + 中间层的专家并行**。

- 每个 GPU 部署 1 个专家，attention 层复制分布
- 输入按数据并行分发，gate 全局可见
- 通过 **All-to-All** 完成 token 在专家间的 dispatch/combine

### All-to-All Map

```python
def get_all_to_all_map(self, idx):
    """统计各 rank 有多少 token 要发送到各专家"""
    for i in range(N):
        pos = torch.where(idx == i)
        nums_to_experts[i] = pos[0].shape[0]
        idx_to_experts.append(pos[0])
    # All-gather：让所有 rank 知道全局分配情况
    all_to_all_map = [torch.zeros_like(nums_to_experts) for _ in range(N)]
    dist.all_gather(all_to_all_map, nums_to_experts)
```

### All-to-All 通信（同步 + 异步）

**同步版本**（避免死锁）：

```python
def all_to_all(self, send_list, recv_list):
    for i in range(self.world_size):
        if i != self.rank:
            if i > self.rank:  # 避免死锁：rank 小的先 send
                dist.send(send_list[i], dst=i)
                dist.recv(recv_list[i], src=i)
            else:
                dist.recv(recv_list[i], src=i)
                dist.send(send_list[i], dst=i)
        else:
            recv_list[i] = send_list[i]  # 本地专家直接赋值
```

**异步版本**：

```python
def all_to_all_async(self, send_list, recv_list):
    comm_ops = []
    for i in range(self.world_size):
        if i != self.rank:
            comm_ops.append(dist.P2POp(dist.isend, send_list[i], i))
            comm_ops.append(dist.P2POp(dist.irecv, recv_list[i], i))
        else:
            recv_list[i] = send_list[i]
    dist.batch_isend_irecv(comm_ops)
    for op in comm_ops:
        op.wait()
```

### GShard Forward Pipeline

```
1. Gate: w_gate(x) → softmax → topk
2. All-to-All dispatch: 按 gate 结果将 token 发送到对应专家所在的 GPU
3. Expert compute: MLP(dispatch_x)
4. All-to-All combine: 将专家输出发送回源 GPU
5. Weighted sum: y_moe = Σ expert_output * gate_weight
```

### GShard Backward

> **关键洞察**：反向的数据流向与前向相同——dispatch dy → expert backward → combine dx。梯度的 All-to-All 路由映射与前向完全一致。Attention 层和 w_gate 的梯度需要 **All-Reduce**（因为它们在所有 rank 上复制）。

---

## B4. DeepSeek-V3 MoE

### 创新点

1. **Sigmoid Gate**（替代 Softmax）+ 归一化
2. **Shared Experts**：不经过 gate，所有 token 共享
3. **Auxiliary-Loss-Free Load Balancing**：可学习 bias 项
4. **Sequence-wise Auxiliary Loss**

### 实现

```python
class DeepSeekV3MoE(nn.Module):
    def __init__(self, dim, expert_nums=8, top_k=2, shared_expert_nums=4):
        super().__init__()
        # Route Experts
        self.experts = nn.ModuleList([SwiGLU(dim) for _ in range(expert_nums)])
        self.w_gate = nn.Linear(dim, expert_nums)
        self.sigmoid = nn.Sigmoid()  # 替代 Softmax

        # Auxiliary-Loss-Free: 可学习 bias
        self.bias = nn.Parameter(torch.zeros(expert_nums))

        # Shared Experts（所有 token 都经过，无 gate）
        self.shared_experts = nn.ModuleList([SwiGLU(dim) for _ in range(shared_expert_nums)])

    def forward(self, x):
        y_route, weight, idx = self.forward_route(x)
        y_shared = self.forward_shared(x)
        y = x + y_route + y_shared  # 残差
        load_loss = self.load_balance_sequence_wise(weight, idx)
        return y, load_loss
```

### 为什么用 Sigmoid 替代 Softmax？

> **关键洞察**：Softmax 的竞争性使得 gate 值之间耦合——一个专家概率升高，其他必然降低。Sigmoid 使各专家独立打分，解耦后更灵活。为保持权重归一化，topk 后做 `weight / weight.sum()`。

```python
def forward_route(self, x):
    g = self.sigmoid(self.w_gate(x))  # 独立打分
    weight, idx = torch.topk(g, k=self.k, dim=-1)
    weight_norm = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)  # 归一化
    ...
```

### Sequence-wise Auxiliary Loss

```python
def load_balance_sequence_wise(self, s, idx):
    """按序列级别计算负载均衡，适用于推理 prefill 阶段"""
    for k in range(bs):
        fi = Nr / (k * seq_len) * seq_expert_count  # 频率
        pi = (s / s.sum(dim=-1, keepdim=True)).sum(dim=0) / seq_len  # 概率
        l_bal += (fi * pi).sum() / seq_len
    return alpha * l_bal
```

---

## B5. 1F1B 计算通信重叠

### 核心思想

标准执行：`F0(attn) → F0(expert_a2a + mlp) → B1(expert_a2a + mlp) → B1(attn)`

通信时间无法隐藏。但如果有**两份数据**（x0 做前向，dy1 做反向），可以将通信与计算交错：

### 重叠方案

```
标准：  F0_attn | F0_disp → F0_mlp → F0_comb | B1_disp → B1_mlp → B1_comb | B1_attn
重叠：  F0_attn  | B1_disp‖  | B1_mlp  | F0_disp‖  | F0_mlp  | B1_comb‖  | B1_attn | F0_comb‖
         计算      通信(后台)    计算      通信(后台)    计算      通信(后台)   计算      通信(后台)
```

> **关键洞察**：All-to-All 的 dispatch 和 combine 可以异步发送（`isend/irecv`），在等待传输完成的同时执行另一条数据路径的计算。代价是需要维护两份中间变量（phase 0 和 phase 1）。

### OverlappedOp 辅助类

```python
class OverlappedOp():
    """分离通信操作，维护两个 phase 的中间变量"""
    def __init__(self, dim, num_experts, rank, world_size):
        self.all_to_all_map = []
        self.idx_to_experts = []
        self.dispatch_x = None   # forward 中间变量
        self.dispatch_dy = None  # backward 中间变量
        self.combine_y = None
        self.combine_dx = None

    def dispatch_isend(self, x):
        """异步发送 dispatch 数据"""
        send_list = [x[self.idx_to_experts[i], :] for i in range(N)]
        recv_list = [torch.zeros(...) for i in range(N)]
        comm_ops = self.all_to_all_async_isend(send_list, recv_list)
        return comm_ops, recv_list

    def combine_isend(self, y):
        """异步发送 combine 数据"""
        send_list = list(torch.split(y, split_sizes, dim=0))
        recv_list = [torch.zeros(...) for i in range(N)]
        comm_ops = self.all_to_all_async_isend(send_list, recv_list)
        return comm_ops, recv_list
```

### 重叠 Step 实现

```python
def step(self, x0, dy1, x1, h1):
    """
    computation:    F0-attn, B1-mlp,  F0-mlp,  B1-attn
    communication:  B1-disp, F0-disp, B1-comb, F0-comb
    """
    # [B1-comm] dispatch dy1 异步发送
    dispatch_dy1 = self.expert.dispatch_isend(dy1, phase=1)
    # [F0-comp] attn forward
    h0 = self.attn(x0)
    reshape_h0 = self.expert.forward_gate(h0, phase=0)

    # [F0-comm] dispatch h0 异步发送
    dispatch_h0 = self.expert.dispatch_isend(reshape_h0, phase=0)
    # [B1-comm] 等待 dy1 接收完成
    self.expert.wait(phase=1)
    # [B1-comp] expert backward mlp
    self.expert.backward_mlp(dispatch_dy1, phase=1)  # → dx

    # [B1-comm] combine dx1 异步发送
    combine_dx1 = self.expert.combine_isend(dispatch_dx, phase=1)
    # [F0-comm] 等待 h0 接收完成
    self.expert.wait(phase=0)
    # [F0-comp] expert forward mlp
    self.expert.forward_mlp(dispatch_h0, phase=0)

    # [F0-comm] combine y0 异步发送
    combine_y0 = self.expert.combine_isend(dispatch_y, phase=0)
    # [B1-comm] 等待 dx1 接收完成
    self.expert.wait(phase=1)
    # [B1-comp] attn backward
    dh1 = self.expert.backward_combine_dx(x1, combine_dx1, phase=1)
    dx1 = dh1 @ self.attn.weight

    # [F0-comm] 等待 y0 接收完成
    self.expert.wait(phase=0)
    # [F0-comp] combine moe output
    y0_moe = self.expert.forward_combine_moe(h0, combine_y0, phase=0)

    return y0_moe, dx1
```

### 推理阶段的重叠

同理可以用 `0F1F` 模式实现 prefill 阶段的计算通信重叠：两个 micro-batch 交替前向。

---

## 关键洞察总结

| 组件 | 核心要点 |
|------|---------|
| **Online Softmax** | 维护 `(m, l)` 两个标量，增量融合新块，修正历史累积 |
| **Ring Attention** | Q 固定 + KV 环形传递 + block-wise Flash Attention + online softmax 融合 |
| **Ring Backward** | dQ 就地累加，dK/dV 需跟随 KV 环形路由回原 rank |
| **Striped Attention** | 交换序列块顺序，利用注意力块计算的无序性，均衡 causal mask 下的计算量 |
| **SMoE Forward** | Gate → TopK → Dispatch → Expert Compute → Combine（加权合并） |
| **SMoE Backward** | 两条梯度分支：Expert 分支 + Gate 分支（含 softmax 雅可比） |
| **GShard** | 数据并行 + All-to-All dispatch/combine，复制层梯度 All-Reduce |
| **DeepSeek-V3** | Sigmoid gate（解耦打分） + Shared Experts + 无辅助损失负载均衡 |
| **1F1B Overlap** | 双数据流，All-to-All 异步发送与计算交错执行，代价是 2× 中间变量 |
