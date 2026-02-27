---
title: xtrain lc7 — MoE 专家并行从零手写
brief: 从零实现 Expert Parallelism（EP）：每张 GPU 持有部分 Expert，用 All-to-All 通信做 Token→Expert 路由（dispatch）和 Expert→Token 归聚（combine）。掌握 Top-K 路由、负载均衡 auxiliary loss、EP+TP 二维并行组合，以及 DeepSeek V3 DualPipe 在 MoE 场景下通信重叠的实现原理。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - distributed-training
  - moe
  - expert-parallel
  - xtrain
related:
  - "[[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-05b-Pipeline-Parallel-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-06-Context并行RingAttention手写]]"
  - "[[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引]]"
  - "[[AI/3-LLM/Infra/分布式训练]]"
---

# xtrain lc7 — MoE 专家并行从零手写

> 来源：`/Users/peterzhang/project/ma-rlhf/xtrain/lecture/lc7_MoE/`
> 系列：[[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引]]
> 难度：★★★★★（MoE 前向/反向 + EP All-to-All + DeepSeek-V3 创新设计）
> 更新：2026-02-26

---

## TL;DR

Sparse Mixture-of-Experts (SMoE) 在保持推理 FLOPs 不变的情况下，通过 top-k 路由让每个 token 只经过 k 个专家，使模型参数量可以大幅扩展。Expert Parallelism (EP) 把不同专家分配到不同 GPU 上，用 **All-to-All 通信**实现 dispatch（输入 → 各 expert GPU）和 combine（各 expert GPU → 输出合并）。

**演化链**：

```
SMoE Forward（单卡 dispatch-combine + 负载均衡）
    ↓
Top-k Gradient（不可导 top-k 的梯度处理）
    ↓
SMoE Backward（手写 expert 梯度 + gate 梯度 + 负载均衡梯度）
    ↓
GShard（多卡 EP，All-to-All 同步/异步）
    ↓
DeepSeek-V3 MoE（Sigmoid Gate + 无辅助损失负载均衡 + Shared Experts）
    ↓
通信-计算重叠（All-to-All 异步 + 双流 overlapped 1F1B）
```

---

## 一、核心问题：MoE 解决了什么？

标准 dense LLM：每个 token 经过所有参数，FLOPs ∝ 参数量。

**SMoE 思路**：
- 有 N 个专家（N 个独立的 FFN/SwiGLU）
- Gate 网络为每个 token 选择 top-k 个专家
- 每个 token 只激活 k/N 的参数，FLOPs 不变但参数量可扩大 N/k 倍

**EP（Expert Parallelism）**：
- 把 N 个专家分布到 N 张 GPU 上
- 每张 GPU 只维护 1 个专家的参数
- 但数据并行——每张 GPU 处理不同的 batch 数据
- 需要 All-to-All 通信：把每张卡上属于 expert_i 的 token 发送给持有 expert_i 的 GPU

---

## 二、SMoE 前向：Dispatch → Compute → Combine

```python
class SMoE(nn.Module):
    def __init__(self, dim, expert_nums=8, top_k=2):
        self.experts = nn.ModuleList([SwiGLU(dim) for _ in range(expert_nums)])
        self.w_gate = nn.Linear(dim, expert_nums)

    def forward(self, x):  # x: [bs, seq_len, dim]
        # Step 1：Gate 计算 top-k 路由
        g = F.softmax(self.w_gate(x), dim=-1)
        weight, idx = torch.topk(g, k=self.k, dim=-1)
        # weight: [bs, seq_len, k] — 专家权重
        # idx:    [bs, seq_len, k] — 选中的专家 id

        # Step 2：Dispatch — 每个专家收集属于自己的 token
        expert_results = []
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i)        # 找出选择专家 i 的所有 (batch, seq) 位置
            x_select = x[cur_pos[0], cur_pos[1], :]  # 对应 token 的 embedding
            y = self.experts[i](x_select) if x_select.shape[0] > 0 else None
            expert_results.append(y)
        
        # Step 3：Combine — 加权合并各专家输出到原始位置
        y_result = torch.zeros_like(x)
        for i in range(self.expert_nums):
            cur_pos = torch.where(idx == i)
            if expert_results[i] is not None:
                gate_weight = weight[cur_pos[0], cur_pos[1], cur_pos[2]].unsqueeze(-1)
                y_result[cur_pos[0], cur_pos[1], :] += expert_results[i] * gate_weight
        
        return y_result, weight, idx
```

**Dispatch 和 Combine 是互逆操作**：
- Dispatch：`x[token_positions] → expert_input`（稀疏gather）
- Combine：`expert_output → y[token_positions]`（稀疏scatter + 加权求和）

---

## 三、Top-k 的梯度：不可导的处理

`torch.topk` 本身不可导（选择操作是离散的）。梯度只流向**被选中的 top-k 位置**，未选中位置梯度为 0。

```python
# PyTorch autograd 自动处理：对 softmax 输出 p，top-k 只反传选中位置
v, idx = torch.topk(p, k=top_k, dim=-1)
loss = ((v - label) ** 2).sum()
loss.backward()  # p.grad 只在 idx 对应位置非零

# 手写等价实现：
dv = 2 * (v - label) / label.numel()
dp = torch.zeros(bs, expert_nums)
for i in range(bs):
    dp[i, idx[i]] = dv[i, :]  # 只填 top-k 位置

# Softmax 反向（Jacobian）：
# d_softmax(x)_i/d_x_j = p_i * (δ_ij - p_j)
# 矩阵形式：d_s = diag(p) - p·p^T
for i in range(bs):
    d_s = torch.diag(p[i]) - torch.outer(p[i], p[i])
    dy[i] = dp[i] @ d_s
```

**关键洞察**：top-k 只是一个"掩码"操作，梯度流向未被截断；gate 参数 `w_gate` 的梯度通过 softmax Jacobian 传递，但非 top-k 位置的 `dp=0` 导致对应分量梯度为 0，形成自然稀疏性。

---

## 四、SMoE 反向：手写三路梯度

```python
def backward(self, x, dy, h):
    """
    三路梯度：
    1. d_expert_x：通过 expert 权重传递的梯度
    2. d_gate_x：通过 gate 权重传递的梯度
    3. expert_w.grad：各专家 MLP 参数的梯度
    """
    L = bs * seq_len
    x_flat, dy_flat = x.reshape(L, dim), dy.reshape(L, dim)
    
    # ① 反向 Dispatch（与前向 Combine 对称）
    dispatch_dy = self.dispatch(dy_flat, self.all_to_all_map, self.idx_to_experts)
    
    # ② Expert MLP 反向
    dispatch_dx = dispatch_dy @ self.MLP.weight       # ∂L/∂expert_input
    self.MLP.weight.grad = dispatch_dy.t() @ self.dispatch_x  # ∂L/∂W_expert
    
    # ③ 反向 Combine（与前向 Dispatch 对称）
    combine_dx = self.combine(dispatch_dx, self.all_to_all_map)
    
    # ④ 加权合并（含 gate 权重）
    dexpert_x = torch.zeros_like(x_flat)
    for i in range(N):
        pos, k_pos = self.idx_to_experts[i], self.idx_to_experts_k[i]
        gi = self.weight[pos, k_pos]
        dexpert_x[pos] += combine_dx[i] * gi.unsqueeze(-1)
    
    # ⑤ Gate 权重梯度（softmax Jacobian）
    d_gi = torch.zeros(L, N)
    for i in range(N):
        pos = self.idx_to_experts[i]
        d_gi[pos, i] = (self.combine_y[i] * self.dispatch_x[pos]).sum(-1)
    
    d_gate = torch.zeros(L, N)
    for i in range(L):
        ds = torch.diag(self.gate_p[i]) - torch.outer(self.gate_p[i], self.gate_p[i])
        d_gate[i] = d_gi[i] @ ds  # softmax Jacobian
    
    self.w_gate.weight.grad = d_gate.t() @ x_flat
    return dexpert_x + d_gate_x  # 两路梯度之和
```

---

## 五、GShard EP：All-to-All 分布式路由

EP 的核心通信原语是 **All-to-All**：每张卡发送不同数据到每张其他卡，也接收来自每张卡的数据。

```
4 张卡，4 个专家，每卡持有 1 个专家：

卡0 有 tokens {a,b} 要给 expert0，{c} 要给 expert1，{d,e} 要给 expert2，{} 给 expert3
卡1 有 tokens {f} 要给 expert0，{g,h} 要给 expert1，{} 要给 expert2，{i,j} 给 expert3
...

All-to-All 后：
GPU_expert0 收到 {a,b}(来自卡0) + {f}(来自卡1) + ...
```

**实现**：

```python
def all_to_all(self, send_list, recv_list):
    """同步 All-to-All（奇偶交错防死锁）"""
    for i in range(self.world_size):
        if i != self.rank:
            if i > self.rank:      # 高序号卡先 send
                dist.send(send_list[i], dst=i)
                dist.recv(recv_list[i], src=i)
            else:                  # 低序号卡先 recv
                dist.recv(recv_list[i], src=i)
                dist.send(send_list[i], dst=i)
        else:
            recv_list[i] = send_list[i]  # 本卡数据直接拷贝

def all_to_all_async(self, send_list, recv_list):
    """异步 All-to-All（batch_isend_irecv，更高效）"""
    comm_ops = []
    for i in range(self.world_size):
        if i != self.rank:
            comm_ops.append(dist.P2POp(dist.isend, send_list[i], i))
            comm_ops.append(dist.P2POp(dist.irecv, recv_list[i], i))
        else:
            recv_list[i] = send_list[i]
    dist.batch_isend_irecv(comm_ops)  # 一次提交所有通信操作
    for op in comm_ops:
        op.wait()
```

**GShard 的 forward 完整流程**：

```python
def forward(self, x):
    # 1. Gate：top-k 路由，得到 idx
    gate = softmax(w_gate(x))
    weight, idx = topk(gate, k)
    
    # 2. 构建 All-to-All mapping（all_gather 全局 token 分布）
    a2a_map, idx_to_experts, idx_to_experts_k = self.get_all_to_all_map(idx)
    # a2a_map[i][j] = 卡 j 要发给 expert_i（本卡）的 token 数量
    
    # 3. Dispatch：All-to-All 发送 token
    dispatch_x = self.dispatch(x, a2a_map, idx_to_experts)
    # 本卡 expert 收到所有需要它处理的 token
    
    # 4. Expert 计算
    y = self.MLP(dispatch_x)
    
    # 5. Combine：All-to-All 返回结果
    combine_y = self.combine(y, a2a_map)
    # 各 token 收到对应 expert 的输出
    
    # 6. 加权合并
    y_moe = 加权sum(combine_y, gate_weight)
    return y_moe
```

**EP 中非 expert 参数的处理**：

```python
# Attention 和 w_gate 在所有卡上相同（广播同步）
model.replica_param()

# 训练时各卡看到不同数据，梯度需要 AllReduce 平均
model.all_reduce_gradient()  # 只对 attn + w_gate 做 AllReduce，expert MLP 不需要
```

---

## 六、DeepSeek-V3 MoE 三大创新

### 1. Sigmoid Gate 替代 Softmax

```python
# Mixtral：softmax gate（所有专家竞争，归一化到 1）
g = F.softmax(w_gate(x), dim=-1)

# DeepSeek-V3：sigmoid gate（每个专家独立打分，不竞争）
g = torch.sigmoid(w_gate(x))
weight_norm = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)  # 事后归一化
```

**为什么 sigmoid 更好？**
- Softmax：专家间是"零和竞争"，提高一个 expert 的分数必然降低其他的
- Sigmoid：每个 expert 独立打分，允许多个 expert 同时获得高分（正交能力可同时激活）
- 代价：权重不再天然归一化，需要事后除以权重和

### 2. Auxiliary-Loss-Free 负载均衡（无辅助损失）

传统负载均衡（Mixtral 等）：加一个 auxiliary loss 惩罚不均衡。

**问题**：auxiliary loss 会干扰主损失梯度，影响模型质量。

DeepSeek-V3 解法：给每个 expert 加一个可学习的 **bias** 参数，仅在路由时使用，不参与 gate 梯度计算：

```python
self.bias = nn.Parameter(torch.zeros(expert_nums))

# 路由时加 bias 影响选择（bias 高的 expert 被更多选中）
g_with_bias = g + self.bias
weight, idx = topk(g_with_bias, k)
weight_norm = weight / weight.sum(...)  # 用原始 g，不含 bias

# bias 更新规则：不用梯度下降，用统计更新
# 如果 expert_i 被过载 → bias_i -= gamma（降低被选概率）
# 如果 expert_i 被欠载 → bias_i += gamma（提高被选概率）
```

**效果**：负载均衡而不损害路由质量（主梯度路径不含 bias）。

### 3. Shared Experts + Sequence-Wise Auxiliary Loss

**Shared Experts**：固定 N_s 个专家，所有 token 都必须经过（不路由），加上 top-k 路由专家：

```python
def forward(self, x):
    y_route, g, idx = self.forward_route(x)   # top-k 路由专家
    y_shared = self.forward_shared(x)          # 共享专家（所有 token 都过）
    y = x + y_route + y_shared
```

**为什么需要 Shared Experts？**
- 强迫模型把"通用知识"放在 shared experts，把"专项知识"放在 routed experts
- 减少不同路由 experts 之间的知识冗余（otherwise 多个 experts 可能学习相同的东西）

**Sequence-Wise Auxiliary Loss**（inference 友好）：

```python
def load_balance_sequence_wise(self, s, idx):
    """
    基于 sequence 而非 batch 统计负载，使 prefill 时每张 GPU 负载均衡
    （推理时 batch=1，必须在 seq 级别均衡）
    fi = Nr/(k * seq_len) × count(expert_i 在 seq 中被选次数)
    pi = mean(s_{i,j}) over all tokens j
    L_bal = alpha × Σ_i (fi × pi)
    """
```

---

## 七、通信-计算重叠（All-to-All Overlap）

EP 的 All-to-All 通信时间 ≈ expert 计算时间（都很长），理想情况是二者重叠。

**关键思路**：同时处理两个 micro-batch，一个做 All-to-All 通信，一个做 expert 计算：

```
时间轴：
Microbatch 0: [Attn_F] [A2A_dispatch] [Expert_0] [A2A_combine] [Attn_B] 
Microbatch 1:           [Attn_F]    [A2A_dispatch↕重叠] [Expert_1] ...
```

**修改 1F1B 调度为 F(mlp)B(attn)F(attn)B(mlp)**：

原 1F1B：
```
[F_attn + F_mlp] → [B_attn + B_mlp]
```

Overlapped 1F1B：
```
[F_mlp_0] [F_attn_0 + A2A_dispatch_1] [B_attn_0 + A2A_combine_1] [B_mlp_0]
```

这里 F_attn 和 A2A_dispatch 并行，B_attn 和 A2A_combine 并行，实现通信掩盖。

**实现要点**（overlapped_1F1B.py 的核心）：
1. 异步 `all_to_all_async`（`batch_isend_irecv`）立即返回 handle
2. 在 handle.wait() 之前执行其他计算（attn forward/backward）
3. `OverlappedOp` 类管理"先发出通信，后等待结果"的生命周期

---

## 八、GShard EP vs DeepSeek-V3 EP 对比

| 维度 | GShard | DeepSeek-V3 |
|------|--------|------------|
| Gate | Softmax | Sigmoid + 事后归一化 |
| 负载均衡 | Auxiliary loss（干扰梯度） | bias 参数 + 统计更新（无辅助损失） |
| Shared experts | 无 | 有（N_s 个，所有 token 必过） |
| 负载均衡粒度 | Batch-level | Sequence-level（inference 友好） |
| 通信 | 同步 All-to-All | 异步 All-to-All + 通信计算重叠 |
| top-k 方式 | 标准 | biased top-k（加 bias 路由，原始权重计算） |

---

## 九、面试考点

**Q1：MoE 的 Dispatch 和 Combine 分别做什么？为什么是互逆的？**

A：Dispatch 是"按 gate 路由，把每个 token 的 embedding 发给对应 expert"；Combine 是"把各 expert 的输出按原始位置加权合并回去"。互逆原因：路由关系是 `token → expert` 的映射，forward 沿这个映射发送，backward 沿反向映射传递梯度。All-to-All 在 forward 是 dispatch（稀疏 gather），在 backward 也是一次 All-to-All（稀疏 scatter），前后向对称。

**Q2：EP 中哪些参数需要 AllReduce 梯度？哪些不需要？**

A：需要 AllReduce 的：Attention 层参数、w_gate 参数（所有卡有相同副本，但各卡数据不同，梯度需要平均）。不需要的：expert MLP 参数（每张卡持有不同 expert，梯度只由选择它的 token 产生，不需要跨卡同步）。

**Q3：DeepSeek-V3 为什么用 Sigmoid 替代 Softmax？**

A：Softmax 强制专家间零和竞争，提高一个 expert 的激活必然压低其他 expert。Sigmoid 让每个 expert 独立评分，使具有"互补能力"的多个 expert 可以同时获得高分。这与 DeepSeek-V3 的 Shared Experts 设计一致——通用知识集中在 shared，专项知识在各 routed expert 独立发展，不需要互相竞争。

**Q4：什么是 Auxiliary-Loss-Free 负载均衡？为什么优于加 aux loss？**

A：传统 aux loss 直接加到总 loss 里，梯度会干扰主损失的优化方向（尤其在 aux_weight 调参困难时）。DeepSeek-V3 用一个不参与反向传播的 bias 参数控制路由倾向：bias 通过统计每个 expert 的实际负载动态调整（过载 → 降 bias，欠载 → 升 bias），与主损失完全解耦。代价是需要引入额外的 bias 更新逻辑，不能用标准 SGD 一次更新。

**Q5：EP 的 All-to-All 通信量是多少？如何估算？**

A：每次 All-to-All（dispatch + combine 各一次）：
- 发送量：`bs × seq_len × (k/N) × dim × 2 bytes`（每个 token 平均发给 k/N 张卡）
- 接收量：相同（每张卡平均收到 `bs × seq_len × (k/N)` 个 token）
- 总通信：约 `4 × bs × seq_len × k/N × dim × 2 bytes`（一个 step 两次 A2A）

这是 EP 的通信瓶颈，DualPipe 和 overlapped 1F1B 的目标都是用计算覆盖这部分通信。

**Q6：Shared Experts 是否也参与 EP？**

A：不参与。Shared Experts 的每张卡都持有完整副本（类似 dense layer），所有 token 直接本地计算，不需要 All-to-All。只有 Routed Experts 参与 EP。这也是 DeepSeek-V3 把"通用能力"集中到 shared experts 的工程动机——避免通用 token 产生大量 All-to-All 通信。

---

## 十、知识关联

- **前置**：[[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操]] — TP 与 EP 是两个正交维度（层内 vs expert 间）
- **前置**：[[Projects/MA-RLHF/xtrain/xtrain-05b-Pipeline-Parallel-手撕实操]] — DualPipe 的设计动机正是 MoE EP 通信重叠
- **深化**：DeepSeek-V3 技术报告 §3.3 — MoE 架构完整描述（256 experts, top-8, N_s=1）
- **深化**：DeepEP 库（github.com/deepseek-ai/DeepEP）— 生产级 EP 通信库
- **横向**：GLM-5 技术报告精读 — 华为昇腾上的 MoE EP 实现（Slime 框架）
- **MA-RLHF 系列**：[[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引]]

## See Also

- [[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操]] — TP 与 EP 正交：TP 切层内矩阵，EP 切 expert 间
- [[Projects/MA-RLHF/xtrain/xtrain-05b-Pipeline-Parallel-手撕实操]] — DualPipe：MoE EP 通信重叠的工程解法
- [[Projects/MA-RLHF/xtrain/xtrain-06-Context并行RingAttention手写]] — CP+EP：超长序列 MoE 训练的四维并行
- [[AI/3-LLM/Architecture/MoE 深度解析]] — MoE 理论全景（gating/负载均衡/capacity factor）
- [[AI/3-LLM/Infra/分布式训练]] — 分布式训练理论全景
- [[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引]] — xtrain 系列课程地图
