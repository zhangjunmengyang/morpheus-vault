---
title: "xtrain lc6 — Context Parallel / Ring Attention 从零手写"
brief: "从零实现 Ring Attention：把序列切分到 N 张 GPU，用 Ring AllReduce 在 GPU 间流式传递 KV 块，同时计算本地 QK^TV 并用 Online Softmax (O,M,L) 三元组合并结果。掌握序列并行（CP）与张量并行（TP）的正交组合，以及 100k+ context 场景下 Ring Attention 与 Flash Attention 的关系。"
date: 2026-02-26
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, distributed-training, context-parallel, ring-attention, xtrain]
related:
  - "[[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]]"
  - "[[AI/3-LLM/Infra/xtrain-lc4-张量并行从零手写]]"
  - "[[AI/3-LLM/Infra/xtrain-lc7-MoE专家并行从零手写]]"
  - "[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]"
  - "[[AI/3-LLM/Inference/FlashAttention-手撕实操]]"
---

# xtrain lc6 — Context Parallel / Ring Attention 从零手写

> 来源：`/Users/peterzhang/project/ma-rlhf/xtrain/lecture/lc6_context_parallelism/`
> 系列：[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]
> 难度：★★★★☆（前向必须，反向为加分项；Online Softmax 是关键数学基础）
> 更新：2026-02-26

---

## TL;DR

Context Parallelism（CP）把**长序列按 seq 维度切分**到 N 张 GPU，每卡只持有 `seq/N` 个 token。核心挑战：Attention 需要**全局 KV**（Q 要能看到所有 K、V）。

**解法：Ring Attention** = 分布式 FlashAttention + 环形 KV 传递。每步：本卡用 Q 块与当前持有的 KV 块做 block attention，然后把 KV 传给下一张卡，循环 N 步后每张卡看到了所有 KV，用 Online Softmax 增量合并结果。

**Online Softmax 是数学基础**：允许分块处理 softmax 而无需看到全部分数，通过维护 `(m_global, l_global)` 状态实现数值稳定的增量归一化。

---

## 一、核心问题：长序列的注意力瓶颈

标准 Attention 的复杂度：O(seq²)，seq=128K 时 attention map 约 16GB，远超单卡。

**CP 的切分策略**：
- 每卡持有 `X_i [bs, seq/N, dim]`
- Q、K、V 都从 X_i 投影：`Q_i, K_i, V_i [bs, heads, seq/N, head_dim]`
- 注意力计算需要 `Q_i @ K_全局`，但 K 全局分散在各卡

**Ring Attention 的思路**：固定 Q 不动，让 KV 在环中传递。每步卡 i 处理 `(Q_i, K_j, V_j)`，N 步后每张卡累积了 `Q_i` 与所有 `K_j` 的完整注意力结果。

---

## 二、Online Softmax：分块增量计算

标准 softmax：`p_i = exp(x_i) / Σ_j exp(x_j)`，需要先扫一遍求全局最大值（数值稳定），再扫一遍求 sum，再扫一遍归一化。**三遍扫描**，无法流式处理。

**Safe Softmax**（两遍）：

```python
m = max(x)          # 全局最大值（减去防溢出）
l = Σ exp(x - m)    # 归一化常数
p = exp(x - m) / l
```

**Online Softmax**（一遍，分块处理）：

```python
# 初始状态
m_global = -∞
l_global = 0

# 对每个块 x_block：
m_local = max(x_block)
m_new = max(m_global, m_local)

# 关键：把旧的 l_global 乘以修正因子后加上新块的贡献
# l_global 的含义随 m_global 变化而变化，修正因子 = exp(m_old - m_new)
l_new = l_global * exp(m_global - m_new) + Σ exp(x_block - m_new)
m_global = m_new
```

**数学证明**（修正因子的来源）：

```
原 l_global = Σ_{已处理} exp(x_i - m_old)

换新基准 m_new 后：
Σ_{已处理} exp(x_i - m_new)
= Σ_{已处理} exp(x_i - m_old) · exp(m_old - m_new)
= l_global · exp(m_old - m_new)   ← 修正因子
```

**代码实现**：

```python
def safe_softmax_incremental(x, m_global, l_global):
    """处理一个新块 x，更新全局状态 (m_global, l_global)"""
    m_local, _ = torch.max(x, dim=-1, keepdim=True)
    m_new = torch.maximum(m_local, m_global)
    
    l_local = torch.sum(torch.exp(x - m_new), dim=-1, keepdim=True)
    l_new = l_global * torch.exp(m_global - m_new) + l_local
    
    p = torch.exp(x - m_new) / l_new  # 当前块的归一化概率（临时，m_global 会继续更新）
    return p, m_new, l_new
```

**Online Softmax 的对应 O 更新（Flash Attention 核心）**：

注意力输出 `O = Σ_i p_i · V_i`，同样需要增量更新：

```
O_new = O_old * exp(m_old - m_new) + exp(S_block - m_new) @ V_block

最终归一化：O_final = O_unnorm / l_global
```

---

## 三、分布式 Ring Online Softmax

环形通信实现分布式 softmax（seq 按卡切分，各卡只有局部 x）：

```python
# 阶段一：链式传递 (m_global, l_global)
if rank != 0:
    dist.recv(m_global, src=rank-1)   # 接收上游的全局状态
    dist.recv(l_global, src=rank-1)

_, m_global, l_global = safe_softmax_incremental(x_local, m_global, l_global)

if rank != world_size - 1:
    dist.send(m_global, dst=rank+1)   # 传给下游
    dist.send(l_global, dst=rank+1)

# 阶段二：广播最终的 (m_final, l_final) 给所有卡
dist.broadcast(m_global, src=world_size-1)  # 最后一张卡有完整状态
dist.broadcast(l_global, src=world_size-1)

# 各卡用全局状态归一化自己的 x_local
local_softmax = torch.exp(x_local - m_global) / l_global
```

**死锁问题**（关键）：若所有卡同时 `send`，会出现死锁。解法：奇偶交错（偶数卡先 send 再 recv，奇数卡先 recv 再 send）：

```python
if rank % 2 == 0:
    dist.send(K, dst=next_rank)
    dist.recv(tmp_K, src=prev_rank)
else:
    dist.recv(tmp_K, src=prev_rank)  # 先 recv（等偶数卡 send）
    dist.send(K, dst=next_rank)
```

---

## 四、Ring Attention 前向

**核心设计**：Q 固定在本卡，KV 每步环形传递一次。

```python
class RingAttention:
    def step_forward(self, Q, K, V):
        """
        Q: [bs, heads, local_seq, head_dim]（本卡固定）
        K/V: [bs, heads, local_seq, head_dim]（每步是不同块）
        """
        L = zeros(...)         # 归一化常数（累积）
        M = ones(...) * -1e4   # 全局最大值
        O = zeros(...)         # 输出（未归一化）
        
        for i in range(world_size):
            # Block attention：Q 与当前 KV 块
            L, M, O = self.block_attention_forward(Q, K, V, L, M, O)
            # 环形传递 KV 到下一张卡
            K, V = self.ring_comm_KV(K, V)
        
        O = O / L              # 最终归一化
        L_b = M + L.log()      # log-sum-exp，保存供反向用
        return O, L_b
    
    def block_attention_forward(self, Q, K, V, L, M, O):
        """单步 block attention + Online Softmax 更新"""
        S = Q @ K.transpose(3, 2) / sqrt(head_dim)  # [bs, heads, q_len, k_len]
        
        M_local = torch.max(S, dim=-1, keepdim=True).values
        M_new = torch.maximum(M, M_local)
        
        L_local = torch.sum(torch.exp(S - M_new), dim=-1, keepdim=True)
        L_new = L * torch.exp(M - M_new) + L_local
        
        # O 更新：旧 O 乘修正因子 + 新块注意力
        O_new = O * torch.exp(M - M_new) + torch.exp(S - M_new) @ V
        
        return L_new, M_new, O_new
```

**环形 KV 通信**（奇偶交错防死锁）：

```python
def ring_comm_KV(self, K, V):
    tmp_K = torch.zeros_like(K)
    next_rank = (self.rank + 1) % self.world_size
    prev_rank = (self.rank - 1) % self.world_size
    
    # 奇偶交错
    if self.rank % 2 == 0:
        dist.send(K, dst=next_rank)
        dist.recv(tmp_K, src=prev_rank)
    else:
        dist.recv(tmp_K, src=prev_rank)
        dist.send(K, dst=next_rank)
    
    # V 类似
    ...
    return tmp_K, tmp_V
```

**完整前向流程**：

```python
def step(self, X, Y=None):
    # 1. 投影（WQ/WK/WV 参数所有卡相同，广播同步过）
    Q, K, V = self.proj(X)   # X [bs, local_seq, dim] → Q/K/V [bs, heads, local_seq, head_dim]
    
    # 2. Ring Attention 前向
    O, L_b = self.step_forward(Q, K, V)
    
    # 3. 输出投影
    O_flat = O.view(bs, local_seq, dim)
    logits, loss = self.loss_fn(O_flat, Y)
    
    # 4. 反向
    ...
```

---

## 五、Ring Attention 反向

反向需要计算 `dQ, dK, dV`，使用 Flash Attention V2 反向的分块技术。

**关键辅助量 D**（Flash Attention 反向的核心）：

```
D_i = rowsum(O_i * dO_i)  = Σ_j O_{ij} · dO_{ij}
```

这是 softmax 反向中的 `p · dp` 项，用于计算 `dS = P * (dP - D)`。

```python
def step_backward(self, Q, K, V, L_b, O, dO):
    dQ = zeros_like(Q)
    dK = zeros_like(K)
    dV = zeros_like(V)
    
    # D = rowsum(O * dO)：每个 query 位置的标量
    D = torch.sum(O * dO, dim=-1, keepdim=True)  # [bs, heads, q_len, 1]
    
    for i in range(world_size):
        dQ_block, dK_block, dV_block = self.block_attention_backward(Q, K, V, L_b, O, dO, D)
        dQ += dQ_block
        dK += dK_block
        dV += dV_block
        K, V = self.ring_comm_KV(K, V)    # KV 继续环形传递
    
    return dQ, dK, dV

def block_attention_backward(self, Q, K, V, L_b, O, dO, D):
    S = Q @ K.transpose(3, 2) / sqrt(head_dim)  # 重算 attention score
    P = torch.exp(S - L_b)                       # 用 L_b = M + log(L) 重算 softmax
    
    # dV = P^T @ dO
    dV = P.transpose(3, 2) @ dO
    
    # dP = dO @ V^T
    dP = dO @ V.transpose(3, 2)
    
    # dS = P * (dP - D)  ← softmax 反向的核心公式
    dS = P * (dP - D)
    
    # dQ = dS @ K / sqrt(d)
    dQ = dS @ K / sqrt(head_dim)
    
    # dK = dS^T @ Q / sqrt(d)
    dK = dS.transpose(3, 2) @ Q / sqrt(head_dim)
    
    return dQ, dK, dV
```

**为什么 `dS = P * (dP - D)`？**

Softmax 反向的完整推导：

```
y = softmax(x)，∂L/∂x_i = y_i · (∂L/∂y_i - Σ_j y_j · ∂L/∂y_j)
                         = y_i · (∂L/∂y_i - D)     其中 D = Σ_j y_j · ∂L/∂y_j

这里 y = P，∂L/∂y = dP，D = rowsum(P * dP) = rowsum(O * dO)（FA优化）
所以 dS = P * (dP - D) ✓
```

---

## 六、权重梯度更新（AllReduce 同步）

CP 中 WQ/WK/WV/WO 参数在所有卡上**相同**，但每张卡只处理了 local_seq 的数据，梯度需要 AllReduce：

```python
def backward_gradient_update(self, X, dQ, dK, dV, O, dO):
    def proj_backward(dY, X, W):
        dY_flat = dY.transpose(2, 1).reshape(bs, seq, dim)
        dW = (dY_flat.transpose(2, 1) @ X).sum(dim=0)  # 累加 batch 维度
        dX = dY_flat @ W
        return dW, dX
    
    d_WQ, dX_Q = proj_backward(dQ, X, self.WQ.weight)
    d_WK, dX_K = proj_backward(dK, X, self.WK.weight)
    d_WV, dX_V = proj_backward(dV, X, self.WV.weight)
    
    # AllReduce 平均各卡的权重梯度（因为各卡数据不同，梯度也不同）
    self.WQ.weight.grad = self.all_reduce_avg(d_WQ)
    self.WK.weight.grad = self.all_reduce_avg(d_WK)
    self.WV.weight.grad = self.all_reduce_avg(d_WV)
    
    # 输入梯度：各分量之和
    dX = dX_Q + dX_K + dX_V
    return dX
```

---

## 七、Causal Mask 的计算不均衡问题

Decoder-Only 模型有因果 mask（下三角），导致各 rank 计算量**严重不均衡**：

```
rank 0 (负责 Q[0:L/4])：只需算 X[0,0]（左上角三角极小）
rank 3 (负责 Q[3L/4:L])：需要算 X[3,0], X[3,1], X[3,2], X[3,3]（几乎完整行）
```

**Striped Attention 解法**：调整 KV 序列顺序，让每个 rank 同时处理"首尾拼接"的 token，使计算量均衡。

```
原始序列：[0,1,2,3,4,5,6,7]，4 卡，每卡 2 token
rank 0: Q[0,1]，K[0,1]   ← 只能看到自己前面，计算少
rank 3: Q[6,7]，K[0..7]  ← 能看到全部，计算多

Striped 重排（rank 0 持有 token 0,7；rank 1 持有 token 1,6；...）：
rank 0: Q[0,7]   ← 一个最短 + 一个最长，均衡
rank 1: Q[1,6]   ← 均衡
```

**实现**：仅在 ring_attention 外层调整输入 token 顺序的 mapping，不改变算法本身。

---

## 八、CP 与其他并行方式的关系

| 并行维度 | 切分对象 | 通信内容 | 适合场景 |
|---------|---------|---------|---------|
| DP | 数据（batch） | AllReduce 梯度 | 普通训练 |
| TP | 权重矩阵行/列 | 激活值 AllReduce | 大模型 inference |
| PP | 模型层 | 激活值 P2P | 超深模型 |
| CP | 序列（seq） | KV 环形传递 | **超长序列**（>32K） |
| ZeRO | 优化器状态 | AllGather 参数 | 大 batch 训练 |

CP 主要用于 **inference prefill** 和超长序列训练：
- Inference：prefill 阶段 seq 很长（RAG/长文档），CP 切分 seq 到多卡并行
- Training：长文档训练（100K+），单卡存不下 attention map

---

## 九、面试考点

**Q1：Ring Attention 和 FlashAttention 的关系？**

A：Ring Attention 是**分布式版 FlashAttention**。FlashAttention 用 Online Softmax 在单卡内分块计算 Attention，避免 O(seq²) 显存；Ring Attention 用同样的 Online Softmax 原理，在多卡间环形传递 KV，每步做一次 block attention，N 步后累积完整注意力，达到等价结果。两者都维护 `(m, l, O)` 三元组增量更新。

**Q2：Online Softmax 的核心公式是什么？为什么需要修正因子？**

A：`l_new = l_old * exp(m_old - m_new) + Σ exp(x_new - m_new)`。修正因子 `exp(m_old - m_new)` 是因为 `l_old` 是相对于旧基准 `m_old` 计算的，切换基准到 `m_new` 时需要重新缩放。本质：`Σ exp(x - m_new) = Σ exp(x - m_old) * exp(m_old - m_new) = l_old * exp(m_old - m_new)`。

**Q3：Ring Attention 中 KV 通信为什么要奇偶交错？**

A：若所有卡同时 `dist.send`，每张卡都在等对方接收，导致死锁。奇偶交错：偶数卡先 send（有卡在接收），奇数卡先 recv（不阻塞偶数卡的 send）；然后角色互换。这是经典的分布式通信死锁解法。

**Q4：Ring Attention 反向中 D 是什么？**

A：`D = rowsum(O * dO)`，是每个 query 位置的标量，来自 softmax 反向公式 `dS = P * (dP - D)`。直觉上：softmax 的反向梯度需要减去"当前 query 对 V 的注意力加权输出与梯度的内积"，防止重复计数。Flash Attention 的关键优化：直接用 `O * dO`（已保存）计算 D，不需要重新 materialize softmax 矩阵 P。

**Q5：Causal Mask 在 Ring Attention 中有什么特殊处理？**

A：Causal Mask 造成计算不均衡——持有高序号 token 的卡（Q 看得到更多 KV）计算量远大于低序号的卡。解法是 Striped Attention：重新排列 token 到各卡，使每卡同时持有靠前和靠后的 token（如 rank 0 持有 token 0 和 token N-1），让各卡的有效计算块数相同。不需要修改 Ring Attention 算法本身，只是输入顺序的 remapping。

**Q6：CP 适合推理还是训练？和 TP 的 Sequence Parallelism 有何区别？**

A：CP 主要适合**超长 seq（>32K）的 prefill 推理**和极长序列训练。TP Sequence Parallelism（Megatron）是在 TP 框架内对 LayerNorm/Dropout 的序列做并行，切的是同一层内不同算子的激活；CP 切的是**整个 attention 的序列维度**，更彻底。CP 需要 KV 全局通信（O(seq × head_dim) per step），适合序列足够长使得 seq² 显存成为瓶颈时使用。

---

## 十、知识关联

- **前置**：[[AI/3-LLM/Inference/vLLM-PageAttention-手撕实操]] — Online Softmax 三元组 (O,M,L) 的 inference 应用
- **前置**：[[AI/3-LLM/Inference/FlashAttention-手撕实操]] — FA 是 Ring Attention 的单卡版
- **前置**：[[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]] — PP 是层间切分，CP 是序列维度切分
- **深化**：DeepSeek-V2 的 MLA — MLA 减少了 KV 头数，CP 通信量随之降低（GQA/MLA 对 CP 友好）
- **生产**：Megatron-LM Context Parallel，LongContext 训练（Llama-3.1-405B 使用 CP）
- **MA-RLHF 系列**：[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]

## See Also

- [[AI/3-LLM/Infra/xtrain-lc5-流水线并行从零手写]] — 前置：PP 层间切分，CP+PP 超长序列训练的两维度
- [[AI/3-LLM/Infra/xtrain-lc4-张量并行从零手写]] — 前置：TP 与 CP 正交可组合（TP 切权重，CP 切序列）
- [[AI/3-LLM/Inference/FlashAttention-手撕实操]] — FA 是 Ring Attention 的单卡版，Online Softmax 同源
- [[AI/3-LLM/Inference/vLLM-PageAttention-手撕实操]] — Online Softmax (O,M,L) 三元组在推理端的应用
- [[分布式训练]] — 分布式训练理论全景
- [[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]] — xtrain 系列课程地图
