---
title: "xtrain · 分布式并行手写专题地图"
type: moc
date: 2026-02-25
source: "https://github.com/dhcode-cpp/MA-RLHF/tree/main/xtrain/lecture"
tags: [moc, ma-rlhf, distributed-training, dp, tp, pp, cp, moe, ep, xtrain]
---

# xtrain · 分布式并行手写专题地图

> **目标**：不依赖 DeepSpeed / Megatron / FSDP，从零手写 5 大并行范式。这是整个 MA-RLHF 课程中**最硬核**的部分。  
> **核心挑战**：每个文件都是纯 `torch.distributed` + P2P 通信实现，理解底层才能理解框架。

---

## 带着这三个问题学

1. **5 大并行（DP / ZeRO / TP / PP / CP / EP）各自切的是什么维度？** 数据、模型、序列、专家——为什么不能只用一种？
2. **DualPipe 比 1F1B 好在哪？** 它解决的是 MoE 网络的什么瓶颈？
3. **Ring Attention 和 FlashAttention 是什么关系？** 一个是分布式版本，一个是单卡版本？

---

## 学习顺序（严格按序，每层依赖前一层）

```
lc1 通信原语          ← 所有并行的基础，必须先掌握
 ↓
lc2 数据并行 DDP      ← 最简单的多卡并行
 ↓
lc3 ZeRO             ← DP 的显存优化（ZeRO-1/2/3）
 ↓
lc4 张量并行 TP       ← 模型内部切分（行列并行）
 ↓
lc5 流水线并行 PP     ← 模型层间切分（GPipe → 1F1B → DualPipe）
 ↓
lc6 上下文并行 CP     ← 序列维度切分（Ring Attention）
 ↓
lc7 MoE 专家并行 EP   ← Expert 维度切分 + 通信计算重叠
```

---

## lc1：通信原语 — 所有并行的根基

**[[AI/LLM/Infra/xtrain-lc1-分布式通信原语从零手写|xtrain-lc1 从零手写]]** · 参考手撕实操：**[[AI/LLM/Infra/分布式训练通信原语-手撕实操|通信原语-MA-RLHF版]]**

| 操作 | 功能 | 通信量 |
|------|------|--------|
| P2P Send/Recv | 点对点通信 | O(d) |
| Broadcast | 一对多广播 | O(d) |
| Gather/Scatter | 收集/分发 | O(N·d) |
| Reduce | 多对一规约（求和等） | O(d) |
| AllReduce | 全部规约（所有 rank 得到相同结果） | O(d) |
| ReduceScatter | 规约 + 分散 | O(d) |
| AllGather | 全部收集 | O(N·d) |
| All2All | 全交换（EP 的核心） | O(N·d) |
| **Ring-AllReduce** | 环形通信实现 AllReduce | O(d·(N-1)/N) |

关键代码：
- `p2p.py` / `p2p_async.py` / `p2p_op.py` — P2P 通信 + 环形通信解死锁
- `fun_ring_allreduce.py` — 🌟 手撕 Ring-AllReduce
- `fun_all2all_scratch.py` — 🌟 手撕 All2All（同步，解死锁）

---

## lc2：数据并行 DDP — 最简单的多卡训练

**[[AI/LLM/Infra/xtrain-lc2-数据并行从零手写|xtrain-lc2 从零手写]]**

- **核心思想**：每个 GPU 有完整模型副本，数据切分 → 各自前向反向 → AllReduce 梯度 → 同步更新
- **关键**：是 reduce **梯度**，不是 reduce **loss**（⚠️ 常见误区）
- **分布式数据集**：手撕 `DistributedSampler`，保证每个 rank 拿到不重叠的数据子集

代码：`torch_ddp_train.py`（API版） · `torch_ddp.py`（手撕版） · `distributed_dataset.py`（并行数据类）

---

## lc3：ZeRO — DP 的显存极致优化

**[[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写|xtrain-lc3 从零手写]]** · 参考：**[[AI/LLM/Infra/ZeRO-手撕实操|ZeRO 手撕]]**

ZeRO 三阶段（本质仍是 DP，但切分了冗余）：

| 阶段 | 切分内容 | 显存节省 | 通信量变化 |
|------|---------|---------|-----------|
| ZeRO-1 | 优化器状态 | ~4x | 不变 |
| ZeRO-2 | + 梯度 | ~8x | ReduceScatter 替代 AllReduce |
| ZeRO-3 | + 参数 | ~N·8x（N=GPU数） | 前向也需 AllGather 参数 |

关键实现：
- `adam.py` — 手撕 Adam optimizer
- `distributed_adam.py` — 分布式 Adam，保证梯度/参数/优化器一致
- `adam_zero1.py` — 优化器切分 + AllGather 同步
- `adam_zero2.py` — 手撕 `loss.backward` + Reduce-Scatter 梯度
- `adam_zero3.py` — 参数切分 + 自定义前向（前向时 AllGather 参数）
- `adam_mix_precision.py` — 🌟 混合精度训练

**核心洞察**：ZeRO 的切分是 **flatten 化**的（不按矩阵/层切分，而是把所有参数 flatten 成一维再均分）→ 对任意模型架构通用。

---

## lc4：张量并行 TP — 模型内部切分

**[[AI/LLM/Infra/xtrain-lc4-张量并行从零手写|xtrain-lc4 从零手写]]** · 参考：**[[AI/LLM/Infra/Tensor-Parallel-手撕实操|TP 手撕]]**

- **列并行 Linear**：Weight 按列切分到各 GPU → 各自计算部分输出 → AllGather 拼接（或直接进下一层行并行）
- **行并行 Linear**：Weight 按行切分 → Input 按列切分 → 各自计算 → AllReduce 求和
- **MLP 并行**：FFN 第一层列并行 + 第二层行并行 → 前向只需 1 次 AllReduce（在行并行的 forward）
- **GQA 并行**：按 head 维度切分，每个 GPU 分配若干 Q/KV 组 → 切分策略比 MHA 复杂
- **Embedding 并行**：词表并行（按 vocab 维度切分，注意 idx offset）和维度并行
- **CrossEntropy 并行**：融合词表并行 + CE 优化反向函数

代码（Part 1 基础）：
- `col_parallel_linear.py` / `row_parallel_linear.py` — 行列并行前后向
- `custom_gradient.py` / `distributed_custom_gradient.py` — autograd 函数中嵌入通信

代码（Part 2 模块 🌟）：
- `mlp.py` — SwiGLU 张量并行
- `attention.py` — 🌟 GQA 张量并行（最难）
- `embedding.py` / `lm_head.py` — 🌟 词表并行 + CrossEntropy 切分
- `decoder.py` / `model.py` — 完整 TP 模型 `XtrainModel`

---

## lc5：流水线并行 PP — 层间切分

**[[AI/LLM/Infra/xtrain-lc5-流水线并行从零手写|xtrain-lc5 从零手写]]** · 参考：**[[AI/LLM/Infra/Pipeline-Parallel-手撕实操|PP 手撕]]**

演进路线：

```
GPipe（批量 F → 批量 B）→ 简单，但 bubble 大
  ↓
1F1B（PipeDream）→ 限制 F 数量，降低显存峰值
  ↓
Zero-Bubble（1F1B1W）→ dW 和 dx 分离，通信计算重叠
  ↓
DualPipe → 双向流水线，MoE EP 通信完全隐藏
```

关键代码：
- `pipeline_parallel_gpipe.py` — micro-batch GPipe
- `pipeline_parallel_pipe_dream.py` / `_2.py` — 🌟 1F1B + 循环队列管理
- `zero_bubble.py` / `zero_bubble_seperate_dx_dw.py` — 🌟 1F1B1W
- `dualpipe_simplest.py` — 🌟 手动 Chimera Schedule
- `dualpipe_xdg.py` — 🌟 Easy-DualPipe（通信 op 管理，wait() 机制）
- `dualpipe.py` — 官方标准实现（难，可选）

**DualPipe 核心**：解决的是 MoE 网络中 EP 的 All2All 通信瓶颈。双向流水线 + 将 F/B 操作拆分为 F(mlp)/B(attn)/F(attn)/B(mlp) → All2All 的 dispatch/combine 与计算完全重叠。

---

## lc6：上下文并行 CP — 序列维度切分

**[[AI/LLM/Infra/xtrain-lc6-Context并行RingAttention手写|xtrain-lc6 从零手写]]** · 参考：**[[AI/LLM/Infra/MoE-Context-Parallel-手撕实操|MoE+CP 手撕]]**（CP 部分）

- **Online Softmax**：分块计算 softmax，每块只需维护 `(max, sum_exp)` → 可增量更新
- **Ring Online Softmax**：分布式环形传递 KV 块 + Online Softmax 更新 → Ring Attention 的前置
- **Ring Attention**：分布式 FlashAttention-V2
  - 各 GPU 持有序列的不同切片（按 seq 维度 scatter）
  - Q 固定在本地，KV 环形传递
  - 每收到一块 KV 就做一次 block-wise attention + online softmax 更新
  - 前向需保存中间变量用于反向
  - **反向**：dQ/dK/dV 块累加计算（🌟 高难度）
- **Striped Attention**：调换 KV 块顺序实现计算均衡（Decoder causal mask 导致各 rank 计算量不平衡）

代码：
- `online_softmax.py` — 分块 Online Softmax
- `ring_online_softmax.py` — 分布式版本（环形通信解死锁）
- `ring_attention.py` — 🌟 完整 Ring FlashAttention V2 前向+反向
- `compute_balance.py` — Striped Attention 均衡方案

---

## lc7：MoE 专家并行 EP — Expert 维度切分

**[[AI/LLM/Infra/xtrain-lc7-MoE专家并行从零手写|xtrain-lc7 从零手写]]** · 参考：**[[AI/LLM/Infra/MoE-Context-Parallel-手撕实操|MoE+CP 手撕]]**（EP 部分）

- **GShard**：每个 GPU 持有不同 Expert，All2All 交换 token → 本地 Expert 计算 → All2All 交换回结果
- **前向**：dispatch (All2All) → local expert compute → combine (All2All)
- **反向**：遵循相同的 All2All map，梯度走相同的 dispatch-combine 路径
- **通信计算重叠**（1F1B 方案）：
  - 将 F/B 拆为 F(mlp)/B(attn)/F(attn)/B(mlp)
  - All2All 异步执行（不等完成）→ 同时做非 MoE 层的计算
  - `OverlappedOp` 管理异步通信，需要数据时先 `wait()`

代码：
- `smoe_forward.py` / `smoe_backward.py` — 单卡 MoE 前后向
- `top_k_gradient.py` — Top-K 不可导的 torch 实现
- `gshard.py` — 🌟 GShard 前后向完整实现
- `fun_all2all_ascyn.py` — 异步 All2All
- `overlapped_1F1B.py` — 🌟 通信计算重叠（最复杂）
- `overlapped_prefill.py` — 🌟 Prefill 版本重叠

---

## 面试高频场景题

**Q：DualPipe 和 1F1B 的区别？**  
A：1F1B 只考虑流水线并行的 bubble，限制 F 数量来降低显存。DualPipe 在此基础上引入双向流水线（从两端同时注入 micro-batch），并将 F/B 操作拆分为 attention 和 mlp 子操作 → MoE 的 All2All 通信可以与非 MoE 计算完全重叠 → 解决 MoE EP 的通信瓶颈。DualPipe 本质是为 DeepSeek V3 的 MoE+EP 训练设计的。

**Q：ZeRO-3 显存节省多少？**  
A：假设模型参数 Φ（FP16），Adam 优化器需要 12Φ 字节（参数 2Φ + 梯度 2Φ + FP32 参数 4Φ + 一阶动量 4Φ + 二阶动量 4Φ = 16Φ，但前向激活不算）。ZeRO-3 将参数+梯度+优化器状态全部切分到 N 个 GPU → 每 GPU 只存 16Φ/N。相比标准 DP（每 GPU 存 16Φ），节省 N 倍。N=64 时每 GPU 只需 0.25Φ 字节。

**Q：Ring Attention 的通信量是多少？**  
A：N 个 GPU，序列长度 S，每个 GPU 持有 S/N 长度的 Q 和 KV。环形传递 KV 需要 N-1 步，每步传递 2×(S/N)×d 大小的 KV 块（K 和 V 各一份）。总通信量 = 2×(N-1)×(S/N)×d ≈ 2Sd（当 N 较大时）。通信量与 GPU 数无关（弱扩展性好），与序列长度线性相关。

**Q：什么时候用 TP，什么时候用 PP？**  
A：TP 切分模型内部（适合单节点多卡，通信带宽需求高，NVLink 级别）；PP 切分模型层间（适合跨节点，通信带宽需求相对低，但有 pipeline bubble）。实践中通常 TP 在节点内 + PP 跨节点 + DP 做数据副本 → 3D 并行。

**Q：5 大并行各自切的是什么维度？**  
A：DP 切 batch 维度；ZeRO 在 DP 基础上切参数/梯度/优化器状态（flatten 后均分）；TP 切 hidden 维度（行列并行）；PP 切 layer 维度（层间）；CP 切 sequence 维度（Ring Attention）；EP 切 expert 维度（All2All 交换 token）。
