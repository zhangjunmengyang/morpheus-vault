---
title: "xtrain lc5 — 流水线并行从零手写"
brief: "从零实现 1F1B 调度（GPipe→1F1B 的 Bubble Rate 优化：(p-1)/(m+p-1)）和 DeepSeek DualPipe（双向流水，bubble 趋近零）。掌握 micro-batch 分块、stage 间通信、forward-backward 交错执行，以及 MoE 场景下 DualPipe+EP 通信重叠的设计思想。"
date: 2026-02-26
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, distributed-training, pipeline-parallel, dualpipe, xtrain]
related:
  - "[[AI/LLM/Infra/xtrain-lc4-张量并行从零手写]]"
  - "[[AI/LLM/Infra/xtrain-lc6-Context并行RingAttention手写]]"
  - "[[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]]"
  - "[[AI/LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]"
  - "[[AI/LLM/Infra/分布式训练]]"
---

# xtrain lc5 — 流水线并行从零手写

> 来源：`/Users/peterzhang/project/ma-rlhf/xtrain/lecture/lc5_pipeline_parallelism/`
> 系列：[[AI/LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]
> 难度：★★★★★（面试必考，GPipe→1F1B→ZeroBubble→DualPipe 完整演化链）
> 更新：2026-02-25

---

## TL;DR

流水线并行（Pipeline Parallelism, PP）把模型**按层切分**到不同 GPU（stage），数据以激活值的形式在 stage 间流动。核心挑战是 **Pipeline Bubble**——一个 stage 等待上游完成时什么都不做，导致利用率下降。

**演化链（bubble 率递减）**：

```
Basic PP（单数据 F→B，bubble=100%）
    ↓ micro-batch
GPipe（N-F-N-B，批量前向再批量反向，bubble=(p-1)/(m+p-1)）
    ↓ 及时反向
1F1B / PipeDream（1前1后交错，显存降低）
    ↓ 分离 dx/dw
ZeroBubble（1F1B1W，dx 通信与 dw 计算重叠，bubble→0）
    ↓ 双向数据流
DualPipe（前半段+后半段互为补充，2流水线 bubble 互填）
```

DualPipe 是 DeepSeek-V3 的训练核心，为 MoE EP 通信与计算重叠设计。

---

## 一、核心问题：为什么 PP 有 Bubble？

单 GPU 模型：前向 → 反向 → 更新，满负荷运行。

4 卡 PP，单批数据：
```
卡0(stage0): [F]  ·  ·  ·  [B]  ·  ·  ·   ← 等待后续stage反向结果
卡1(stage1):  ·  [F]  ·  · [B]   ·  ·  ·
卡2(stage2):  ·   ·  [F]  ·  [B]  ·  ·  ·
卡3(stage3):  ·   ·   ·  [F][B]  ·  ·  ·  ← 最后一张卡串行 F+B
```

卡 0 大部分时间空转：**bubble = 2(p-1)/2(p-1)+p = 高**。

**解法方向**：用 **micro-batch** 填充 bubble——把一个 batch 分成 M 个 micro-batch，在等待的时候处理下一个 micro-batch。

---

## 二、Basic PP：手写 P2P 通信

```python
class PipeModel(nn.Module):
    """每张卡只存 num_blocks/world_size 层"""
    def __init__(self, dim, num_blocks, rank=0, world_size=1):
        self.local_num_blocks = num_blocks // world_size
        self.layers = nn.ModuleList([MLP(dim) for _ in range(self.local_num_blocks)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # residual
        return x
```

**前向 P2P 通信**：

```python
if rank == 0:
    stage_output = pipe_model(x)
    dist.send(stage_output, dst=rank+1)           # 发给下一个 stage
else:
    dist.recv(x, src=rank-1)                      # 阻塞等待上一个 stage
    stage_output = pipe_model(x)
    if rank != world_size - 1:
        dist.send(stage_output, dst=rank+1)
```

**反向 P2P 通信**（梯度向左传递）：

```python
if rank == world_size - 1:
    loss = loss_fn(stage_output, label)
    loss.backward()
    dist.send(x.grad.clone(), dst=rank-1)         # 发梯度给上游
else:
    dist.recv(stage_output_grad, src=rank+1)      # 接收下游传来的梯度
    stage_output.backward(gradient=stage_output_grad)  # 用接收的梯度反向
    if rank != 0:
        dist.send(x.grad.clone(), dst=rank-1)
```

**关键**：`stage_output.backward(gradient=...)` 而非 `loss.backward()`——中间 stage 没有标量 loss，用来自下游的梯度作为"外部梯度"触发反向传播。

**Bubble 问题**：单 batch 时，F 前进到末尾卡才开始 B，所有中间卡的 B 都在等。

---

## 三、GPipe：Micro-batch + 批量 F 再批量 B

**思路**：把 batch 分成 M 个 micro-batch，先做所有 micro-batch 的前向（F0,F1,...,Fm），再做所有反向（B0,B1,...,Bm）。

```
卡0:  F0 F1 F2 F3 ·  ·  ·  · B3 B2 B1 B0
卡1:   ·  F0 F1 F2 F3 ·  ·  · B3 B2 B1 B0
卡2:   ·   ·  F0 F1 F2 F3 ·  · B3 B2 B1 B0
卡3:   ·   ·   ·  F0 F1 F2 F3[B3 B2 B1 B0]
```

**Bubble 率**：`bubble = (p-1) / (m+p-1)`，当 `m >> p` 时趋近于 0。

```python
# 前向：所有 micro-batch
for i in range(world_size):  # world_size = M（micro-batch数量）
    if rank != 0:
        dist.recv(x_list[i], src=rank-1)                    # 阻塞接收
    
    stage_output = pipe_model(x_list[i])
    stage_output_list[i] = stage_output
    
    if rank != world_size - 1:
        req = dist.isend(stage_output.clone(), dst=rank+1)  # 异步发送
        reqs.append(req)

# 反向：所有 micro-batch（逆序）
for i in range(world_size - 1, -1, -1):
    if rank != world_size - 1:
        dist.recv(stage_output_grad_list[i], src=rank+1)
    
    if rank == world_size - 1:
        loss = loss_fn(stage_output_list[i], label_list[i])
        loss /= world_size  # 梯度累积：各 micro-batch loss 除以 M
        loss.backward()     # torch 自动累积梯度（不 zero_grad）
    else:
        stage_output_list[i].backward(gradient=stage_output_grad_list[i])
    
    if rank != 0:
        req = dist.isend(x_list[i].grad.clone(), dst=rank-1)
```

**GPipe 的问题**：需要存储**所有 micro-batch 的中间激活值**（F0,F1,...,Fm 的 activations 同时在内存里），显存峰值 = M × 单批激活值。

---

## 四、1F1B / PipeDream：及时反向降显存

**1F1B 思路**：不等所有 micro-batch 都前向完，每做完一个前向立刻触发对应反向——**最后一张卡**立刻就能算 B，不需要等所有 F 完成。

```
卡0:  F0 F1 F2 F3 · · · B3 B2 B1 B0
卡1:   ·  F0 F1 F2 F3 · B3 B2 B1 B0
卡2:   ·   ·  F0 F1 [F2 B2] [F3 B3] B1 B0
卡3:   ·   ·   ·  [F0 B0] [F1 B1] [F2 B2] [F3 B3]
```

**调度器实现**（时间步 `i` 的策略）：

```python
for i in range(world_size + micro_batch_size - 1):
    # 时间步 i >= rank 时执行前向
    if i >= rank and f_idx < micro_batch_size:
        if rank != 0:
            dist.recv(x_list[f_idx], src=rank-1, tag=10010)
        
        stage_output = pipe_model(x_list[f_idx])
        stage_output_list.append(stage_output)
        
        if rank != world_size - 1:
            req = dist.isend(stage_output.clone(), dst=rank+1, tag=10010)
        f_idx += 1
    
    # 时间步 i >= (world_size - 1) 开始执行反向
    if i >= world_size - 1 and b_idx < micro_batch_size:
        if rank != world_size - 1:
            dist.recv(stage_output_grad_list[b_idx], src=rank+1, tag=10086)
        
        if rank == world_size - 1:
            loss = loss_fn(stage_output_list[b_idx], label_list[b_idx])
            loss /= world_size
            loss.backward()
        else:
            stage_output_list[b_idx].backward(gradient=stage_output_grad_list[b_idx])
        
        if rank != 0:
            req = dist.isend(x_list[b_idx].grad.clone(), dst=rank-1, tag=10086)
        b_idx += 1
```

**关键时序调度**：前向时序 = `i >= rank`（波浪线推进）；反向时序 = `i >= (world_size - 1)`（从最后一张卡开始）。

**1F1B vs GPipe**：
- GPipe：存 M 份激活（显存 ∝ M）
- 1F1B：steady state 时每张卡最多存 P 份激活（显存 ∝ P，与 M 无关）

---

## 五、ZeroBubble：dx/dw 分离，实现通信-计算重叠

**核心思路**：把反向传播拆成两步：
1. **backward_for_input（dx 计算）**：算出 `dx = dh @ W1`，**立刻通信**（isend dx 给上游）
2. **backward_for_weight（dw 计算）**：在通信进行的同时，计算 `dw1 = dh.T @ x`

```
通信和计算重叠：
t1: 计算 dx → isend(dx) [异步，不等待]
t2: 计算 dw [与通信并行进行]
```

**ZeroBubbleMLP 的分离实现**：

```python
class ZeroBubbleMLP(nn.Module):
    def forward(self, x):
        h = self.w1(x)
        o = self.w2(h)
        return h, o   # 返回中间激活 h，供反向用
    
    def backward_for_input(self, do):
        """只算 dx，不算 dw（先通信）"""
        dh = do @ self.w2.weight          # [bs, 4d] @ [4d, d] = [bs, d]，但需存 dh
        dx = dh @ self.w1.weight          # [bs, 4d] @ [4d, d] = [bs, d]
        return dh, dx                     # 保存 dh 供后面算 dw 用（增加显存）
    
    def backward_for_weight(self, do, dh, h, x):
        """算 dw，在 dx 通信期间并发执行"""
        self.w2.weight.grad = do.t() @ h  # [d, bs] @ [bs, 4d] = [d, 4d]
        self.w1.weight.grad = dh.t() @ x  # [4d, bs] @ [bs, d] = [4d, d]
```

**ZeroBubble 调度的关键执行流**：

```python
def backward_zero_bubble(self, layers_output, do, b_idx, is_send=True):
    # Step 1：所有层反向算 dx
    dx = None
    for layer, layer_output in zip(reversed(self.layers), reversed(layers_output)):
        dh, dx = layer.backward_for_input(do)
        layer_output.append(dh)   # 保存 dh，供 Step 2 算 dw
        layer_output.append(dx)
    
    # Step 2：isend(dx) 到上游（异步，立即返回）
    if self.rank != 0 and is_send:
        req = dist.isend(dx, dst=self.rank-1, tag=10086)
        req.wait()  # 注意：这里 wait() 会阻塞，生产实现应真正异步
    
    # Step 3：算 dw（理想情况与通信重叠）
    for layer, layer_output in zip(reversed(self.layers), reversed(layers_output)):
        layer.backward_for_weight(
            x=layer_output[0], h=layer_output[1],
            dh=layer_output[2], do=do
        )
```

**ZeroBubble 的代价**：
- 必须**额外保存** `dh`（中间层梯度）供 `backward_for_weight` 用
- 显存：比 1F1B 略增（存 dh），但通过通信-计算重叠获得接近零 bubble

**ZeroBubble vs 1F1B**：

| 阶段 | 1F1B | ZeroBubble |
|------|------|-----------|
| Backward | dx+dw 一起算，然后通信 | dx 先算 → 通信，dw 在通信期间计算 |
| 通信时 GPU 状态 | 空闲（bubble） | 计算 dw（有效利用） |
| 显存 | 低（不存 dh） | 略高（存 dh） |
| Bubble | 有 | 趋近于 0 |

---

## 六、DualPipe：双向数据流，专为 MoE EP 通信设计

**背景**：DeepSeek-V3 使用 MoE 架构，MoE 的 Expert Parallelism（EP）会产生大量全局 token 路由通信（All-to-All）。DualPipe 的设计目标是**将这些 EP 通信隐藏在计算中**。

**核心思路**：将 PP stage 分为前半段和后半段（共 2P 张卡），**两路数据流同时运行**：
- 流 0（chunk 0）：从 rank 0 → rank P-1 前进
- 流 1（chunk 1）：从 rank P-1 → rank 0 前进

```
rank:  0   1   2   3 | 4   5   6   7
      [前半段 →→→   ] [←←← 后半段]

时序：
rank0: F0(c0) FF(c0,c1) FB(c1) B0(c0)
rank3: ·      F0(c0)    FB(c1,c0) B0(c1)
rank4: ·      F0(c1)    FB(c0,c1) B0(c0)
rank7: ··     F0(c1)    FB(c1,c0) B0(c1)
```

**DualPipe 调度的五种操作**（dualpipe_simplest.py）：
- `F`（Forward for input，chunk i）
- `FF`（Forward for input，两路同时）
- `FB`（Forward + Backward 同一时间步的两路混合）
- `BB`（Backward backward，两路）
- `B`（Backward for weight，chunk i）

**调度策略**（以 8 卡为例，每组 4 卡）：

```python
def step_schedule(x, phase, rank, world_size):
    is_in_first_half = rank < world_size // 2    # rank 0-3
    is_in_second_half = rank >= world_size // 2  # rank 4-7

    # Step1: f0（单路预热，bubble 填充）
    step = abs(world_size//2 - rank) + is_in_second_half
    # 前半段 rank 越靠近中间，预热步数越少

    # Step2: f0f1（两路同时前进）
    # 前半段：先 chunk1 再 chunk0（利用反向路由）
    # 后半段：先 chunk0 再 chunk1

    # Step3: f1b1（一路前向，一路反向交织）
    # 这里实现 EP 通信与计算重叠的关键窗口

    # Step4: b0b1（两路反向）

    # Step5: b1（单路收尾）
```

**DualPipe 的本质**：把原来线性流水线 `0→1→...→P` 变成双向菊花链 `0→...→P/2←...←P`，使得前半段和后半段的 F/B 操作在时间上互补，相互填充对方的 bubble。

---

## 七、Gradient Checkpoint（重计算）

PP 中间激活显存开销大，Gradient Checkpoint 用**重计算**换显存：

```python
# 简化实现：forward 时不存激活，backward 时重新前向计算一遍
import torch.utils.checkpoint as cp

class CheckpointMLP(nn.Module):
    def forward(self, x):
        # 用 checkpoint 包裹，forward 时不保留中间激活
        return cp.checkpoint(self._forward_impl, x)
    
    def _forward_impl(self, x):
        h = self.w1(x)
        return self.w2(h)
```

**手撕重计算**（checkpoint_scratch.py 的思路）：

```python
class ScratchCheckpoint(autograd.Function):
    @staticmethod
    def forward(ctx, run_fn, *args):
        ctx.run_fn = run_fn
        ctx.save_for_backward(*args)
        with torch.no_grad():
            output = run_fn(*args)
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        args = ctx.saved_tensors
        # 重新前向（这次打开梯度跟踪）
        with torch.enable_grad():
            args = tuple(x.detach().requires_grad_(x.requires_grad) for x in args)
            output = ctx.run_fn(*args)
        
        # 用重算的结果做反向
        torch.autograd.backward(output, grad_outputs)
        return (None,) + tuple(a.grad if isinstance(a, torch.Tensor) else None for a in args)
```

**内存-计算权衡**：GC 节省 ≈ 每层激活大小，代价是每层额外做一次前向。PP 中每个 stage 如果配合 GC，整体显存从 O(M×L) 降至 O(L) 量级。

---

## 八、PP 各方案对比

| 方案 | Bubble 率 | 显存占用 | 实现复杂度 | 适用场景 |
|------|---------|---------|---------|---------|
| Basic PP | 高 (p-1)/p | 低 | 简单 | 教学 |
| GPipe | (p-1)/(m+p-1) | 高（存 M 份激活） | 中 | 小模型/大 M |
| 1F1B | ≈GPipe | 低（存 P 份激活） | 中 | 主流训练 |
| ZeroBubble | ≈0 | 略高（存 dh） | 高 | 大模型训练 |
| DualPipe | ≈0 | 略高 | 很高 | MoE + EP（DeepSeek-V3） |

---

## 九、面试考点

**Q1：GPipe 和 1F1B 的显存差距在哪里？**

A：GPipe 在开始反向之前需要保存**所有 M 个 micro-batch 的中间激活**，显存 ∝ M（批次数）。1F1B 进入稳定状态后，每张卡最多只需同时持有 P 份激活（与流水线深度相关，与 M 无关）。大 M 时 GPipe 显存爆，1F1B 不会。

**Q2：ZeroBubble 为什么能减少 bubble？代价是什么？**

A：普通 1F1B 中，一张卡算完 dx 后要把 dx 通信给上游，通信期间 GPU 空闲（bubble）。ZeroBubble 把 backward 拆成 `backward_for_input`（算 dx → 立即通信）和 `backward_for_weight`（算 dw），在 dx 通信期间并发计算 dw，用计算填充通信时间。代价是需要额外存储 `dh`（中间层梯度），显存略增。

**Q3：DualPipe 解决的是什么问题？为什么 1F1B 解决不了？**

A：DeepSeek-V3 使用 MoE + Expert Parallelism，EP 的 All-to-All 通信（token 路由到各个 expert）很难在 1F1B 框架内隐藏——1F1B 是单向数据流，通信和计算不能在同一时间步被两路数据同时利用。DualPipe 引入双向流水线，前半段和后半段同时运行互补操作，使得 EP 通信可以被另一路的计算覆盖。本质是"资源消耗不会消失，只会转移"。

**Q4：`stage_output.backward(gradient=grad_output)` 和 `loss.backward()` 的区别？**

A：只有最后一张卡持有标量 loss，可以调 `loss.backward()`。中间 stage 的输出是向量，没有标量 loss；需要用从下游接收来的梯度 `grad_output`（即 `∂L/∂stage_output`）作为外部梯度调用 `tensor.backward(gradient=grad_output)`，PyTorch 会把这个梯度作为 chain rule 的起点往上游反向传播。

**Q5：流水线并行的 bubble 能完全消除吗？**

A：理论上 ZeroBubble 和 DualPipe 可以让 bubble→0，但：(1) ZeroBubble 额外存 dh 增加显存；(2) DualPipe 实现极复杂且仅在 MoE EP 通信足够大时才划算；(3) 实际系统中通信和计算很难完美重叠（硬件 overlap 支持、调度精度等）。实践中是 bubble 率和工程复杂度的权衡。

**Q6：PP 适合和哪些并行方式组合？**

A：PP（层间切分）+ TP（层内切分）+ DP（数据复制）= 3D 并行，是 Megatron-LM 和 DeepSeek-V3 的标准配置。PP 负责解决模型层数超出单机、TP 负责解决单层参数超出单卡、DP 负责扩展吞吐量。加 ZeRO 后变成 4D。

---

## 十、知识关联

- **前置**：[[AI/LLM/Infra/xtrain-lc4-张量并行从零手写]] — TP 和 PP 是两个维度的并行
- **前置**：[[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — ZeRO + PP 减少每个 stage 的显存
- **深化**：DeepSeek-V3 技术报告 — DualPipe 的正式描述（Figure 5，Schedule 图）
- **横向**：Megatron-LM 的 Virtual Stage PP — 1F1B 变体，进一步减少 bubble
- **生产**：torchpippy / DeepSpeed pipeline engine / Megatron-LM pipeline scheduler
- **MA-RLHF 系列**：[[AI/LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]

## See Also

- [[AI/LLM/Infra/xtrain-lc4-张量并行从零手写]] — 前置：TP 层内并行，与 PP 层间切分正交可组合
- [[AI/LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]] — 前置：ZeRO+PP 减少每 stage 显存
- [[AI/LLM/Infra/xtrain-lc6-Context并行RingAttention手写]] — 后置：CP 序列维度切分，PP+CP 超长序列训练
- [[AI/LLM/Infra/xtrain-lc7-MoE专家并行从零手写]] — DualPipe 设计动机正是 MoE EP 通信重叠
- [[AI/LLM/Infra/分布式训练]] — 分布式训练理论全景
- [[AI/LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]] — xtrain 系列课程地图
