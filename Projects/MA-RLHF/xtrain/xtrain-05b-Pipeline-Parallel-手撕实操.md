---
title: "Pipeline Parallel 手撕实操"
brief: "流水线并行完整实现：GPipe（朴素前向/micro-batch梯度累积）、1F1B调度（内存从O(N)降至O(1)）、交错式1F1B（bubble率降低1/m）、PipeDream异步流水线，bubble率公式推导，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, pipeline-parallel, gpipe, 1f1b, distributed-training, pytorch]
related:
  - "[[Projects/MA-RLHF/xtrain/xtrain-04b-Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-03b-ZeRO-手撕实操|ZeRO-手撕实操]]"
  - "[[Projects/MA-RLHF/lc-comm/lc-comm-01-分布式训练通信原语-手撕实操|分布式训练通信原语-手撕实操]]"
---

# 流水线并行（Pipeline Parallelism）手撕实操

> 来源：MA-RLHF xtrain (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

1. [Naive PP（基础流水线）](#1-naive-pp基础流水线)
2. [GPipe（micro-batch 流水线）](#2-gpipemicro-batch-流水线)
3. [PipeDream（1F1B）](#3-pipedream1f1b)
4. [Zero Bubble（dx/dw 分离）](#4-zero-bubbledxdw-分离)
5. [DualPipe（DeepSeek 双向流水线）](#5-dualpipedeepseek-双向流水线)
6. [Activation Checkpointing](#6-activation-checkpointing)

---

## 1. Naive PP（基础流水线）

### 基本思路

将模型按层切分到不同 GPU，串行执行前向和反向传播。

```
模型共 N 层，P 个 GPU
每个 GPU 持有 N/P 层
Forward:  GPU0 → GPU1 → GPU2 → GPU3
Backward: GPU3 → GPU2 → GPU1 → GPU0
```

### Bubble 问题

**Bubble 率 = (P-1) / (P-1+1) = (P-1) / P**

4 卡示例（时间步）：
```
GPU0: [F] [idle] [idle] [idle] [idle] [idle] [B] 
GPU1:     [F]    [idle] [idle] [idle] [B]
GPU2:            [F]    [idle] [B]
GPU3:                   [F|B]
```

大量 GPU 空闲等待，利用率极低。

### 实现要点

1. 数据流：rank 0 发送激活值到 rank 1，依次传递
2. 反向时：最后一个 rank 计算 loss，将 `grad_output` 回传
3. 使用 `tensor.backward(gradient=stage_output_grad)` 对中间变量做反向传播

### 代码实现

```python
class PipeModel(nn.Module):
    def __init__(self, dim, num_blocks, rank=0, world_size=1):
        super().__init__()
        self.local_num_blocks = num_blocks // world_size
        self.layers = nn.ModuleList([
            MLP(dim, rank, world_size) for _ in range(self.local_num_blocks)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x   # 残差连接
        return x


# Forward: 逐 rank 串行
if rank == 0:
    stage_output = pipe_model(x)
    dist.send(stage_output, dst=rank + 1)
else:
    dist.recv(x, src=rank - 1)
    stage_output = pipe_model(x)
    if rank != world_size - 1:
        dist.send(stage_output, dst=rank + 1)

# Backward: 反向串行，传递梯度
if rank == world_size - 1:
    loss = loss_fn(stage_output, label)
    loss.backward()
    dist.send(x.grad.clone(), dst=rank - 1)
else:
    dist.recv(stage_output_grad, src=rank + 1)
    stage_output.backward(gradient=stage_output_grad)
    if rank != 0:
        dist.send(x.grad.clone(), dst=rank - 1)
```

---

## 2. GPipe（micro-batch 流水线）

### 核心改进

将一个 mini-batch 切分为 M 个 **micro-batch**，让不同 GPU 同时处理不同 micro-batch：

```
P=4 卡, M=4 micro-batch:
GPU0: F₀ F₁ F₂ F₃                B₃ B₂ B₁ B₀
GPU1:    F₀ F₁ F₂ F₃          B₃ B₂ B₁ B₀
GPU2:       F₀ F₁ F₂ F₃    B₃ B₂ B₁ B₀
GPU3:          F₀ F₁ F₂ [F₃ B₃] B₂ B₁ B₀
```

### Bubble 率

**Bubble 率 = (P-1) / (P-1+M)**

当 M >> P 时，bubble 率趋近于 0。

### 梯度累积

多个 micro-batch 的梯度自动累积（torch 的 `.backward()` 叠加 `.grad`），最后统一 `optimizer.step()`。

### 实现要点

- Forward：异步发送 `dist.isend`，同步接收 `dist.recv`
- Backward：反向遍历 micro-batch，梯度累积后统一更新
- loss 需要除以 micro-batch 数量：`loss /= world_size`

### 代码实现

```python
# Forward: 逐 micro-batch 流水
for i in range(world_size):
    if rank != 0:
        dist.recv(x_list[i], src=rank - 1)          # 同步接收

    x_list[i].retain_grad()
    stage_output = pipe_model(x_list[i])
    stage_output_list[i] = stage_output

    if rank != world_size - 1:
        req = dist.isend(stage_output.clone(), dst=rank + 1)  # 异步发送
        reqs.append(req)

# Backward: 反向遍历 micro-batch，梯度累积
for i in range(world_size - 1, -1, -1):
    if rank != world_size - 1:
        dist.recv(stage_output_grad_list[i], src=rank + 1)

    if rank == world_size - 1:
        loss = loss_fn(stage_output_list[i], label_list[i])
        loss /= world_size   # 梯度累积：除以 micro-batch 数量
        loss.backward()
    else:
        stage_output_list[i].backward(gradient=stage_output_grad_list[i])

    if rank != 0:
        req = dist.isend(x_list[i].grad.clone(), dst=rank - 1)

optimizer.step()  # 统一更新
```

### GPipe 的问题

**显存占用高**：最后一个 rank 在开始 backward 前需要存储所有 micro-batch 的中间激活值（F₀~F₃）。

---

## 3. PipeDream（1F1B）

### 核心思想

1F1B = 1 Forward 1 Backward，**及时消化 forward 产生的显存**。

最后一个 rank 完成一个 micro-batch 的前向后立即做反向，而不是等所有前向都完成：

```
GPipe（存 4 份激活）：
GPU3:  F₀ F₁ F₂ F₃ B₃ B₂ B₁ B₀

PipeDream（最多存 1 份）：
GPU3:  F₀ B₀  F₁ B₁  F₂ B₂  F₃ B₃
```

### Schedule 设计

总时序为 `P + M - 1` 步，每步包含条件性的 1F 和 1B：

```
4 卡, M=4:
        it0   it1   it2   it3   it4   it5   it6
GPU0:  [F0]  [F1]  [--]  [--]  [--]  [B1]  [B0]
       [--]  [--]  [--]  [B0]  [B1]  [--]  [--]
GPU1:  [--]  [F0]  [F1]  [--]  [B1]  [B0]  [--]
       [--]  [--]  [B0]  [B1]  [--]  [--]  [--]
GPU2:  [--]  [--]  [F0]  [F1]  [B0]  [--]  [--]
       [--]  [B0]  [B1]  [--]  [--]  [--]  [--]
GPU3:  [--]  [--]  [--]  [F0 B0] [F1 B1] [--] [--]
```

### Bubble 率

与 GPipe 相同：**(P-1) / (P-1+M)**

但**峰值显存大幅下降**：只需保存当前活跃的中间变量，不需要积压所有 micro-batch。

### 代码实现

```python
for i in range(world_size + micro_batch_size - 1):
    # 1F: 满足时序条件就前向
    if i >= rank and f_idx < micro_batch_size:
        if rank != 0:
            dist.recv(x_list[f_idx], src=rank - 1, tag=10010)
        x_list[f_idx].retain_grad()
        stage_output = pipe_model(x_list[f_idx])
        stage_output_list.append(stage_output)
        if rank != world_size - 1:
            req = dist.isend(stage_output.clone(), dst=rank + 1, tag=10010)
        f_idx += 1

    # 1B: 最后一个 rank 完成前向后立刻反向
    if i >= world_size - 1 and b_idx < micro_batch_size:
        if rank != world_size - 1:
            dist.recv(stage_output_grad_list[b_idx], src=rank + 1, tag=10086)
        if rank == world_size - 1:
            loss = loss_fn(stage_output_list[b_idx], label_list[b_idx])
            loss /= world_size
            loss.backward()
        else:
            stage_output_list[b_idx].backward(gradient=stage_output_grad_list[b_idx])
        if rank != 0:
            req = dist.isend(x_list[b_idx].grad.clone(), dst=rank - 1, tag=10086)
        b_idx += 1
```

---

## 4. Zero Bubble（dx/dw 分离）

### 核心洞察

反向传播可以分解为两个独立步骤：

```
Forward:   y₁ = x₁ * w₁ → y₂ = y₁ * w₂

Backward(dx):          # 计算输入梯度，需要通信
  dy₁ = dy₂ * w₂
  dx₁ = dy₁ * w₁
  → 立即 isend(dx) 给上一个 rank

Backward(dw):          # 计算权重梯度，纯本地计算
  dw₂ = dy₂ * y₁
  dw₁ = dy₁ * x₁
  → 与通信重叠执行
```

**Schedule**: 1F-1B-1W（F=Forward, B=Backward_dx, W=Backward_dw）

### 通信-计算重叠

```
时间线:
t1: backward_dx → 计算 dx
t2: [通信] isend(dx)    |  [计算] backward_dw（与通信并行）
```

**关键**：资源消耗不会消失，只会转移。dx/dw 分离增加了中间变量的显存占用（需要存储用于计算 dw 的中间数据），但实现了计算-通信隐藏。

### Bubble 率

理论上可以达到**近零 bubble**，代价是更高的显存峰值。

### 代码实现

```python
class ZeroBubbleMLP(nn.Module):
    def __init__(self, dim, rank=0, world_size=1):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        h = self.w1(x)
        o = self.w2(h)
        return h, o            # 返回中间变量 h，供 dw 计算

    def backward_for_input(self, do):
        """只计算 dx，不计算 dw"""
        dh = do @ self.w2.weight
        dx = dh @ self.w1.weight
        return dh, dx

    def backward_for_weight(self, do, dh, h, x):
        """只计算 dw，与通信重叠"""
        self.w2.weight.grad = do.t() @ h   # dim×bs @ bs×4dim
        self.w1.weight.grad = dh.t() @ x


class ZeroBubbleModel(nn.Module):
    def backward_zero_bubble(self, layers_output, do, b_idx, is_send=True, dst=None):
        # Step 1: 逐层计算 dx
        dx = None
        for layer, layer_output in zip(reversed(self.layers), reversed(layers_output)):
            dh, dx = layer.backward_for_input(do)
            layer_output.append(dh)    # 保存中间梯度，供后续 dw 使用
            layer_output.append(dx)

        # Step 2: 异步发送 dx（通信）
        if self.rank != 0 and is_send:
            req = dist.isend(dx, dst=self.rank - 1, tag=10086)
            req.wait()

        # Step 3: 计算 dw（与通信重叠）
        for layer, layer_output in zip(reversed(self.layers), reversed(layers_output)):
            layer.backward_for_weight(
                x=layer_output[0], h=layer_output[1],
                dh=layer_output[2], do=do,
            )
        return dx
```

---

## 5. DualPipe（DeepSeek 双向流水线）

### 背景

DualPipe 是 DeepSeek 提出的双向流水线方案，专门解决 **MoE 模型中 EP（Expert Parallelism）通信重叠**导致的 bubble 问题。

### 核心思想

将 micro-batch 分为两半，从**两端同时**发起流水线：

```
正向流: 数据从 rank 0 → rank P-1
反向流: 数据从 rank P-1 → rank 0
双向同时执行，最大化 GPU 利用率
```

### Schedule 设计（8 步）

对于 P 个 rank（要求 P 为偶数），num_chunks 个 micro-batch（要求为偶数且 ≥ 2P）：

```
Step 1: nF0          — 填充正向流水线
Step 2: nF0F1        — 同时启动正/反向流
Step 3: nB1W1F1      — Zero Bubble 式 dw 分离
Step 4: nF0B1F1B0    — 主循环：双向 forward+backward 交错（最大吞吐）
Step 5: nB1F1B0      — 排空阶段
Step 6: nB1B0        — 后半段使用 Zero Bubble
Step 7: nWB0         — 排空 + weight 更新
Step 8: nW           — 剩余 weight 更新
```

其中 phase 0 = 正向流，phase 1 = 反向流。前半 rank 和后半 rank 的 phase 含义互换。

### 操作集合

DualPipe 将每个计算步骤拆解为 5 种原子操作：

| 操作 | 含义 |
|---|---|
| **F** | 纯前向（只发不收反向梯度） |
| **FF** | Forward-Forward（双向同时前向） |
| **FB** | Forward-Backward（正向前向 + 反向后向重叠） |
| **BB** | Backward-Backward（双向后向） |
| **B** | 纯后向 |

### 关键实现

```python
class DualPipe(nn.Module):
    def __init__(self, modules, batch_dim=0, process_group=None):
        super().__init__()
        self.module = nn.ModuleList(modules)  # (module_forward, module_backward)
        # 前半 rank: phase 0=正向, phase 1=反向
        # 后半 rank: phase 0=反向, phase 1=正向
        self.is_in_second_half = self.rank >= self.num_ranks // 2

    def _forward_backward_compute_chunk(self, phase0, phase1):
        """同时执行一个 forward chunk 和一个 backward chunk"""
        if self.overlapped_forward_backward:
            # 利用 overlapped_forward_backward 接口实现计算重叠
            outputs0, loss0 = type(module0).overlapped_forward_backward(
                module0, inputs0, criterion0, labels0,
                module1, loss1, outputs1, output_grads1,
            )
        else:
            self._forward_compute_chunk(phase0)
            self._backward_compute_chunk(phase1)

    def step(self, *inputs, num_chunks, criterion=None, labels=[]):
        """完整的 DualPipe 训练步骤"""
        # Step 1-8 按 schedule 编排
        # Step 1: nF0
        for i in range((num_half_ranks - half_rank - 1) * 2):
            self._forward_chunk(0)

        # Step 4 (Main): nF0B1F1B0 — 最大并行度
        for i in range(step_4):
            self._forward_backward_chunk(0, 1)  # 正向F + 反向B 重叠
            self._forward_backward_chunk(1, 0)  # 反向F + 正向B 重叠

        # Step 6: nB1B0 with Zero Bubble
        for i in range(half_rank + 1):
            self._backward_chunk(1, enable_zb=True)   # enable Zero Bubble
            self._backward_chunk(0, enable_zb=True)

        # Step 8: nW — flush weight gradients
        for i in range(half_rank + 1):
            self._weight_chunk()

        loss = torch.stack(self.loss_chunks)
        return loss, outputs
```

### 通信管理

DualPipe 使用 `batch_isend_irecv` 批量通信，并用 `WeightGradStore` 延迟 dw 计算：

```python
def _commit_and_wait_comm(self):
    """提交所有待发通信，阻塞等待完成"""
    if not self.comm_ops:
        return
    reqs = dist.batch_isend_irecv(self.comm_ops)
    for req in reqs:
        req.wait()
    self.comm_ops = []
    self._free_tensors()
```

### 思考题（来自原仓库）

1. 当前 MoE 类网络训练瓶颈是什么？→ EP 通信成为主要 bottleneck
2. DualPipe 解决了什么？→ 通过双向流水线，将 EP 通信与计算重叠
3. 计算-通信重叠的输入输出是什么？→ 输入为 activation/gradient tensor，输出为通信后的完整 tensor
4. DualPipe schedule 设计思路？→ 双向填充 → 稳态双向交错 → 排空
5. 1F1B / 0B1B / 0F1F 重叠是否可行？→ 需要分析数据依赖

---

## 6. Activation Checkpointing

### 原理

标准反向传播需要保存所有中间激活值用于梯度计算，内存开销 O(N)。Activation Checkpointing 的策略：

- **前向时**：不保存中间激活值（`torch.no_grad()`）
- **反向时**：重新执行前向计算（重计算），获取中间激活值

**时间换空间**：计算量约增加 33%，但内存降至 O(√N)。

### PyTorch API 用法

```python
from torch.utils.checkpoint import checkpoint

def checkpointed_forward(module, *inputs):
    return checkpoint(module, *inputs, use_reentrant=False)

class MyModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        x = checkpointed_forward(self.layer2, x)  # 只对 layer2 做 checkpoint
        x = self.layer3(x)
        return x
```

`use_reentrant=True`：计算过程不记录在计算图里
`use_reentrant=False`：记录在计算图里（推荐）

### 手写 Checkpoint 实现

核心：自定义 `autograd.Function`，forward 时用 `torch.no_grad()` 不保存中间变量，backward 时用 `torch.enable_grad()` 重计算。

```python
class CustomCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, *args):
        ctx.func = func
        ctx.save_for_backward(*args)
        with torch.no_grad():
            # 不保存中间 activation，只保存输入
            outputs = func(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        # 重计算：重新执行前向，这次记录计算图
        with torch.enable_grad():
            outputs = ctx.func(*inputs)
        # 基于重计算的输出，反向传播
        torch.autograd.backward(outputs, grad_outputs)
        return (None,) + tuple(inp.grad for inp in inputs)
```

### 验证：hook 检测重计算

```python
intermediates = []
def hook(module, input, output):
    intermediates.append(output[0, :4])

model.w1.register_forward_hook(hook)

# 前向 → 记录 1 次
output = CustomCheckpointFunction.apply(model.forward, input)
print(len(intermediates))  # 1

# 反向 → 重计算触发第 2 次前向
output.backward(gradient=grad_output)
print(len(intermediates))  # 2（两次数据相同，第二次带 grad_fn）
```

---

## 总结：PP 方案演进

| 方案 | Bubble 率 | 峰值显存 | 关键技术 |
|---|---|---|---|
| **Naive PP** | (P-1)/P | 低 | 串行 F→B |
| **GPipe** | (P-1)/(P-1+M) | **高**（存所有 micro-batch 激活） | micro-batch + 梯度累积 |
| **PipeDream 1F1B** | (P-1)/(P-1+M) | **中**（及时释放激活） | 前向后立即反向 |
| **Zero Bubble** | **≈0** | 高（dx/dw 中间变量） | dx/dw 分离 + 计算-通信重叠 |
| **DualPipe** | **≈0** | 高 | 双向流水线 + 计算-通信重叠 |

**演进主线**：减少 bubble → 降显存 → 计算-通信重叠 → 双向流水线（为 MoE EP 通信而生）。
