---
title: xtrain lc3 — ZeRO 优化器从零手写
brief: 从零实现 ZeRO-1/2/3 三阶段优化器切分：理解 optimizer state（fp32参数+梯度+一阶+二阶矩）在不同阶段的分片策略，掌握 AllReduce/ReduceScatter/AllGather 通信原语在 ZeRO 各阶段的应用，以及 ZeRO 与 TP 的根本差异（存储分摊 vs 计算并行）。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - distributed-training
  - zero
  - xtrain
related:
  - "[[AI/3-LLM/Infra/xtrain-lc2-数据并行从零手写]]"
  - "[[AI/3-LLM/Infra/xtrain-lc4-张量并行从零手写]]"
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]"
  - "[[分布式训练]]"
---

# xtrain lc3 — ZeRO 优化器从零手写

> 来源：`/Users/peterzhang/project/ma-rlhf/xtrain/lecture/lc3_ZeRO/`
> 系列：[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]
> 难度：★★★★★（面试必考，手写 ZeRO1/2/3 完整流程）
> 更新：2026-02-25

---

## TL;DR

ZeRO 本质仍是数据并行（DP），但通过将**优化器状态 / 梯度 / 参数**沿 N 个 rank 切分，把每张卡的显存占用降低至原来的 1/N。xtrain lc3 从零实现了完整的五层演化链，没有任何框架 magic，每一步通信都显式可见。

**演化链**：
```
adam.py          → 单卡手写 Adam（基准）
distributed_adam → 多卡 + AllReduce 梯度（Naive DDP Adam，存储 xN）
adam_zero1       → ZeRO-1：切分 optimizer state，AllReduce梯度 + AllGather参数
adam_zero2       → ZeRO-2：切分 optimizer state + 梯度，AllReduce + Scatter梯度 + AllGather参数
adam_zero3       → ZeRO-3：切分 optimizer state + 梯度 + 参数，前向 AllGather 层参数，即用即丢
adam_mix_precision → 混合精度 Adam（BF16 计算 + FP32 主参数备份 + Loss Scaling）
```

---

## 一、核心问题：分布式训练里为什么要 ZeRO？

训练一个 Transformer 模型，每个参数需要：
- **FP16 参数**：2 bytes
- **FP16 梯度**：2 bytes  
- **FP32 optimizer states（Adam m+v + fp32参数备份）**：12 bytes

合计：**16 bytes per parameter**（混合精度 Adam 标准配置）

7B 参数模型 = **112 GB**，远超单卡 80GB A100。

**Naive DDP 的问题**：每张卡都存完整的 optimizer state，N 卡 = N 倍冗余。

**ZeRO 的核心思路**：切分（shard），在计算时按需聚合（all-gather），计算完立刻丢弃冗余部分。

---

## 二、手写 Adam 基础

### `MyAdam.step()` 核心逻辑

```python
class MyAdam:
    def __init__(self, params, lr=1e-3, beta1=0.90, beta2=0.999, eps=1e-8):
        self.params = list(params)  # 引用，不占额外存储
        self.t = 0.0
        # M[i], V[i] 与 param[i] 形状相同
        self.M = [torch.zeros_like(param.data, dtype=torch.float32) for param in self.params]
        self.V = [torch.zeros_like(param.data, dtype=torch.float32) for param in self.params]

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # 一阶矩（动量）
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            # 二阶矩（自适应学习率）
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            # Bias correction
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            # 参数更新（AdamW 则额外加 weight_decay * param.data）
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
```

> **面试考点**：为什么需要 bias correction？
> 初始化 M=0/V=0，前几步的 m_hat/v_hat 被显著低估（t=1 时 m_hat = grad/(1-beta1)），correction 修复了这一冷启动偏差。

### `zero_grad()` 的正确写法

```python
# ❌ 原始代码的 BUG（in-place操作不正确）：
param.grad *= torch.zeros_like(param.grad)

# ✅ 正确写法：
param.grad = None  # 释放内存（preferred）
# 或：param.grad.zero_()  # 清零但保留 tensor
```

---

## 三、分布式 Adam（Naive，存储 xN 倍冗余）

```python
class DistributedAdam:
    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # 所有卡看到相同的平均梯度（all-reduce 后除以 world_size）
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= self.world_size
            
            # 每张卡独立计算相同的 optimizer update（计算冗余但结果一致）
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
```

**问题**：每卡都存完整 M, V，显存 = 单卡 N 倍（只是计算分摊了，存储没有）。

---

## 四、ZeRO-1：切分 Optimizer State

**切分策略**：每层参数 flatten 后，rank i 只负责 `[i/N, (i+1)/N]` 区间的 optimizer state。

```python
class MyAdamZeRO1:
    def __init__(self, params, world_size, rank, ...):
        # 每层 optimizer state 只存 1/world_size
        for param in self.params:
            shared_size = param.data.numel() // world_size  # 假设整除
            self.M.append(torch.zeros(shared_size, dtype=torch.float32))
            self.V.append(torch.zeros(shared_size, dtype=torch.float32))

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # Step 1：AllReduce 梯度（全局平均）
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= self.world_size

            # Step 2：只计算自己负责的那 1/N 区间
            shared_size = param.grad.numel() // self.world_size
            shared_grad = param.grad.view(-1)[
                self.rank * shared_size : (self.rank + 1) * shared_size
            ]
            M = self.beta1 * M + (1 - self.beta1) * shared_grad
            V = self.beta2 * V + (1 - self.beta2) * shared_grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # Step 3：只更新自己负责的参数区间
            weight_slice = param.data.view(-1)[
                self.rank * shared_size : (self.rank + 1) * shared_size
            ]
            weight_slice -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # Step 4：AllGather 同步所有卡的参数（各卡更新了不同区间，需要聚合）
            gather_tensor = torch.zeros(param.grad.numel(), dtype=param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_slice)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()
```

**通信分析**：
- AllReduce（梯度）：2x梯度大小（reduce + broadcast）
- AllGather（参数）：1x参数大小
- 总计：ZeRO-1 vs Naive DDP 通信量略增，但 optimizer state 显存降至 1/N

---

## 五、ZeRO-2：切分 Optimizer State + 梯度

ZeRO-2 在 ZeRO-1 基础上，进一步避免每卡存完整梯度。

**关键改变**：AllReduce 梯度后立刻 Scatter（每卡只保留自己的 1/N 份梯度）。

```python
def backward_zero2(model, loss, rank, world_size):
    """
    手写 ZeRO-2 的 backward：loss.backward() + reduce-scatter 梯度
    实现方案二（实现方便）：先完整 backward，再逐层 scatter
    方案一（最优）：hook 住每层 backward，边算边 scatter（类似 DDP bucket）
    """
    for param in model.parameters():
        if param.grad is None:
            continue
        
        # Step 1：AllReduce 得到全局平均梯度
        dist.all_reduce(param.grad, dist.ReduceOp.SUM)
        param.grad /= world_size
        tmp_grad = deepcopy(param.grad)

        # Step 2：Scatter 梯度，每卡只保留 1/N 份
        shared_size = param.grad.numel() // world_size
        param.grad.data = torch.zeros(shared_size)  # 收缩梯度 tensor
        
        if rank == 0:
            grad_chunks = list(torch.split(tmp_grad.view(-1), shared_size))
            dist.scatter(param.grad.data, grad_chunks, src=0)
        else:
            dist.scatter(param.grad.data, [], src=0)
```

**ZeRO-2 Optimizer（step 更简单了）**：

```python
class MyAdamZeRO2:
    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # 梯度已经是散射后的 1/N 份，直接用
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # 参数也只更新自己的 1/N 区间
            shared_size = param.data.numel() // self.world_size
            weight_slice = param.data.view(-1)[
                self.rank * shared_size : (self.rank + 1) * shared_size
            ]
            weight_slice -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # AllGather 同步完整参数
            gather_tensor = torch.zeros(param.data.numel(), dtype=param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_slice)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()
```

**ZeRO-1 vs ZeRO-2 对比**：

| 项目 | ZeRO-1 | ZeRO-2 |
|------|--------|--------|
| Optimizer state | 1/N | 1/N |
| 梯度存储 | 每卡完整（AllReduce前） | 1/N（Scatter后） |
| 通信量 | AllReduce(梯度) + AllGather(参数) | AllReduce+Scatter(梯度) + AllGather(参数) |
| 通信量总计 | ≈ 3x参数 | ≈ 3x参数（相同，但梯度峰值显存更低） |

---

## 六、ZeRO-3：切分 Optimizer State + 梯度 + 参数

ZeRO-3 是最激进的：**参数本身也被切分**，每卡只存 1/N 的参数。

代价：前向传播时需要 AllGather 每层参数，用完即丢。

### 关键步骤

**1. 初始化：参数切分（shared_model）**

```python
def shared_model(self):
    """把完整模型参数切分到各 rank，每卡只保留 1/N"""
    for param in self.parameters():
        shared_size = param.data.numel() // self.world_size
        weight_data = deepcopy(param.data).view(-1)
        param.data = torch.zeros(shared_size)  # 收缩参数 tensor
        
        if self.rank == 0:
            scatter_list = list(weight_data.split(shared_size, dim=0))
        else:
            scatter_list = None
        dist.scatter(param.data, scatter_list, src=0)
```

**2. 前向计算：逐层 AllGather，用完即丢**

```python
def forward_zero3(self, x):
    """ZeRO-3 前向：每层 AllGather 完整参数 → 计算 → 参数留着到 backward"""
    # 设计选择：完整 forward 后 backward，而非逐层即时丢弃
    # （即时丢弃 = MR-ZeRO / ZeRO-Infinity 的激进版本）
    
    w1_gather = torch.zeros(self.shape_list[0])
    dist.all_gather_into_tensor(w1_gather.view(-1), self.w1.weight.data)
    self.w1.weight.data = w1_gather  # 临时替换为完整参数
    hidden = self.w1(x)

    w2_gather = torch.zeros(self.shape_list[1])
    dist.all_gather_into_tensor(w2_gather.view(-1), self.w2.weight.data)
    self.w2.weight.data = w2_gather
    output = self.w2(hidden)
    
    return output, hidden
```

**3. 反向计算：AllReduce + Scatter 梯度，然后重新切分参数**

```python
def backward_zero3(model, loss, rank, world_size):
    """ZeRO-3 backward 与 ZeRO-2 相同，但额外在末尾重新 shared_model"""
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, dist.ReduceOp.SUM)
        dist.barrier()
        tmp_grad = deepcopy(param.grad) / world_size

        shared_size = param.grad.numel() // world_size
        param.grad.data = torch.zeros(shared_size)
        
        if rank == 0:
            grad_chunks = list(torch.split(tmp_grad.view(-1), shared_size))
        else:
            grad_chunks = []
        dist.scatter(param.grad.data, grad_chunks, src=0)
    
    # backward 后重新切分参数（前向 AllGather 了完整参数）
    # model.shared_model()  ← 注意：在 train() 里调用，不在这里
```

**4. ZeRO-3 Optimizer（最简单）**：

```python
class MyAdamZeRO3:
    """参数已经是 shard 的，直接更新，不需要 AllGather"""
    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            # ZeRO-3 不需要 AllGather（参数本身就是 shard 的）
```

**ZeRO-3 的 train loop**：

```python
def train(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model.forward_zero3(input)   # AllGather 层参数
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        dist.barrier()
        model.shared_model()           # 前向后重新切分参数（丢弃冗余）
        backward_zero3(model, loss, rank, world_size)  # scatter 梯度
        optimizer.step()               # 在 shard 参数上直接更新
        dist.barrier()
```

---

## 七、混合精度 Adam

**核心设计三角**：
```
BF16 参数（forward/backward 计算）
    ↓ 更新
FP32 主参数备份（optimizer 内部精度累积，防 underflow）
    + FP32 M, V（optimizer state）
```

```python
class MixPrecisionAdam:
    def step(self, scale=10.0):
        self.t += 1
        for param_bf16, param_fp32, M, V in zip(
            self.params, self.params_target, self.M, self.V
        ):
            # 重要：梯度除以 scale（因为 forward 时 loss 乘了 scale）
            param_bf16.grad *= scale  # ← 原代码有 BUG：应该是 /= scale
            
            M = self.beta1 * M + (1 - self.beta1) * param_bf16.grad
            V = self.beta2 * V + (1 - self.beta2) * param_bf16.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # 1. 用 FP32 精度更新 FP32 主参数
            param_fp32.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            
            # 2. 同步回 BF16 参数（用于下次 forward）
            param_bf16.data = param_fp32.data.clone().to(torch.bfloat16)
```

**Loss Scaling 原理**：
- BF16 指数位只有 8 位（vs FP32 的 23 位），极小梯度会 underflow 到 0
- `loss_scale = loss * scale` 放大 loss → 梯度相应放大 → optimizer 内部再除以 scale 恢复量纲
- 动态 scaling（AMP 标准实现）：先尝试 scale=65536，如果出现 inf/nan 则减半，稳定后倍增

---

## 八、ZeRO 三阶段对比

| 阶段 | 切分内容 | 每卡显存 | 额外通信 | 实际部署 |
|------|---------|---------|---------|---------|
| DDP（基准） | 无 | 16 bytes/param × N | AllReduce 梯度 | 小模型 |
| ZeRO-1 | Optimizer state | (4+2+2) bytes/param + 12/N | +AllGather 参数 | 常见 |
| ZeRO-2 | Optimizer + 梯度 | (4+2) bytes/param + 12/N | ≈ZeRO-1 | 主流 |
| ZeRO-3 | Optimizer + 梯度 + 参数 | 16/N bytes/param | +AllGather 前向 | 超大模型 |

> 实际 DeepSpeed ZeRO-3：每卡显存 ≈ **16B/N**，其中 N=64（A100集群）即约 250MB/7B 模型。

---

## 九、面试考点

**Q1：ZeRO-1/2/3 各切分什么？通信量如何变化？**

A：见上表。关键结论：ZeRO-1/2 通信量与 DDP 相当（AllReduce≈AllGather+ReduceScatter）；ZeRO-3 额外增加前向的 AllGather。

**Q2：ZeRO-2 的 backward_zero2 为什么用两步（AllReduce + Scatter）而不是 ReduceScatter？**

A：xtrain 的教学实现用 AllReduce + Scatter（代码更简单）；生产实现（DeepSpeed/FSDP）用 `dist.reduce_scatter`（单步完成，减少通信）。两者语义等价但效率不同。

**Q3：ZeRO-3 前向时为什么是"完整 forward 后再 shared_model"，而不是"逐层即时丢弃"？**

A：逐层即时丢弃（Activation Recomputation + ZeRO-3 联合）实现复杂（需要 hook 住每层 backward）；先完整 forward + backward，再 shared_model 是更简单的两段式实现。ZeRO-Infinity 的激进实现支持逐层丢弃。

**Q4：混合精度训练为什么需要 FP32 主参数备份？**

A：BF16 精度不足以累积微小更新量（如 lr=1e-4 的更新相对参数值可能 < BF16 精度阈值），FP32 备份保证数值稳定性。M, V 也是 FP32 因为同样原因（累积历史梯度统计需要精度）。

**Q5：ZeRO 和 Tensor Parallelism 的本质区别？**

A：ZeRO 切的是**存储**（同一层参数的 replica 被切分到不同卡，前向时 all-gather 重组），不切计算图。TP 切的是**计算**（矩阵乘法沿某维切分，每卡做不同的矩阵块，结果通过 all-reduce 合并）。ZeRO-3 在 all-gather 前参数分散在多卡，本质是**参数共管**；TP 的参数各卡持有不同"条"，永远不 gather，是**参数分工**。

**Q6：deepcopy 梯度 vs 直接操作有什么区别？**

A：`tmp_param = deepcopy(param.grad)` 是为了在 scatter 之前保存完整梯度。直接 `param.grad.data = torch.zeros(shared_size)` 收缩了 tensor，如果不先备份就无法 scatter。这是教学代码的做法；生产代码用 `reduce_scatter` 一步完成无需 deepcopy。

---

## 十、知识关联

- **前置**：[[AI/3-LLM/Infra/xtrain-lc2-数据并行从零手写]] — DDP AllReduce 基础
- **后置**：[[AI/3-LLM/Infra/xtrain-lc4-张量并行从零手写]] — TP 切计算不切存储
- **横向**：[[AI/3-LLM/Infra/ZeRO-手撕实操]] — 更早的 ZeRO 原理笔记（更偏理论）
- **深化**：DeepSpeed ZeRO-Infinity — 把 ZeRO-3 扩展到 CPU/NVMe offload
- **生产对照**：PyTorch FSDP（Fully Sharded Data Parallel）= ZeRO-3 的 PyTorch 原生实现
- **MA-RLHF 系列**：[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]

## See Also

- [[AI/3-LLM/Infra/xtrain-lc2-数据并行从零手写]] — 前置：DDP 数据并行基础
- [[AI/3-LLM/Infra/xtrain-lc4-张量并行从零手写]] — 后置：TP 切计算不切存储（vs ZeRO 切存储）
- [[AI/3-LLM/Infra/ZeRO-手撕实操]] — 横向：ZeRO 原理版（MA-RLHF lc9 版）
- [[分布式训练]] — 分布式训练理论全景
- [[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]] — xtrain 课程地图

---

## 十一、代码文件速查

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `adam.py` | 单卡 Adam 基准 | `MyAdam` |
| `distributed_adam.py` | Naive 分布式 Adam | `DistributedAdam`（显存冗余） |
| `adam_zero1.py` | ZeRO-1 优化器状态切分 | `MyAdamZeRO1`（AllReduce+AllGather） |
| `adam_zero2.py` | ZeRO-2 梯度切分 | `backward_zero2` + `MyAdamZeRO2` |
| `adam_zero3.py` | ZeRO-3 参数切分 | `ToyModel.shared_model` + `forward_zero3` + `backward_zero3` + `MyAdamZeRO3` |
| `adam_mix_precision.py` | 混合精度 Adam | `MixPrecisionAdam`（BF16计算+FP32备份） |
| `zero_io.py` | 模型分块保存 | ZeRO-3 checkpoint 实现 |
| `zero_io_shared_load.py` | 超大模型加载 | 逐层 CPU→GPU 分块传输 |
