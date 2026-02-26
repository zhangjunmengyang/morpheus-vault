---
title: "xtrain lc2 — 数据并行从零手写"
brief: "从零实现 DP（DataParallel）和 DDP（DistributedDataParallel）：朴素参数同步 vs Ring-AllReduce gradient sync。理解 PyTorch DDP 内部工作机制，ZeRO 的出发点。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, distributed-training, data-parallel, ddp, xtrain]
related:
  - "[[AI/3-LLM/Infra/xtrain-lc1-分布式通信原语从零手写]]"
  - "[[AI/3-LLM/Infra/xtrain-lc3-ZeRO优化器从零手写]]"
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/xtrain-分布式并行手写-MOC]]"
---

# xtrain lc2 — 数据并行从零手写

> 来源：MA-RLHF xtrain lecture/lc2_data_parallelism/（3个py文件）
> 核心内容：从手写梯度 AllReduce 到 PyTorch DDP API，再到分布式数据加载

---

## 1. DDP 工作原理

### 核心流程（必须理清的顺序）

```
各 rank 持有相同模型参数
       ↓
各 rank 拿到不同的 mini-batch 数据
       ↓
各 rank 独立 forward → 得到不同的 loss
       ↓
各 rank 独立 backward → 得到不同的 gradient
       ↓
AllReduce 梯度（SUM → 除以 world_size = AVG）  ← 这是唯一的通信点
       ↓
各 rank 用相同梯度执行 optimizer.step()
       ↓
各 rank 参数保持一致
```

> **关键纠正**：DDP 不是"AllReduce loss 再 backward"。那样做数学上等价（因为 backward 是线性算子），但实际实现是各 rank **独立 backward 后 AllReduce 梯度**。原因：
> 1. AllReduce loss 再 backward → 每个 rank 仍需独立 backward（loss 标量 AllReduce 没有意义）
> 2. 实际 DDP 用 **gradient hook** 在 backward 过程中 overlap 通信与计算

### DDP 中的5个关键性质

```python
# 原始代码注释精华
# 各个 rank 上：
# 1. 参数同步 — 初始化时 broadcast 保证一致
# 2. 数据不同 — DistributedSampler 保证不重叠
# 3. loss不同 — 因为数据不同
# 4. gradient相同 — AllReduce 后一致
# 5. optimizer后更新的参数相同 — 因为梯度相同 + 参数相同 + 同一个 optimizer
```

---

## 2. 手写 DP 梯度同步

### 2.1 手写版完整代码

以下是课程中手写的 DDP 实现——不用 `loss.backward()`，而是**手动推导每层梯度并 AllReduce**：

```python
def train_my_ddp(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    """
    手写数据并行梯度同步
    关键思路：每个 rank 独立计算各层梯度 → AllReduce 求均值 → 手动更新参数
    """
    lr = optimizer.param_groups[0]['lr']
    bs, _ = input.shape
    _, dim_out = labels.shape

    with torch.no_grad():
        for i in range(epochs):
            # === Forward ===
            output, hidden = model(input)       # output = w2 @ (w1 @ input)
            loss = loss_fn(output, labels)       # MSELoss

            # === Backward（手动求导）===
            # MSELoss 的梯度：d_loss/d_output = 2(output - labels) / (bs * dim_out)
            do = 2 * (output - labels) / (bs * dim_out)

            # w2 层梯度
            grad_w2 = do.t() @ hidden             # [dim_out, dim_hidden]
            grad_hidden = do @ model.w2.weight     # hidden 的梯度（用于继续反传）

            # AllReduce w2 梯度 → 所有 rank 得到一致的聚合梯度
            dist.all_reduce(grad_w2, dist.ReduceOp.SUM)
            grad_w2 = grad_w2 / world_size        # 求均值
            model.w2.weight -= lr * grad_w2        # 手动 SGD 更新

            # w1 层梯度
            grad_w1 = grad_hidden.t() @ input      # [dim_hidden, dim_in]

            # AllReduce w1 梯度
            dist.all_reduce(grad_w1, dist.ReduceOp.SUM)
            grad_w1 = grad_w1 / world_size
            model.w1.weight -= lr * grad_w1
```

### 2.2 代码解析

| 步骤 | 操作 | 是否通信 |
|------|------|---------|
| forward | 各 rank 用自己的 mini-batch 前向 | ❌ |
| 手动求 grad_w2 | `do.t() @ hidden`（局部梯度） | ❌ |
| AllReduce grad_w2 | `dist.all_reduce(grad_w2, SUM)` → 除以 N | ✅ |
| 更新 w2 | `w2 -= lr * grad_w2` | ❌ |
| 手动求 grad_w1 | `grad_hidden.t() @ input` | ❌ |
| AllReduce grad_w1 | `dist.all_reduce(grad_w1, SUM)` → 除以 N | ✅ |
| 更新 w1 | `w1 -= lr * grad_w1` | ❌ |

> **注意**：`grad_hidden`（中间激活的梯度）**不需要** AllReduce，因为它只是反传链路的中间值，不用于参数更新。只有**参数的梯度**需要 AllReduce。

### 2.3 验证：手写 vs PyTorch DDP 结果一致

```python
# 课程代码的验证方式
print(f'rank {rank}: my model', my_model.w1.weight.data[0, :4])
print(f'rank {rank}: ddp model', ddp_model.module.w1.weight.data[0, :4])
# 两者输出一致，证明手写实现正确
```

---

## 3. DistributedSampler：确保各 rank 数据不重叠

### 3.1 PyTorch DistributedSampler 方案

每个 rank 都持有完整数据集，但 `DistributedSampler` 确保各 rank 只取自己的那份索引。

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

dataset = MyDataset(N=1024, dim=16, classes=10)
sampler = DistributedSampler(dataset)  # 自动根据 rank 和 world_size 分配索引
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

# DistributedSampler 内部逻辑：
# 总共 1024 个样本，4个 rank
# rank 0 → indices [0, 4, 8, 12, ...]
# rank 1 → indices [1, 5, 9, 13, ...]
# rank 2 → indices [2, 6, 10, 14, ...]
# rank 3 → indices [3, 7, 11, 15, ...]
# 等价于：indices[rank::world_size]
```

**关键约束**：
- `steps × world_size × batch_size = N`
- 每个 epoch 应调用 `sampler.set_epoch(epoch)` 以改变 shuffle seed

### 3.2 手写分布式数据集（Scatter 方案）

课程还实现了另一种方案：rank 0 持有完整数据，通过 `scatter` 分发到各 rank。

```python
def run_distributed_dataset(rank, ...):
    # 每个 rank 创建空壳数据集
    distributed_dataset = MyDataset(N // world_size, dim_in, dim_out)

    if rank == 0:
        # rank 0 持有完整数据并切分
        dataset = MyDataset(N, dim_in, dim_out)
        distributed_datas = list(dataset.data.split(N // world_size, dim=0))
        distributed_labels = list(dataset.labels.split(N // world_size, dim=0))
    else:
        distributed_datas = None
        distributed_labels = None

    # 用 scatter 将数据分发到各 rank
    dist.scatter(distributed_dataset.data, distributed_datas, src=0)
    dist.scatter(distributed_dataset.labels, distributed_labels, src=0)
    dist.barrier()

    # 每个 rank 只用自己的数据子集
    dataloader = DataLoader(distributed_dataset, batch_size=2)
```

### 3.3 两种方案对比

| 维度 | DistributedSampler | Scatter 分发 |
|------|-------------------|-------------|
| 数据存储 | 每个 rank 存全量数据 | 每个 rank 只存 1/N |
| 内存开销 | 较大（冗余存储） | 较小 |
| 初始化通信 | 无 | 需要一次 scatter |
| shuffle 灵活性 | 每 epoch 可重新 shuffle | 需重新分发 |
| 生产环境 | ✅ 标准方案 | 较少使用 |
| 适用场景 | 数据量不太大 | 数据量巨大，内存敏感 |

---

## 4. PyTorch DDP API vs 手写实现对比

### 4.1 PyTorch DDP 用法

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = ToyModel()
ddp_model = DDP(model)  # 包装模型

optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs, _ = ddp_model(input)        # forward（自动处理）
    loss = loss_fn(outputs, labels)
    loss.backward()                       # backward 时自动触发 AllReduce
    optimizer.step()                      # 用同步后的梯度更新
```

### 4.2 DDP 内部做了什么

1. **构造时**：`DDP(model)` 会 broadcast rank 0 的参数到所有 rank，确保初始参数一致
2. **Forward 时**：直接调用 `model.forward()`，无特殊处理
3. **Backward 时**：
   - DDP 注册了 **autograd hook**（梯度钩子）
   - 当某个参数的梯度计算完成时，hook 自动触发 AllReduce
   - 使用 **bucket** 机制：将参数梯度打包成若干 bucket，一个 bucket 满了就立即 AllReduce
   - 这实现了 **计算-通信 overlap**（下一层在算梯度时，上一层的梯度已经在 AllReduce 了）
4. **Optimizer.step()**：因为梯度已经同步，各 rank 独立更新得到相同参数

### 4.3 对比总结

| 维度 | 手写实现 | PyTorch DDP |
|------|---------|-------------|
| 梯度计算 | 手动逐层推导 | autograd 自动 |
| 梯度同步 | 手动逐层 AllReduce | hook + bucket 自动 |
| 通信-计算 overlap | ❌ 串行 | ✅ bucket 满即发 |
| 代码复杂度 | 高（需要推导每层梯度） | 低（3行包装） |
| 性能 | 较差（无 overlap） | 好（overlap + bucket） |
| 教学价值 | 高（理解本质） | — |

---

## 5. Gradient Accumulation 与 DDP 配合

### 5.1 问题

梯度累积（Gradient Accumulation）意味着多个 micro-step 做 forward + backward 但不 step，最后一步才 step。

在 DDP 中，**每次 backward 都会触发 AllReduce**。如果你只想在最后一步同步，中间的 AllReduce 就是浪费通信。

### 5.2 解决方案：no_sync()

```python
# 梯度累积 K 步
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # 前 K-1 步：禁用 AllReduce
    if (i + 1) % accumulation_steps != 0:
        with ddp_model.no_sync():  # 上下文管理器：跳过 AllReduce
            output = ddp_model(batch)
            loss = loss_fn(output, labels)
            loss.backward()  # 梯度只在本地累积，不通信
    else:
        # 第 K 步：正常 backward（触发 AllReduce）
        output = ddp_model(batch)
        loss = loss_fn(output, labels)
        loss.backward()     # 这次会 AllReduce 累积的梯度
        optimizer.step()
        optimizer.zero_grad()
```

**`no_sync()` 的原理**：临时取消 autograd hook 中的 AllReduce 操作。梯度在本地 `.grad` 中累积，直到退出 `no_sync()` 后的下一次 backward 才同步。

---

## 6. DDP vs FSDP vs ZeRO 的定位差异

| 方案 | 一句话概括 | 内存占用 | 通信量 |
|------|-----------|---------|--------|
| **DDP** | 每个 rank 存完整参数+梯度+优化器状态，只同步梯度 | 高（全量冗余） | AllReduce 梯度 |
| **ZeRO-1** | 分片优化器状态，参数和梯度仍全量存 | 省 ~4× 优化器内存 | AllReduce 梯度 |
| **ZeRO-2** | 分片优化器状态 + 梯度，参数仍全量存 | 省更多 | ReduceScatter 梯度 |
| **ZeRO-3 / FSDP** | 参数+梯度+优化器全部分片，用时 AllGather 恢复 | 最省（~1/N） | AllGather 参数 + ReduceScatter 梯度 |

**简版定位**：
- **DDP**：参数放得下就用 DDP，通信最少、实现最简单
- **FSDP / ZeRO-3**：参数放不下时切参数，用通信换内存
- **ZeRO-1/2**：折中方案，先省优化器内存

---

## 7. 面试考点

### Q1: DDP 中梯度同步发生在哪个时机？是 AllReduce loss 还是 AllReduce gradient？

**答**：AllReduce **gradient**，不是 loss。各 rank 独立 forward + backward 得到各自的梯度，然后 AllReduce 梯度求均值。DDP 利用 autograd hook 在 backward 过程中异步触发通信，实现计算-通信 overlap。

### Q2: DistributedSampler 如何保证各 rank 数据不重叠？

**答**：`DistributedSampler` 先对数据集索引 shuffle（seed 固定），然后按 `indices[rank::world_size]` 切分，使得每个 rank 取到互不重叠的子集。每个 epoch 调用 `set_epoch()` 改变 shuffle seed。

### Q3: DDP 的 no_sync() 是做什么的？什么时候用？

**答**：`no_sync()` 是一个上下文管理器，进入时临时禁用 backward 中的 AllReduce。用于 **梯度累积**场景：多个 micro-step 只在本地累积梯度，最后一步才同步，避免中间步的无效通信开销。

### Q4: DDP 的 bucket 机制是什么？为什么需要它？

**答**：DDP 将模型参数分成若干 **bucket**（默认 25MB），当一个 bucket 内的所有参数梯度都算完后，立即对该 bucket 发起 AllReduce。好处：
- **计算-通信 overlap**：靠近输出层的梯度先算完，先 AllReduce，同时底层还在算
- **减少通信次数**：将多个小 tensor 打包成大 tensor 一次性通信，提高带宽利用率
- Bucket 按参数注册顺序的**逆序**排列（因为 backward 从输出往输入算）

---

## 附：完整 DDP 训练模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def run(rank, world_size):
    # 1. 初始化
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. 模型 → DDP
    model = MyModel().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 3. 数据 → DistributedSampler
    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 4. 训练循环
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        sampler.set_epoch(epoch)  # 必须！确保每 epoch shuffle 不同
        for batch in dataloader:
            optimizer.zero_grad()
            output = ddp_model(batch['input'].cuda(rank))
            loss = loss_fn(output, batch['label'].cuda(rank))
            loss.backward()   # 自动 AllReduce 梯度
            optimizer.step()

    dist.destroy_process_group()
```
