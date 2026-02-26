---
title: "ZeRO 手撕实操"
brief: "ZeRO-1/2/3 完整实现：显存分析（参数/梯度/优化器状态三阶段分片）、分布式Adam（all-reduce/reduce-scatter/all-gather通信模式）、ZeRO-3参数异步gather，量化对比DDP冗余，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, zero, distributed-training, memory-optimization, pytorch]
related:
  - "[[AI/LLM/Infra/Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]]"
  - "[[AI/LLM/Infra/分布式训练通信原语-手撕实操|分布式训练通信原语-手撕实操]]"
  - "[[AI/LLM/Infra/Pipeline-Parallel-手撕实操|Pipeline-Parallel-手撕实操]]"
  - "[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]]"
---

# ZeRO 手撕实操

> 来源：MA-RLHF xtrain (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 核心思想

**ZeRO（Zero Redundancy Optimizer）本质仍然是数据并行（DP）**，但通过切分冗余状态来节省显存。

关键洞察：传统 DDP 中每个 rank 都保存完整的模型参数 + 梯度 + 优化器状态，大量冗余。ZeRO 的策略是 **Shard（分片）数据，即算即取**——将数据 flatten 化后均分到各 rank，需要时再通信获取。

---

## 目录

1. [显存分析基础](#1-显存分析基础)
2. [手撕 Adam 优化器](#2-手撕-adam-优化器)
3. [分布式 Adam（DDP baseline）](#3-分布式-adam-ddp-baseline)
4. [混合精度 Adam](#4-混合精度-adam)
5. [ZeRO-1：优化器状态分片](#5-zero-1-优化器状态分片)
6. [ZeRO-2：梯度 + 优化器分片](#6-zero-2-梯度--优化器分片)
7. [ZeRO-3：参数 + 梯度 + 优化器全分片](#7-zero-3-参数--梯度--优化器全分片)
8. [各阶段对比总结](#8-各阶段对比总结)

---

## 1. 显存分析基础

设模型参数量为 Ψ（以参数个数计），使用混合精度训练（fp16/bf16 前向 + fp32 优化器）。

**单卡显存占用（传统 DDP，每个 rank）**：

| 组件 | 精度 | 显存 |
|------|------|------|
| 模型参数（fp16） | 2 bytes | 2Ψ |
| 梯度（fp16） | 2 bytes | 2Ψ |
| 参数备份（fp32） | 4 bytes | 4Ψ |
| Adam M（一阶矩，fp32） | 4 bytes | 4Ψ |
| Adam V（二阶矩，fp32） | 4 bytes | 4Ψ |
| **合计** | | **16Ψ** |

其中优化器状态 = fp32参数备份 + M + V = **12Ψ**，占总显存 75%。

---

## 2. 手撕 Adam 优化器

### 原理

Adam 更新公式（element-wise）：

```
M_t = β₁ · M_{t-1} + (1 - β₁) · g_t
V_t = β₂ · V_{t-1} + (1 - β₂) · g_t²
M̂_t = M_t / (1 - β₁^t)
V̂_t = V_t / (1 - β₂^t)
θ_t = θ_{t-1} - lr · M̂_t / (√V̂_t + ε)
```

**关键特性：element-wise 更新** → 每个参数的更新只依赖自身的 M、V → 天然可切分。

### 完整代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(128, 512, bias=False)
        self.w2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        hidden = self.w1(x)
        output = self.w2(hidden)
        return output, hidden

class MyAdam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)  # 引用，不拷贝——内部更新即外部更新
        self.t = 0.0
        self.M = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]
        self.V = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad *= 0

def run(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)
    model = ToyModel()
    optimizer = MyAdam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    input = torch.randn(128, 128)
    labels = torch.randn(128, 10)

    for i in range(1000):
        outputs, _ = model(input)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(loss)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 1), nprocs=1)
```

---

## 3. 分布式 Adam（DDP baseline）

### 原理

传统 DDP 的问题：**梯度先 reduce 还是更新量先 reduce？**

正确答案：**梯度先 all_reduce 求均值，再各自计算参数更新**。这保证所有 rank 的参数更新一致。

> 如果先各自计算更新量再 reduce，由于 Adam 是非线性运算（含 sqrt），reduce 后的结果 ≠ 正确结果。

### 完整代码

```python
class DistributedAdam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, world_size=1):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)
        self.world_size = world_size
        self.t = 0.0
        self.M = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]
        self.V = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # 关键：先 all_reduce 梯度
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= self.world_size

            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad *= 0
```

**显存**：每个 rank 保存完整的 params + grads + M + V = **16Ψ**（全冗余）。

---

## 4. 混合精度 Adam

### 原理

混合精度训练的存储策略：
1. **bf16 参数**：用于前向/反向（2Ψ + 2Ψ 梯度）
2. **fp32 参数备份**：用于优化器更新（4Ψ）
3. **fp32 M 和 V**：Adam 状态（8Ψ）

训练流程：
1. bf16 前向 → bf16 梯度
2. Loss scaling 防下溢：`loss *= scale`
3. 优化器在 fp32 精度更新参数
4. fp32 参数 cast 回 bf16 同步

### 完整代码

```python
class ToyModel(nn.Module):
    '''参数权重和梯度用 bf16 存储，fp32 备份用于更新'''
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(128, 512, bias=False, dtype=torch.float32)    # fp32 备份
        self.w2 = nn.Linear(512, 10, bias=False, dtype=torch.float32)     # fp32 备份
        self.w1_bf16 = nn.Linear(128, 512, bias=False, dtype=torch.bfloat16)  # 计算用
        self.w2_bf16 = nn.Linear(512, 10, bias=False, dtype=torch.bfloat16)   # 计算用

    def forward(self, x):
        hidden = self.w1_bf16(x)
        output = self.w2_bf16(hidden)
        return output, hidden

class MixPrecisionAdam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, scale=10):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0.0

        # 只优化 bf16 参数，fp32 参数仅用于备份和精确更新
        self.params = []
        self.params_target = []  # 对应的 fp32 备份
        params_dict = dict(params)
        for name, param in params_dict.items():
            if param.requires_grad and 'bf16' in name:
                self.params.append(param)
                target_name = name.replace('_bf16', '')
                self.params_target.append(params_dict[target_name])

        self.M = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]
        self.V = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]

    def step(self, scale=10.0):
        self.t += 1
        for param, param_target, M, V in zip(
                self.params, self.params_target, self.M, self.V):
            if param.grad is None:
                continue
            param.grad *= scale  # unscale gradient

            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            # 在 fp32 精度上更新
            param_target -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            # 同步回 bf16
            param.data = param_target.data.clone().to(torch.bfloat16)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad *= 0
```

---

## 5. ZeRO-1：优化器状态分片

### 原理

**切什么**：将 Adam 的 M 和 V 均分到 N 个 rank。每个 rank 只维护 1/N 的优化器状态。

**流程**：
1. 各 rank 计算本地梯度
2. `all_reduce` 梯度（保证梯度一致）
3. 每个 rank 只更新自己负责的那 1/N 参数
4. `all_gather` 恢复完整参数

**显存节省**：

| 组件 | 传统 DDP | ZeRO-1 |
|------|----------|--------|
| 参数（fp16） | 2Ψ | 2Ψ |
| 梯度（fp16） | 2Ψ | 2Ψ |
| 优化器状态（fp32） | 12Ψ | **12Ψ / N** |
| **合计** | 16Ψ | **4Ψ + 12Ψ/N** |

4 卡时：16Ψ → 7Ψ，节省 56%。

**通信量**：all_reduce（2Ψ）+ all_gather（Ψ）= 3Ψ per step。

### 完整代码

```python
class MyAdamZeRO1:
    '''
    优化器参数 flatten 化后均分到各 rank。
    Adam 更新是 element-wise 的，天然支持切分。
    '''
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, world_size=1, rank=0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)
        self.t = 0.0
        self.world_size = world_size
        self.rank = rank

        # 每个 rank 只初始化 1/N 的优化器状态
        self.M, self.V = [], []
        for param in self.params:
            shared_size = param.data.numel() // self.world_size
            self.M.append(torch.zeros(shared_size, dtype=torch.float32))
            self.V.append(torch.zeros(shared_size, dtype=torch.float32))

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            # Step 1: all_reduce 梯度
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= self.world_size

            # Step 2: 取出本 rank 负责的梯度分片
            shard_size = param.grad.numel() // self.world_size
            shared_grad = param.grad.view(-1)[
                self.rank * shard_size : (self.rank + 1) * shard_size]

            # Step 3: 只更新本 rank 的 M, V, 参数
            M = self.beta1 * M + (1 - self.beta1) * shared_grad
            V = self.beta2 * V + (1 - self.beta2) * shared_grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            weight_data = param.data.view(-1)[
                self.rank * shard_size : (self.rank + 1) * shard_size]
            weight_data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # Step 4: all_gather 恢复完整参数
            gather_tensor = torch.zeros(param.grad.numel(), dtype=param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_data)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad *= 0

def run_zero1(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    model = ToyModel()
    optimizer = MyAdamZeRO1(model.parameters(), lr=0.001,
                            world_size=world_size, rank=rank)
    loss_fn = nn.MSELoss()
    input = torch.randn(128, 128)
    labels = torch.randn(128, 10)

    for i in range(1000):
        outputs, _ = model(input)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(loss)

    # 验证参数同步
    print(rank, model.w1.weight.data)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_zero1, args=("127.0.0.1", "12801", 4), nprocs=4)
```

---

## 6. ZeRO-2：梯度 + 优化器分片

### 原理

**在 ZeRO-1 基础上，额外切分梯度**。

核心改变：backward 后不再 all_reduce 梯度，而是 **reduce_scatter**——每个 rank 只保留自己负责的那 1/N 梯度分片。

**流程**：
1. 各 rank 计算本地梯度
2. **reduce_scatter 梯度**：每个 rank 只拿到 1/N 的聚合梯度
3. 每个 rank 用局部梯度更新 1/N 的参数
4. all_gather 恢复完整参数

**手撕 backward_zero2**：

```python
from copy import deepcopy

def backward_zero2(model, loss, rank, world_size):
    '''
    ZeRO-2 的梯度切分：
    1. 先 all_reduce 梯度（保证正确性）
    2. 再 scatter 到各 rank（每个 rank 只保留 1/N）
    '''
    for param in model.parameters():
        if param.grad is not None:
            # all_reduce 求均值
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= world_size
            tmp_param = deepcopy(param.grad)

            # scatter：rank 0 切分后分发
            shared_size = param.grad.numel() // world_size
            param.grad.data = torch.zeros(shared_size)
            if rank == 0:
                grad_list = list(torch.split(tmp_param.view(-1), shared_size))
                dist.scatter(param.grad.data, grad_list, src=0)
            else:
                dist.scatter(param.grad.data, [], src=0)
```

**显存节省**：

| 组件 | ZeRO-1 | ZeRO-2 |
|------|--------|--------|
| 参数（fp16） | 2Ψ | 2Ψ |
| 梯度（fp16） | 2Ψ | **2Ψ / N** |
| 优化器状态 | 12Ψ / N | 12Ψ / N |
| **合计** | 4Ψ + 12Ψ/N | **2Ψ + 14Ψ/N** |

4 卡时：7Ψ → 5.5Ψ。

### 完整代码

```python
class MyAdamZeRO2:
    '''梯度已经被 backward_zero2 切分，直接在分片上更新'''
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, world_size=1, rank=0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)
        self.t = 0.0
        self.world_size = world_size
        self.rank = rank

        self.M, self.V = [], []
        for param in self.params:
            shared_size = param.data.numel() // self.world_size
            self.M.append(torch.zeros(shared_size, dtype=torch.float32))
            self.V.append(torch.zeros(shared_size, dtype=torch.float32))

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            shared_size = param.data.numel() // self.world_size

            # 梯度已经是切分后的，直接用
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            weight_data = param.data.view(-1)[
                self.rank * shared_size : (self.rank + 1) * shared_size]
            weight_data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))

            # all_gather 恢复完整参数
            gather_tensor = torch.zeros(param.data.numel(), dtype=param.data.dtype)
            dist.all_gather_into_tensor(gather_tensor, weight_data)
            param.data = gather_tensor.reshape(param.data.shape)
            dist.barrier()

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = torch.zeros_like(param.data)

def run_zero2(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    model = ToyModel()
    optimizer = MyAdamZeRO2(model.parameters(), lr=0.001,
                            world_size=world_size, rank=rank)
    loss_fn = nn.MSELoss()
    input, labels = torch.randn(128, 128), torch.randn(128, 10)

    for i in range(1000):
        outputs, _ = model(input)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        backward_zero2(model, loss, rank, world_size)
        optimizer.step()
        if rank == 0 and i % 10 == 0:
            print(loss)

    print(rank, model.w1.weight.data)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_zero2, args=("127.0.0.1", "12801", 4), nprocs=4)
```

---

## 7. ZeRO-3：参数 + 梯度 + 优化器全分片

### 原理

**ZeRO 的终极形态：连模型参数本身也切分**。

每个 rank 只存 1/N 的参数。前向/反向需要完整参数时，临时 all_gather 拿过来用，用完丢弃。

**与 Tensor Parallelism 的核心区别**：
- TP：每个 rank 算矩阵的一部分（按行/列切分），通信发生在每层的前向/反向中
- ZeRO-3：每个 rank 存参数的一部分（flatten 后均分），前向时临时 all_gather 完整参数再做完整矩阵乘法

**流程**：
1. 初始化后 `shared_model()`：scatter 参数到各 rank
2. **前向**：逐层 all_gather 参数 → 计算 → （可选）丢弃冗余参数
3. **反向**：梯度 reduce_scatter 到各 rank
4. **更新**：各 rank 在局部参数分片上直接更新，无需 all_gather

### 完整代码

```python
from copy import deepcopy

class ToyModel(nn.Module):
    def __init__(self, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.w1 = nn.Linear(128, 512, bias=False)
        self.w2 = nn.Linear(512, 10, bias=False)
        self.shape_list = [self.w1.weight.data.shape,
                           self.w2.weight.data.shape]

    def shared_model(self):
        '''将完整参数 scatter 到各 rank，每个 rank 只保留 1/N'''
        for param in self.parameters():
            shared_size = param.data.numel() // self.world_size
            weight_data = deepcopy(param.data).view(-1)
            param.data = torch.zeros(shared_size)
            if self.rank == 0:
                scatter_list = list(weight_data.split(shared_size, dim=0))
            else:
                scatter_list = None
            dist.scatter(param.data, scatter_list, src=0)

    def forward_zero3(self, x):
        '''
        前向时逐层 all_gather 恢复完整参数，计算后继续
        '''
        # Layer 1: all_gather → forward
        w1_gather = torch.zeros(self.shape_list[0])
        dist.all_gather_into_tensor(w1_gather.view(-1), self.w1.weight.data)
        self.w1.weight.data = w1_gather
        hidden = self.w1(x)

        # Layer 2: all_gather → forward
        w2_gather = torch.zeros(self.shape_list[1])
        dist.all_gather_into_tensor(w2_gather.view(-1), self.w2.weight.data)
        self.w2.weight.data = w2_gather
        output = self.w2(hidden)

        return output, hidden

def backward_zero3(model, loss, rank, world_size):
    '''
    ZeRO-3 反向：梯度 reduce → scatter → 恢复参数分片
    注意：反向时参数仍是完整的（前向 all_gather 过），
    反向完成后需要 shared_model() 恢复分片状态
    '''
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            dist.barrier()
            tmp_param = deepcopy(param.grad) / world_size

            shared_size = param.grad.numel() // world_size
            param.grad.data = torch.zeros(shared_size)
            if rank == 0:
                grad_list = list(torch.split(tmp_param.view(-1), shared_size))
            else:
                grad_list = []
            dist.scatter(param.grad.data, grad_list, src=0)

class MyAdamZeRO3:
    '''
    参数已经是分片的，M/V 直接在分片尺寸上建立。
    更新后无需 all_gather（下一个前向会做）。
    '''
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, world_size=1, rank=0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)
        self.t = 0.0
        self.world_size = world_size
        self.rank = rank

        self.M, self.V = [], []
        for param in self.params:
            self.M.append(torch.zeros(param.data.shape, dtype=torch.float32))
            self.V.append(torch.zeros(param.data.shape, dtype=torch.float32))

    def step(self):
        self.t += 1
        for param, M, V in zip(self.params, self.M, self.V):
            M = self.beta1 * M + (1 - self.beta1) * param.grad
            V = self.beta2 * V + (1 - self.beta2) * param.grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)
            param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            # ZeRO-3 不需要 all_gather 参数，下次前向时再 gather

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = None

def train_zero3(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    for i in range(epochs):
        optimizer.zero_grad()
        outputs, _ = model.forward_zero3(input)  # 分布式前向
        loss = loss_fn(outputs, labels)
        loss.backward()
        dist.barrier()
        model.shared_model()  # 恢复参数分片
        backward_zero3(model, loss, rank, world_size)
        optimizer.step()
        dist.barrier()
        if rank == 0 and i % 10 == 0:
            print(loss)

def run_zero3(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    model = ToyModel(world_size=world_size, rank=rank)
    model.shared_model()  # 初始化分片
    print(model.w1.weight.shape)  # [512/N, 128] → 分片后
    print(model.w2.weight.shape)  # [10/N, 512] → 分片后

    optimizer = MyAdamZeRO3(model.parameters(), lr=0.001,
                            world_size=world_size, rank=rank)
    loss_fn = nn.MSELoss()
    input, labels = torch.randn(128, 128), torch.randn(128, 10)

    train_zero3(rank, world_size, model, input, labels, loss_fn, optimizer, 1000)

    # 每张卡上的参数是不同的分片
    print(rank, model.w1.weight.data)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_zero3, args=("127.0.0.1", "12801", 4), nprocs=4)
```

---

## 8. 各阶段对比总结

### 显存对比（混合精度，N 个 rank）

| 阶段 | 参数 | 梯度 | 优化器状态 | 总计 (per rank) | 4卡时 |
|------|------|------|-----------|----------------|-------|
| **DDP** | 2Ψ | 2Ψ | 12Ψ | **16Ψ** | 16Ψ |
| **ZeRO-1** | 2Ψ | 2Ψ | 12Ψ/N | **4Ψ + 12Ψ/N** | 7Ψ |
| **ZeRO-2** | 2Ψ | 2Ψ/N | 12Ψ/N | **2Ψ + 14Ψ/N** | 5.5Ψ |
| **ZeRO-3** | 2Ψ/N | 2Ψ/N | 12Ψ/N | **16Ψ/N** | 4Ψ |

### 通信对比

| 阶段 | 通信操作 | 通信量 (per step) | 说明 |
|------|---------|-------------------|------|
| **DDP** | all_reduce (grad) | 2Ψ | baseline |
| **ZeRO-1** | all_reduce (grad) + all_gather (param) | 3Ψ | +50% 通信 |
| **ZeRO-2** | reduce_scatter (grad) + all_gather (param) | 2Ψ | ≈ DDP |
| **ZeRO-3** | all_gather (param, fwd) + reduce_scatter (grad) + all_gather (param, bwd) | 3Ψ | +50% 通信 |

### 关键 Tradeoff

| 维度 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| **显存节省** | 中 | 大 | 极大（线性缩放） |
| **通信开销** | +50% | ≈ DDP | +50% |
| **实现复杂度** | 低 | 中 | 高 |
| **推荐场景** | 常规大模型 | 显存紧张 | 超大模型（单卡放不下） |

### 核心洞察

1. **ZeRO 的本质**：用通信换显存。切分越多、显存越省、通信越贵。
2. **Adam 的 element-wise 特性**是 ZeRO 可行的理论基础——每个参数的更新独立。
3. **Flatten 而非矩阵切分**：ZeRO 将参数展平后按元素数均分，而非按矩阵维度切分（这是与 TP 的本质区别）。
4. **ZeRO-3 的前向开销**：每层都需要 all_gather 完整参数，这是 ZeRO-3 的主要瓶颈。DeepSpeed 通过 prefetching 和通信-计算重叠来缓解。
5. **ZeRO-2 是性价比最高的选择**：通信量与 DDP 相当，但显存节省显著。

---

## See Also

**分布式训练四维并行谱系**
- [[AI/LLM/Infra/分布式训练通信原语-手撕实操|分布式训练通信原语手撕]] — ZeRO 的底层：AllReduce/ReduceScatter/AllGather 的 ring 实现
- [[AI/LLM/Infra/Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]] — TP（列/行切分 Linear，通信模式与 ZeRO 正交）
- [[AI/LLM/Infra/Pipeline-Parallel-手撕实操|Pipeline-Parallel-手撕实操]] — PP（层间切分，与 ZeRO-3 可叠加）
- [[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]] — verl 工程实践层（ZeRO 作为 actor 后端）

**理论对应**
- [[训练后端|verl 训练后端]] — ZeRO（FSDP）在 verl 中的工程集成
- [[MA-RLHF-手撕实操-系列索引|MA-RLHF 手撕实操系列索引]] — ZeRO 在整个课程中的位置（lc9 分布式RL训练）

## 推荐阅读

1. **ZeRO 原始论文**：[arXiv:1910.02054](https://arxiv.org/abs/1910.02054) — Rajbhandari et al.，三级分析
2. **DeepSpeed 文档**：[deepspeed.ai/training](https://deepspeed.ai/training/) — ZeRO 配置实战
