---
title: "分布式训练通信原语手撕实操"
brief: "NCCL核心通信原语完整实现与可视化：broadcast/reduce/all-reduce（Ring-AllReduce）/gather/scatter/all-gather/reduce-scatter，Ring-AllReduce传输量公式推导，各并行范式通信选型分析，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, nccl, distributed-training, all-reduce, ring-allreduce, communication]
related:
  - "[[AI/3-LLM/Infra/Tensor-Parallel-手撕实操|Tensor-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操|ZeRO-手撕实操]]"
  - "[[AI/3-LLM/Infra/Pipeline-Parallel-手撕实操|Pipeline-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Infra/MoE-Context-Parallel-手撕实操|MoE-Context-Parallel-手撕实操]]"
---

# 分布式训练通信原语手撕实操

> 来源：MA-RLHF xtrain (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

1. [环境初始化](#1-环境初始化)
2. [P2P 同步通信](#2-p2p-同步通信)
3. [P2P 异步通信](#3-p2p-异步通信)
4. [Broadcast 广播](#4-broadcast-广播)
5. [Reduce 规约](#5-reduce-规约)
6. [Gather 收集](#6-gather-收集)
7. [ReduceScatter 规约分散](#7-reducescatter-规约分散)
8. [Ring-AllReduce 环形全规约](#8-ring-allreduce-环形全规约)
9. [AllToAll 全交换](#9-alltoall-全交换)
10. [DeviceMesh 设备网格](#10-devicemesh-设备网格)
11. [torchrun 启动](#11-torchrun-启动)

---

## 1. 环境初始化

### 原理

分布式训练的第一步：建立进程组。`torch.distributed` 支持三种后端：
- **gloo**：CPU 通信，跨平台，调试友好
- **nccl**：NVIDIA GPU 专用，性能最优
- **mpi**：传统 HPC 后端

初始化方式：
- `env://`：从环境变量读取 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`
- `tcp://host:port`：直接指定 TCP 地址
- `init_device_mesh`：支持多维设备拓扑（如 DP×TP 的 2D mesh）

### 完整代码

```python
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12801 RANK=0 WORLD_SIZE=1 python initial.py

import torch
import torch.distributed as dist

def main():
    '''环境变量方式初始化'''
    print('torch distributed is available:', dist.is_available())
    print('gloo available:', dist.is_gloo_available())
    print('nccl available:', dist.is_nccl_available())

    dist.init_process_group(backend="gloo",
                            init_method='env://',
                            world_size=1,
                            rank=0,
                            group_name='basic_distributed_env')

    print('is_initialized:', dist.is_initialized())
    dist.destroy_process_group()

def init_tcp():
    '''TCP 方式初始化'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:12801',
                            rank=0, world_size=1)
    print('is_initialized:', dist.is_initialized())
    dist.destroy_process_group()

def init_device_mesh():
    '''DeviceMesh 方式 — 支持 2D 拓扑'''
    # 1D mesh
    dist.init_device_mesh(device_type='cpu', mesh_shape=(1,1))
    dist.destroy_process_group()

    # 2D mesh（DP × TP）
    dist.init_device_mesh(device_type='cpu',
                          mesh_shape=(1,1),
                          mesh_dim_names=("dp", "tp"))
    dist.destroy_process_group()

def get_env():
    '''查看分布式环境信息'''
    dist.init_process_group(backend="gloo", init_method='env://',
                            world_size=1, rank=0)
    print('backend:', dist.get_backend())
    print('rank:', dist.get_rank())
    print('world_size:', dist.get_world_size())
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    init_tcp()
    get_env()
    init_device_mesh()
```

### 何时使用

**所有分布式训练的起点**。每个进程必须先 `init_process_group`，才能使用任何通信原语。

---

## 2. P2P 同步通信

### 原理

点对点（Point-to-Point）是最基础的通信原语：
- `dist.send(tensor, dst)`：阻塞式发送，直到接收方 recv 完成才返回
- `dist.recv(tensor, src)`：阻塞式接收，in-place 写入 tensor

**所有集合通信都可以用 P2P 手撕实现**——这是理解分布式通信的基石。

### 完整代码

```python
# python p2p.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''同步 P2P 通信：rank 0 → rank 1'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)  # 阻塞发送
        tensor -= 1
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)  # 阻塞接收

    print('Rank', rank, 'has data', tensor[0])
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- 需要精确控制哪个 rank 向哪个 rank 传数据
- 实现自定义通信拓扑（如 pipeline parallelism 中相邻 stage 通信）
- 手撕集合通信操作的基础构件

---

## 3. P2P 异步通信

### 原理

异步通信用 `isend` / `irecv`，立即返回一个 `Request` 对象：
- 发送方不必等接收方完成即可继续发送
- 通过 `req.wait()` 显式同步
- **关键**：`dist.barrier()` 防止某个 rank 先完成后 destroy 进程组导致其他 rank hang

### 完整代码

```python
# python p2p_async.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    异步通信：rank 0 连续发送 10 个数据给 rank 2
    发送方不必等待接收方完成，通过 buffer 收集所有 request 最后统一 wait
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://' + master_addr + ':' + master_port,
                            rank=rank, world_size=world_size)

    tensor = torch.zeros(1)
    buffer = []

    if rank == 0:   # 发送方：连续异步发送
        for i in range(10):
            tensor = torch.ones(1) * i
            req = dist.isend(tensor=tensor, dst=2)
            buffer.append(req)
    elif rank == 2: # 接收方：逐个异步接收
        for i in range(10):
            tmp_tensor = torch.zeros(1)
            req = dist.irecv(tensor=tmp_tensor, src=0)
            req.wait()  # 等待单个接收完成
            tensor += tmp_tensor

    if rank == 0:
        for req in buffer:
            req.wait()  # 确保所有发送完成

    dist.barrier()  # 防止先完成的 rank destroy 导致其他 rank hang
    print('Rank', rank, 'has data', tensor[0])
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **通信-计算重叠**：发出数据后继续计算，最后再同步
- 批量发送场景，避免逐个阻塞
- Pipeline parallelism 中异步发送 activation

---

## 4. Broadcast 广播

### 原理

将 src rank 的数据复制到组内所有 rank。所有 rank 调用同一函数，src rank 提供数据，其他 rank 接收。

**数据流**：`src → all ranks`

### API 版本

```python
# python fun_broadcast.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''API 版 broadcast'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    # 广播 object list
    if dist.get_rank() == 0:
        objects = ["foo", 12, {1: 2}]
    else:
        objects = [None, None, None]
    dist.broadcast_object_list(objects, src=0)
    print('Rank', rank, 'has data', objects)

    time.sleep(1)

    # 广播 tensor
    tensor = torch.zeros(2)
    if dist.get_rank() == 0:
        tensor = tensor + 100
    dist.broadcast(tensor, src=0)
    print('Rank', rank, 'has data', tensor)

    time.sleep(1)

    # 组内广播（子 group）
    tensor = torch.ones(2) * rank * 2
    group = dist.new_group(ranks=[0, 1])
    dist.broadcast(tensor, src=1, group=group)
    print('Rank', rank, 'has data', tensor)

    dist.destroy_process_group()
```

### P2P 手撕版本

```python
def p2p_broadcast(rank, master_addr, master_port, world_size, backend='gloo'):
    '''手撕 broadcast：rank 0 逐个 send 给其他 rank'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)
    group = dist.new_group(ranks=[0, 1, 2, 3])

    tensor = torch.zeros(2)
    if dist.get_rank() == 0:
        tensor = tensor + 100
        ranks = dist.get_process_group_ranks(group)
        for r in ranks:
            if r != 0:
                dist.send(tensor=tensor, dst=r)
    else:
        dist.recv(tensor=tensor, src=0)

    print('Rank', rank, 'has data', tensor)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
    mp.spawn(p2p_broadcast, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- 分发初始模型参数到所有 rank
- 分发超参数、配置、随机种子
- ZeRO-3 前向计算中 all-gather 完整参数后的分发

---

## 5. Reduce 规约

### 原理

将所有 rank 的数据按指定操作（SUM、MAX 等）合并：
- `all_reduce`：结果存在**所有 rank** 上（最常用）
- `reduce`：结果仅存在 **dst rank** 上

**数据流**：
- all_reduce: `all ranks → op → all ranks`
- reduce: `all ranks → op → dst rank`

### API 版本

```python
# python fun_reduce.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    # all_reduce SUM：所有 rank 结果一致
    tensor = torch.ones(1) * 2 * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank', rank, 'all_reduce SUM:', tensor)

    # all_reduce MAX
    tensor = torch.ones(1) * 2 * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    print('Rank', rank, 'all_reduce MAX:', tensor)

    time.sleep(1)

    # reduce：结果仅在 dst=0
    tensor = torch.ones(1) * 2 * rank
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print('Rank', rank, 'reduce to rank 0:', tensor)

    dist.destroy_process_group()
```

### P2P 手撕版本

```python
def p2p_reduce(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    手撕 all_reduce：
    1. rank 0 作为参数服务器，收集其他 GPU 数据
    2. 在 rank 0 上计算 SUM
    3. broadcast 结果给所有 rank
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)
    group = dist.new_group(ranks=[0, 1, 2, 3])

    tensor = torch.ones(2) * rank * 2
    tensor_sum = torch.ones(2) * rank * 2
    time.sleep(1)

    # Step 1: rank 0 收集并求和
    if dist.get_rank() == 0:
        ranks = dist.get_process_group_ranks(group)
        tmp = torch.zeros(2)
        for r in ranks:
            if r != 0:
                dist.recv(tensor=tmp, src=r)
                tensor_sum += tmp
    elif rank != 0:
        dist.send(tensor=tensor, dst=0)

    # Step 2: 广播结果
    dist.broadcast(tensor_sum, src=0)
    print('Rank', rank, 'has data', tensor_sum)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
    mp.spawn(p2p_reduce, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **DDP 梯度同步**：所有 rank 的梯度 all_reduce 求平均
- 分布式 metric 聚合（loss、accuracy 求均值）
- 参数服务器架构

---

## 6. Gather 收集

### 原理

将所有 rank 的数据收集在一起：
- `all_gather`：所有 rank 都拿到完整数据
- `all_gather_into_tensor`：收集后 concat 成单个 tensor

**数据流**：`all ranks → concatenate → all ranks`

### API 版本

```python
# python fun_gather.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    # all_gather → list of tensors
    tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    dist.all_gather(tensor_list=tensor_list, tensor=tensor)
    print('Rank', rank, 'all_gather:', tensor_list)

    # all_gather_into_tensor → 单个大 tensor
    tensor_trg = torch.zeros(2 * world_size, dtype=torch.int64)
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    dist.all_gather_into_tensor(output_tensor=tensor_trg, input_tensor=tensor)
    print('Rank', rank, 'all_gather_into_tensor:', tensor_trg)

    time.sleep(1)

    # all_gather_object → 任意 picklable 对象
    gather_objects = ["foo", 12, {1: 2}, 'xiaodongguaAIGC']
    output = [None for _ in gather_objects]
    dist.all_gather_object(output, gather_objects[rank])
    print(f'rank:{rank}', output)

    dist.destroy_process_group()
```

### P2P 手撕版本

```python
def p2p_gather(rank, master_addr, master_port, world_size, backend='gloo'):
    '''手撕 all_gather_into_tensor'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)
    group = dist.new_group(ranks=[0, 1, 2, 3])

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    tensor_trg = torch.arange(2 * world_size)
    tensor_list = [torch.arange(2, dtype=torch.int64) for _ in range(world_size)]

    # Step 1: rank 0 收集各 rank 数据
    if rank == 0:
        tensor_list[0] = tensor
        ranks = dist.get_process_group_ranks(group)
        for r in ranks:
            if r != 0:
                dist.recv(tensor_list[r], src=r)
    else:
        dist.send(tensor, dst=0)

    # Step 2: rank 0 合并
    if rank == 0:
        tensor_trg = torch.concat(tensor_list, dim=0)

    # Step 3: broadcast 完整结果
    dist.broadcast(tensor_trg, src=0)
    print(f'rank:{rank} all-gather result', tensor_trg)

    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
    mp.spawn(p2p_gather, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **ZeRO-1/2 参数同步**：各 rank 更新局部参数后 all_gather 恢复完整参数
- **ZeRO-3 前向**：前向前 all_gather 层参数
- Sequence parallelism 中收集完整 sequence
- 评估时收集所有 rank 的预测结果

---

## 7. ReduceScatter 规约分散

### 原理

先 Reduce（聚合），再 Scatter（切分分发）。每个 rank 只拿到结果的一个分片。

**数据流**：`all ranks → reduce → split → rank_i gets chunk_i`

**核心应用**：ZeRO-2 的梯度分片——先 all_reduce 梯度，再把不同分片分给不同 rank。

### API 版本

```python
# python fun_reduce_scatter.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    reduce_scatter 适用于 ZeRO-2：
    收集所有梯度 → reduce → 切分 → 各 GPU 维护部分梯度/优化器参数
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    # 基础 reduce_scatter
    tensor_out = torch.zeros(2, dtype=torch.int64)
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64)
    dist.reduce_scatter_tensor(tensor_out, tensor_in, op=dist.ReduceOp.SUM)
    print('Rank', rank, 'reduce_scatter:', tensor_out)

    time.sleep(1)

    # stack 形式
    tensor_in = torch.reshape(
        torch.arange(world_size * 2, dtype=torch.int64), (world_size, 2))
    tensor_out = torch.zeros(1, 2, dtype=torch.int64)
    dist.reduce_scatter_tensor(tensor_out, tensor_in)
    print('Rank', rank, 'reduce_scatter (stack):', tensor_out)

    dist.destroy_process_group()
```

### P2P 手撕版本

```python
def p2p_reduce_scatter(rank, master_addr, master_port, world_size, backend='gloo'):
    '''手撕 reduce_scatter：先 reduce 到 rank 0，再 scatter 各分片'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)
    group = dist.new_group(ranks=[0, 1, 2, 3])

    tensor_out = torch.zeros(2, dtype=torch.int64)
    tensor_in = torch.arange(world_size * 2, dtype=torch.int64)
    tensor_reduce = torch.arange(world_size * 2, dtype=torch.int64)

    # Step 1: rank 0 收集并 reduce
    if rank == 0:
        tensor_tmp = torch.zeros(world_size * 2, dtype=torch.int64)
        for r in dist.get_process_group_ranks(group):
            if r != 0:
                dist.recv(tensor_tmp, src=r)
                tensor_reduce += tensor_tmp
    else:
        dist.send(tensor_in, dst=0)

    time.sleep(1)

    # Step 2: rank 0 切分并 scatter
    if rank == 0:
        scatter_list = list(tensor_reduce.split(split_size=2, dim=0))
        for r in dist.get_process_group_ranks(group):
            if r != 0:
                dist.send(scatter_list[r], dst=r)
            else:
                tensor_out = scatter_list[0]
    else:
        dist.recv(tensor_out, src=0)

    print('Rank', rank, 'p2p reduce_scatter:', tensor_out)
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
    mp.spawn(p2p_reduce_scatter, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **ZeRO-2 梯度切分**：backward 后 reduce_scatter 梯度到各 rank
- Tensor parallelism 的列并行输出聚合
- 任何"聚合后分片"的场景

---

## 8. Ring-AllReduce 环形全规约

### 原理

**Ring-AllReduce 是分布式训练的核心算法**，分两个阶段：

1. **ReduceScatter 阶段**：每个 rank 将数据切成 N 份，沿环形拓扑传递并累加，N-1 轮后每个 rank 持有一个完整的 reduce 结果分片
2. **AllGather 阶段**：再沿环传 N-1 轮，每个 rank 拿到所有分片

**通信量**：每个 rank 发送 `2(N-1)/N × data_size`，接近理论最优。

**死锁处理**：偶数 rank 先发后收，奇数 rank 先收后发（或 rank 0 特殊处理）。

### 完整代码

```python
# python fun_ring_allreduce.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, master_addr, master_port, world_size, backend='gloo'):
    '''
    Ring AllReduce = ReduceScatter + AllGather
    注意：偶数 rank 先 send 再 recv，防止死锁
    '''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    batch = torch.zeros(2 * world_size, dtype=torch.int64) + 1
    micro_batch = list(torch.split(batch, 2, dim=0))
    tmp_tensor = torch.zeros(2, dtype=torch.int64)

    # ======== Stage 1: ReduceScatter ========
    for i in range(world_size - 1):
        cur_idx = (rank - i) % world_size
        next_idx = (rank - i - 1) % world_size
        if rank % world_size == 0:
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
        micro_batch[next_idx] += tmp_tensor

    dist.barrier()
    if rank == 0:
        print('Stage 1 (ReduceScatter) complete')
    print('Rank', rank, 'after ReduceScatter:', micro_batch)
    dist.barrier()

    # ======== Stage 2: AllGather ========
    for i in range(world_size - 1):
        cur_idx = (i + rank + 1) % world_size
        next_idx = (cur_idx + 1) % world_size
        if rank % world_size == 0:
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
        micro_batch[next_idx] = tmp_tensor

    dist.barrier()
    if rank == 0:
        print('Stage 2 (AllGather) complete')
    print('Rank', rank, 'after AllGather:', micro_batch)
    dist.barrier()

    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **DDP 梯度同步的底层实现**：NCCL 后端的 all_reduce 就是 Ring-AllReduce
- 理解大规模训练通信瓶颈的关键
- 大模型训练中通信量分析的基础

---

## 9. AllToAll 全交换

### 原理

每个 rank 向每个 rank 发送不同的数据分片。第 i 个 rank 发给第 j 个 rank 的数据 = 原数据的第 j 个分片。

**数据流**：rank_i 的 chunk_j → rank_j（转置关系）

支持两种形式：
- `all_to_all_single`：输入单个 tensor，按 split 切分收发
- `all_to_all`：输入 tensor list，逐个交换

### 完整代码

```python
# python fun_all2all.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_single(rank, master_addr, master_port, world_size, backend='gloo'):
    '''all_to_all_single：输入单个 tensor'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    input = torch.arange(4) + rank * 4
    output = torch.empty([4], dtype=torch.int64)
    dist.all_to_all_single(output, input)
    print('Rank', rank, 'input:', input, '→ output:', output)
    dist.destroy_process_group()

def run_list(rank, master_addr, master_port, world_size, backend='gloo'):
    '''all_to_all：输入 tensor list'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    input = torch.arange(4) + rank * 4
    input = list(input.chunk(4))
    output = list(torch.empty([4], dtype=torch.int64).chunk(4))
    dist.all_to_all(output, input)
    print('Rank', rank, 'output:', output)
    dist.destroy_process_group()

def run_variable_size(rank, master_addr, master_port, world_size, backend='gloo'):
    '''支持不等长分片的 all_to_all_single'''
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    input = [torch.tensor([0,1,2,3,4,5], dtype=torch.int64),
             torch.tensor([10,11,12,13,14,15,16,17,18], dtype=torch.int64),
             torch.tensor([20,21,22,23,24], dtype=torch.int64),
             torch.tensor([30,31,32,33,34,35,36], dtype=torch.int64)]

    input_splits = torch.tensor([[2,2,1,1],
                                 [3,2,2,2],
                                 [2,1,1,1],
                                 [2,2,2,1]], dtype=torch.int64)
    output_splits = input_splits.t()
    output = [torch.zeros(torch.sum(output_splits[i,:]), dtype=torch.int64)
              for i in range(world_size)]

    dist.all_to_all_single(output=output[rank], input=input[rank],
                           output_split_sizes=output_splits[rank,:].tolist(),
                           input_split_sizes=input_splits[rank,:].tolist())
    print('Rank', rank, 'variable output:', output[rank])
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(run_single, args=("127.0.0.1", "12801", 4), nprocs=4)
    mp.spawn(run_variable_size, args=("127.0.0.1", "12801", 4), nprocs=4)
```

### 何时使用

- **MoE（Mixture of Experts）**：token 按 expert 路由，all_to_all 将 token 发送到对应 expert 所在的 rank
- Expert parallelism 中的 token dispatch / combine
- Sequence parallelism 中的序列重排

---

## 10. DeviceMesh 设备网格

### 原理

`DeviceMesh` 提供多维设备拓扑抽象，自动创建子通信组：
- 1D mesh：等价于普通 process_group
- 2D mesh（如 `dp × tp`）：自动在 DP 维度和 TP 维度各创建通信组

### 完整代码

```python
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12801 RANK=0 WORLD_SIZE=6 python device_mesh.py

import torch
import torch.distributed as dist
import os

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def device_map():
    '''创建 2D mesh（2×3 = 6 devices）'''
    dist.device_mesh.init_device_mesh(
        device_type='cpu',
        mesh_shape=(2, 3)
    )
    print('is_initialized:', dist.is_initialized())
    print('current rank:', dist.get_rank())
    dist.destroy_process_group()

if __name__ == '__main__':
    master_port = find_free_port()
    os.environ['MASTER_PORT'] = master_port
    device_map()
```

### 何时使用

- **混合并行**：同时使用 DP + TP（或 DP + TP + PP）时，用 DeviceMesh 管理多维通信组
- FSDP 2D sharding
- 任何需要在不同维度做不同通信的场景

---

## 11. torchrun 启动

### 原理

`torchrun` 是 PyTorch 推荐的分布式启动器，自动设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量。

### 完整代码

```python
# torchrun --standalone --nnodes=1 --nproc-per-node="cpu" torchrun.py

import os

local_rank = int(os.environ["LOCAL_RANK"])
print(f'LOCAL_RANK: {local_rank}')

if local_rank == 0:
    print('world_size:', os.environ["WORLD_SIZE"])
```

启动命令：

```bash
# 单机多进程
torchrun --standalone --nnodes=1 --nproc-per-node=4 torchrun.py

# 多机（node 0）
torchrun --standalone --master-addr="127.0.0.1" --node-rank=0 --nnodes=2 --nproc-per-node=4 torchrun.py &

# 多机（node 1）
torchrun --standalone --master-addr="127.0.0.1" --node-rank=1 --nnodes=2 --nproc-per-node=4 torchrun.py &
```

### 何时使用

- 生产环境启动分布式训练的标准方式
- 替代手动 `mp.spawn`，支持弹性训练和故障恢复

---

## 通信原语速查表

| 原语 | 数据流 | 通信量 (per rank) | 典型应用 |
|------|--------|-------------------|----------|
| **Send/Recv** | 1 → 1 | O(data) | Pipeline parallelism |
| **Broadcast** | 1 → N | O(data) | 参数初始化 |
| **Reduce** | N → 1 (op) | O(data) | 聚合到 master |
| **AllReduce** | N → N (op) | O(data) | DDP 梯度同步 |
| **Gather** | N → 1 (concat) | O(N × data) at dst | 收集评估结果 |
| **AllGather** | N → N (concat) | O(N × data) | ZeRO 参数恢复 |
| **Scatter** | 1 → N (split) | O(data) | 数据分发 |
| **ReduceScatter** | N → N (op + split) | O(data) | ZeRO-2 梯度切分 |
| **AllToAll** | N → N (transpose) | O(data) | MoE token routing |
| **Ring-AllReduce** | N → N (op) | 2(N-1)/N × data | NCCL 底层实现 |
