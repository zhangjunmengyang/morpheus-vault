---
title: "Ray 推理系统实操"
brief: "基于Ray Actor的LLM推理服务完整实现：Generator Actor（vLLM封装）、负载均衡、流式输出、多实例并发，结合Ray Data的批量推理pipeline，是理解RL训练中rollout生成侧的核心设计，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, ray, inference-serving, vllm, rollout, training-infra]
related:
  - "[[AI/LLM/Infra/Ray-分布式RL训练实操|Ray-分布式RL训练实操]]"
  - "[[AI/LLM/Inference/vLLM-手撕实操|vLLM-手撕实操]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
---

# Ray 推理系统实操

> 来源：MA-RLHF xtrain (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 核心定位

Ray 是分布式 LLM 系统的基础设施层，掌握三个关键能力：
1. **数据 / Tensor 通信传输**
2. **PD 分离**（Prefill-Decode 分离）
3. **训推分离**（Training-Inference 分离）

本文从最简单的 Remote Function 出发，逐步构建到分布式 LLM 系统级通信。

---

## 目录

1. [Remote Function：远程函数调用](#1-remote-function远程函数调用)
2. [Actor 通信 v1：同步收发](#2-actor-通信-v1同步收发)
3. [Actor 通信 v2：异步接收](#3-actor-通信-v2异步接收)
4. [Actor 通信 v3：异步发送 + 同步接收](#4-actor-通信-v3异步发送--同步接收)
5. [Actor 通信 v4：多发送者](#5-actor-通信-v4多发送者)
6. [同步 vs 异步任务](#6-同步-vs-异步任务)
7. [AllReduce 操作](#7-allreduce-操作)
8. [Tensor Transfer：零拷贝传输](#8-tensor-transfer零拷贝传输)
9. [Ray + torch.distributed 集成](#9-ray--torchdistributed-集成)
10. [构建路径总结](#10-构建路径总结)

---

## 1. Remote Function：远程函数调用

### 原理

`@ray.remote` 将普通函数提交到远程 Worker 执行。数据通过 Ray 的对象存储（Object Store）在进程间传输。

- `.remote()` 返回 `ObjectRef`（future/句柄）
- `ray.get()` 阻塞获取结果
- 支持跨设备：CPU → GPU 的数据搬运

### 完整代码

```python
import ray
import torch

ray.init()

# device = 'cuda:0'
device = 'mps'  # Mac

@ray.remote
def process(data, device):
    '''将数据提交到远程设备上计算'''
    from_device = data.device
    data.to(device)
    return data.sum().to(from_device), data.max().to(from_device)

# 创建测试数据
A = torch.randn(10, 3, 4, device='cpu')

# 并行处理多个子任务
futures = [process.remote(A[i], device) for i in range(10)]
results = ray.get(futures)

for i, (sum_val, max_val) in enumerate(results):
    print(f"结果{i+1}: 总和={sum_val}, 最大={max_val}")
```

### 关键点

- `process.remote()` 是异步的，10 个调用立即返回
- `ray.get(futures)` 批量等待所有结果
- 这是 Ray 最基础的并行模式：**无状态函数并行**

---

## 2. Actor 通信 v1：同步收发

### 原理

Actor 是 Ray 的有状态计算单元。与 Remote Function 不同，Actor 保持持久状态，支持进程间通信。

**v1 实现**：发送方逐个同步发送，接收方逐个同步接收。

### 完整代码

```python
import ray
import time

ray.init()

@ray.remote
class SenderActor:
    def __init__(self):
        self.datas = [12, 32, 4, 8, 10]

    def send_to(self, receiver_actor, data):
        # 调用接收方的 remote 方法，返回句柄
        return receiver_actor.receive.remote(data)

    def get_datas(self):
        return self.datas

@ray.remote
class RecActor:
    def __init__(self):
        self.datas = []

    def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self):
        return self.datas

send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

datas = ray.get(send_actor.get_datas.remote())
print(datas)

for i in datas:
    future = ray.get(send_actor.send_to.remote(rec_actor, i))
    result = ray.get(future)
    print('receive actor len:', result)

    result = ray.get(rec_actor.get_datas.remote())
    print('receive actor datas:', result)
```

### 关键点

- `send_actor.send_to.remote(rec_actor, i)` 返回的是"句柄的句柄"（nested future）
- 需要两次 `ray.get()` 才能拿到最终结果
- **效率低**：每次发送都阻塞等待接收完成

---

## 3. Actor 通信 v2：异步接收

### 原理

接收方标记为 `async def receive`，使接收方能并发处理多个请求。但发送方仍然是批量发送后统一等待。

### 完整代码

```python
import ray

ray.init()

@ray.remote
class SenderActor:
    def __init__(self):
        self.datas = [1, 2, 3, 4, 5]

    def send_to(self, receiver_actor):
        # 批量异步发送，返回 handle list
        futures = []
        for i in self.datas:
            future = receiver_actor.receive.remote(i)
            futures.append(future)
        return futures

    def get_datas(self):
        return self.datas

@ray.remote
class RecActor:
    def __init__(self):
        self.datas = []

    async def receive(self, data):  # 异步接收
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self):
        return self.datas

print('---异步接收通信---')
send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

futures = ray.get(send_actor.send_to.remote(rec_actor))
result = ray.get(futures)  # 类似 barrier，阻塞直到所有接收完成
print('receive actor len:', result)

result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)
```

### 关键点

- `async def receive` 使接收方具备并发能力
- 发送方 `send_to` 一次性发出所有数据，不等待每个完成
- `ray.get(futures)` 作为 barrier，确保全部接收完毕

---

## 4. Actor 通信 v3：异步发送 + 同步接收

### 原理

发送方异步发送（不等待），接收方同步接收。关键问题：**异步发送时数据顺序会打乱吗？**

**答案：不会。** Ray Actor 的核心保证是 **同一 Actor 的方法调用串行执行**。即使发送是异步的，接收方按入队顺序逐个处理。

### 完整代码

```python
import ray

ray.init()

@ray.remote
class SenderActor:
    def __init__(self):
        self.datas = [1, 2, 3, 4, 5]

    def send_to(self, receiver_actor):
        futures = []
        for i in self.datas:
            future = receiver_actor.receive.remote(i)
            futures.append(future)
        return futures

    def get_datas(self):
        return self.datas

@ray.remote
class RecActor:
    def __init__(self):
        self.datas = []

    def receive(self, data):  # 同步接收
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self):
        return self.datas

send_actor = SenderActor.remote()
rec_actor = RecActor.remote()

print('---同步接收通信---')
futures = ray.get(send_actor.send_to.remote(rec_actor))
result = ray.get(futures)
print('receive actor len:', result)

result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)

# 为什么异步发送 + 同步接收不会打乱顺序？
# 1. Ray Actor 内部方法执行是串行的
# 2. 发送方按顺序调用 .remote()，接收方按入队顺序处理
# 3. 即使发送是非阻塞的，调用顺序是确定的
```

### 关键洞察

> **Actor 内部方法执行是串行的**——这是 Ray Actor 模型最重要的保证。即使有多个并发调用，RecActor 也会按顺序一个一个执行 `receive()`。

---

## 5. Actor 通信 v4：多发送者

### 原理

多个 SenderActor 同时向同一个 RecActor 发送数据。接收方内部仍然串行处理，但来自不同发送者的数据交错顺序不确定。

### 完整代码

```python
import ray

ray.init()

@ray.remote
class SenderActor:
    def __init__(self, datas):
        self.datas = datas

    async def send_to(self, receiver_actor):
        futures = []
        for i in self.datas:
            future = receiver_actor.receive.remote(i)
            futures.append(future)
        return futures

    def get_datas(self):
        return self.datas

@ray.remote
class RecActor:
    def __init__(self):
        self.datas = []

    def receive(self, data):
        self.datas.append(data)
        return len(self.datas)

    def get_datas(self):
        return self.datas

send_actor_1 = SenderActor.remote(list(range(1, 5)))
send_actor_2 = SenderActor.remote(list(range(15, 20)))
rec_actor = RecActor.remote()

print('---2 actor 异步发送---')
futures_1 = ray.get(send_actor_1.send_to.remote(rec_actor))
futures_2 = ray.get(send_actor_2.send_to.remote(rec_actor))

result = ray.get(futures_2 + futures_1)
print('2 send actor to recv actor, datalen:', result)

result = ray.get(rec_actor.get_datas.remote())
print('receive actor datas:', result)
# 注意：来自不同 sender 的数据交错顺序不确定
# 但来自同一 sender 的数据保持有序
```

### 关键点

- 同一 Sender 的数据在 Receiver 中保持有序
- 不同 Sender 之间的数据交错顺序不确定
- 这是分布式系统中的典型"因果序"保证

---

## 6. 同步 vs 异步任务

### 原理

理解 `ray.get(fn.remote())` 和 `fn.remote()` 的根本区别：

- `future = fn.remote()`：**异步**——立即返回 ObjectRef，任务在后台运行
- `result = ray.get(future)`：**同步阻塞**——等待任务完成
- `result = ray.get(fn.remote())`：等价于同步执行

### 完整代码

```python
import ray
import time
import random
import torch

random.seed(42)
torch.manual_seed(42)
ray.init()

@ray.remote
def fun_mul(x, y):
    for i in range(10):
        b = x @ y
        time.sleep(random.random())
        print(f'[MUL] step:{i}')
    return b

@ray.remote
def fun_add(x, y):
    for i in range(10):
        b = x + y
        time.sleep(random.random())
        print(f'\t\t[ADD] step:{i}')
    return b

A = torch.randn(2048, 2048)
B = torch.randn(2048, 2048)

# ===== 方式1：同步执行（串行）=====
print('--- example 1: 同步执行任务 ---')
result_mul = ray.get(fun_mul.remote(A, B))  # 阻塞等待
print('>>> Return mul shape:', result_mul.shape)

result_add = ray.get(fun_add.remote(A, B))  # 阻塞等待
print('>>> Return add shape:', result_add.shape)

# ===== 方式2：异步执行（并行）=====
print('--- example 2: 异步执行任务 ---')
future = fun_mul.remote(A, B)  # 不阻塞，立即返回

# fun_add 运行过程中，fun_mul 也在后台运行
result_add = ray.get(fun_add.remote(A, B))
print('>>> Return add shape:', result_add.shape)

result_mul = ray.get(future)  # 此时 mul 可能已经完成
print('>>> Return mul shape:', result_mul.shape)
```

### 关键洞察

> **`fn.remote()` 是 Ray 并行的核心**：它将任务提交到调度器后立即返回。只有在需要结果时才 `ray.get()` 同步。这是实现通信-计算重叠的基础。

---

## 7. AllReduce 操作

### 原理

用 Ray 实现分布式 AllReduce：
1. 各 "GPU" 的数据通过 `ray.put()` 放入 Object Store
2. 传递引用（ObjectRef）而非数据本身
3. 在目标设备上 `ray.get()` 取出数据并计算

### 完整代码

```python
import ray
import torch
import time

ray.init()

device = 'mps' if torch.mps.is_available() else 'cuda:0'

@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
def all_reduce_mean(refs):
    '''
    接收 ObjectRef 列表，在目标设备上做 reduce mean
    '''
    tensors = ray.get(refs)
    tensors = [tensor.to(device) for tensor in tensors]

    tensors_cat = torch.cat(tensors, dim=0)
    result = tensors_cat.mean(dim=0)
    return result

# 模拟 8 个 GPU 的数据
data_list = [torch.randn(1, 3, 4, device='cpu') for _ in range(8)]

# 存入 Object Store，获取引用
refs = [ray.put(data) for data in data_list]

# AllReduce mean
results = ray.get(all_reduce_mean.remote(refs)).to('cpu')
print(results)

time.sleep(3)
```

### 关键点

- `ray.put(data)` 将数据存入共享 Object Store，返回引用
- 传递引用而非数据：**零拷贝**（同节点）或高效序列化（跨节点）
- `@ray.remote(num_gpus=0.5)` 声明 GPU 资源需求

---

## 8. Tensor Transfer：零拷贝传输

### 原理

Ray 的 Object Store 基于共享内存（Apache Arrow）。同一节点上的 tensor 通过 `ray.put()` 存入后，其他进程 `ray.get()` 时是**零拷贝**——直接读取共享内存中的数据。

### 完整代码

```python
import ray
import torch
import time

ray.init()

# GPU tensor
tensor_gpu = torch.zeros(4, 4, device='mps')  # 或 'cuda:0'
tensor_cpu = torch.zeros(2, 2, device='cpu')

# 存入共享内存
tensor_ref = ray.put(tensor_gpu)
tensor_ref_cpu = ray.put(tensor_cpu)

# 零拷贝获取
result = ray.get(tensor_ref)
print(result.device)
print("通过对象引用传输的张量:", result)

time.sleep(2)

# 查看共享对象信息
print(tensor_ref)
print(tensor_ref_cpu)
result = ray._private.internal_api.memory_summary()
print(result)
```

### 关键点

- 同节点：共享内存 → 零拷贝
- 跨节点：自动序列化传输
- GPU tensor 会先拷贝到 CPU 共享内存，`ray.get()` 后在 CPU 上（需手动 `.to(device)`）
- 这是 Ray 大规模 tensor 传输的基础机制

---

## 9. Ray + torch.distributed 集成

### 原理

Ray 负责资源管理和进程编排，`torch.distributed` 负责高性能通信。两者结合：
1. Ray 创建 Worker Actor → 分配资源
2. 每个 Worker 内部初始化 `torch.distributed` 进程组
3. Worker 之间通过 GLOO/NCCL 高速通信

这是 **vLLM、OpenRLHF 等系统的底层架构**。

### 完整代码

```python
import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time

def init_distributed_gloo():
    '''Ray 编排 + torch.distributed 通信'''

    ray.init(_node_ip_address="0.0.0.0")

    # 创建 Placement Group：4 个 Worker，每个 1 CPU
    pg = placement_group([{"CPU": 1}] * 4)
    ray.get(pg.ready())

    @ray.remote(
        num_cpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True
        )
    )
    class Worker:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size

        def init_process_group(self, master_addr, master_port):
            import os
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['RANK'] = str(self.rank)
            os.environ['LOCAL_RANK'] = str(self.rank)

            # 在 Worker 内部初始化 torch.distributed
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )

            print(f"Worker {self.rank}: {dist.get_rank()}/{dist.get_world_size()}")

            # 验证通信
            A = torch.randn(100)
            dist.all_reduce(A)
            return True

    # 创建 Workers
    world_size = 4
    master_addr = ray.util.get_node_ip_address()
    master_port = "29500"

    workers = [Worker.remote(i, world_size) for i in range(world_size)]

    # 并行初始化所有 Worker 的通信
    results = ray.get([
        w.init_process_group.remote(master_addr, master_port)
        for w in workers
    ])

    time.sleep(5)
    print(f"All workers initialized: {results}")

if __name__ == "__main__":
    init_distributed_gloo()
```

### 关键点

- **PlacementGroup**：确保 Worker 的资源分配满足约束（同机/跨机）
- **Worker Actor 内部初始化 `dist.init_process_group`**：每个 Worker 独立初始化
- **master_addr/port 通过 Ray 获取**：`ray.util.get_node_ip_address()` 自动发现
- 这个模式就是 OpenRLHF 和 vLLM 的骨架

---

## 10. 构建路径总结

从简单到分布式 LLM 系统的演进路径：

```
Remote Function（无状态并行）
    ↓
Actor Model（有状态计算单元）
    ↓
Actor 通信（同步→异步→多发送者）
    ↓
Object Store + Tensor Transfer（零拷贝数据传输）
    ↓
Ray AllReduce（分布式聚合操作）
    ↓
Ray + torch.distributed（高性能通信集成）
    ↓
分布式 LLM 系统（训推分离、PD 分离）
```

### 每一层解决的问题

| 层级 | 解决的问题 | 对应 LLM 场景 |
|------|-----------|---------------|
| Remote Function | 数据并行处理 | 批量推理预处理 |
| Actor Model | 有状态服务 | 模型服务（KV Cache 管理） |
| Actor 通信 | 进程间数据传递 | Prefill → Decode 传递 |
| Tensor Transfer | 高效 tensor 传输 | 大模型权重分发 |
| AllReduce | 梯度/参数聚合 | 分布式训练 |
| Ray + torch.distributed | 高性能集合通信 | RLHF 训练系统 |

### Ray Actor 通信演进对比

| 版本 | 发送方式 | 接收方式 | 特点 |
|------|---------|---------|------|
| v1 | 逐个同步 | 同步 | 最简单，效率最低 |
| v2 | 批量异步 | 异步 | 接收方并发能力 |
| v3 | 批量异步 | 同步 | Actor 串行保证顺序 |
| v4 | 多 sender 异步 | 同步 | 多源数据交错 |

### 核心设计原则

1. **Actor 内部串行**：同一 Actor 的方法调用按序执行 → 天然保证数据一致性
2. **`.remote()` 是异步的**：提交任务后立即返回 → 通信-计算重叠的基础
3. **Object Store 是数据总线**：`ray.put()` / `ray.get()` → 同节点零拷贝
4. **Ray 管编排，torch.distributed 管通信**：各司其职 → 生产级 LLM 系统的标准架构
