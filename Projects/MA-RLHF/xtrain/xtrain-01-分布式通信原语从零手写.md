---
title: xtrain lc1 — 分布式通信原语从零手写
brief: 从零实现 NCCL 8大通信原语（Broadcast/Reduce/AllReduce/AllGather/ReduceScatter/Scatter/Gather/AllToAll）及 Ring-AllReduce。理解 GPU 集群通信的底层机制，面试分布式训练必备。
date: 2026-02-25
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - distributed-training
  - nccl
  - communication-primitives
  - xtrain
related:
  - "[[Projects/MA-RLHF/lc-comm/lc-comm-01-分布式训练通信原语-手撕实操]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-02-数据并行从零手写]]"
  - "[[AI/3-LLM/Infra/分布式训练]]"
  - "[[Projects/MA-RLHF/MA-RLHF-手撕实操-系列索引]]"
---

# xtrain lc1 — 分布式通信原语从零手写

> 来源：MA-RLHF xtrain lecture/lc1_basic/（21个py文件）
> 核心内容：PyTorch `torch.distributed` 从环境初始化到8大集合通信原语，全部手写实现

---

## 1. 分布式训练环境初始化

### 1.1 init_process_group — 基础初始化

每个分布式程序的起点。创建一个 **ProcessGroup**，所有参与的进程（rank）通过它协调通信。

```python
import torch.distributed as dist

# 方式一：环境变量（需提前设置 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE）
dist.init_process_group(
    backend="gloo",          # gloo（CPU）/ nccl（GPU）/ mpi
    init_method='env://',    # 默认值，从环境变量读取
    world_size=1,
    rank=0,
    group_name='basic_distributed_env',
)

# 方式二：TCP 显式指定（适合手动多进程调试）
dist.init_process_group(
    backend='gloo',
    init_method='tcp://127.0.0.1:12801',  # master 地址
    rank=rank,
    world_size=world_size
)

# 常用查询
dist.is_initialized()    # 是否已初始化
dist.get_backend()       # 当前后端
dist.get_rank()          # 当前进程号
dist.get_world_size()    # 总进程数

# 销毁
dist.destroy_process_group()
```

**后端选择**：
- `gloo`：CPU 通信，开发调试首选，所有平台可用
- `nccl`：GPU 通信，NVIDIA 专用，生产环境标配
- `mpi`：需要独立安装，用得较少

### 1.2 ProcessGroup — 进程分组

可以在全局 world 中创建子组，不同子组独立通信。这是实现混合并行（如 DP 组 + TP 组）的基础。

```python
dist.init_process_group(backend='gloo',
                        init_method='tcp://127.0.0.1:12803',
                        rank=rank, world_size=6)

# 创建两个子组：ranks [0,1,2] 和 [3,4,5]
group_0 = dist.new_group(ranks=[0, 1, 2], backend='gloo')
group_1 = dist.new_group(ranks=[3, 4, 5], backend='gloo')

# 查询映射关系
dist.get_group_rank(group=group_0, global_rank=2)  # → 2（组内rank）
dist.get_global_rank(group=group_0, group_rank=0)  # → 0（全局rank）
dist.get_process_group_ranks(group_0)               # → [0, 1, 2]
```

> **注意**：`new_group` 是**集合操作**，所有 rank 都必须调用，即使某些 rank 不属于该组。

### 1.3 DeviceMesh — 多维设备拓扑

`DeviceMesh` 是对 ProcessGroup 的更高层抽象，用于描述 2D/3D 并行拓扑。内部自动调用 `init_process_group`。

```python
# 1D mesh：等价于普通 init_process_group
dist.init_device_mesh(device_type='cpu', mesh_shape=(1, 1))

# 2D mesh：定义 DP × TP 拓扑
dist.init_device_mesh(
    device_type='cpu',
    mesh_shape=(2, 3),          # 2个DP组 × 3个TP组
    mesh_dim_names=("dp", "tp") # 命名维度
)

# 手动构造（指定设备映射矩阵）
dist.device_mesh.DeviceMesh(
    device_type='cpu',
    mesh=[[0, 1], [2, 3]]  # 2×2 拓扑
)
```

### 1.4 多进程启动方式

```python
import torch.multiprocessing as mp

# 方式一：mp.spawn（开发调试用）
mp.spawn(run_fn, args=("127.0.0.1", "12801", 4), nprocs=4)

# 方式二：torchrun（生产环境）
# torchrun --nproc_per_node=4 train.py
```

---

## 2. P2P 点对点通信

### 2.1 同步 P2P：send / recv

阻塞式通信：`send` 阻塞直到数据被对方 `recv`，`recv` 阻塞直到收到数据。

```python
def run(rank, master_addr, master_port, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)  # 阻塞发送到 rank 1
        tensor -= 1                       # 发送完成后才执行
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)   # 阻塞接收来自 rank 0

    print(f'Rank {rank} has data {tensor[0]}')
    # Rank 0 → 0.0（发送后减了1）
    # Rank 1 → 1.0（收到了1）
    dist.destroy_process_group()
```

### 2.2 异步 P2P：isend / irecv

非阻塞通信：立即返回一个 `Work` 对象（请求句柄），后续通过 `req.wait()` 同步。

```python
def run(rank, master_addr, master_port, world_size):
    dist.init_process_group(...)

    tensor = torch.zeros(1)
    buffer = []

    if rank == 0:  # 发送者：连续发送10个tensor
        for i in range(10):
            tensor = torch.ones(1) * i
            req = dist.isend(tensor=tensor, dst=2)  # 异步发送，立即返回
            buffer.append(req)
        for req in buffer:
            req.wait()  # 最终确保全部发送完成

    elif rank == 2:  # 接收者：逐个接收并累加
        for i in range(10):
            tmp_tensor = torch.zeros(1)
            req = dist.irecv(tensor=tmp_tensor, src=0)  # 异步接收
            req.wait()  # 等待本次接收完成
            tensor += tmp_tensor

    dist.barrier()  # 防止先完成的rank执行destroy导致其他rank hang住
    dist.destroy_process_group()
```

### 2.3 批量异步 P2P：batch_isend_irecv

一次性提交多个 P2P 操作（典型用于**环形通信**）：

```python
def run(rank, master_addr, master_port, world_size):
    dist.init_process_group(...)

    send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
    recv_tensor = torch.zeros(2, dtype=torch.float32)

    # 定义环形通信：每个rank向右发、从左收
    send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size)

    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    print(f'Rank {rank} received {recv_tensor}')
    dist.destroy_process_group()
```

---

## 3. 集合通信 8 大原语

### 3.1 Broadcast — 一对多广播

**功能**：将 `src` rank 的数据复制到组内所有 rank。

**典型场景**：模型参数初始化同步（确保所有 rank 从相同参数起步）。

```python
# === PyTorch API ===
tensor = torch.zeros(2)
if dist.get_rank() == 0:
    tensor = tensor + 100
dist.broadcast(tensor, src=0)
# 所有 rank 上 tensor 都变成 [100., 100.]

# 也支持广播 Python 对象
objects = ["foo", 12, {1: 2}] if rank == 0 else [None, None, None]
dist.broadcast_object_list(objects, src=0)

# 支持子组广播
group = dist.new_group(ranks=[0, 1])
dist.broadcast(tensor, src=1, group=group)  # 只在组内广播

# === 手写实现（用 P2P）===
if rank == 0:
    ranks = dist.get_process_group_ranks(group)
    for r in ranks:
        if r != 0:
            dist.send(tensor=tensor, dst=r)
else:
    dist.recv(tensor=tensor, src=0)
```

### 3.2 Scatter — 一对多分发

**功能**：`src` rank 将一个列表中的数据，按 rank 顺序分发给组内每个 rank（每个 rank 收到不同的片段）。

**典型场景**：将训练数据切分分发到各 worker。

```python
# === PyTorch API ===
tensor_tmp = torch.zeros(2, dtype=torch.int64)
tensor_list = []
if rank == 0:
    tensor_total = torch.arange(2 * world_size, dtype=torch.int64) + 1
    tensor_list = list(tensor_total.split(split_size=2, dim=0))
    # [(1,2), (3,4), (5,6), (7,8)]
dist.scatter(tensor=tensor_tmp, scatter_list=list(tensor_list), src=0)
# rank 0 → [1,2], rank 1 → [3,4], rank 2 → [5,6], rank 3 → [7,8]

# 也支持 Python 对象
dist.scatter_object_list(output_list, objects, src=0)

# === 手写实现（用 P2P）===
if rank == 0:
    tensor_list = list(tensor_total.split(split_size=2, dim=0))
    ranks = dist.get_process_group_ranks(group)
    for r in ranks:
        if r != 0:
            dist.send(tensor_list[r], dst=r)
        else:
            tensor_tmp = tensor_list[0]
else:
    dist.recv(tensor_tmp, src=0)
```

### 3.3 Gather / AllGather — 多对一/多对多收集

**Gather**：所有 rank 的数据收集到某个 `dst` rank。
**AllGather**：所有 rank 的数据收集到**每个** rank（= Gather + Broadcast）。

**典型场景**：
- Gather：收集各 rank 的推理结果
- AllGather：**ZeRO 参数恢复**（forward 前用 AllGather 把分片参数拼回完整参数）

```python
# === AllGather API ===
tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
dist.all_gather(tensor_list=tensor_list, tensor=tensor)
# 每个 rank 都得到 [[1,2], [3,4], [5,6], [7,8]]

# AllGather into 连续 tensor（更高效）
output = torch.zeros(2 * world_size, dtype=torch.int64)
dist.all_gather_into_tensor(output_tensor=output, input_tensor=tensor)
# output = [1,2,3,4,5,6,7,8]

# === 手写 AllGather（Gather + Broadcast）===
tensor_list = [torch.arange(2, dtype=torch.int64) for _ in range(world_size)]
# Step 1: Gather to rank 0
if rank == 0:
    tensor_list[0] = tensor
    for r in range(1, world_size):
        dist.recv(tensor_list[r], src=r)
else:
    dist.send(tensor, dst=0)
# Step 2: rank 0 拼接
if rank == 0:
    tensor_trg = torch.concat(tensor_list, dim=0)
# Step 3: Broadcast 给所有 rank
dist.broadcast(tensor_trg, src=0)
```

### 3.4 Reduce — 多对一规约

**功能**：对所有 rank 的 tensor 做规约运算（SUM/MAX/MIN/PRODUCT），结果只存在 `dst` rank。

**典型场景**：收集 loss 求平均用于 logging。

```python
# === Reduce API ===
tensor = torch.ones(1) * 2 * rank
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
# 只有 rank 0 上 tensor = 0 + 2 + 4 + 6 = 12
```

**支持的 ReduceOp**：`SUM`, `MAX`, `MIN`, `PRODUCT`, `AVG`（部分后端）。

### 3.5 AllReduce — 多对多规约

**功能**：对所有 rank 的 tensor 做规约运算，结果存在**每个** rank（= Reduce + Broadcast）。

**典型场景**：**DDP 梯度同步**（所有 rank 得到相同的聚合梯度）。

```python
# === AllReduce API ===
tensor = torch.ones(1) * 2 * rank
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# 所有 rank 上 tensor 都 = 12

# 手动求 AVG（SUM 后除以 world_size）
dist.all_reduce(tensor, dist.ReduceOp.SUM)
tensor = tensor / world_size

# === 手写 AllReduce（Reduce + Broadcast）===
# Step 1: 所有 rank 把数据发给 rank 0 求和
if rank == 0:
    for r in range(1, world_size):
        tmp = torch.zeros_like(tensor)
        dist.recv(tensor=tmp, src=r)
        tensor_sum += tmp
else:
    dist.send(tensor=tensor, dst=0)
# Step 2: Broadcast 结果
dist.broadcast(tensor_sum, src=0)
```

> 朴素 AllReduce 的**瓶颈**：rank 0 是通信热点，总通信量 = 2(N-1) × D。Ring-AllReduce 解决这个问题。

### 3.6 ReduceScatter — 规约后分片

**功能**：先对所有 rank 的数据做 Reduce，再将结果 Scatter 到各 rank（每个 rank 只保留一份分片）。等价于 AllReduce 的"一半"。

**典型场景**：**ZeRO-2 梯度分片**（每个 rank 只维护一部分梯度/优化器状态）。

```python
# === ReduceScatter API ===
tensor_in = torch.arange(world_size * 2, dtype=torch.int64)  # 每个rank相同输入
tensor_out = torch.zeros(2, dtype=torch.int64)
dist.reduce_scatter_tensor(tensor_out, tensor_in, op=dist.ReduceOp.SUM)
# rank 0 → 前2个元素的sum，rank 1 → 中间2个的sum...

# === 手写实现（Reduce + Scatter）===
# Step 1: Reduce to rank 0
if rank == 0:
    for r in range(1, world_size):
        tensor_tmp = torch.zeros(world_size * 2, dtype=torch.int64)
        dist.recv(tensor_tmp, src=r)
        tensor_reduce += tensor_tmp
else:
    dist.send(tensor_in, dst=0)

# Step 2: Scatter 分片
if rank == 0:
    scatter_list = list(tensor_reduce.split(split_size=2, dim=0))
    for r in range(1, world_size):
        dist.send(scatter_list[r], dst=r)
    tensor_out = scatter_list[0]
else:
    dist.recv(tensor_out, src=0)
```

### 3.7 All2All — 全交换

**功能**：每个 rank 向每个 rank 发送一份（可能不同大小的）数据，同时从每个 rank 接收一份数据。可以理解为**转置操作**：输入矩阵的第 i 行第 j 列的数据，从 rank i 发往 rank j。

**典型场景**：**MoE 专家路由**（token 按专家分发→计算→结果回收）。

```python
# === All2All single（等大小交换）===
input = torch.arange(4) + rank * 4   # rank 0: [0,1,2,3], rank 1: [4,5,6,7], ...
output = torch.empty([4], dtype=torch.int64)
dist.all_to_all_single(output, input)
# rank 0 output = [0,4,8,12]（每个rank的第0个元素）
# rank 1 output = [1,5,9,13]（每个rank的第1个元素）

# === All2All list（不等大小交换）===
input = list(torch.arange(4).chunk(4))   # 4个size-1的tensor
output = list(torch.empty([4], dtype=torch.int64).chunk(4))
dist.all_to_all(output, input)

# === 灵活版本：指定 split sizes ===
dist.all_to_all_single(
    output=output_tensor,
    input=input_tensor,
    output_split_sizes=[2, 3, 2, 2],  # 从各rank接收的大小
    input_split_sizes=[2, 2, 1, 1],   # 向各rank发送的大小
)

# === 用 Scatter 等价实现 All2All ===
scatter_list = list(input.chunk(world_size))
gather_list = list(output.chunk(world_size))
for i in range(world_size):
    # rank i 作为 src，scatter 自己的数据给所有人
    dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)
```

---

## 4. Ring-AllReduce 手写实现

### 4.1 为什么需要 Ring-AllReduce？

**朴素 AllReduce（参数服务器模式）**：
- rank 0 作为中心节点，收集所有 rank 的数据 → 求和 → 广播回去
- 通信量：收集 `(N-1)×D` + 广播 `(N-1)×D` = `2(N-1)×D`
- **瓶颈**：rank 0 的带宽成为瓶颈，其他 rank 空闲等待

**Ring-AllReduce**：
- 所有 rank 组成环形拓扑，每个 rank 只与左右邻居通信
- 通信量均匀分摊到每个 rank，无中心瓶颈

### 4.2 完整实现

Ring-AllReduce 分两个阶段：**ReduceScatter 阶段** + **AllGather 阶段**。

```python
def run(rank, master_addr, master_port, world_size):
    """
    Ring AllReduce 完整实现
    将长度为 2*world_size 的 tensor 切成 world_size 个 micro_batch
    """
    dist.init_process_group(backend='gloo',
                            init_method='tcp://127.0.0.1:' + master_port,
                            rank=rank, world_size=world_size)

    # 初始数据：每个 rank 持有全 1 的 tensor [1,1,1,1,1,1,1,1]（world_size=4时）
    batch = torch.zeros(2 * world_size, dtype=torch.int64) + 1
    micro_batch = list(torch.split(batch, 2, dim=0))  # 切成 4 个 chunk
    tmp_tensor = torch.zeros(2, dtype=torch.int64)

    # ============ Stage 1: Reduce-Scatter ============
    # 经过 N-1 轮，每个 rank 最终持有一个 chunk 的完整 reduce 结果
    for i in range(world_size - 1):
        cur_idx = (rank - i) % world_size       # 当前要发送的 chunk
        next_idx = (rank - i - 1) % world_size  # 下一轮要累加到的 chunk

        # 防死锁：rank 0 先发后收，其他先收后发
        if rank % world_size == 0:
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
        micro_batch[next_idx] += tmp_tensor  # 累加到下一个 chunk

    # 此时每个 rank 的 micro_batch 中有一个 chunk 是所有 rank 的 sum
    # rank 0 持有 chunk[(0-3+1)%4]=chunk[1] 的完整 sum
    # rank 1 持有 chunk[2], rank 2 持有 chunk[3], rank 3 持有 chunk[0]

    dist.barrier()

    # ============ Stage 2: All-Gather ============
    # 经过 N-1 轮，每个 rank 把自己持有的完整 chunk 传播给所有其他 rank
    for i in range(world_size - 1):
        cur_idx = (i + rank + 1) % world_size
        next_idx = (cur_idx + 1) % world_size

        if rank % world_size == 0:
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp_tensor, src=(rank - 1) % world_size)
            dist.send(micro_batch[cur_idx], dst=(rank + 1) % world_size)
        micro_batch[next_idx] = tmp_tensor  # 直接覆盖（不再累加）

    # 最终每个 rank 都持有完整的 AllReduce 结果
    dist.barrier()
    dist.destroy_process_group()
```

### 4.3 通信量分析

设数据总量为 `D`，`N` 个 rank 组成环：

| 指标 | 朴素 AllReduce | Ring-AllReduce |
|------|---------------|----------------|
| 每 rank 每轮发送 | D | D/N |
| 总轮数 | 2 轮（收集+广播） | 2(N-1) 轮 |
| **每 rank 总通信量** | **2(N-1)×D** | **2(N-1)/N × D ≈ 2D** |
| 瓶颈节点 | rank 0（中心） | 无（均匀分摊） |

**关键结论**：
- Ring-AllReduce 每个 rank 的通信量 ≈ `2D`，**与 N 无关**（当 N 较大时）
- 朴素版 rank 0 的通信量 = `2(N-1)×D`，随 N 线性增长
- 因此 Ring-AllReduce 的通信量是**带宽最优**的

### 4.4 防死锁技巧

由于 `send/recv` 是阻塞的，如果所有 rank 都先 send 会死锁（没人 recv）。解法：
- **奇偶交替**：rank 0 先 send 后 recv，其他 rank 先 recv 后 send
- 或使用 `isend/irecv` + `batch_isend_irecv` 避免阻塞

---

## 5. 并行范式 × 通信原语对照表

| 并行范式 | 核心通信原语 | 说明 |
|----------|-------------|------|
| **DDP**（数据并行） | AllReduce | 各 rank 独立 forward/backward，AllReduce 同步梯度 |
| **ZeRO-1**（优化器分片） | ReduceScatter + AllGather | ReduceScatter 梯度→各 rank 更新分片参数→AllGather 恢复 |
| **ZeRO-2**（梯度+优化器分片） | ReduceScatter + AllGather | 同上，梯度也不全存 |
| **ZeRO-3 / FSDP** | ReduceScatter + AllGather | forward 时 AllGather 参数，backward 时 ReduceScatter 梯度 |
| **TP**（张量并行） | AllReduce / AllGather | 列并行用 AllGather，行并行用 AllReduce |
| **PP**（流水线并行） | Send / Recv（P2P） | stage 间传递 activation / gradient |
| **EP / MoE**（专家并行） | All2All | token dispatch 到对应专家 → 计算 → All2All 回收 |
| **SP**（序列并行） | AllGather + ReduceScatter | 与 TP 配合，减少 activation 内存 |

---

## 6. 面试考点

### Q1: Ring-AllReduce 每个 rank 的通信量是多少？为什么比朴素方式好？

**答**：设数据量 D、N 个 rank。
- Ring-AllReduce：分 ReduceScatter 和 AllGather 两阶段，每阶段 N-1 轮，每轮发送 D/N → 每 rank 总通信 `2(N-1)/N × D`。
- 朴素方式：中心节点通信 `2(N-1) × D`。
- Ring 的优势：通信量 ≈ 2D 与 N 无关，且负载均匀无热点。

### Q2: All2All 用在哪里？为什么 MoE 需要它？

**答**：All2All 用于 **MoE 专家并行**。每个 rank 上有部分 token 和部分专家。Gate 路由后，token 需要发送到对应专家所在的 rank（dispatch All2All），专家计算完再发回来（combine All2All）。All2All 是唯一支持"每个 rank 向每个 rank 发送不同数据"的原语。

### Q3: ReduceScatter 和 AllGather 怎么配合实现 ZeRO/FSDP？

**答**：
- **Forward**：每个 rank 只存 1/N 参数，forward 前 **AllGather** 把完整参数拼回来用于计算，用完即丢
- **Backward**：计算出完整梯度后，**ReduceScatter** 聚合梯度并分片到各 rank
- 每个 rank 只用自己分片的梯度更新自己分片的参数
- 两个原语合起来 = AllReduce 的通信量，但内存从 O(D) 降到 O(D/N)

### Q4: gloo 和 nccl 后端的区别？什么时候用哪个？

**答**：
- `gloo`：纯 CPU 实现，支持所有平台，开发调试用
- `nccl`：NVIDIA GPU 专用，利用 NVLink/NVSwitch/RDMA，生产环境必选
- 实际训练中：GPU 通信走 `nccl`，CPU tensor（如 object_list）走 `gloo`
- NCCL 不支持某些操作（如 `send/recv` 在某些版本中有限制），此时 fallback 到 gloo

### Q5: Ring-AllReduce 有什么局限？实际中还有哪些 AllReduce 算法？

**答**：
- **局限**：延迟 = 2(N-1) 轮，小数据量下延迟占主导（带宽利用率低）；只适合同构带宽拓扑
- **改进**：
  - **Recursive Halving-Doubling**：O(log N) 轮次，适合小数据
  - **2D-Ring**：多机多卡时，机内 NVLink ring + 机间 IB ring
  - **树形 AllReduce**：非对称拓扑下更优
  - NCCL 内部自动根据数据大小和拓扑选择最优算法

---

## 附：通信原语数据流一图流

```
Broadcast:    [A] ──→ [A] [A] [A] [A]        （src → all）

Scatter:      [A B C D] ──→ [A] [B] [C] [D]  （src 切分 → 各rank）

Gather:       [A] [B] [C] [D] ──→ [A B C D]  （各rank → dst 拼接）

AllGather:    [A] [B] [C] [D] ──→ [ABCD] [ABCD] [ABCD] [ABCD]

Reduce:       [1] [2] [3] [4] ──→ [10]       （各rank → dst 求和）

AllReduce:    [1] [2] [3] [4] ──→ [10] [10] [10] [10]

ReduceScatter:[AB][AB][AB][AB] ──→ [sum(A)] [sum(B)] ...

All2All:      [ab][cd][ef][gh] ──→ [aceg] [bdfh]（转置）
```
