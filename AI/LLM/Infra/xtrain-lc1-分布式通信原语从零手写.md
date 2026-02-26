---
title: "xtrain lc1：分布式通信原语从零手写"
brief: "MA-RLHF xtrain课程lc1：Ring-AllReduce原理 + PyTorch dist 8大通信原语（broadcast/reduce/gather/scatter/all_reduce/all_gather/reduce_scatter/all2all + P2P异步）从零手写，含NCCL多进程实战和防死锁面试Q&A。"
type: hands-on-coding
source: MA-RLHF xtrain lecture/lc1_basic/
date: 2026-02-26
rating: ★★★★★
tags:
  - distributed-training
  - communication-primitives
  - ring-allreduce
  - nccl
  - pytorch-dist
  - hands-on
  - 面试必考
related:
  - AI/LLM/Infra/xtrain-lc2-数据并行从零手写
  - AI/LLM/Architecture/分布式训练
  - AI/LLM/Infra/verl-数据流与训练循环
---

# xtrain lc1：分布式通信原语从零手写

**来源**：MA-RLHF xtrain `lecture/lc1_basic/`（21个py文件）
**评级**：★★★★★
**标签**：#分布式通信 #PyTorch分布式 #集合通信 #通信原语 #面试必考
**核心文件**：`initial.py` / `fun_broadcast.py` / `fun_reduce.py` / `fun_gather.py` / `fun_scatter.py` / `fun_reduce_scatter.py` / `fun_ring_allreduce.py` / `fun_all2all.py` / `p2p.py` / `p2p_async.py`

---

## 一、分布式初始化（`initial.py`）

### 三种后端

```python
dist.init_process_group(
    backend="gloo",        # CPU/调试用，支持所有操作
    # backend="nccl",      # GPU专用，生产必用，不支持gloo部分操作
    # backend="mpi",       # MPI后端，HPC场景
    init_method='env://',  # 从环境变量读取 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE
    world_size=4,
    rank=0,
)
```

**三种 init_method**：
- `env://`：从环境变量读（torchrun默认）
- `tcp://127.0.0.1:12801`：手动指定 rendezvous 地址
- `file:///path/to/shared/file`：文件存储（共享文件系统）

### 1D vs 2D DeviceMesh

```python
# 1D mesh（传统分布式）
dist.init_process_group(backend="gloo", ...)

# 2D mesh（TP+DP 混合并行必用）
dist.init_device_mesh(
    device_type='cpu',   # 'cuda' for GPU
    mesh_shape=(2, 4),   # (dp_size, tp_size)
    mesh_dim_names=("dp", "tp")
)
# 4x4 GPU集群：mesh_shape=(4,4)，自动路由dp组和tp组的通信
```

**实用 API**：
```python
dist.get_rank()           # 当前进程 rank
dist.get_world_size()     # 总进程数
dist.get_backend()        # 当前后端
dist.is_initialized()     # 是否已初始化
dist.barrier()            # 同步屏障
dist.destroy_process_group()  # 清理（防止hang）
```

### Group 管理

```python
# 创建子通信组（TP内部通信只在组内进行）
group = dist.new_group(ranks=[0, 1], backend='gloo')
dist.all_reduce(tensor, group=group)  # 只在 rank 0,1 间规约
dist.get_group_rank(group, global_rank=0)  # 全局 rank → 组内 rank
dist.get_process_group_ranks(group)   # 组内所有 global rank
```

---

## 二、P2P 通信（点对点）

### 同步 P2P（`p2p.py`）

```python
# 阻塞：send 完才继续，recv 到才继续
if rank == 0:
    dist.send(tensor=tensor, dst=1)   # 发送给 rank 1
elif rank == 1:
    dist.recv(tensor=tensor, src=0)   # 从 rank 0 接收
```

**注意**：send/recv 必须配对，否则死锁。Pipeline Parallel 的前向/反向传播就是这个原语。

### 异步 P2P（`p2p_async.py`）

```python
if rank == 0:
    buffer = []
    for i in range(10):
        req = dist.isend(tensor=tensor_i, dst=2)  # 非阻塞，立即返回
        buffer.append(req)
    for req in buffer:
        req.wait()  # 最后统一等待

elif rank == 2:
    for i in range(10):
        req = dist.irecv(tensor=tmp, src=0)  # 非阻塞接收
        req.wait()  # 等待这一条接收完
        tensor += tmp

dist.barrier()  # 关键！防止某 rank 提前 destroy 导致其他 rank hang
```

**为什么异步？** Pipeline Parallel 的 microbatch 流水线：发送完当前 microbatch 就立即准备下一个，不等接收方处理完。

---

## 三、集合通信原语（Collective Operations）

### 3.1 Broadcast

```
rank0: [A] → rank0: [A]
rank1: [?] → rank1: [A]
rank2: [?] → rank2: [A]
```

```python
# 官方 API
tensor = torch.zeros(2)
if rank == 0: tensor += 100
dist.broadcast(tensor, src=0)  # 所有 rank 都有 100

# 手动实现（p2p 模拟）
if rank == 0:
    for r in ranks:
        if r != 0: dist.send(tensor, dst=r)
else:
    dist.recv(tensor, src=0)
```

**用途**：初始化时同步模型参数（DDP 的第一步）；广播超参配置。

### 3.2 Reduce / All-Reduce

```
Reduce（结果只在 dst）：
rank0: [2] → rank0: [2+4+6+8=20]
rank1: [4] → rank1: [4]（不变）

All-Reduce（结果在所有 rank）：
rank0: [2] → rank0: [20]
rank1: [4] → rank1: [20]
```

```python
# All-Reduce（DDP 梯度同步核心）
tensor = torch.ones(1) * rank
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# 或 MAX / MIN / PRODUCT

# Reduce（结果只到 dst）
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
```

**手动实现 All-Reduce（参数服务器模式）**：
```python
if rank == 0:  # 参数服务器
    for r in ranks:
        if r != 0:
            dist.recv(tmp, src=r)
            tensor_sum += tmp
else:
    dist.send(tensor, dst=0)  # 发给参数服务器

dist.broadcast(tensor_sum, src=0)  # 再广播回去
```
**注意**：参数服务器模式 = reduce + broadcast。Ring All-Reduce 更高效（无中心节点瓶颈）。

### 3.3 Gather / All-Gather

```
Gather（结果只在 dst）：
rank0: [1,2] → rank0: [[1,2],[3,4],[5,6],[7,8]]
rank1: [3,4] → rank1: [3,4]（不变）

All-Gather（所有 rank 都有完整拼接）：
rank0: [1,2] → rank0: [[1,2],[3,4],[5,6],[7,8]]
rank1: [3,4] → rank1: [[1,2],[3,4],[5,6],[7,8]]
```

```python
# All-Gather 列表形式
tensor_list = [torch.zeros(2) for _ in range(world_size)]
tensor = torch.arange(2) + 1 + 2 * rank
dist.all_gather(tensor_list=tensor_list, tensor=tensor)

# All-Gather 直接输出到 tensor（更高效）
tensor_trg = torch.zeros(2 * world_size)  # 行拼接
dist.all_gather_into_tensor(output_tensor=tensor_trg, input_tensor=tensor)

# 手动实现（rank0收集 + broadcast）
if rank == 0:
    tensor_list[0] = tensor
    for r in ranks:
        if r != 0: dist.recv(tensor_list[r], src=r)
    tensor_trg = torch.concat(tensor_list, dim=0)
else:
    dist.send(tensor, dst=0)
dist.broadcast(tensor_trg, src=0)
```

**用途**：ZeRO-3 的参数 gather（前向时拼回完整参数）；TP 的 output all-gather。

### 3.4 Scatter

```
Scatter（从 src 分发）：
rank0: [1,2,3,4,5,6,7,8] → rank0: [1,2]
                           → rank1: [3,4]
                           → rank2: [5,6]
                           → rank3: [7,8]
```

```python
tensor_tmp = torch.zeros(2)
if rank == 0:
    tensor_total = torch.arange(2 * world_size)
    tensor_list = tensor_total.split(split_size=2, dim=0)  # 切分
dist.scatter(tensor=tensor_tmp, scatter_list=list(tensor_list), src=0)
```

**用途**：初始化时分发数据分片；ZeRO 参数/梯度的初始分配。

### 3.5 Reduce-Scatter（ZeRO 核心）

```
每个 rank 有完整梯度 [g0,g1,g2,g3]：
         rank0 rank1 rank2 rank3
     → reduce([g0_sum]) → rank0: g0_sum
     → reduce([g1_sum]) → rank1: g1_sum
     → reduce([g2_sum]) → rank2: g2_sum
     → reduce([g3_sum]) → rank3: g3_sum
```

```python
tensor_in = torch.arange(world_size * 2)  # 完整梯度
tensor_out = torch.zeros(2)               # 分配到自己的梯度分片
dist.reduce_scatter_tensor(tensor_out, tensor_in, op=dist.ReduceOp.SUM)
```

**手动实现（Reduce + Scatter）**：
```python
# 先 reduce 到 rank0
if rank == 0:
    for r in ranks:
        if r != 0:
            dist.recv(tmp, src=r)
            tensor_reduce += tmp
else:
    dist.send(tensor_in, dst=0)

# 再 scatter 分块
if rank == 0:
    scatter_list = list(tensor_reduce.split(2, dim=0))
    for r in ranks:
        if r != 0: dist.send(scatter_list[r], dst=r)
        else: tensor_out = scatter_list[0]
else:
    dist.recv(tensor_out, src=0)
```

**用途**：**ZeRO-2/3 梯度规约的核心原语**。每个 rank 只维护自己负责的梯度分片，节省 world_size 倍内存。

### 3.6 All-to-All（MoE 核心）

```
每个 rank 给所有其他 rank 发不同的数据：
rank0: [a→0, b→1, c→2, d→3] → rank0收到: [a_r0, a_r1, a_r2, a_r3]
rank1: [e→0, f→1, g→2, h→3] → rank1收到: [b_r0, b_r1, b_r2, b_r3]
...（全排列转置）
```

```python
input = torch.arange(4) + rank * 4     # [0,1,2,3] / [4,5,6,7] / ...
output = torch.empty([4], dtype=torch.int64)
dist.all_to_all_single(output, input)
# 等价于：对每个 rank i，scatter input[i] 到 rank i

# 不等长数据（MoE dispatch 实际场景）
input_splits = torch.tensor([[2,2,1,1],[3,2,2,2],[2,1,1,1],[2,2,2,1]])
output_splits = input_splits.t()  # 转置！input 的 split 决定 output 的 split
dist.all_to_all_single(
    output=output_buf,
    input=input_buf,
    output_split_sizes=output_splits[rank].tolist(),
    input_split_sizes=input_splits[rank].tolist(),
)
```

**用途**：**MoE 的 Expert Dispatch**（将每个 token 路由到对应 Expert 所在 GPU）和 **Expert Gather**（Expert 输出汇回 token 的原始 GPU）。

---

## 四、Ring All-Reduce 手写（`fun_ring_allreduce.py`）

Ring All-Reduce = Reduce-Scatter + All-Gather，无中心节点，通信量恒定。

```python
micro_batch = list(batch.split(2, dim=0))  # 切成 world_size 份

# Stage 1: Reduce-Scatter（N-1 轮）
for i in range(world_size - 1):
    cur_idx = (rank - i) % world_size       # 当前要发送的分片
    next_idx = (rank - i - 1) % world_size  # 接收后累加的位置
    
    if rank == 0:  # 防死锁：rank0先发再收
        dist.send(micro_batch[cur_idx], dst=(rank+1) % world_size)
        dist.recv(tmp, src=(rank-1) % world_size)
    else:          # 其他rank先收再发（防止所有rank同时send导致死锁）
        dist.recv(tmp, src=(rank-1) % world_size)
        dist.send(micro_batch[cur_idx], dst=(rank+1) % world_size)
    micro_batch[next_idx] += tmp

# Stage 2: All-Gather（N-1 轮）
for i in range(world_size - 1):
    cur_idx = (i + rank + 1) % world_size
    next_idx = (cur_idx + 1) % world_size
    # 同上，rank0先发再收，其他先收再发
    dist.recv(tmp, src=(rank-1) % world_size)
    dist.send(micro_batch[cur_idx], dst=(rank+1) % world_size)
    micro_batch[next_idx] = tmp  # All-Gather 是赋值，不是累加！
```

**关键工程细节**：
1. **死锁防御**：rank0 先 send 再 recv，其他先 recv 再 send。否则所有 rank 同时 send，陷入循环等待。
2. **Stage 1**：`micro_batch[next_idx] += tmp`（累加 = reduce）
3. **Stage 2**：`micro_batch[next_idx] = tmp`（赋值 = gather，不是累加！）
4. 通信量：每个 GPU 发 N-1 次，收 N-1 次，每次 M/N 数据量 → 总通信 = 2*(N-1)/N * M ≈ 2M（与 world_size 无关）

---

## 五、torchrun vs mp.spawn

```python
# mp.spawn：手动管理进程，调试用
mp.spawn(run, args=("127.0.0.1", "12801", 4,), nprocs=4)

# torchrun：生产推荐，自动设置 RANK/WORLD_SIZE/MASTER_ADDR
# torchrun --nproc_per_node=4 script.py
# 启动后每个 rank 调用 dist.init_process_group(init_method='env://')
```

---

## 六、原语汇总表

| 操作 | 方向 | 数据变化 | 核心用途 |
|------|------|---------|---------|
| Broadcast | 1→all | 1份→N份 | 参数初始化同步 |
| Reduce | all→1 | N份→1份（聚合） | 梯度汇总到参数服务器 |
| **All-Reduce** | all→all | N份→N份（各持聚合） | **DDP 梯度同步** |
| Gather | all→1 | N份→1处存N份 | 调试/日志收集 |
| **All-Gather** | all→all | 1份→all持N份 | **ZeRO-3 参数 gather；TP output** |
| Scatter | 1→all | N份→各1份 | 数据/参数分发 |
| **Reduce-Scatter** | all→all | N份全量→各1份分片 | **ZeRO-2/3 梯度分片** |
| **All-to-All** | all→all | 全排列转置 | **MoE Expert Dispatch/Gather** |
| P2P Send/Recv | 1→1 | 点对点 | **Pipeline Parallel** |

---

## 七、面试必备

**Q：DDP 的梯度同步用哪个原语？为什么？**
- All-Reduce（通常是 Ring All-Reduce）
- 每个 rank 各算一份 loss/梯度，All-Reduce 后每个 rank 都有全局平均梯度
- 不用 Reduce 是因为 Reduce 只有 dst 有结果，其他 rank 还要广播，多了一步

**Q：ZeRO-2 和 ZeRO-3 分别用什么原语？**
- ZeRO-1：optimizer state 分片，梯度 All-Reduce 后各自更新自己的分片
- ZeRO-2：梯度 Reduce-Scatter（各持一部分聚合梯度），optimizer state 分片更新
- ZeRO-3：梯度 Reduce-Scatter，参数 All-Gather（前向时临时拼回），参数 Scatter（前向完 scatter 回各 rank）

**Q：Ring All-Reduce 的通信量是多少？与 world_size 有什么关系？**
- 每个 GPU 总发送 = (N-1)/N * M ≈ M，总接收 ≈ M（Reduce-Scatter）
- All-Gather 再各 ≈ M → 总通信量 ≈ 2M，与 world_size N 无关
- 对比参数服务器（中心节点通信量 ∝ N）：Ring All-Reduce 是 O(1) vs O(N)

**Q：All-to-All 在 MoE 中的具体作用？**
- Expert Dispatch：每个 GPU 持有全部 token，需要把每个 token 发到对应 Expert 所在 GPU
  - 不等长：input_splits[rank][dst] = 要发给 GPU dst 的 token 数
- Expert Gather：Expert 计算完输出，需要发回 token 原始 GPU
  - = 第一次 All-to-All 的转置（output_splits = input_splits.t()）

**Q：异步 P2P 为什么要在最后加 barrier？**
- isend 方可能已执行完 barrier，但 irecv 方还未接收完
- 如果 sender 先 destroy_process_group，receiver 会 hang 死
- barrier 保证所有 rank 都完成后才销毁进程组

**Q：send/recv 防死锁的模式？**
- 奇偶交替：rank 为偶数先 send，奇数先 recv（或 rank0 先 send，其他先 recv）
- 使用 isend/irecv（异步不阻塞，天然防死锁但需要显式 wait）
- 使用 dist.batch_isend_irecv（BatchIsendIrecv，一次提交一组 p2p 操作）
