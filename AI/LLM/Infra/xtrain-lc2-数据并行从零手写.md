---
title: "xtrain lc2：数据并行 DDP 从零手写"
brief: "MA-RLHF xtrain课程lc2：DP vs DDP原理对比，DDP梯度同步机制（Bucket All-Reduce）+ DistributedSampler实现，从零手撕torch_ddp.py，含面试必考Q&A（为什么除以world_size / barrier时机）。"
type: hands-on-coding
source: MA-RLHF xtrain lecture/lc2_data_parallelism/
date: 2026-02-26
rating: ★★★★★
tags:
  - distributed-training
  - DDP
  - data-parallelism
  - gradient-sync
  - bucket-all-reduce
  - hands-on
  - 面试必考
related:
  - AI/LLM/Infra/xtrain-lc1-分布式通信原语从零手写
  - AI/LLM/Architecture/分布式训练
---

# xtrain lc2：数据并行 DDP 从零手写

**来源**：MA-RLHF xtrain `lecture/lc2_data_parallelism/`
**评级**：★★★★★
**标签**：#DDP #数据并行 #梯度同步 #DistributedSampler #面试必考
**核心文件**：`torch_ddp.py` / `torch_ddp_train.py` / `distributed_dataset.py`

---

## 一、DDP 的基本理念

```
数据并行 = 每个 GPU 持有完整模型 + 各自处理不同数据

rank0: model_copy → forward(data_0) → grad_0
rank1: model_copy → forward(data_1) → grad_1
rank2: model_copy → forward(data_2) → grad_2
rank3: model_copy → forward(data_3) → grad_3
             ↓ All-Reduce (SUM / N)
             所有 rank 同步梯度 → 同步更新参数
```

**五个不变量**：
1. 参数初始化相同（DDP 第一次 forward 前广播）
2. 每个 rank 数据不同（DistributedSampler 保证无重叠）
3. 每个 rank loss 不同（因为数据不同）
4. 梯度相同（All-Reduce 后）
5. optimizer 更新后参数相同（因为梯度相同）

---

## 二、官方 DDP 使用（`torch_ddp.py`）

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='gloo', init_method='tcp://...', rank=rank, world_size=world_size)

# 构建模型，包装成 DDP
model = ToyModel()
ddp_model = DDP(model)  # DDP 自动在 backward 后 hook All-Reduce

# 正常训练（DDP 透明同步梯度）
optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()
    output, _ = ddp_model(input)   # 每个 rank 的 input 不同
    loss = loss_fn(output, labels)
    loss.backward()               # 反向传播时自动触发 All-Reduce
    optimizer.step()              # 梯度已同步，更新结果一致
```

**DDP 的 Hook 机制**：
- DDP 在每个参数上注册 `autograd hook`
- 反向传播计算完一个参数的梯度后立即触发 All-Reduce（不等所有层算完）
- **Bucketing**：实际上把多个小参数打包成 bucket，攒够一批再 All-Reduce（减少通信次数）

**访问模型参数**：`ddp_model.module.w1.weight`（`.module` 剥掉 DDP 包装）

---

## 三、手写 DDP（`train_my_ddp` 函数）

```python
def train_my_ddp(rank, world_size, model, input, labels, loss_fn, optimizer, epochs):
    lr = optimizer.param_groups[0]['lr']
    bs, _ = input.shape
    _, dim_out = labels.shape
    
    with torch.no_grad():
        for i in range(epochs):
            output, hidden = model(input)
            loss = loss_fn(output, labels)

            # 手动反向传播：两层 MLP
            do = 2 * (output - labels) / (bs * dim_out)   # dL/d_output
            
            # 第二层梯度
            grad_w2 = do.t() @ hidden                      # [dim_out, dim_hidden]
            grad_hidden = do @ model.w2.weight             # [bs, dim_hidden]
            
            # All-Reduce 梯度 w2（所有 rank 梯度求和再除以 world_size）
            dist.all_reduce(grad_w2, dist.ReduceOp.SUM)
            grad_w2 = grad_w2 / world_size
            model.w2.weight -= lr * grad_w2
            
            # 第一层梯度（grad_hidden 不需要 All-Reduce！只是中间量）
            grad_w1 = grad_hidden.t() @ input             # [dim_hidden, dim_in]
            dist.all_reduce(grad_w1, dist.ReduceOp.SUM)
            grad_w1 = grad_w1 / world_size
            model.w1.weight -= lr * grad_w1
```

**关键问题：梯度 All-Reduce 在每层反向后，还是全部层反向完后？**

**答案：全部层反向完后，对每一层的参数梯度单独 All-Reduce**（方案 1）。

理由：
- 中间激活值（如 `grad_hidden`）不需要跨 GPU 同步，每个 GPU 各自算
- 只有**参数梯度**（`grad_w1`, `grad_w2`）需要同步
- 方案 2（逐层 backward + 逐层 All-Reduce）理论上可行但实现复杂，实际与方案 1 等价

---

## 四、分布式数据集（`distributed_dataset.py` + `torch_ddp_train.py`）

### 方案 A：全量数据 + DistributedSampler（推荐）

```python
dataset = MyDataset(N=1024, dim=16, classes=10)  # 每个 GPU 存完整数据集

# DistributedSampler 自动分配无重叠的索引子集
sampler = DistributedSampler(dataset)
# 默认：rank i 负责 dataset[i::world_size] 的 index

dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
# 每个 GPU 只遍历自己的 1/world_size 数据

# steps * world_size * batch_size = N
```

**DistributedSampler 内部逻辑**：
- `total_size = len(dataset)` 向上取整到 world_size 的倍数
- `indices = range(total_size)[rank::world_size]`（步长 world_size 的均匀分配）
- 不足时循环补充（padding），保证每个 rank batch 数相同

**多 epoch 使用时必须调用**：
```python
sampler.set_epoch(epoch)  # 每个 epoch 重新洗牌，否则数据顺序固定
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

### 方案 B：数据 Scatter 分发（`run_distributed_dataset`）

```python
# rank0 有完整数据，scatter 分给各 GPU
distributed_dataset = MyDataset(N // world_size, dim, classes)  # 小数据集

if rank == 0:
    datas = list(full_dataset.data.split(N // world_size, dim=0))
    labels = list(full_dataset.labels.split(N // world_size, dim=0))
else:
    datas = None
    labels = None

dist.scatter(distributed_dataset.data, datas, src=0)    # 分发数据
dist.scatter(distributed_dataset.labels, labels, src=0) # 分发标签
dist.barrier()

dataloader = DataLoader(distributed_dataset, batch_size=2)  # 无需 sampler
```

**方案 A vs B**：
| | 方案 A（DistributedSampler） | 方案 B（Scatter分发） |
|--|--|--|
| 内存 | 每 GPU 存全量（N条） | 每 GPU 存 1/world_size（N/W条）|
| 通信 | 无（sampler只传索引） | 初始化时 Scatter |
| 灵活性 | 高（shuffle/epoch管理简单） | 低（固定分配） |
| 实际使用 | **生产推荐** | 特殊场景（数据加载受限）|

---

## 五、完整 DDP 训练程序模板（`torch_ddp_train.py`）

```python
def run(rank, master_addr, master_port, world_size):
    # 1. 初始化
    dist.init_process_group(backend='gloo', 
                            init_method=f'tcp://{master_addr}:{master_port}',
                            rank=rank, world_size=world_size)
    
    # 2. 模型
    model = ToyModel(dim_in=16, dim_hidden=512, classes=10)
    ddp_model = DDP(model)
    
    # 3. 数据（DistributedSampler 分片）
    dataset = MyDataset(N=1024, dim=16, classes=10)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    
    # 4. 损失 + 优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 5. 训练循环
    for epoch in range(100):
        # sampler.set_epoch(epoch)  # 生产中必须
        for batch in dataloader:
            optimizer.zero_grad()
            output, _ = ddp_model(batch['input_ids'])
            loss = loss_fn(output, batch['lables'])
            loss.backward()      # 触发 All-Reduce
            optimizer.step()
        
        # 汇报全局 loss（可选）
        dist.all_reduce(loss.data, dist.ReduceOp.SUM)
        if rank == 0:
            print(epoch, loss.data / world_size)
    
    dist.destroy_process_group()

# 启动
mp.spawn(run, args=("127.0.0.1", "12801", 4,), nprocs=4)
# 或 torchrun --nproc_per_node=4 script.py（生产推荐）
```

---

## 六、DDP 的局限性与 FSDP/ZeRO 的出现

**DDP 的问题**：
- 每个 GPU 持有完整模型 → 大模型装不下（如 70B LLaMA 需要 ~140GB，8xA100 每卡只有 80GB）
- 梯度和 optimizer state 同样完整 → 峰值显存 ≈ 参数 × 16（fp16: param+grad+adam_m+adam_v）

**FSDP/ZeRO 如何解决**：
- ZeRO-1：分片 optimizer state（节省 4x）
- ZeRO-2：分片 optimizer state + 梯度（节省 8x）
- ZeRO-3/FSDP：分片 optimizer state + 梯度 + 参数（节省 N*world_size 倍）
  - 前向时 All-Gather 拼回参数 → 计算 → 反向后 Reduce-Scatter 分梯度 → Scatter 参数

**DDP 仍是主力**：对于能放入单卡的模型，DDP 是最简单高效的方案；ZeRO 是内存受限时的选择。

---

## 七、面试必备

**Q：DDP 和 DP（DataParallel）的区别？**
- DP：单进程多线程，GIL 限制效率；参数服务器模式（rank0 汇总，再广播）→ rank0 成瓶颈
- DDP：多进程，每进程一个 GPU；Ring All-Reduce 去中心化同步梯度 → 线性扩展

**Q：DDP 的 backward 是如何同步梯度的？**
- autograd hook：每个参数注册梯度完成的回调
- Bucket 机制：参数按注册顺序打包，bucket 满了就触发 All-Reduce（等于边算边通信）
- 通信和计算重叠：后面的 layer 还在 backward 时，前面 layer 的 bucket 已经在 All-Reduce

**Q：DistributedSampler 怎么保证无数据重叠？**
- 每个 rank 取 `indices[rank::world_size]`，步长 world_size 均匀分配
- 必须 `set_epoch(epoch)` 保证每 epoch 洗牌后分配不同

**Q：手写 DDP 的梯度同步逻辑（两层 MLP）？**
```python
# 关键代码
dist.all_reduce(grad_w2, dist.ReduceOp.SUM)
grad_w2 = grad_w2 / world_size  # 除以 world_size 得到平均梯度

dist.all_reduce(grad_w1, dist.ReduceOp.SUM)
grad_w1 = grad_w1 / world_size
```
- 中间激活梯度（`grad_hidden`）不需要同步，只需参数梯度

**Q：为什么 All-Reduce 后要除以 world_size？**
- All-Reduce 用 SUM，结果是所有 rank 梯度之和
- 数学等价于 batch_size * world_size 的 mini-batch 梯度总和
- 除以 world_size 得到等价于大 batch 的平均梯度（与 lr 的关系保持一致）
- 也可以在计算 loss 时除以 world_size，两种等价

**Q：在 loss.backward() 之前还是之后需要 dist.barrier()？**
- 不需要额外 barrier——DDP 的 bucket All-Reduce 内置了同步语义
- optimizer.step() 只在梯度就绪后执行（DDP 保证这点）
- 只在打印/汇报全局 loss 时，需要先 all_reduce loss 数值
