---
title: P1：xtrain——分布式训练基础设施，从零实现
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, pre-training, distributed-training, TP, PP, ZeRO]
brief: 系统从零实现分布式训练全栈（通信原语→ZeRO→TP→PP→MoE），不是调库，是真正理解每一层。故事线：在后训练实验里反复被分布式问题卡住，倒逼自己把底层搞清楚。
related:
  - "[[Projects/MA-RLHF/xtrain/xtrain-03-ZeRO优化器从零手写]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-04-张量并行从零手写]]"
  - "[[Projects/MA-RLHF/xtrain/xtrain-05-流水线并行从零手写]]"
  - "[[AI/3-LLM/Architecture/DeepSeek-V3-手撕实操]]"
---

# P1：xtrain——分布式训练基础设施，从零实现

---

## 故事线（面试完整版，4-6 分钟）

---

### 第一幕：为什么要从零写，不直接用框架

在后训练项目（P2）里用 verl 跑 GRPO 的时候，遇到了一个让我很沮丧的处境——

训练过程中，GPU 之间的 weight sync 会偶尔卡住，卡的时间从几秒到几分钟不等，没有规律。报错信息完全看不出根因，只说 timeout。我当时的调试方法是：改改配置，重启，看看好没好。

这是工程师最怕的状态：碰运气。

我意识到问题在哪：我知道怎么用 verl，但我不知道 verl 在底层做了什么。当 weight sync 卡住，我不知道是 NCCL 的问题、还是 Ray 的调度问题、还是 vLLM 和 Actor 进程之间的通信问题。不知道是哪一层，就没办法系统排查。

所以我决定倒过来：**先把底层搞清楚，再用框架**。xtrain 这个项目，就是从零实现分布式训练的全栈，不调 NCCL、不调 DeepSpeed，自己写。

---

### 第二幕：通信原语——第一步就很硬

从最底层开始：AllReduce。

AllReduce 是数据并行训练的核心——多个 GPU 各自算完梯度，需要聚合成全局梯度，再各自更新参数。怎么聚合，就是 AllReduce 要解决的问题。

我先实现了最朴素的版本：所有 GPU 把梯度发给主 GPU，主 GPU 加完再发回去。跑了一下，确实能工作，但很慢。瓶颈很明显：主 GPU 是单点，所有流量都过它，带宽打满了。

然后实现了 Ring AllReduce：把 GPU 排成一个环，每个 GPU 只和相邻的 GPU 通信，分多轮把数据传完。带宽利用率从单点的 1/N 变成接近 100%。

实现完之后，我才真正理解了为什么 NCCL 快——它不只是 Ring AllReduce，还根据硬件拓扑（NVLink vs PCIe vs InfiniBand）自动选最优通信路径，同一台机器里的 GPU 用 NVLink（带宽 600GB/s），跨机器用 InfiniBand。这个选择是 NCCL 在运行时做的，你调 API 的时候感知不到。但当通信出问题时，理解这一层非常关键——是 NVLink 降速了，还是跨机器带宽被占满了，排查方向完全不同。

---

### 第三幕：ZeRO——显存的数学账

通信搞清楚之后，下一个核心问题是显存。

7B 参数的模型，全精度（fp32）存参数就要 28GB。但训练时还有梯度（28GB）和 AdamW 的 optimizer states（动量 + 方差 + master weights，大约 84GB）。加起来 140GB，单卡 A100（80GB）完全放不下。

ZeRO 的思路是：这些数据在多卡 DDP 里是完全冗余的——每张 GPU 都存了一份一模一样的 optimizer state。把冗余去掉，每张 GPU 只存 1/N 的 optimizer state，需要用的时候 AllGather 拿过来，用完 ReduceScatter 分发出去。

我把三个级别都实现了：ZeRO-1（只分片 optimizer state）、ZeRO-2（再分片梯度）、ZeRO-3（连参数也分片）。

实现的过程里遇到了一个很细节但很重要的问题：ZeRO-3 的 AllGather 和 ReduceScatter 必须和 forward/backward 的计算严格交错，不然要么显存峰值反而更高（一次性 AllGather 所有参数），要么计算等待通信变成串行。这个 overlap 是性能的关键，也是实际工程里最难调的部分。

做完这个之后，我对"为什么不是所有场景都用 ZeRO-3"有了真实的理解：ZeRO-3 的通信开销很高，当节点间带宽低时（跨机器），每次 forward 都要 AllGather 全部参数，反而比 ZeRO-1 更慢。工程上的选择总是 trade-off，不是越激进越好。

---

### 第四幕：张量并行——把模型切开

ZeRO 解决了"参数怎么省"，张量并行（TP）解决的是"一个模型本身就太大，一张 GPU 放不下怎么办"。

TP 的核心是把矩阵乘法按列或者按行切开，不同 GPU 算不同的部分，再用通信聚合结果。

实现 MLP 层的 TP 时，有一个让我觉得很优雅的地方：Column Parallel（按列切）→ Row Parallel（按行切）这对组合，整个 MLP 只需要两次 AllReduce 通信，而不是每层都通信。因为 Column 的输出正好就是 Row 的输入格式，中间不需要通信。这个设计让 TP 的通信开销变得可接受。

但 TP 有个硬约束：必须用在节点内，不能跨节点。原因是 TP 每个 transformer layer 都有通信，频率极高。NVLink 的带宽是 600GB/s，跨节点的 InfiniBand 是 100-400GB/s，差了几倍。频率高 × 带宽低，延迟会在整个训练里累积，吞吐量大幅下降。

这个约束当时让我踩了一个坑：在双机实验里启用了 TP，结果比单机还慢。查了很久，才意识到是跨节点 TP 的问题。关掉 TP，用流水线并行（PP）处理跨节点，正常了。

---

### 第五幕：这些底层理解，怎么用回去

xtrain 做完之后，回去继续跑 verl，那些之前排查不了的问题，大部分都能找到方向了。

最典型的是最初的 weight sync 卡住问题。理解了 Actor（训练进程）和 Rollout（vLLM 进程）的通信机制之后，发现问题出在 weight sync 时 tensor 的 sharding 策略不一致——Actor 用 TP 分片了某些层的参数，Rollout 期望的是完整参数，shape 对不上，通信卡住。解法是 sync 时先做 AllGather 把分片参数聚合成完整参数，再发给 Rollout。

这个 bug，如果不理解 TP 的分片机制，光看报错信息根本找不到。

**这就是为什么要从零实现一遍：不是为了在生产里用自己写的代码，而是为了在框架出问题的时候知道该往哪看。**

---

## 技术路径深化（面试追问完整版）

### a. Ring AllReduce 的带宽分析，Tree AllReduce 什么时候更好

**Ring AllReduce 带宽分析：**

设 N 个 GPU，每个 GPU 有 D 大小的数据（梯度），Ring AllReduce 分两个阶段：

1. **ReduceScatter**（N-1 轮）：每个 GPU 把数据分成 N 份，轮流发送/接收/累加，结果是每个 GPU 持有 1/N 的最终 reduce 结果
2. **AllGather**（N-1 轮）：每个 GPU 把自己持有的 1/N 数据广播出去，结果是所有 GPU 都有完整的 reduce 结果

每个阶段，每个 GPU 发送和接收的总数据量都是 `(N-1)/N × D ≈ D`（N 大时趋近 D）。

**总通信量 = 2D（发送 D + 接收 D），与 N 无关。**

这是 Ring AllReduce 的核心优势：带宽利用率不随 GPU 数量降低。对比朴素的 Reduce-Broadcast：

```
朴素方案：
  - 所有 GPU → 主 GPU：(N-1) × D 数据流入主 GPU
  - 主 GPU → 所有 GPU：(N-1) × D 数据流出主 GPU
  - 主 GPU 的带宽是瓶颈，实际每 GPU 有效带宽 = B/N
Ring AllReduce：
  - 每 GPU 有效带宽 ≈ B（全额利用）
```

**Tree AllReduce 什么时候更好：**

Ring AllReduce 的假设是所有 GPU 之间带宽均等。但现实里，节点内 GPU 之间有 NVLink（600GB/s），节点间通过 InfiniBand（100-400GB/s），带宽相差 2-6 倍。

如果直接用 Ring AllReduce 且 Ring 跨了节点，跨节点的那条边会成为瓶颈，整个 Ring 的速度被这条边限制。

Tree AllReduce（更准确说是"分层 AllReduce"）：
1. 先在节点内做 AllReduce（走 NVLink，快）
2. 每节点选一个代表，节点间做 AllReduce（走 InfiniBand）
3. 节点间结果广播回各节点内

NCCL 在检测到多节点时自动做类似的分层处理。**手动选择 Tree vs Ring 的场景**：超大集群（1000+ GPU）下，Ring 的轮次数是 O(N)，延迟累积严重；Tree 的轮次是 O(log N)，延迟更低，但带宽利用率稍差。单机或小规模多机，Ring 更好。

---

### b. ZeRO-3 的通信量化分析：比 DDP 多了多少通信

**DDP 的通信量（基准）：**

DDP 在 backward 结束后做一次 AllReduce，通信量 = 2D（每 GPU 发 D，收 D），与 ZeRO-1 相同。

**ZeRO-3 的额外通信（每次 forward + backward）：**

ZeRO-3 把参数也分片，每个 GPU 只持有 1/N 的参数。因此每次 forward 和 backward 都需要 AllGather 当前层的完整参数：

```
forward：对每个 transformer layer
  AllGather 参数（通信量 = D）
  计算 forward
  释放聚合后的参数（只保留本 GPU 的 1/N 分片）

backward：对每个 transformer layer（逆序）
  AllGather 参数（通信量 = D）  ← 相比 DDP 多出的
  计算梯度
  ReduceScatter 梯度（通信量 = D）
  释放参数分片

optimizer step：本地更新（无通信）
```

**量化对比（7B 模型，bf16，N=8 GPU）：**

| 方案 | 每步总通信量 | 说明 |
|------|-----------|------|
| DDP | ~2D ≈ 28GB | backward 一次 AllReduce |
| ZeRO-1 | ~2D ≈ 28GB | 同 DDP，只省显存不加通信 |
| ZeRO-2 | ~2D ≈ 28GB | 梯度分片用 ReduceScatter 替代 AllReduce，通信量不变 |
| ZeRO-3 | ~4D ≈ 56GB | 额外的 forward AllGather（每层参数聚合一次）|

**ZeRO-3 多了 2D 的通信，但省了显存。什么带宽条件下值得：**

ZeRO-3 多了约 2 倍通信，如果通信带宽足够，overhead 可以被计算 overlap 掩盖（通信和计算同时进行）。经验规则：
- 节点内（NVLink 600GB/s）：几乎总是值得
- 跨节点（InfiniBand 100-400GB/s）：要看通信/计算时间比。当 GPU 算力高（A100/H100）而网络带宽相对不足时，ZeRO-3 的通信会成为瓶颈，不如用 ZeRO-1/2 + 模型并行

---

### c. TP + PP + ZeRO 的组合策略，实际配置里各负责什么

**三者的分工：**

| 技术 | 解决什么 | 通信位置 | 推荐配置 |
|------|---------|---------|---------|
| TP（张量并行） | 单层参数太大，一张 GPU 放不下 | 节点内（NVLink） | TP=8（节点内全部 GPU） |
| PP（流水线并行） | 模型层数太多，层叠放不下 | 跨节点（IB） | PP=节点数 |
| ZeRO | 参数+梯度+优化器状态总量超过显存 | 同 TP 的节点内 | ZeRO-1 或 ZeRO-2（ZeRO-3 与 TP 配合有冲突） |

**实际配置示例（70B 模型，32×A100，4 个节点）：**
```yaml
# 节点内 TP=8（8 个 GPU 切一层参数）
tensor_parallel_size: 8
# 跨节点 PP=4（4 个节点各跑不同的层段）
pipeline_parallel_size: 4  
# ZeRO-1（分片 optimizer state，省显存，通信开销低）
zero_stage: 1
# 数据并行：TP × PP = 32，只剩 1，即 1 个数据并行副本
data_parallel_size: 1
```

**为什么 ZeRO-3 与 TP 配合有冲突：**
ZeRO-3 把参数切成 1/N，TP 也把参数切成 1/T，两套切法叠加在一起，AllGather 时的 reshape 逻辑非常复杂，且 NCCL 通信 group 设置容易冲突。实践中大多数框架（Megatron-LM、verl）在 TP+PP 场景下只用 ZeRO-1/2，不用 ZeRO-3。

---

### d. MoE 的 expert load balancing：auxiliary loss 设计 + collapse 机制

**MoE 路由问题的根本矛盾：**

Router 是可学习的，它的优化目标是"让 token 被路由到能给出最佳输出的 expert"。但如果某个 expert 一开始稍微强一点，router 就会给它更多 token → 它变得更强 → router 给它更多……这是正反馈循环，最终导致 **expert collapse**：1-2 个 expert 处理了 90%+ 的 token，其余 expert 几乎从不被激活（essentially dead experts）。

**Auxiliary loss（负载均衡辅助损失）：**

```python
# 标准 auxiliary loss（来自 Switch Transformer）
def aux_loss(router_probs, expert_load, alpha=0.01):
    # router_probs: [batch, seq, n_experts] — 路由概率
    # expert_load: [n_experts] — 实际每个 expert 被选中的 token 比例
    
    # 目标：所有 expert 的平均路由概率 × 实际负载尽量均匀
    mean_routing = router_probs.mean(dim=[0,1])  # [n_experts]
    
    # 两者点积：如果分布均匀，点积最小（均匀分布下最小）
    loss = n_experts * (mean_routing * expert_load).sum()
    
    return alpha * loss  # alpha 控制辅助损失的权重
```

**alpha 的调法：**
- alpha 太小：auxiliary loss 太弱，collapse 仍然发生
- alpha 太大：强制均匀路由，破坏了"让强的 expert 处理更多"的优化意图，模型质量下降
- 经验范围：0.001 到 0.1，从 0.01 开始调

**DeepSeek 的改进（来自 DeepSeek-V3）：**
DeepSeek 用 "Expert-level Balance Loss" 替代全局 auxiliary loss——不要求全局均匀，允许不同 expert 有专长，但要求每个 expert 不要严重过载（设上限）。同时用 expert-affinity score 让 router 对同类 token 倾向同一组 expert（提高专业化程度）。

**Collapse 发生时的诊断：**
```python
# 监控：每隔 N 步统计 expert 激活分布
expert_activation_rate = (expert_indices == k).float().mean()  # 对每个 k
# 如果某个 expert 的 activation_rate > 0.4，其他 < 0.05 → collapse 信号
# 应对：临时调高 alpha，或者重新初始化 dead experts 的参数
```

---

## 快速技术速查（追问备用）

**"ZeRO-1/2/3 分别解决了什么？"**
ZeRO-1 分片 optimizer state（省最多，因为 Adam 的 state 是参数量的 3 倍）；ZeRO-2 再分片梯度；ZeRO-3 连参数也分片，显存几乎正比缩放，但通信开销最大。

**"TP 的两次通信具体在哪里？"**
Column Parallel Linear 的输出端（一次 AllReduce 或 AllGather），Row Parallel Linear 的输出端（一次 AllReduce）。整个 MLP 只有这两次，因为两层的 sharding 方式正好接续，不需要中间通信。

**"PP 的 bubble 怎么计算和优化？"**
GPipe bubble ratio = (p-1)/(m+p-1)，p 是流水线段数，m 是 micro-batch 数量。m 越大，bubble 越小。1F1B 调度（One Forward One Backward）不减小 bubble ratio，但把显存峰值从 O(p×m) 降到 O(p)，允许用更大的 m。

---

## See Also

- [[Projects/项目故事/P2-后训练大项目-MA-RLHF工程实战]]
- [[Projects/MA-RLHF/xtrain/xtrain-03-ZeRO优化器从零手写]]
- [[Projects/MA-RLHF/xtrain/xtrain-04-张量并行从零手写]]
- [[Projects/MA-RLHF/xtrain/xtrain-05-流水线并行从零手写]]
- [[AI/3-LLM/Infra/分布式训练通信原语-手撕实操]]
