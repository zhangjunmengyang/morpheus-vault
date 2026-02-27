---
title: P1：xtrain——分布式训练基础设施，从零实现
type: project-story
status: active
date: 2026-02-28
updated: 2026-02-28
tags: [career, interview, pre-training, distributed-training, TP, PP, ZeRO]
brief: 系统从零实现分布式训练全栈（通信原语→ZeRO→TP→PP→MoE），不是调库，是真正理解每一层。故事线：在后训练实验里反复被分布式问题卡住，倒逼自己把底层搞清楚。
related:
  - "[[AI/3-LLM/Infra/ZeRO-手撕实操]]"
  - "[[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]]"
  - "[[AI/3-LLM/Infra/Pipeline-Parallel-手撕实操]]"
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
- [[AI/3-LLM/Infra/ZeRO-手撕实操]]
- [[AI/3-LLM/Infra/Tensor-Parallel-手撕实操]]
- [[AI/3-LLM/Infra/Pipeline-Parallel-手撕实操]]
- [[AI/3-LLM/Infra/分布式训练通信原语-手撕实操]]
