---
title: "3. Flink CheckPoint"
type: concept
domain: engineering/flink/checkpoint
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/checkpoint
  - type/concept
---
# 3. Flink CheckPoint

## 一、状态

Barrier对齐：Exactly Once

Barrier非对齐：At Least Once

### 2.1、对齐 CP

Checkpoint表示实时任务在**某一时刻的全局状态快照**，包含了所有算子的状态，可以理解为状态定时持久化存储。

JM构建ExecutionGraph时，会调用`ExecutionGraph.enableCheckpointing()`，该方法会创建CheckpointCoordinator，同时添加监听器CheckpointCoordinatorDeActivator，当算子状态转为running时，监听器调用`CheckpointCoordinator.startCheckpointScheduler()`部署定时任务ScheduledTrigger，用于周期性的触发cp，触发前检查：

- 当前并发cp数量是否超过阈值。
- cp时间间隔是否达到设置的两次cp间隔。
- 所有算子是否是running状态。
没有问题后，触发cp流程：

1. 生成唯一自增cp id。
1. 初始化CheckpointStorageLocation，存储本次cp的路径。
1. 生成PendingCheckpoint，表示一个处于中间状态的cp，保存在cp id -> PendingCheckpoint map中。
1. 向所有source算子触发cp，调用TM的`triggerCheckpoint()`先获取到要触发cp的算子，再调用`StreamTask.performCheckpoint()`，执行：
1. 向下游所有算子广播CheckpointBarrier（同步执行，执行过程中不能处理数据）
1. 进行状态快照操作，每个算子快照被抽象为OperatorSnapshotFutures：
1. OperatorSnapshotFutures持有RunnableFuture，RunnableFuture中定义算子状态快照操作过程任务FutureTask，快照制作分为同步和异步。
1. 同步阶段，调用算子`StreamOperator.snapshotState()`，将快照操作过程写入到RunnableFuture，但不会立即执行。
1. 异步阶段，调用`RunnableFuture.run()`，创建OperatorSnapshotFinalizer执行快照操作，生成状态快照，并将状态数据写入到文件系统，执行完毕后向CheckpointCoordinator上报cp信息。
1. CheckpointCoordinator调用`PendingCheckpoint.acknowledgeTask()`，将上报的cp元数据（cp路径、状态大小等）添加到PendingCheckpoint。
1. 如果收到了所有算子的上报ack，执行`completePendingCheckpoint()`，将PendingCheckpoint转为CompletedCheckpoint，并将cp元数据持久化，同时删除过期cp。
1. 通知所有算子进行commit操作，一般用于两阶段提交事务。
假设算子有两个上游a、b：

1. 算子先收到输入a的Barrier，将a中的数据放入缓存暂不处理，等待输入b的Barrier到达。
1. 输入b的Barrier到达，算子异步制作（不等待快照执行完毕）快照并报告CheckpointCoordinator，然后将**两个Barrier合并成一个**，向下游所有算子广播。
1. 算子先处理缓存中的积压数据，然后再从输入流中获取数据
**为什么在对齐过程中要阻塞消费？**

为了防止正在处理的数据修改快照内容，影响快照的准确性。

**为什么要进行Barrier对齐？不对齐有什么问题？**

- 为了保证Exactly Once，则必须要保证Barrier对齐，如果不对齐就变成了At Least Once。
- 举例：source在offset=100时下发Barrier，下发到下游task时的状态聚合值count=100，如果下游收到此Barrier后，仍继续消费数据，那么等集齐Barrier之后在制作快照，状态聚合值很可能count>100。如果作业故障，source会从offset=100开始恢复，下游task会从count>100恢复，offset=100之后的消息会被重复消费。
- [Flink--Checkpoint机制原理](https%3A%2F%2Fwww.jianshu.com%2Fp%2F4d31d6cddc99)
每个需要checkpoint的应用在启动时，Flink的JobManager为其创建一个 CheckpointCoordinator，CheckpointCoordinator全权负责本应用的快照制作。

1. JobManager端的 CheckPointCoordinator向所有 SourceTask 周期性发送 CheckPointTrigger，Source Task会在数据流中安插 CheckPoint barrier。
1. 当 task 收到一个barrier时，便暂停数据处理过程，收到所有的barrier后，为了不阻塞下游快照，**先向自己的下游继续传递barrier，然后自身执行快照**，并将自己的状态异步写入到状态后端。增量CheckPoint只是把最新的一部分更新写入到 外部存储；
1. 当task完成备份后，会将备份数据的地址（state handle）通知给JobManager的CheckPointCoordinator；如果CheckPoint 的持续时长超过 了CheckPoint 设定的超时时间，CheckPointCoordinator 还没有收集完所有的 State Handle，CheckPointCoordinator就会认为本次 CheckPoint 失败，会把这次 CheckPoint 产生的所有状态数据全部删除。
1. 当CheckpointCoordinator收到所有算子的报告之后，认为该周期的快照制作成功; 否则，如果在规定的时间内没有收到所有算子的报告，则认为本周期快照制作失败 ;
1. 最后 CheckPoint Coordinator 会把整个 StateHandle 封装成 completed CheckPoint Meta，写入到hdfs。
### 2.2、非对齐 CP

#### 2.2.1、原理

- [Flink Unaligned Checkpoint 在 Shopee 的优化和实践-阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1114464)
- [Flink 1.11 Unaligned Checkpoint 解析-阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F768710)
反压严重时，Aligned Checkpoint（下文简称 AC）超时主要在于 Barrier 在数据流中排队。反压严重时，数据流动很慢导致 Barrier 流动很慢，最终导致 AC 超时。因此可以通过 Unaligned Checkpoint 来进行调优。从语义层面上：

- AC：Exactly Once
- UC：At Least Once
由于我们用的是 Flink 1.12，开启非对齐的ck的作业会有问题。更改并发之后，不可使用平台的CK持久化来从CK恢复作业。因为社区在1.12版本上没有实现改并发后的buffer数据回放问题（需要保证同一个key，hash到同一个并发中），所以在1.12版本直接抛了异常，社区在1.13实现了这个功能。所以没有在生产环境中实际使用UC

[Flink Unaligned Checkpoint 在 Shopee 的优化和实践-阿里云开发者社区](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1114464)

原理：当数据流动很慢时，Barrier 通过某些机制超越数据，从而使得 Barrier 可以快速地从 Source 一路超车到 Sink

流程：

**阶段一 UC 同步阶段**

1. **当某一个 InputChannel 接收到 Barrier 时，会直接开启 UC 同步阶段**
- 只要有任意一个 Barrier 进入 Task 网络层的输入缓冲区，Task 直接开始 UC；
- 不用等其他 InputChannel 接收到 Barrier，也不需要处理完 InputChannel 内 Barrier 之前的数据。
为了保证数据一致性，UC 同步阶段 Task 不能处理数据，同步阶段会做以下几个事情：

- Barrier 超车：发送 Barrier 到所有的 ResultSubPartition 的头部，超越所有的 input&output buffer，Barrier 可以被快速发到下游 Task；为了保证这些 Buffer 不丢失，需要对 buffer 进行快照
- 状态快照：调用算子的 snapshotState 方法，对算子内部的 State 进行快照。
**阶段二 Barrier 对齐**

等其他InputChannel的Barrier到达，对Barrier之前的buffer都需要快照。

**阶段三UC 异步阶段**

**异步阶段将同步阶段浅拷贝的 State 以及 buffer 写到 HDFS 中。**

为什么 UC 还有 Barrier 对齐呢？

当 Task 开始 UC 时，有很多 InputChannel 没接收到 Barrier，这些 InputChannel 的 Barrier 之前可能还会有 network buffer 需要进行快照，所以 UC 第二阶段需要等所有 InputChannel 的 Barrier 都到达，且 Barrier 之前的 buffer 都需要快照。可以认为 UC 需要写三类数据到 HDFS 上：

- 同步阶段引用的所有 input&output buffer；
- 同步阶段引用的算子内部的 State；
- 同步阶段后其他 InputChannel Barrier 之前的 buffer。
**异步阶段把这三部分数据全部写完后，将文件地址汇报给 JobManager，当前 Task 的 UC 结束。**

注：理论上 UC 异步阶段的 Barrier 对齐会很快。如上述 Task 所示，Barrier 可以快速超越所有的 input&output buffer，优先发送 Barrier 给下游 Task，所以上游 Task 也类似：Barrier 超越上游所有的 buffer，快速发送给当前 Task。

#### 2.2.2、额外风险

1. 在很多场景，任务反压严重时，UC 仍然不能成功，导致 UC 预期收益大打折扣；
1. UC 会显著增加写 HDFS 的文件数，造成 Namenode 压力，对线上服务的稳定性有影响，增加了大范围应用的难度；
1. 非对齐是 At Least Once 语义
主要在于 UC 相比 AC 会写 network buffer 到 Checkpoint 中，所以引入了一些额外风险：

- 会写更多的文件到 HDFS，给 NameNode 造成额外压力（默认 Flink 每个 Subtask 为 buffer 写一个文件，假设任务有 10 个 Task，每个 Task 并发为 1000，则 UC 可能会额外写 1 万个小文件。假设 Kafka 集群出现故障或瓶颈，大量 Flink Job 写 Kafka 慢，会导致大量 Flink 任务从 AC 切换成 UC。这种情况大量任务瞬间写数十万的小文件到 HDFS，可能导致 NameNode 雪崩。）；
- 数据的 schema 升级以后，如果序列化不兼容，则数据无法恢复；
- 当算子之间的连接发生变化时，算子之间的 buffer 数据无法恢复（例如：从 rebalance 改为 forward）。
调整完UC后，也需要调整networkbuffer，避免barrier堵塞

使用非对齐的checkpoint需要注意：对于1.12版本，开启非对齐的ck的作业，更改并发之后，不可使用平台的CK持久化来从CK恢复作业。因为社区在1.12版本上没有实现改并发后的buffer数据回放问题（需要保证同一个key，hash到同一个并发中），所以在1.12版本直接抛了异常，社区在1.13实现了这个功能。[https://issues.apache.org/jira/browse/FLINK-17979](https%3A%2F%2Fissues.apache.org%2Fjira%2Fbrowse%2FFLINK-17979) 短期解决办法：1，通过AB部署启动作业；2，手动触发Savepoint恢复作业。 未来解决办法（23年H2）：平台会将Flink1.12升级到Flink1.16，就可以正常通过CK持久化功能正常恢复作业。

**优化：**[https://developer.aliyun.com/article/1114464](https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1114464)

- Legacy Source 的提升：source端也会检查buffer
- 处理一条数据需要多个 buffer 场景的提升：透支buffer机制
- InputChannel 支持从 AC 切换为 UC：AC timeout，避免AC超时，CP制作失败
- Output buffer 支持从 AC 切换为 UC：避免barrier在Output buffer中排队而无法发送到下游task的InputChannel中
#### 2.2.3、优化

**预留 buffer 和透支 buffer 机制**

Task 在处理数据的过程中不能处理 Checkpoint，必须将当前处理的这条数据处理完 并将结果写入到 OutputBufferPool 以后，才会检查是否 InputChannel 有接收到 UC Barrier，如果有则开始 UC。

如果 Task 处理一条数据并写结果到 OutputBufferPool 超过 10 分钟，那么 UC 还是会超时。通常处理一条数据不会很慢，但写结果到 OutputBufferPool 可能会比较耗时。

从 OutputBufferPool 的视角来看，上游 Task 是生产者，下游 Task 是消费者。所以下游 Task 有瓶颈时，上游 Task 输出结果到 OutputBufferPool 会卡在等待 buffer，不能开始 UC。

等 OutputBufferPool 中有空闲 buffer 了才去处理数据，来保证 Task 处理完数据后可以顺利地将结果写入到 OutputBufferPool 中，不会卡在第 5 步数据输出的环节。优化后如果没有空闲 buffer，Task 会卡在第 3 步等待空闲 buffer 和 UC Barrier 的环节，在这个环节当接收到 UC Barrier 时可以快速开始 UC。

由于只预留了一个 buffer，当处理一条数据需要多个 buffer 的场景，Task 处理完数据输出结果到 OutputBufferPool 时可能仍然会卡在第 5 步，导致 Task 不能处理 UC。

- 例如：单条数据较大、flatmap、window 触发以及广播 watermark 都是处理一条数据需要多个 buffer 场景，这些场景下 Task 卡在第 5 步数据输出环节，导致 UC 表现不佳。解决这个问题的核心思路还是如何让 Task 不要卡在第 5 步而是卡在第 3 步的等待环节。
-  基于上述问题，Shopee 在 [FLIP-227](https%3A%2F%2Fcwiki.apache.org%2Fconfluence%2Fdisplay%2FFLINK%2FFLIP-227%253A%2BSupport%2Boverdraft%2Bbuffer) 提出了 overdraft（透支） buffer 的提议，思路是：处理数据过程中，如果 buffer 不足且 TaskManager 有空余的 network 内存，则当前 Task 的 OutputBufferPool 会向 TM 透支一些 buffer，从而完成第 5 步数据处理环节。
- 注：OutputBufferPool 一定是在没有空闲 buffer 时才会使用透支 buffer。所以一旦透支 buffer 被使用，Task 在进行下一轮第 3 步进入等待 Barrier 和空闲 buffer 的环节时，Task 会认为 OutputBufferPool 没有空闲 buffer，直到所有透支 buffer 都被下游 Task 消费完且 OutputBufferPool 至少有一个空闲 buffer 时，Task 才会继续处理数据。
-  默认 taskmanager.network.memory.max-overdraft-buffers-per-gate=5，即：Task 的每个 OutputBufferPool 可以向 TM 透支 5 个 buffer。引入透支 buffer 机制后，当 TM network 内存足够时，如果处理一条数据需要 5 个 buffer，则 UC 完全不会卡住。如果 TM 的 network 内存比较多，可以调大参数兼容更多的场景。
-  Flink-1.16 开始支持透支 buffer 的功能，涉及到的 JIRA 有：[FLINK-27522](https%3A%2F%2Fissues.apache.org%2Fjira%2Fbrowse%2FFLINK-27522)、[FLINK-26762](https%3A%2F%2Fissues.apache.org%2Fjira%2Fbrowse%2FFLINK-26762)、[FLINK-27789](https%3A%2F%2Fissues.apache.org%2Fjira%2Fbrowse%2FFLINK-27789)。
### 2.3、SavePoint

sp是任务的全局快照，其底层使用代码跟cp一致，可以看做是特殊的cp。使用sp最好为每个算子分配uid，只要算子uid不变就能从sp恢复。

默认情况下，从sp恢复时会尝试将所有状态分配给新作业，如果删除了算子，可以通过`allowNonRestoredState`跳过，否则无法从sp恢复；如果添加了新算子，那么它将会在没有任何状态的情况下初始化。

与cp区别：

- cp侧重容错，用于意外失败重启后快速恢复；sp侧重维护，用于程序修改、版本升级或作业迁移后恢复。
- cp生命周期由Flink管理，无需用户干预，用户终止作业后，会自动删除cp（除非明确配置为保留的cp）；sp根据用户需要触发和清理。
- cp创建轻量级，可以从cp快速恢复；sp以二进制形式存储所有状态和元数据，执行起来慢且开销大，但是保证了可移植性。
- cp支持增量，对于大状态作业可以降低写入成本；sp不支持增量。
## 三、状态后端

**状态后端 StateBackend**

可以存在堆内存或堆外内存，也可以第三方，Flink 提供三种：

- **MemorySB**：运行所需的**状态都在 TM 堆内存中**，执行 CP 会将状态保存到 **JM 进程内存**中，执行 SavePoint 会**从 JM 同步到远端。**
- 适用：本地调试
- 每个 State 默认 5MB,可通过 MemoryStateBackend 构造函数调整
- 状态大小不能超过 akka 的 Framesize 大小
- 不能超过 JM 内存。
- 代码中指定大小、**关闭异步快照机制**
- **FsSB**：运行所需的**状态都在 TM 堆内存中**，CP 时将快照**写入到配置好的系统目录**，**少量元数据信息**写入 JM 内存中，从 TM 异步同步到远端。
- 适用：消状态、短窗口、或小键值状态的有状态处理任务。
- 不能超过 TM 内存。
- 代码中显示指定路径
- **RocksDBSB**：将状态数据保存在 RocksDB，RocksDB 数据库默认将数据存在 TM 运行节点的数据目录下，CP 时将整个 RocksDB 保存的 state 数据**全量或增量**持久化到远端
- 适合大状态、长窗口、大键值状态的有状态处理任务，比如计算 DAU 这种大数据量去重。
- 可以存储远超 fs 的状态，避免 OOM，但吞吐量会下降（**由于存在磁盘，序列化**）
- 唯一支持增量快照的 SB
状态后端选择：状态大——RKDB，状态不大——fs

三种状态后端和两种状态的对应关系：

![image](assets/JulIdybfQouxkkxnSyJcOGznnHh.png)

## 5、状态

### 5.1、状态分类

状态用于保存中间计算结果或缓存数据，分为keyedState和operatorState：

- keyedState：
- 只能用于keyedStream，状态跟key绑定。
- 支持ListState、MapSate、ValueSate等。
- operatorState：
- 用于所有算子任务，每个算子子任务共享一个状态，算子子任务之间的状态不能相互访问。
- 只支持ListState、Union ListState（发生故障时可以从sp恢复）和BroadcastState。
- 广播状态（BroadcastState），来自一个流的数据需要被广播到下游所有任务，在下游算子内存存储，处理另一个流时依赖广播状态，比如规则流和事件流。
### 5.2、状态存储

![image](assets/P2prdMmI4orZfqxAudncXUB4nlr.png)

状态存储在StateBackend上，作用：

- 计算过程中提供访问状态的能力。
- 将状态持久化到外部存储，提高容错。
三种存储方式：

- MemoryStateBackend：
- 运行时所需状态保存在TM内存，执行cp时，会把状态快照保存在JM的内存中。
- 适合测试，不推荐生产使用。
- FSStateBackend：
- 运行时所需状态保存在TM内存，执行cp时，会把状态快照保存在配置的文件系统中，可以是分布式或本地文件系统。
- 适合处理长周期、大状态任务。
- RocksDBStateBackend：
- 使用嵌入式本地数据库RocksDB，将状态存储在堆外内存和本地磁盘中，不受限于TM内存大小，执行cp时，会把RocksDB中存储的状态全量或增量持久化到配置的文件系统中。
- 相比基于内存的StateBackend，访问性能下降（序列化存储），但支持增量cp，且不受内存限制以及JVM垃圾回收影响。
- 从1.10开始，Flink默认将RocksDB的内存大小配置为每个task slot的托管内存。
- 适合处理长周期、大状态任务。
状态过期：使用状态清理策略StateTtlConfig设置。

- 过期时间：超过多长时间未访问，则视为状态过期。
- 过期时间更新策略：
- Disabled：不更新。
- OnCreateAndWrite：状态创建或每次写入时更新。
- OnReadAndWrite：状态创建、写入、读取均会更新。
- 过期状态处理策略（已过期但还未清理的状态）：
- ReturnExpiredIfNotCleanedUp：即使状态过期，但如果还未被清理，返回调用方。
- NeverReturnExpired：一旦状态过期， 就不会返回。
**状态-后端-CP 关系**：状态是数据，状态后端决定以什么数据结构和存储方式来管理状态，CP 提供定时将状态后端中的状态同步到远端存储系统的能力。

**Flink 四种状态数据结构：ValueState、MapState、AppendingState、ReadOnlyBroadcastState，其中 Appending 还分为 Reducing、Aggregate 和 List**

1. 访问状态：使用 StateDescriptor 方法专门访问
1. 比如，通过 getruntimecontext（）.getstate(descriptor)来获取状态句柄，真正访问就放在 map 里面，也可以设置 TTL 什么的