---
title: "6. Flink 运行"
type: concept
domain: engineering/flink/运行
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/运行
  - type/concept
---
# 6. Flink 运行

### 1.5、执行流程

1. 用户提交Flink Application。
1. Optimizer和Graph Builder解析代码，生成StreamGraph，提交给客户端。
1. StreamGraph在客户端转为JobGraph，客户端提交作业到Yarn，包括jars包和JobGraph等。
1. YarnRM分配Container启动AM，AM中启动JM和FlinkRM，并将作业提交给JobMaster。
1. JobMaster向FlinkRM申请资源slots。
1. FlinkRM向YarnRM请求Container启动TM。
1. TM启动后向FlinkRM注册自己可用的slot数量，并与JobMaster通信。
1. JobMaster将任务分发给TM，执行任务。
### 1.6、Flink on Yarn

分为3种模式：session、per-job和application

session：

- 预分配资源，根据配置初始化一个集群，拥有一个JM和固定数量的TM。
- 所有提交的作业可以直接运行（资源用满后需要等待资源释放），共享JM和TM。
- 节省作业提交资源开销，但所有作业之间竞争资源，如果一个TM宕机，它上面运行的所有作业都会失败；作业越多，JM负载越大。
- 适合部署运行时间短的作业。
per-job：

- 每个提交的作业会形成单独的集群，拥有专属JM和TM，并在各自作业完成后销毁。
- 好处是资源隔离，一个作业的TM失败不会影响其他作业，JM负载分散。但每个作业维护一个集群，启动、销毁以及资源请求耗时长。
- 适合部署长时间运行的作业。
- 1.15版本废弃。
Deployer代表向Yarn发起部署请求的节点，该节点作为所有作业的提交入口（客户端）。在`main()`开始执行到`env.execute()`之前，客户端工作量：

- 获取作业依赖。
- JobGraph解析。
- 将依赖项和JobGraph上传到集群。
如果所有用户都在同一个客户端提交作业，较大的依赖会消耗更多带宽；复杂的作业逻辑翻译成JobGraph也会占用更多cpu和内存，客户端会成为瓶颈。

application：

![image](TIildI1mVodBB8xDaU9c63bBn6d.png)

- 为每个applicantion创建独立集群，允许每个application中包含多个作业提交，当应用执行完成后关闭集群。
- `main()`方法在JM中执行，降低客户端压力。
### 1.7、运行状态

1. 首先为创建状态（created），然后切换到运行状态（running），完成所有工作后，切换到完成状态（finished）。
1. 失败的情况下，切换到失败状态（failed），取消所有正在运行的作业。
1. 如果作业可以重新启动，完成重启进入创建状态（created）。
1. 用户取消作业后进入取消状态（canceled）。
### 1.8、并行度设置

4个层面设置并行度：

- 操作算子（Operator Level）：算子.setParallelism(3)。
- 执行环境（Execution Environment Level）：构建Flink环境时，getExecutionEnvironment.setParallelism(1)。
- 客户端（Client Level）：提交Flink run -p时设置。
- 系统（System Level）：客户端flink-conf.yaml文件中parallelism.default配置项设置。
优先级：操作算子>执行环境>客户端>系统

### 1.9、序列化/反序列化

使用TypeInformation接口调用`createSerialize()`方法，创建TypeSerializer提供序列化和反序列化能力。

Flink会将每条记录以序列化的形式存储在一个或多个MemorySegment中。

- MemorySegment是一个固定长度的内存，默认32KB，是Flink最小内存分配单元。
- 借助MemorySegment可以直接基于二进制数据进行比较操作，避免反复序列化和反序列化。