---
title: "Flink"
type: reference
domain: engineering/flink
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink
  - type/reference
---
# Flink

## 一、架构

在Flink中，Container、TaskManager和Slot关系如下：

- **Container**：在分布式环境下，Flink作业会运行在容器化的环境中，比如Docker容器或者Kubernetes Pod。Container是一个独立的执行环境，可以包含一个或多个TaskManager。
- **TaskManager**：TaskManager是Flink作业的执行引擎，负责执行任务和管理任务的资源。一个TaskManager可以运行在一个独立的进程中，也可以运行在一个容器中。一个Container可以包含一个或多个TaskManager。
- **Slot**：Slot是TaskManager中的资源单位，用于执行任务。每个TaskManager可以有一个或多个Slot。一个Slot可以执行一个或多个任务，具体取决于作业的并行度和资源配置。
**在一个Container中，可以有多个TaskManager，每个TaskManager可以有多个Slot。**作业的并行度决定了需要多少个Slot来执行任务，而资源配置决定了每个TaskManager可以提供多少个Slot。

通过合理配置Container的数量、TaskManager的数量和每个TaskManager的Slot数量，可以有效地管理和分配作业的资源，提高作业的执行效率和吞吐量。

### 1.1、概念

Flink是一个分布式计算引擎，用于对无界和有界流进行状态计算，提供了数据分布、容错机制及资源管理等功能，提供了高抽象层API：

- DataSet API：对静态数据进行批处理，支持Java、Scala和Python。
- DataStream API：对数据流进行流处理，支持Java和Scala。
- Table API：通过类SQL语言处理结构化数据，支持Java和Scala。
### 1.2、架构

JobManager：负责决定何时调度task，对执行完成或失败的task做出反应，协调Checkpoint、故障恢复。由3个组件组成：

- ResourceManager：负责资源申请和释放，管理slot。实现多种RM以适配不同资源管理框架，如Yarn、K8s或Standalone。
- Dispatcher：提供提交作业所需的rest接口，为每个提交的作业启动JobMaster，运行web ui提供执行信息。
- JobMaster：管理单个JobGraph的执行，每个作业都有自己的JobMaster。
- 至少有一个JM，HA中有多个JM，一个是Leader其他都是Standby。
TaskManager：执行任务。

### 1.3、slot

![image](Q6D0dbngKoblAhxMWy5cE6LDn9W.png)

每个TM是一个JVM进程，内部多个线程执行多个任务，通过slot控制一个TM能接收多少个任务。每个task slot代表TM中一部分固定资源，不同任务之间不会进行资源抢占。

![image](ZXH4dvxbIoa7UJxme7zcQA0DnCg.png)

Flink支持slot共享，把不同任务根据依赖关系分配到同一个slot中，好处：

- 方便统计最大资源配置。
- 避免slot频繁申请与释放，提高slot使用率。
### 1.4、执行图

生成顺序：StreamGraph -> JobGraph -> ExecutionGraph -> 物理执行图

StreamGraph：StreamAPI生成的流图，包含算子信息。

![image](BPLOdlta0oJkmnx7bGLchQL1nPe.png)

JobGraph：由StreamAPI优化生成，通过OperationChain机制将算子合并起来，在执行时，调度在同一个线程上，避免跨线程网络传输、减少序列化/反序列化开销，减少延迟提高吞吐。

![image](WLK4djvNYopmnfxBZM0ckveCnZg.png)

ExecutionGraph：

- 由JobGraph转化，包含了作业中所有并行执行的Task信息、Task之间的关联关系、数据流转关系。
- StreamGraph和JobGraph都在客户端生成，ExecutionGraph在JM生成。
![image](PF3XdqqCCoCnSWx9jjFcH8BDn9c.png)

Flink vs Spark？

1. Spark 只针对 Driver 的故障恢复做了数据和元数据的 checkpoint，而 flink 采用轻量级快照，实现了对每个算子及数据流的快照。