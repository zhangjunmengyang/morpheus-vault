---
title: "Spark CheckPoint"
category: "工程"
tags: [Checkpoint, OCR, RDD, Spark, YARN]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark CheckPoint

Checkpoint机制

通过上述分析可以看出在以下两种情况下，RDD需要加检查点。

DAG中的Lineage过长，如果重算，则开销太大（如在PageRank中）。

在宽依赖上做Checkpoint获得的收益更大。

由于RDD是只读的，所以Spark的RDD计算中一致性不是主要关心的内容，内存相对容易管理，这也是设计者很有远见的地方，这样减少了框架的复杂性，提升了性能和可扩展性，为以后上层框架的丰富奠定了强有力的基础。

在RDD计算中，通过检查点机制进行容错，传统做检查点有两种方式：通过冗余数据和日志记录更新操作。在RDD中的doCheckPoint方法相当于通过冗余数据来缓存数据，而之前介绍的血统就是通过相当粗粒度的记录更新操作来实现容错的。

检查点（本质是通过将RDD写入Disk做检查点）是为了通过lineage做容错的辅助，lineage过长会造成容错成本过高，这样就不如在中间阶段做检查点容错，如果之后有节点出现问题而丢失分区，从做检查点的RDD开始重做Lineage，就会减少开销。

一个 Streaming Application 往往需要7*24不间断的跑，所以需要有抵御意外的能力（比如机器或者系统挂掉，JVM crash等）。为了让这成为可能，Spark Streaming需要 checkpoint 足够多信息至一个具有容错设计的存储系统才能让 Application 从失败中恢复。Spark Streaming 会 checkpoint 两种类型的数据。

Metadata（元数据） checkpointing - 保存定义了 Streaming 计算逻辑至类似 HDFS 的支持容错的存储系统。用来恢复 driver，元数据包括：

配置 - 用于创建该 streaming application 的所有配置

DStream 操作 - DStream 一些列的操作

未完成的 batches - 那些提交了 job 但尚未执行或未完成的 batches

Data checkpointing - 保存已生成的RDDs至可靠的存储。这在某些 stateful 转换中是需要的，在这种转换中，生成 RDD 需要依赖前面的 batches，会导致依赖链随着时间而变长。为了避免这种没有尽头的变长，要定期将中间生成的 RDDs 保存到可靠存储来切断依赖链

具体来说，metadata checkpointing主要还是从drvier失败中恢复，而Data Checkpoing用于对有状态的transformation操作进行checkpointing

Checkpointing具体的使用方式时通过下列方法：

streamingContext.checkpoint(checkpointDirectory)

什么时候需要启用 checkpoint？

什么时候该启用 checkpoint 呢？满足以下任一条件：

使用了 stateful 转换 - 如果 application 中使用了updateStateByKey或reduceByKeyAndWindow等 stateful 操作，必须提供 checkpoint 目录来允许定时的 RDD checkpoint

希望能从意外中恢复 driver

如果 streaming app 没有 stateful 操作，也允许 driver 挂掉后再次重启的进度丢失，就没有启用 checkpoint的必要了。

如何使用 checkpoint？

启用 checkpoint，需要设置一个支持容错 的、可靠的文件系统（如 HDFS、s3 等）目录来保存 checkpoint 数据。通过调用 streamingContext.checkpoint(checkpointDirectory) 来完成。另外，如果你想让你的 application 能从 driver 失败中恢复，你的 application 要满足：

若 application 为首次重启，将创建一个新的 StreamContext 实例

如果 application 是从失败中重启，将会从 checkpoint 目录导入 checkpoint 数据来重新创建 StreamingContext 实例

通过 StreamingContext.getOrCreate 可以达到目的：

def functionToCreateContext(): StreamingContext = {

    val ssc = new StreamingContext(...)    val lines = ssc.socketTextStream(...)

    ...

    ssc.checkpoint(checkpointDirectory)    ssc

}

val context = StreamingContext.getOrCreate(checkpointDirectory, functionToCreateContext _)

context. ...

context.start()

context.awaitTermination()

如果 checkpointDirectory 存在，那么 context 将导入 checkpoint 数据。如果目录不存在，函数 functionToCreateContext 将被调用并创建新的 context

除调用 getOrCreate 外，还需要你的集群模式支持 driver 挂掉之后重启之。例如，在 yarn 模式下，driver 是运行在 ApplicationMaster 中，若 ApplicationMaster 挂掉，yarn 会自动在另一个节点上启动一个新的 ApplicationMaster。

需要注意的是，随着 streaming application 的持续运行，checkpoint 数据占用的存储空间会不断变大。因此，需要小心设置checkpoint 的时间间隔。设置得越小，checkpoint 次数会越多，占用空间会越大；如果设置越大，会导致恢复时丢失的数据和进度越多。一般推荐设置为 batch duration 的5~10倍。

导出 checkpoint 数据

上文提到，checkpoint 数据会定时导出到可靠的存储系统，那么

在什么时机进行 checkpoint

checkpoint 的形式是怎么样的

checkpoint 的时机

在 Spark Streaming 中，JobGenerator 用于生成每个 batch 对应的 jobs，它有一个定时器，定时器的周期即初始化 StreamingContext 时设置的 batchDuration。这个周期一到，JobGenerator 将调用generateJobs方法来生成并提交 jobs，这之后调用 doCheckpoint 方法来进行 checkpoint。doCheckpoint 方法中，会判断当前时间与 streaming application start 的时间之差是否是 checkpoint duration 的倍数，只有在是的情况下才进行 checkpoint。

checkpoint 的形式

最终 checkpoint 的形式是将类 Checkpoint的实例序列化后写入外部存储，值得一提的是，有专门的一条线程来做将序列化后的 checkpoint 写入外部存储。类 Checkpoint 包含以下数据：

除了 Checkpoint 类，还有 CheckpointWriter 类用来导出 checkpoint，CheckpointReader 用来导入 checkpoint

Checkpoint 的局限

Spark Streaming 的 checkpoint 机制看起来很美好，却有一个硬伤。上文提到最终刷到外部存储的是类 Checkpoint 对象序列化后的数据。那么在 Spark Streaming application 重新编译后，再去反序列化 checkpoint 数据就会失败。这个时候就必须新建 StreamingContext。

针对这种情况，在我们结合 Spark Streaming + kafka 的应用中，我们自行维护了消费的 offsets，这样一来及时重新编译 application，还是可以从需要的 offsets 来消费数据，这里只是举个例子，不详细展开了。
