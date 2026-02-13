---
title: "1. Spark 存储"
type: concept
domain: engineering/spark/存储
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/存储
  - type/concept
---
# 1. Spark 存储

## 存储体系概览

Spark 的存储体系服务于两个目标：**缓存中间数据加速计算** 和 **Shuffle 中间数据持久化**。理解存储机制是调优的基础。

```
Spark 存储体系
├── Block Manager          ← 核心组件，管理所有数据块
│   ├── MemoryStore        ← 堆内/堆外内存存储
│   ├── DiskStore          ← 本地磁盘存储
│   └── BlockTransferService ← 跨节点数据传输
├── Storage Level          ← persist/cache 策略
└── Broadcast Manager      ← 广播变量管理
```

## BlockManager 架构

BlockManager 是 Spark 存储的核心，每个 Executor 都有一个实例。它负责管理本地内存和磁盘上的数据块（Block）。

```
Driver: BlockManagerMaster
    ↕ RPC
Executor1: BlockManager     Executor2: BlockManager
  ├── MemoryStore             ├── MemoryStore
  ├── DiskStore               ├── DiskStore
  └── BlockTransferService    └── BlockTransferService
         ↕ Netty
```

**数据块类型**：
- `rdd_<rddId>_<partitionId>`：RDD 缓存的分区数据
- `shuffle_<shuffleId>_<mapId>_<reduceId>`：Shuffle 中间数据
- `broadcast_<broadcastId>`：广播变量
- `taskresult_<taskId>`：Task 结果（大于阈值时不走 RPC）

## Storage Level 详解

```scala
// 常用存储级别
MEMORY_ONLY          // 反序列化对象存内存，不够就不存（默认 cache）
MEMORY_ONLY_SER      // 序列化后存内存，省空间但需要反序列化开销
MEMORY_AND_DISK      // 内存放不下的溢出到磁盘
MEMORY_AND_DISK_SER  // 序列化 + 磁盘溢出
DISK_ONLY            // 只存磁盘
OFF_HEAP             // 堆外内存（Tungsten）

// 加 _2 后缀表示 2 副本，如 MEMORY_ONLY_2
```

**选择策略**：

```scala
// 场景1：数据量小于可用内存 → MEMORY_ONLY
val smallDF = spark.read.parquet("small_table").cache()

// 场景2：数据量中等，内存紧张 → MEMORY_ONLY_SER
import org.apache.spark.storage.StorageLevel
val mediumDF = spark.read.parquet("medium_table")
  .persist(StorageLevel.MEMORY_ONLY_SER)

// 场景3：数据量大，重算代价高 → MEMORY_AND_DISK
val expensiveDF = spark.read.parquet("big_table")
  .join(anotherDF, "key")
  .persist(StorageLevel.MEMORY_AND_DISK)
```

实际经验：**大多数生产场景用 `MEMORY_AND_DISK_SER`**。纯内存模式听起来快，但如果内存不够导致频繁重算，总体反而更慢。

## 内存管理（统一内存模型）

Spark 1.6+ 采用统一内存管理（Unified Memory Manager），Execution 和 Storage 共享内存池：

```
Executor JVM Heap
├── Reserved Memory (300MB 固定)
├── User Memory (1 - spark.memory.fraction) × (Heap - Reserved)
│   └── 用户数据结构、UDF 临时对象
└── Unified Memory (spark.memory.fraction × (Heap - Reserved))
    ├── Storage Memory（初始各占 50%，可互相借用）
    │   └── 缓存的 RDD/DataFrame
    └── Execution Memory
        └── Shuffle、Sort、Aggregation 缓冲区
```

关键参数：

```properties
spark.memory.fraction=0.6           # Unified Memory 占比
spark.memory.storageFraction=0.5    # Storage 初始占比
spark.memory.offHeap.enabled=false  # 堆外内存开关
spark.memory.offHeap.size=0         # 堆外内存大小
```

**核心规则**：Execution 可以抢占 Storage 的内存（驱逐缓存），但 Storage 不能抢占正在使用的 Execution 内存。设计原因很简单——Execution 内存被占着说明有 Task 在运行，强制回收会导致 Task 失败。

## 缓存机制深入

### 什么时候该 cache

```scala
// 该 cache：同一个 DataFrame 被多次 action 使用
val filtered = rawDF.filter($"status" === "active").cache()
filtered.count()          // 触发缓存
filtered.groupBy("city").count().show()  // 从缓存读
filtered.write.parquet("output")         // 从缓存读

// 不该 cache：只用一次的 DataFrame
val result = rawDF.filter($"x" > 0).groupBy("y").count()
result.show()  // 只用一次，cache 反而浪费内存
```

### 缓存驱逐

采用 **LRU（Least Recently Used）** 策略，最近最少使用的 Block 最先被驱逐。

需要注意：`unpersist()` 是懒执行的，`unpersist(blocking = true)` 才会立即释放。

```scala
// 用完记得释放
filtered.unpersist(blocking = true)
```

## Broadcast 变量

Broadcast 是一种特殊的存储机制，把 Driver 端的数据广播到所有 Executor：

```scala
val dimMap = spark.sparkContext.broadcast(
  dimDF.collect().map(row => row.getString(0) -> row.getString(1)).toMap
)

// 在 Task 中使用
resultDF = factDF.map(row => {
  val dimValue = dimMap.value.getOrElse(row.getKey, "unknown")
  // ...
})
```

广播过程：
1. Driver 将数据序列化，存入 BlockManager
2. 使用类似 BitTorrent 的 P2P 协议分发（TorrentBroadcastFactory）
3. 每个 Executor 本地缓存一份

**大小限制**：默认上限 `spark.broadcast.blockSize=4m`（每块），总体不超过 Driver 内存。实践中 **广播变量不要超过几百 MB**，太大的话 Driver 序列化时间和内存压力都扛不住。

## 本地磁盘存储

Shuffle 数据和 Spill 数据写入本地磁盘：

```properties
# 配置多块磁盘轮转写入，提升 I/O 吞吐
spark.local.dir=/data1/spark,/data2/spark,/data3/spark
```

生产建议：
- 用 SSD 存储 Shuffle 数据
- 配置多块磁盘做 round-robin
- 监控磁盘使用率，Shuffle 产生的临时文件可能撑爆磁盘

## 相关

- [[Spark 内存模型]]
- [[Engineering/Spark/SQL/Spark Shuffle|Spark Shuffle]]
- [[Engineering/Spark/存储/RDD/Spark RDD|Spark RDD]]
- [[Engineering/Spark/生产运维/调优/Spark 调优|Spark 调优]]
- [[Engineering/Spark/生产运维/小文件问题/小文件问题|小文件问题]]
- [[Engineering/Spark/Spark 概述|Spark 概述]]
