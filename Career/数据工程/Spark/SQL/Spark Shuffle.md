---
title: "Spark Shuffle"
type: concept
domain: engineering/spark/sql
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/sql
  - type/concept
---
# Spark Shuffle

> 参考：
> - https://blog.csdn.net/XMZHSY/article/details/132756068
> - https://developer.aliyun.com/article/927120

## 为什么 Shuffle 是性能瓶颈

Shuffle 是 Spark 中 **跨分区数据重新分布** 的过程。每当涉及 `groupByKey`、`reduceByKey`、`join`、`distinct`、`repartition` 等操作，都会触发 Shuffle。

Shuffle 的代价：
- **磁盘 I/O**：Map 端写中间文件，Reduce 端读取
- **网络传输**：跨节点的数据搬运
- **序列化/反序列化**：数据在传输前后需要 ser/deser
- **内存压力**：排序和聚合需要内存缓冲区

一条经验法则：**Shuffle 数据量每减少 50%，作业耗时大约减少 30-40%**。优化 Spark 作业，首先看 Shuffle。

## Shuffle 演进

### Hash Shuffle（已废弃）

早期实现，每个 Map Task 为每个 Reduce Task 生成一个文件：

```
文件数 = Map Tasks × Reduce Tasks
```

1000 个 Map × 1000 个 Reduce = 100 万个小文件，直接把磁盘打爆。

### Sort Shuffle（默认）

Spark 1.2+ 默认。每个 Map Task 只产生 **一个数据文件 + 一个索引文件**：

```
文件数 = 2 × Map Tasks
```

核心流程：
1. Map Task 将输出写入内存缓冲区（`AppendOnlyMap` 或 `PartitionedPairBuffer`）
2. 缓冲区满时 Spill 到磁盘，按 partition id 排序
3. 最终 Merge 所有 Spill 文件为一个有序文件
4. 同时生成索引文件，记录每个 partition 的偏移量

### Tungsten Sort Shuffle

在 Sort Shuffle 基础上进一步优化：
- 直接操作 **二进制数据**，避免反序列化
- 使用 **Unsafe 内存操作**，减少 GC 压力
- 利用 **cache-friendly** 的排序算法

触发条件：不需要 map-side aggregation + 序列化器支持 relocation + partition 数 < 16777216。

## Shuffle Write 详解

```
Map Task
    ↓
ExternalSorter（内存缓冲 + Spill）
    ↓
Spill File 1, Spill File 2, ...
    ↓
Merge → shuffle_0_0_0.data + shuffle_0_0_0.index
```

关键参数：

```properties
# 每个 Task 的 Shuffle Write 缓冲区
spark.shuffle.file.buffer=32k          # 默认 32KB，建议调到 64-128KB
# Spill 前的内存阈值
spark.shuffle.spill.initialMemoryThreshold=5242880  # 5MB
# 排序使用的内存占 execution memory 的比例
spark.shuffle.sort.bypassMergeThreshold=200  # partition 数低于此值用 bypass 模式
```

**Bypass 模式**：当 partition 数较少且不需要 map-side combine 时，跳过排序，直接按 partition 写文件再合并。适用于简单的 `repartition` 场景。

## Shuffle Read 详解

```
Reduce Task
    ↓
MapOutputTracker 获取数据位置
    ↓
BlockTransferService 拉取数据（本地短路读 / 远程网络）
    ↓
ExternalAppendOnlyMap / ExternalSorter 聚合排序
    ↓
输出 Iterator
```

关键参数：

```properties
# 每个 Reduce Task 同时拉取的数据块数
spark.reducer.maxSizeInFlight=48m      # 默认 48MB
# 拉取失败重试次数
spark.shuffle.io.maxRetries=3
# 重试间隔
spark.shuffle.io.retryWait=5s
```

## 常见 Shuffle 问题与优化

### 1. 数据倾斜

某些 partition 数据量远大于其他 partition，导致个别 Task 执行时间极长：

```sql
-- 诊断：查看 key 分布
SELECT key, count(*) as cnt FROM table GROUP BY key ORDER BY cnt DESC LIMIT 20;

-- 方案1：加盐打散
SELECT split_key, sum(value) FROM (
  SELECT concat(key, '_', floor(rand() * 10)) as split_key, value FROM table
) GROUP BY split_key;

-- 方案2：AQE 自动处理
SET spark.sql.adaptive.skewJoin.enabled=true;
```

### 2. Shuffle 分区数不合理

```properties
# 默认 200，通常需要根据数据量调整
spark.sql.shuffle.partitions=200

# 经验公式：shuffle 数据量 / 128MB
# 10GB shuffle → ~80 partitions
# 1TB shuffle → ~8000 partitions
```

### 3. 减少不必要的 Shuffle

```scala
// 坏：两次 Shuffle
val result = rdd.groupByKey().mapValues(_.sum)

// 好：一次 Shuffle，map-side combine
val result = rdd.reduceByKey(_ + _)

// 更好：如果两个 RDD 已经同分区，cogroup 避免额外 Shuffle
val rdd1 = data1.partitionBy(new HashPartitioner(100))
val rdd2 = data2.partitionBy(new HashPartitioner(100))
val joined = rdd1.join(rdd2)  // 无 Shuffle
```

### 4. External Shuffle Service

```properties
spark.shuffle.service.enabled=true
```

让 NodeManager 托管 Shuffle 数据，而不是 Executor。好处：
- Executor 可以动态回收，Shuffle 数据不丢
- 配合 Dynamic Allocation 使用

## Shuffle 监控

Spark UI 的 Stage 详情页：
- **Shuffle Write**：关注 Records 和 Size
- **Shuffle Read**：关注 Fetch Wait Time（网络瓶颈）和 Records Read
- **Task Duration 分布**：看是否有明显的长尾 Task（数据倾斜）

## 相关

- [[Spark SQL]]
- [[Spark SQL 执行过程]]
- [[数据倾斜优化|数据倾斜优化]]
- [[一次特殊的数据倾斜优化|一次特殊的数据倾斜优化]]
- [[Spark AQE + DDP]]
- [[Spark Partitioner|Spark Partitioner]]
- [[Spark 调优|Spark 调优]]
