---
title: "Spark 运行"
type: concept
domain: engineering/spark/生产运维/运行
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/生产运维/运行
  - type/concept
---
# Spark 运行

## 作业运行全流程

理解 Spark 作业从提交到完成的全链路，是排查线上问题的基础。

```
spark-submit
    ↓
Driver 启动（ApplicationMaster / Client）
    ↓
SparkContext 初始化
    ↓
DAGScheduler 构建 DAG → 切分 Stage
    ↓
TaskScheduler 调度 Task → 分配到 Executor
    ↓
Executor 执行 Task
    ↓
结果返回 / 写入存储
```

### Stage 划分原理

Stage 的边界由 **Shuffle 依赖（宽依赖）** 决定。DAGScheduler 从最终的 RDD 反向遍历，遇到 ShuffleDependency 就切一刀：

```
RDD A → map → RDD B → groupByKey → RDD C → map → RDD D → join → RDD E
                       ^                          ^
                    Stage 边界                  Stage 边界

Stage 1: A → map → B (ShuffleMapStage)
Stage 2: B → groupByKey → C → map → D (ShuffleMapStage)
Stage 3: D + other → join → E (ResultStage)
```

Stage 内的 Task 可以 **pipeline 执行**（无需物化中间结果），Stage 之间必须等上游完成（需要 Shuffle 数据）。

### Task 调度策略

```properties
# FIFO（默认）：先来先服务
spark.scheduler.mode=FIFO

# FAIR：公平调度，适合多用户共享集群
spark.scheduler.mode=FAIR
```

Task 分配遵循数据本地性优先级：

```
PROCESS_LOCAL  → 数据在同一个 Executor 的内存中
NODE_LOCAL     → 数据在同一节点的磁盘上
RACK_LOCAL     → 数据在同一机架的其他节点上
ANY            → 任意节点
```

```properties
# 本地性等待时间（默认 3s）
spark.locality.wait=3s
spark.locality.wait.node=3s
spark.locality.wait.rack=3s
# 设为 0 跳过本地性等待，适合计算密集型作业
```

## 运行模式

### Client 模式

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --class com.example.MyApp \
  my-app.jar
```

Driver 在提交机器上运行。适合 **交互式调试**，日志直接输出到终端。

### Cluster 模式

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --class com.example.MyApp \
  --num-executors 20 \
  --executor-cores 4 \
  --executor-memory 8g \
  --driver-memory 4g \
  --conf spark.yarn.maxAppAttempts=2 \
  my-app.jar
```

Driver 在 YARN ApplicationMaster 中运行。生产必须用 Cluster 模式——提交机器挂了不影响作业。

### Dynamic Allocation

```properties
spark.dynamicAllocation.enabled=true
spark.dynamicAllocation.minExecutors=5
spark.dynamicAllocation.maxExecutors=100
spark.dynamicAllocation.executorIdleTimeout=60s
spark.dynamicAllocation.schedulerBacklogTimeout=1s
spark.shuffle.service.enabled=true  # 必须配合 External Shuffle Service
```

动态分配让空闲 Executor 自动释放，忙时自动申请。配合 External Shuffle Service 使用，否则 Executor 被回收后 Shuffle 数据就丢了。

## 运行时监控

### Spark UI 关键页面

1. **Jobs 页**：整体作业进度，是否有失败重试
2. **Stages 页**：每个 Stage 的 Task 执行时间分布
   - 看 Duration 的 25th/50th/75th/Max → 判断是否倾斜
   - 看 Shuffle Read/Write Size → 判断 Shuffle 开销
3. **Storage 页**：缓存的 RDD 占用空间
4. **Executors 页**：各 Executor 的内存、GC 时间、Task 完成数
5. **SQL 页**：查询执行计划，每个算子的 metrics

### 事件日志

```properties
spark.eventLog.enabled=true
spark.eventLog.dir=hdfs:///spark-events
spark.eventLog.compress=true

# 通过 History Server 查看已完成作业
spark.history.fs.logDirectory=hdfs:///spark-events
```

### 关键 metrics 告警

```
# 必须告警的指标
- 作业失败/重试
- Stage 长尾 Task（max duration > 5× median）
- GC 时间 > 总执行时间 10%
- Executor OOM / Lost
- Shuffle Fetch Failed
```

## 常见运行问题

### Executor Lost

```
ExecutorLostFailure (executor 5 exited caused by one of the running tasks) 
Reason: Container killed by YARN for exceeding memory limits. 
16.2 GB of 16 GB physical memory used.
```

原因：堆外内存（native memory、Netty buffer）超出 YARN Container 限制。

```properties
# 解决：增加 overhead
spark.executor.memoryOverhead=4g  # 默认是 max(384MB, 0.1 × executor.memory)
```

### Fetch Failed

```
FetchFailedException: Unable to fetch Block reduce_0_1_0
```

原因：Shuffle 数据丢失或 Executor 挂了。通常伴随 Executor Lost。

```properties
# 增加重试
spark.shuffle.io.maxRetries=10
spark.shuffle.io.retryWait=10s
# 开启 Shuffle Service
spark.shuffle.service.enabled=true
```

### Task 超时

```properties
# Task 心跳超时（默认 120s）
spark.task.maxFailures=4
spark.network.timeout=600s
spark.executor.heartbeatInterval=20s
```

## 推测执行

```properties
spark.speculation=true
spark.speculation.interval=100ms
spark.speculation.quantile=0.75       # 75% Task 完成后开始推测
spark.speculation.multiplier=1.5      # 比中位数慢 1.5 倍就推测
```

推测执行对长尾 Task 有效，但在写入有副作用的场景（写数据库、写文件）要小心——可能导致重复写入。

## 相关

- [[Career/数据工程/Spark/Spark 概述|Spark 概述]]
- [[Career/数据工程/Spark/生产运维/运行/Spark 提交流程]]
- [[Career/数据工程/Spark/生产运维/运行/Spark on Yarn]]
- [[Career/数据工程/Spark/生产运维/Spark 生产运维|Spark 生产运维]]
- [[Career/数据工程/Spark/存储/Spark 内存模型|Spark 内存模型]]
- [[Career/数据工程/Spark/生产运维/调优/Spark 调优|Spark 调优]]
- [[Career/数据工程/Spark/SQL/Spark Shuffle|Spark Shuffle]]
