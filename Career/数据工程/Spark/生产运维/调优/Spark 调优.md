---
title: "Spark 调优"
type: concept
domain: engineering/spark/生产运维/调优
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/生产运维/调优
  - type/concept
---
# Spark 调优

### 内存调优

如果直接 OOM，调大内存可以解决，如果没有 OOM，但是有比如heartbeat timeout 等这类报错，也可能是内存导致，具体可能需要看日志

- Executor 内存
- 堆内存
现象：Exit status: 143、OOM、heartbeat timeout

- 如果executor-cores 比较大（>2）, 首先减小executor-cores
- 如果executor-cores 比较小，增大spark.executor.memory
- 堆外内存
现象：Consider boosting spark.yarn.executor.memoryOverhead

- 如果executor-cores 比较大（>2）, 首先减小executor-cores
- 如果executor-cores 比较小，增大 spark.yarn.executor.memoryOverhead
- Diver 内存不足
现象：

- WARN Client: Fail to get RpcResponse: Timeout
- timed out. Failing the application.
- Diver 压力大，不能及时响应 Executor 信息
- Executor heartbeat timed out 
- Exit status: 56
- Exit status: 1
1. 怀疑大表被广播，检查一下是否有将**spark.sql.autoBroadcastJoinThreshold**设置比较大，或者对大表加了广播的hint（mapjoin、broadcast）
1. SQL执行图中搜索如果有 **BroadcastNestedLoopJoin **说明有非等值join（join关联条件是or，<、>等，或忘记写on条件）
1. 检查是否向Driver collect大量数据
1. 增大spark.driver.memory