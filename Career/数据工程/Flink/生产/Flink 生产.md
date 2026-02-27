---
title: "1. Flink 生产"
type: concept
domain: engineering/flink/生产
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/生产
  - type/concept
---
# 1. Flink 生产

## 生产部署的核心关注点

Flink 作业从开发到生产，最大的认知差距在于：开发时关注 **逻辑正确性**，生产时关注 **稳定性和可恢复性**。一个跑通了单元测试的作业，离能在生产上 7×24 运行，还差十万八千里。

生产环境的核心指标：
- **稳定性**：不能隔三差五挂掉重启
- **数据正确性**：Exactly-Once 或 At-Least-Once 语义保证
- **可恢复性**：挂了之后能从 Checkpoint 恢复，不丢数据
- **可观测性**：出问题能快速定位

## 部署模式选择

```
Standalone      ← 测试环境
YARN Session    ← 多作业共享集群，资源利用率高
YARN Per-Job    ← 已废弃（Flink 1.15+）
YARN Application ← 推荐，作业隔离性好
K8s Native      ← 云原生趋势
```

生产建议用 **YARN Application 模式**，每个作业独立的 ApplicationMaster，互不干扰。Session 模式虽然省资源，但一个作业 OOM 可能拖垮整个 Session。

## Checkpoint 配置（生产级）

```java
env.enableCheckpointing(60000); // 1 分钟一次
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000); // 两次 CP 间最少 30s
env.getCheckpointConfig().setCheckpointTimeout(600000); // 10 分钟超时
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);
env.getCheckpointConfig().setTolerableCheckpointFailureNumber(3); // 容忍 3 次失败

// 取消作业时保留 Checkpoint
env.getCheckpointConfig().setExternalizedCheckpointCleanup(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);

// State Backend
env.setStateBackend(new EmbeddedRocksDBStateBackend(true)); // 增量 CP
env.getCheckpointConfig().setCheckpointStorage("hdfs:///flink/checkpoints");
```

**关键决策**：
- 状态小（< 1GB）：HashMapStateBackend，CP 快
- 状态大（> 1GB）：RocksDBStateBackend + 增量 CP，否则每次 CP 都要序列化全量状态
- `setTolerableCheckpointFailureNumber` 不要设太大，CP 连续失败说明有问题要排查

## 资源配置

### TaskManager 内存

```properties
# 总内存
taskmanager.memory.process.size=4096m
# Framework + Task Heap
taskmanager.memory.task.heap.size=2048m
# Managed Memory（RocksDB 用）
taskmanager.memory.managed.fraction=0.4
# Network Memory
taskmanager.memory.network.fraction=0.1
taskmanager.memory.network.min=128m
taskmanager.memory.network.max=512m
```

### 并行度设置

```
并行度 = Kafka partition 数 × 消费能力因子

# 一般原则：
# - Source 并行度 = Kafka partition 数
# - 算子并行度 = Source 并行度 × (1~3)，取决于计算复杂度
# - Sink 并行度 = 下游系统能承受的写入并发
```

一个常见错误：所有算子设同一个并行度。实际上 Source 和 Sink 的瓶颈往往不在计算而在 I/O，应该分开设置。

## 监控与报警

### 核心指标

```
# 必须监控的指标
flink_jobmanager_job_uptime                    # 作业运行时间（频繁重启会很短）
flink_taskmanager_job_task_checkpointAlignmentTime  # CP 对齐时间
flink_taskmanager_job_task_buffers_outPoolUsage     # 输出缓冲池使用率（反压信号）
flink_taskmanager_Status_JVM_GC_Time_G1_Old_Generation  # GC 时间

# Kafka Consumer 指标
flink_taskmanager_job_task_operator_KafkaSourceReader_currentOffset
flink_taskmanager_job_task_operator_KafkaSourceReader_committedOffset
```

### 反压排查

反压是生产中最常见的性能问题。排查路径：

1. **Flink Web UI** → Back Pressure 标签：看哪个算子标红
2. **确认是 Source 慢还是某个算子慢**：如果 Source 都反压了，说明下游某处有瓶颈
3. **Thread Dump**：看阻塞线程在做什么
4. **常见原因**：
   - 外部系统写入慢（数据库、ES）→ 加异步 I/O 或 batch 写
   - 数据倾斜 → 某个 key 数据量远超其他
   - GC 频繁 → 增加堆内存或优化状态使用

```java
// 异步 I/O 解决外部系统瓶颈
AsyncDataStream.unorderedWait(
    inputStream,
    new AsyncDatabaseRequest(),
    5000,              // 超时 5s
    TimeUnit.MILLISECONDS,
    100                // 最大并发请求数
);
```

## 作业升级与状态兼容

生产作业升级是高风险操作，核心原则：

1. **先做 Savepoint**：`flink savepoint <jobId> hdfs:///savepoints/`
2. **状态兼容性检查**：
   - 不能改 Operator UID（`uid("my-operator")`）
   - 不能删除有状态的算子
   - 新增算子无状态默认值
3. **从 Savepoint 恢复**：`flink run -s hdfs:///savepoints/savepoint-xxx`

```java
// 务必给每个有状态的算子设置 UID！
stream
    .keyBy(Event::getKey)
    .process(new MyProcessFunction())
    .uid("my-process-function")    // 这行忘了，升级时状态就丢了
    .name("Process Events");
```

## 生产 Checklist

- [ ] 所有有状态算子都设置了 `uid()`
- [ ] Checkpoint 间隔和超时合理
- [ ] 取消作业时保留 Checkpoint
- [ ] RocksDB 增量 CP（状态 > 1GB）
- [ ] 核心指标接入监控报警
- [ ] 反压报警阈值设置
- [ ] Consumer Lag 报警
- [ ] 作业自动重启策略配置
- [ ] Savepoint 定期备份
- [ ] 状态 TTL 设置（防止状态无限增长）

## 相关

- [[Career/数据工程/Flink/Flink 概述|Flink 概述]]
- [[Career/数据工程/Flink/CheckPoint/Flink CheckPoint|Flink CheckPoint]]
- [[Career/数据工程/Flink/运行/Flink 运行|Flink 运行]]
- [[Career/数据工程/Flink/生产/调优/Flink 调优|Flink 调优]]
- [[Career/数据工程/Flink/运行/Flink 重启策略|Flink 重启策略]]
- [[Career/数据工程/Flink/Flink 内存机制|Flink 内存机制]]
- [[反压|反压]]
- [[Career/数据工程/Flink/CheckPoint/Exactly Once 语义|Exactly Once 语义]]
- [[Career/数据工程/Flink/CheckPoint/RocksDB 原理|RocksDB 原理]]
- [[Career/数据工程/Flink/开发/Flink 开发|Flink 开发]]
- [[AI/3-LLM/Frameworks/verl/grafana 看板|grafana 看板]]
