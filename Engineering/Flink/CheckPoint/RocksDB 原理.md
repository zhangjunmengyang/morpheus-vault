---
title: "RocksDB 原理"
type: concept
domain: engineering/flink/checkpoint
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/checkpoint
  - type/concept
---
# RocksDB 原理

> 参考：RocksDB 的常用调优参数：https://blog.csdn.net/weixin_36145588/article/details/78541070

## 为什么 Flink 用 RocksDB

Flink 有两种 State Backend：HashMapStateBackend（纯内存）和 EmbeddedRocksDBStateBackend。当状态规模超过可用内存时，RocksDB 是唯一选择——它能把状态存在本地磁盘上，突破内存限制。

核心权衡：
- **HashMapStateBackend**：快（内存直接访问），但状态量受内存限制
- **RocksDB**：慢（涉及序列化 + 磁盘 I/O），但可以管理 TB 级状态

生产中 **大部分有状态的 Flink 作业都用 RocksDB**，因为内存够用只是当前够用，业务增长后状态量往往超预期。

## LSM-Tree 核心架构

RocksDB 基于 **LSM-Tree（Log-Structured Merge-Tree）** 构建：

```
Write Path:
    Write → MemTable（内存，跳表结构）
                ↓ flush
           Immutable MemTable
                ↓ flush
           Level 0 SST Files（可能有重叠）
                ↓ compaction
           Level 1 SST Files（key range 不重叠）
                ↓ compaction
           Level 2 SST Files
                ↓ ...
           Level N SST Files

Read Path:
    Read → MemTable → Level 0 → Level 1 → ... → Level N
    (通过 Bloom Filter 跳过不包含目标 key 的 SST)
```

### 写入流程

```
1. 写入 WAL（Write-Ahead Log）→ 崩溃恢复用
2. 写入 MemTable（默认 64MB）
3. MemTable 满了 → 变成 Immutable MemTable
4. 后台线程 Flush → 生成 Level 0 SST 文件
5. Level 0 积累到阈值 → 触发 Compaction → 合并到 Level 1
```

**写放大**：一条数据从写入到最终落盘，可能经历多次 Compaction 重写。写放大比 = 实际磁盘写入量 / 用户写入量，典型值 10-30x。

### 读取流程

```
1. 查 MemTable（O(log n)，跳表）
2. 查 Immutable MemTable
3. 查 Level 0（每个 SST 都要查，因为 key range 可能重叠）
4. 查 Level 1+（二分查找定位 SST，Bloom Filter 快速排除）
```

**读放大**：最坏情况需要查每一层，但 Bloom Filter 能跳过 99%+ 的无效 SST 查找。

## SST 文件结构

```
SST File:
┌──────────────────┐
│   Data Blocks    │  ← 实际 KV 数据，有序排列
│   Filter Block   │  ← Bloom Filter
│   Index Block    │  ← Data Block 的索引
│   Meta Index     │
│   Footer         │  ← 指向 Index 和 Meta Index
└──────────────────┘
```

每个 Data Block 默认 4KB，内部按 key 排序。Index Block 记录每个 Data Block 的最后一个 key 和偏移量。

## Compaction 策略

### Level Compaction（默认）

```
Level 0: [SST][SST][SST][SST]    ← 4 个文件触发 compaction
Level 1: [SST][SST][SST]          ← 大小限制 256MB
Level 2: [SST][SST][SST][SST]     ← 大小限制 2.56GB（10× Level 1）
Level 3: ...                       ← 每层 10× 上一层
```

优点：读性能好（每层 key range 不重叠），空间放大小。
缺点：写放大大（反复 compaction）。

### Universal Compaction

适合写密集场景，减少写放大但增加空间放大和读放大。

### FIFO Compaction

简单地按时间淘汰旧文件，适合 TTL 场景（如 Flink 状态 TTL）。

## Flink 中的 RocksDB 调优

### 内存管理

```java
// Flink 通过 Managed Memory 控制 RocksDB 内存
// 默认占 TaskManager Managed Memory 的全部

// 配置方式1：通过 flink-conf
state.backend.rocksdb.memory.managed=true  // 使用 Flink Managed Memory
taskmanager.memory.managed.fraction=0.4    // Managed Memory 占比

// 配置方式2：精细控制
state.backend.rocksdb.block.cache-size=256m      // Block Cache 大小
state.backend.rocksdb.writebuffer.size=128m       // 单个 MemTable 大小
state.backend.rocksdb.writebuffer.count=3         // MemTable 数量
state.backend.rocksdb.writebuffer.number-to-merge=2  // 合并写入的 MemTable 数
```

RocksDB 的内存组成：

```
RocksDB 内存 = Block Cache + MemTable × count + Index/Filter Cache

Block Cache: 缓存读取的 Data Block（最影响读性能）
MemTable:    写入缓冲区（最影响写性能）
Index/Filter: SST 文件的索引和 Bloom Filter
```

### Checkpoint 优化

RocksDB 支持 **增量 Checkpoint**，这是大状态场景的救命稻草：

```java
// 开启增量 Checkpoint
EmbeddedRocksDBStateBackend backend = new EmbeddedRocksDBStateBackend(true);
env.setStateBackend(backend);
```

增量 CP 的原理：
1. 每次 CP 只上传自上次 CP 以来新增/变更的 SST 文件
2. 利用 RocksDB 的 SST 文件不可变特性——已经上传过的文件不会变
3. 状态 10GB，增量部分可能只有几百 MB

**注意**：增量 CP 依赖 SST 文件引用计数，如果 Compaction 频繁导致大量 SST 文件变更，增量 CP 的优势会减弱。

### 常用调优参数

```properties
# Compaction 线程数（默认 1，建议增加）
state.backend.rocksdb.thread.num=4

# 压缩算法
state.backend.rocksdb.compression.per.level=no_compression;no_compression;lz4;lz4;lz4;zstd;zstd

# Bloom Filter（加速点查）
state.backend.rocksdb.bloom-filter.bits-per-key=10
state.backend.rocksdb.bloom-filter.block-based=false  # full filter 比 block-based 更好

# Write Buffer
state.backend.rocksdb.writebuffer.size=128m
state.backend.rocksdb.writebuffer.count=3
```

### 性能诊断

```java
// 开启 RocksDB 原生统计
state.backend.rocksdb.metrics.actual-delayed-write-rate=true
state.backend.rocksdb.metrics.estimate-pending-compaction-bytes=true
state.backend.rocksdb.metrics.num-running-compactions=true
```

关键指标：
- `rocksdb.actual-delayed-write-rate > 0`：写入被限流，Compaction 跟不上
- `rocksdb.estimate-pending-compaction-bytes` 持续增长：Compaction 积压
- `rocksdb.num-running-compactions`：Compaction 并发数

## 常见问题

1. **Checkpoint 超时**：状态太大 + 全量 CP → 改增量 CP，增加 Compaction 线程
2. **写入限流**：RocksDB 内部的 Write Stall → 增加 MemTable 数量、增大 L0 触发 Compaction 的阈值
3. **读取延迟高**：Block Cache 命中率低 → 增大 Block Cache，确保 Bloom Filter 生效
4. **磁盘空间暴涨**：Compaction 积压或空间放大 → 检查 Compaction 线程是否足够

## 相关

- [[Flink CheckPoint]]
- [[Checkpoint]]
- [[Flink 内存机制]]
- [[Flink 调优]]
- [[Flink 生产]]
- [[存储结构]]
