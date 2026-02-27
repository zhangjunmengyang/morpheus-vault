---
title: "2. Doris 底层存储"
type: concept
domain: engineering/doris/底层存储
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris/底层存储
  - type/concept
---
# 2. Doris 底层存储

## 存储架构概览

Doris 的存储引擎是自研的列式存储，设计思路融合了 Google Mesa 和 Apache ClickHouse 的理念。BE 节点负责实际的数据存储和计算。

```
Table
├── Partition（按时间/范围分区）
│   ├── Tablet（分桶，数据调度的最小单位）
│   │   ├── Rowset（一次导入/Compaction 的产物）
│   │   │   ├── Segment（物理文件，默认 256MB 一个）
│   │   │   │   ├── Column Data（列式存储）
│   │   │   │   ├── Column Index（索引）
│   │   │   │   └── Footer（元数据）
```

**核心设计决策**：
- 数据按 Partition → Tablet 两级划分
- Tablet 是副本管理和负载均衡的最小单位
- 每个 Tablet 独立管理自己的 Rowset 版本链

## 列式存储格式

Doris 使用自研的列式存储格式，每个 Segment 文件结构：

```
Segment File Layout:
┌──────────────────────┐
│     Column 1 Data    │  ← Page 粒度存储
│     Column 1 Index   │  ← Ordinal Index + Zone Map + Bitmap Index
│     Column 2 Data    │
│     Column 2 Index   │
│         ...          │
│     Short Key Index  │  ← 前缀索引
│     Footer           │  ← 元信息、各列偏移量
└──────────────────────┘
```

### Page 存储

每列数据被切分成固定大小的 Page（默认 64KB），每个 Page 内部可以单独压缩和编码：

```
Page 编码方式：
- Plain Encoding          ← 无编码
- Dictionary Encoding     ← 低基数列（性别、状态）
- Run-Length Encoding      ← 连续相同值多的列
- Bit-Shuffle Encoding    ← 数值列，利用位模式相似性
- Frame-Of-Reference      ← 时间戳等单调递增列
```

编码方式在数据导入时自动选择，依据列的数据分布。Dictionary Encoding 对低基数列效果极好——10 亿行数据，如果只有 5 种取值，字典就 5 个 entry。

### 压缩

```sql
-- 建表时指定压缩算法
CREATE TABLE example (
    id INT,
    name VARCHAR(100)
) PROPERTIES (
    "compression" = "zstd"  -- 支持 lz4, zlib, zstd, snappy
);
```

| 压缩算法 | 压缩率 | 压缩速度 | 解压速度 | 适用场景 |
|---------|--------|---------|---------|---------|
| LZ4 | 低 | 极快 | 极快 | 实时查询优先 |
| ZSTD | 高 | 快 | 快 | 存储成本敏感 |
| ZLIB | 高 | 慢 | 中 | 归档数据 |
| Snappy | 中 | 快 | 快 | 平衡选择 |

生产建议：**默认用 LZ4**（Doris 2.0+ 默认已改为 ZSTD），查询密集型场景压缩率不是第一优先级。

## 索引体系

Doris 的索引是存储层性能的关键。多层索引配合，实现高效的数据过滤：

### 1. Short Key Index（前缀索引）

```sql
-- 取前 36 字节的列组合作为前缀索引
-- 列顺序很重要！高频查询条件的列放前面
CREATE TABLE user_events (
    user_id INT,        -- 查询最常用的过滤条件
    event_date DATE,    -- 第二常用
    event_type VARCHAR(50),
    payload TEXT
) DUPLICATE KEY(user_id, event_date)
-- 前缀索引 = user_id + event_date（前 36 字节）
```

Short Key Index 是稀疏索引，每 1024 行一个 entry。查询时先用前缀索引定位到数据范围，再读取对应的 Page。

### 2. Zone Map Index

每个 Page 自动维护最小值、最大值、NULL 数量：

```
Page Zone Map:
  min=100, max=999, has_null=false

查询 WHERE id = 50 → 跳过此 Page（50 < min=100）
查询 WHERE id = 500 → 需要读此 Page（100 ≤ 500 ≤ 999）
```

Zone Map 对于有序或近似有序的列效果最好。如果数据完全随机分布，Zone Map 过滤率极低。

### 3. Bloom Filter Index

```sql
ALTER TABLE user_events SET ("bloom_filter_columns" = "user_id,event_type");
```

适用于 **高基数列的等值查询**。注意：
- 占额外存储空间
- 对范围查询无效
- 误判率可通过 `bloom_filter_fpp` 控制（默认 0.05）

### 4. Bitmap Index

```sql
CREATE INDEX idx_status ON user_events (status) USING BITMAP;
```

适用于 **低基数列的等值/IN 查询**。状态字段（如 "paid"/"pending"/"cancelled"）用 Bitmap 索引效果极好。

### 5. Inverted Index（2.0+）

```sql
CREATE INDEX idx_payload ON user_events (payload) USING INVERTED;
```

Doris 2.0 引入倒排索引，支持全文检索和复杂条件过滤，替代部分 ES 场景。

## Compaction

Doris 的 LSM-like 存储模型需要后台 Compaction 来合并 Rowset：

```
导入产生的 Rowset:
[v0-v0] [v1-v1] [v2-v2] [v3-v3] [v4-v4]

Cumulative Compaction（小版本合并）:
[v0-v0] [v1-v4]

Base Compaction（大版本合并）:
[v0-v4]
```

**两级 Compaction 策略**：
- **Cumulative Compaction**：合并最近的小 Rowset，频率高，开销小
- **Base Compaction**：将 Cumulative 结果合并到 Base Rowset，频率低，开销大

```sql
-- 查看 Tablet 的 Compaction 状态
SHOW TABLET FROM db.table;

-- Compaction 相关配置（be.conf）
-- cumulative_compaction_num_threads_per_disk=4
-- base_compaction_num_threads_per_disk=2
-- cumulative_compaction_policy=size_based
```

Compaction 滞后（Rowset 版本数过多）会导致：
- 查询需要 Merge 多个版本，性能下降
- `-235` 错误：版本数超过上限（默认 500），拒绝新导入

排查思路：检查 BE 的 Compaction 线程是否繁忙、磁盘 I/O 是否打满。

## 副本管理

```sql
-- 默认 3 副本
CREATE TABLE example (...) 
PROPERTIES ("replication_num" = "3");
```

Tablet 的副本分布在不同 BE 上，FE 通过 **心跳检测 + 版本一致性检查** 管理副本状态：
- `NORMAL`：正常
- `CLONE`：正在从其他副本恢复
- `DECOMMISSION`：BE 下线中，正在迁移

写入时采用 **Quorum 协议**：多数副本写成功即返回成功。读取时只读一个副本（由 FE 选择健康副本）。

## 相关

- [[Career/数据工程/Doris/Doris 架构|Doris 架构]]
- [[Career/数据工程/Doris/底层存储/写入流程/数据导入/Doris 数据导入|Doris 数据导入]]
- [[Career/数据工程/Doris/底层存储/存储结构]]
- [[Career/数据工程/Doris/底层存储/读写原理待归档]]
- [[Career/数据工程/Doris/底层存储/写入流程/写入流程|写入流程]]
- [[Career/数据工程/Doris/底层存储/读取流程]]
- [[Career/数据工程/Doris/治理/Doris 查询计划|Doris 查询计划]]
- [[Career/数据工程/Doris/Doris 概述|Doris 概述]]
- [[Career/数据工程/Doris/治理/Doris 治理|Doris 治理]]
