---
title: "2. Spark SQL"
type: concept
domain: engineering/spark/sql
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/sql
  - type/concept
---
# 2. Spark SQL

## 核心定位

Spark SQL 是 Spark 生态中最常用的模块，它把 **SQL 的声明式语义** 和 **Spark 的分布式执行引擎** 桥接在一起。对于数仓工程师来说，90% 的日常工作都在跟 Spark SQL 打交道——写查询、调性能、排查数据质量问题。

核心价值：不用关心底层 RDD 怎么跑，Catalyst Optimizer 帮你做逻辑优化和物理优化。但 **"帮你做"不代表"做得好"**，理解优化器行为是调优的前提。

## Catalyst 优化器流程

```
SQL / DataFrame API
        ↓
   Unresolved Logical Plan   ← Parser
        ↓
   Resolved Logical Plan     ← Analyzer（绑定 catalog/schema）
        ↓
   Optimized Logical Plan    ← Optimizer（规则 + CBO）
        ↓
   Physical Plans            ← Planner（生成候选物理计划）
        ↓
   Selected Physical Plan    ← Cost Model 选最优
        ↓
   RDD DAG                   ← Code Generation（Whole-Stage CodeGen）
```

关键阶段拆解：

1. **Analyzer**：解析列名、表名、函数，绑定到 Catalog。常见报错 `AnalysisException` 就出在这一步。
2. **Optimizer**：应用一系列 Rule，比如谓词下推（PushDownPredicate）、列裁剪（ColumnPruning）、常量折叠（ConstantFolding）。这些 Rule 是确定性的，不依赖统计信息。
3. **CBO（Cost-Based Optimization）**：需要 `ANALYZE TABLE` 收集统计信息才生效。Join 顺序调整、Build Side 选择都依赖 CBO。
4. **Whole-Stage CodeGen**：把多个算子融合成一个 Java 方法，避免虚函数调用开销。通过 `explain(true)` 可以看到带 `*` 前缀的算子就是被 CodeGen 的。

## 常用调优手段

### 谓词下推

```sql
-- 好：分区过滤在 scan 阶段就生效
SELECT * FROM orders WHERE dt = '2026-02-13' AND status = 'paid';

-- 坏：对分区列做函数运算，谓词下推失效
SELECT * FROM orders WHERE date_format(dt, 'yyyy-MM') = '2026-02';
```

用 `EXPLAIN EXTENDED` 确认 `PushedFilters` 是否包含你的过滤条件。

### Broadcast Join

```sql
-- 小表 < spark.sql.autoBroadcastJoinThreshold（默认 10MB）自动 broadcast
-- 手动 hint：
SELECT /*+ BROADCAST(dim) */ *
FROM fact_table f
JOIN dim_table dim ON f.dim_id = dim.id;
```

Broadcast Join 避免了 Shuffle，但要注意 **Driver 内存**——小表先 collect 到 Driver 再广播。如果"小表"其实有几百 MB，Driver OOM 是常见翻车现场。

### AQE（Adaptive Query Execution）

Spark 3.x 的杀手特性，运行时根据 Shuffle 统计动态调整：

```properties
spark.sql.adaptive.enabled=true
spark.sql.adaptive.coalescePartitions.enabled=true       # 自动合并小分区
spark.sql.adaptive.skewJoin.enabled=true                  # 倾斜 join 自动拆分
spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=256m
```

AQE 的 **合并小分区** 功能特别实用——上游 Shuffle 写了 200 个分区但大部分很小，AQE 自动合并成合理数量，减少 Task 调度开销。

## DataFrame vs SQL vs Dataset

| 维度 | DataFrame | SQL | Dataset |
|------|-----------|-----|---------|
| 类型安全 | 运行时 | 运行时 | 编译时 |
| 优化 | Catalyst 全链路 | Catalyst 全链路 | Catalyst（但 lambda 可能阻断优化） |
| 适用场景 | ETL 管道 | 临时查询 / 可读性 | 需要类型安全的 Scala 场景 |

实际项目中的经验：**优先用 DataFrame API**，可读性和优化效果兼得。Dataset 的 `map`/`filter` 用 lambda 时，Catalyst 无法穿透优化，容易产生性能陷阱。

## Spark SQL 执行计划阅读

```sql
EXPLAIN FORMATTED
SELECT department, count(*) as cnt
FROM employees
WHERE salary > 10000
GROUP BY department;
```

重点关注：
- **Scan 算子**：`PushedFilters` 里有没有你的条件
- **Exchange 算子**：Shuffle 发生的位置和 partition 数
- **Sort 算子**：是否有不必要的排序
- **BroadcastHashJoin vs SortMergeJoin**：Join 策略是否合理

一个实用技巧：`spark.sql.planChangeLog.level=WARN` 可以在日志里看到 Optimizer 每一步 Rule 做了什么变换，排查"为什么没走谓词下推"之类的问题非常有用。

## 常见坑

1. **隐式类型转换**：`WHERE id = '123'` 中 `id` 是 INT 类型，会触发 Cast，索引/分区裁剪可能失效。
2. **COUNT(DISTINCT) 性能**：单个 Stage 完成，数据全部 Shuffle 到一个 Reducer。大数据量用 `approx_count_distinct` 或两阶段 GROUP BY。
3. **小文件问题**：动态分区写入时容易产生大量小文件，用 `DISTRIBUTE BY` 或 AQE 的 coalesce 来控制。
4. **NULL 语义**：`NOT IN` 子查询遇到 NULL 时结果可能出乎意料。优先用 `NOT EXISTS` 或 `LEFT ANTI JOIN`。

## 相关

- [[Spark SQL 执行过程]]
- [[Spark Shuffle]]
- [[Spark AQE + DDP]]
- [[谓词下推]]
- [[Spark 调优|Spark 调优]]
- [[Spark Join]]
- [[Spark 概述|Spark 概述]]
