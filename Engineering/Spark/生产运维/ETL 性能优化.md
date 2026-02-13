---
title: "ETL 性能优化"
type: concept
domain: engineering/spark/生产运维
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/生产运维
  - type/concept
---
# ETL 性能优化

## 优化的思考框架

ETL 性能优化不是东调一个参数西改一行 SQL 的散弹枪式操作。需要一个系统化的排查框架：

```
定位瓶颈 → 分析原因 → 制定方案 → 验证效果 → 固化经验

瓶颈在哪？
├── I/O 密集？（Scan 慢、Shuffle 大、写入慢）
├── CPU 密集？（复杂计算、UDF、序列化）
├── 内存不足？（GC 频繁、Spill 到磁盘）
└── 调度开销？（Task 太多、Stage 太多）
```

**第一步永远是看 Spark UI**，不要凭感觉调参。

## 数据读取优化

### 分区裁剪

```sql
-- 好：利用分区字段过滤
SELECT * FROM ods.orders WHERE dt = '2026-02-13';

-- 坏：分区字段上做计算
SELECT * FROM ods.orders WHERE substr(dt, 1, 7) = '2026-02';

-- 验证：EXPLAIN 看 PartitionFilters
EXPLAIN EXTENDED SELECT * FROM ods.orders WHERE dt = '2026-02-13';
```

### 列裁剪

```sql
-- 坏：SELECT *（读取所有列）
SELECT * FROM ods.orders WHERE dt = '2026-02-13';

-- 好：只读需要的列（Parquet/ORC 列式存储优势）
SELECT order_id, user_id, amount FROM ods.orders WHERE dt = '2026-02-13';
```

列式存储格式（Parquet/ORC）下，列裁剪可以显著减少 I/O。一张 100 列的表只读 5 列，数据量可能只有原来的 5%。

### 小文件合并

```sql
-- 读取前先合并小文件
SET spark.sql.files.maxPartitionBytes=256m;     -- 单个分区最大数据量
SET spark.sql.files.openCostInBytes=8388608;    -- 小文件的"虚拟膨胀"成本
```

如果上游产出大量小文件（比如实时写入的 Hive 表），先跑一个 compaction 作业合并，再做下游 ETL。

## Shuffle 优化

Shuffle 通常是 ETL 最大的性能杀手。优化策略：

### 减少 Shuffle 数据量

```sql
-- 先过滤再 Join（谓词下推）
-- Spark 优化器通常会自动做，但复杂子查询可能失效
SELECT a.*, b.name
FROM (SELECT * FROM orders WHERE dt = '2026-02-13' AND amount > 0) a
JOIN users b ON a.user_id = b.user_id;
```

### 选择合适的 Join 策略

```sql
-- 小表 Broadcast，避免 Shuffle
SELECT /*+ BROADCAST(dim) */ *
FROM fact_orders f
JOIN dim_city dim ON f.city_id = dim.city_id;

-- Bucket Join：两表按相同 key 分桶，Join 时无需 Shuffle
-- 前提：两表都按 join key 做了 bucket
CREATE TABLE bucketed_orders (
    order_id STRING, user_id STRING, amount DOUBLE
) USING parquet
CLUSTERED BY (user_id) INTO 256 BUCKETS;
```

### 合理设置 Shuffle 分区数

```properties
# 静态设置
spark.sql.shuffle.partitions=500

# 更好的方式：开启 AQE 自动调整
spark.sql.adaptive.enabled=true
spark.sql.adaptive.coalescePartitions.enabled=true
spark.sql.adaptive.coalescePartitions.initialPartitionNum=2000
spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
```

## 写入优化

### 控制输出文件数

```sql
-- 方法1：DISTRIBUTE BY 控制分区数
INSERT OVERWRITE TABLE dwd.orders PARTITION(dt)
SELECT * FROM staging_orders
DISTRIBUTE BY dt;

-- 方法2：coalesce / repartition
df.coalesce(100).write.mode("overwrite").parquet("output_path")

-- 方法3：AQE 自动合并
SET spark.sql.adaptive.coalescePartitions.enabled=true;
```

### 动态分区写入优化

```properties
# 限制动态分区数，防止生成海量小文件
spark.sql.shuffle.partitions=500
hive.exec.max.dynamic.partitions=10000
hive.exec.max.dynamic.partitions.pernode=1000
```

动态分区 + 高基数分区键 = 小文件灾难。比如按 `user_id` 分区，100 万用户就是 100 万个目录。

## 代码级优化

### 避免低效 UDF

```python
# 坏：Python UDF（跨进程序列化开销巨大）
@udf(returnType=StringType())
def my_udf(value):
    return value.upper()

# 好：用 Spark 内置函数
from pyspark.sql.functions import upper
df.withColumn("upper_name", upper(col("name")))

# 折中：Pandas UDF（Arrow 批量传输，减少序列化开销）
@pandas_udf(StringType())
def my_pandas_udf(s: pd.Series) -> pd.Series:
    return s.str.upper()
```

Spark 内置函数走 Catalyst 优化 + Tungsten CodeGen，性能比 Python UDF 快 **10-100 倍**。

### 避免重复计算

```python
# 坏：同一个 DataFrame 被多个 action 触发，每次都重新计算
expensive_df = raw_df.join(dim_df, "key").filter(...)
print(expensive_df.count())      # 触发计算
expensive_df.write.parquet(...)  # 再次触发计算

# 好：缓存中间结果
expensive_df = raw_df.join(dim_df, "key").filter(...)
expensive_df.persist(StorageLevel.MEMORY_AND_DISK_SER)
print(expensive_df.count())
expensive_df.write.parquet(...)
expensive_df.unpersist()
```

## 资源配置模板

```properties
# 中等规模 ETL 作业（日增 10-50GB）
spark.executor.instances=20
spark.executor.cores=4
spark.executor.memory=8g
spark.executor.memoryOverhead=2g
spark.driver.memory=4g
spark.sql.shuffle.partitions=400
spark.sql.adaptive.enabled=true

# 大规模 ETL 作业（日增 100GB+）
spark.executor.instances=50
spark.executor.cores=4
spark.executor.memory=16g
spark.executor.memoryOverhead=4g
spark.driver.memory=8g
spark.sql.shuffle.partitions=2000
spark.sql.adaptive.enabled=true
```

## 优化效果度量

优化前后必须量化对比：

```
- 作业总耗时
- Shuffle Read/Write 数据量
- GC 时间占比
- Spill 到磁盘的数据量
- Stage 执行时间分布（是否有长尾 Task）
```

## 相关

- [[Spark SQL]]
- [[Spark Shuffle]]
- [[Spark 调优]]
- [[数据倾斜优化]]
- [[小文件问题]]
- [[Spark AQE + DDP]]
- [[Spark 生产运维]]
- [[谓词下推]]
