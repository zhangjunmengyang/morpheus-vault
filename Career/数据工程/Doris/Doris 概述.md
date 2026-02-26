---
title: "Doris"
type: reference
domain: engineering/doris
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris
  - type/reference
---
# Doris

多看文档：[快速开始 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fget-starting%2Fquick-start%2F)

## 使用问题

### 数据导入导出

- 支持的导入方式：目前只支持 kafka 和 hive 进行导入，mt 内部都是基于olap 平台进行操作。建立 Doris 数据模型后可以新建离线或实时任务进行导入
- k2d：[订阅 Kafka 日志 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-operate%2Fimport%2Fimport-scenes%2Fkafka-load)
- h2d：[Spark Load - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-operate%2Fimport%2Fimport-way%2Fspark-load-manual)
- 支持的导出方式：常见常用的方式如下
![image](YUMMdU3dkoalXlxLc4ccCqLsntb.png)

- d2d 任务，通常用于单层简单简单加工，或满足不同的调度 or 数据可见性要求
- d2h 任务会用于给分析师提供近实时看板，或者直接暴露用于查询，防止错误的 SQL 的写法造成 Doris 集群压力，同时，d2h + h2MySQL 也在以往用于解决数据同步冲突问题，比如原本是 Doris2Mysql，但因为导入较慢，如果恰好赶上 Doris 正在写入，就会导致同步失败，出现大量重试（case：上游 Doris 15min 调度，MySQL 每天调度一次，可能正好赶上 Doris 正在写入，数据索引等变化，最差就可能会重试 3 次最终失败），Hive 能避免这个问题，因为 hive2mysql 执行快
![image](TfZzdlqtgoMUo2xPvCzcmY3tnWe.png)

![image](S6CidUhLlolhXsxnQ7AcbBksnhe.png)

- 问题：为什么 d2m 比 h2m 执行得更慢？实现原理分别是什么样的？
- d2Cellar：新 KV，和 Squrriel 都是一般用于读写维表
- d2Squrriel：老 KV，有一部分还在用
- blade：mt 的分布式关系型数据库，不太了解，没遇到过使用的业务下游
### 支持的数据量

从经验测试结果来看，百万级数据可以做到喵姐延迟，如果设置合理的索引和 rollup，结合有效的过滤（比如日期分区）聚合可以支持亿级数据，但不保证时延，可能10s 以上

### 是否支持 UDF

- 不支持 Java UDF，因为 Doris BE 是基于 C++实现
- 但 Doris 本身支持通过 http 获取 UDF包，但公司工具栏目前不支持，也没有遇到必须要用的业务场景
## 索引

[索引概述 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-table%2Findex%2Findex-overview)

### 前缀索引

不同于传统的数据库设计，Doris 不支持在任意列上创建索引。Doris 这类 MPP 架构的 OLAP 数据库，通常都是通过提高并发，来处理大量数据的。

本质上，Doris 的数据存储在类似 SSTable（Sorted String Table）的数据结构中。该结构是一种有序的数据结构，可以按照指定的列进行排序存储。在这种数据结构上，以排序列作为条件进行查找，会非常的高效。

在 Aggregate、Unique 和 Duplicate 三种数据模型中。底层的数据存储，是按照各自建表语句中，AGGREGATE KEY、UNIQUE KEY 和 DUPLICATE KEY 中指定的列进行排序存储的。

而前缀索引，即在排序的基础上，实现的一种根据给定前缀列，快速查询数据的索引方式。

注意：当遇到 VARCHAR 类型时，前缀索引会直接截断

当我们的查询条件，是前缀索引的前缀时，可以极大的加快查询速度。比如在第一个例子中，我们执行如下查询：

```
*SELECT* * *FROM* *table* *WHERE* user_id=1829239 and age=20；
```

该查询的效率会远高于如下查询：

```
*SELECT* * *FROM* *table* *WHERE* age=20；
```

所以在建表时，正确的选择列顺序，能够极大地提高查询效率。

因为建表时已经指定了列顺序，所以一个表只有一种前缀索引。这对于使用其他不能命中前缀索引的列作为条件进行的查询来说，效率上可能无法满足需求。因此，我们可以通过创建 ROLLUP 来人为的调整列顺序。详情可参考 [ROLLUP](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-table%2Fhit-the-rollup)。

### 倒排索引

[倒排索引 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-table%2Findex%2Finverted-index)

从2.0.0版本开始，Doris支持倒排索引，可以用来**进行文本类型的全文检索**、**普通数值日期类型的等值范围查询**，快速从海量数据中过滤出满足条件的行。本文档主要介绍如何倒排索引的创建、删除、查询等使用方式。

在Doris的倒排索引实现中，table的一行对应一个文档、一列对应文档中的一个字段，因此利用倒排索引可以根据关键词快速定位包含它的行，达到WHERE子句加速的目的。

与Doris中其他索引不同的是，在存储层倒排索引使用独立的文件，跟segment文件有逻辑对应关系、但存储的文件相互独立。这样的好处是可以做到创建、删除索引不用重写tablet和segment文件，大幅降低处理开销。

### BloomFilter

[BloomFilter 索引 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2Fdata-table%2Findex%2Fbloomfilter)

Doris的BloomFilter索引可以通过建表的时候指定，或者通过表的ALTER操作来完成。Bloom Filter本质上是一种位图结构，用于快速的判断一个给定的值是否在一个集合中。这种判断会产生小概率的误判。即如果返回false，则一定不在这个集合内。而如果范围true，则有可能在这个集合内。

BloomFilter索引也是以Block为粒度创建的。每个Block中，指定列的值作为一个集合生成一个BloomFilter索引条目，用于在查询时快速过滤不满足条件的数据。

1. 不支持对Tinyint、Float、Double 类型的列建Bloom Filter索引。
1. Bloom Filter索引只对 in 和 = 过滤查询有加速效果。
1. 如果要查看某个查询是否命中了Bloom Filter索引，可以通过查询的Profile信息查看。
## 模型设计

[数据库建表最佳实践 - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2F2.0%2Ftable-design%2Fbest-practice)

### 分桶 Tablet

在 Doris 的存储引擎中，用户数据被水平划分为若干个数据分片（Tablet，也称作数据分桶）。每个 Tablet 包含若干数据行。各个 Tablet 之间的数据没有交集，并且在物理上是独立存储的。

### 分区 Partition

多个 Tablet 在逻辑上归属于不同的分区（Partition）。一个 Tablet 只属于一个 Partition。而一个 Partition 包含若干个 Tablet。因为 Tablet 在物理上是独立存储的，所以可以视为 Partition 在物理上也是独立。Tablet 是数据移动、复制等操作的最小物理存储单元。

若干个 Partition 组成一个 Table。Partition 可以视为是逻辑上最小的管理单元。数据的导入与删除，仅能针对一个 Partition 进行。

# 常用命令

## 查询

### **查看前端节点**

SHOW PROC '/frontends'

### **查看后端节点**

SHOW PROC '/backends'

### **查看表数据大小**

SHOW DATA  # 查看所有表大小

SHOW DATA FROM org_project_data_2 # 查看指定表大小

### **查看表信息**

DESCRIBE hbdata.app_phx_goods_refund_delete;

### **查看列修改情况**

SHOW ALTER TABLE COLUMN;

### **查看建表语句**

show create table ${table_name}

### 查看查询计划

可以通过以下三种命令查看一个 SQL 的执行计划。

- `EXPLAIN GRAPH select ...;` 或者 `DESC GRAPH select ...;`
- `EXPLAIN select ...;`
- `EXPLAIN VERBOSE select ...;`
## 创建 / 修改

### **重命名表**

ALTER TABLE test_table RENAME new_table_name

### 新增列

ALTER TABLE t1 ADD COLUMN c1 BIGINT REPLACE AFTER c2;

### **查看表分区**

SHOW PARTITIONS FROM new_table_name

### **新增分区**

ALTER TABLE t1 ADD PARTITION p20230301 VALUES[("20230228"), ("20230301"));

### **导入分区数据**

INSERT INTO t1 PARTITION(p20230301)  WITN my_label

SELECT * FROM t2 PARTITION(p20230301);

不同列类型不能直接插入

**一般情况下不能直接通过insert into将不同类型的列的数据互相导入，因为Doris对于不同的列类型都有对应的存储方式和计算方法。如果数据类型不匹配，会导致写入或读取数据时出现问题。**

在Doris中，Bitmap和HLL属于两种不同的基数统计方法，它们的实现原理和存储方式有较大的差异。如果你想要将Bitmap类型的数据插入到HLL类型的列中，需要先将Bitmap值经过转换后再插入到表B中。具体的转换方式可以使用UDF函数或者ETL工具进行实现。

### **查看导入情况**

SHOW LOAD WHERE label = "my_label";

### **修改分桶**

alter table ${table_name} modify distribution DISTRIBUTED BY HASH(${hash_column}) BUCKETS ${new_bucket_num}

### ROLLUP

创建rollup：ALTER TABLE t1 ADD ROLLUP t1_rollup(c1, c2, c3);

删除rollup：ALTER TABLE t1 DROP ROLLUP rollup;

查看rollup创建情况：SHOW ALTER TABLE ROLLUP;

查看rollup更新频率和间隔：SHOW ALTER TABLE ROLLUP;

SHOW ROLLUP TASKS IN hbdataflow.app_rt_flow_poi_intent_uv_agg5min;
