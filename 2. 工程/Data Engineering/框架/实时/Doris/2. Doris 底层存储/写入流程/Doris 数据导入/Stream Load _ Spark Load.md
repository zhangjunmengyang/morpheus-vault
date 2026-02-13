---
title: "Stream Load / Spark Load"
category: "工程"
tags: [Doris, ETL, Hive, Join, Rollup]
created: "2026-02-13"
updated: "2026-02-13"
---

# Stream Load / Spark Load

# **一、**Doris Spark Load

Spark load 通过外部的 Spark 资源实现对导入数据的预处理，提高 Doris 大数据量的导入性能并且节省 Doris 集群的计算资源。主要用于初次迁移，大数据量导入 Doris 的场景。

Spark load 是一种异步导入方式，用户需要通过 MySQL 协议创建 Spark 类型导入任务，并通过 SHOW LOAD 查看导入结果。

**E**xtract from Hive, **T**ransform in Spark, **L**oad to Doris

## 1.1、适用场景

- 源数据在 Spark 可以访问的存储系统中， Hive。
- 数据量在 **几十 GB 到 TB **级别。数据条数在百亿级别
### 1.1.1、spark load 适合什么

1. 当前导入任务数据量大，比如并发数在30以上的作业，尤其适合聚合模型
1. 导入的目标集群事务数压力大的场景，导入槽位紧张
1. hive2doris导入时长较长，并且作业中load to doris 这部分占比大的作业
### 1.1.2、spark load 不适合什么

1. bitmap 字典构建时间长
1. doris表中分桶设置不合理导致数据倾斜严重
1. 导入数据量非常小的作业
1. 冗余模型，非聚合，或者聚合度非常小的表
## 1.2、导入流程

用户通过 MySQL 客户端提交 Spark 类型导入任务，FE 记录元数据并返回用户提交成功。

Spark Load 任务的执行主要分为以下 5 个阶段。

1. FE 调度提交 ETL 任务到 Spark 集群执行。
1. Spark 集群执行 ETL 完成对导入数据的预处理，包括全局字典构建（ Bitmap 类型）、分区、排序、聚合等。
1. ETL 任务完成后，FE 获取预处理过的每个分片的数据路径，并调度相关的 BE 执行 Push 任务。
1. BE 通过 Broker 读取数据，转化为 Doris 底层存储格式。
1. Broker 为一个独立的无状态进程。封装了文件系统接口，提供 Doris 读取远端存储系统中文件的能力。
1. FE 调度生效版本，完成导入任务。
![image](assets/IX1Hd0ZaboJFB3xBjr8cisxGnWg.png)

### 1.2.1、全局字典

目前 Doris 中 Bitmap 列是使用类库 `Roaringbitmap` 实现的，而 `Roaringbitmap` 的输入数据类型只能是整型，因此如果要在导入流程中实现对于 Bitmap 列的预计算，那么就需要将输入数据的类型转换成整型。

在 Doris 现有的导入流程中，全局字典的数据结构是基于 Hive 表实现的，保存了原始值到编码值的映射。

1. 读取上游数据源的数据，生成一张 Hive 临时表，记为 `hive_table`。
1. 从 `hive_table `中抽取待去重字段的去重值，生成一张新的 Hive 表，记为 `distinct_value_table`。
1. 新建一张全局字典表，记为 `dict_table` ，一列为原始值，一列为编码后的值。
1. 将 `distinct_value_table` 与 `dict_table` 做 Left Join，计算出新增的去重值集合，然后对这个集合使用窗口函数进行编码，此时去重列原始值就多了一列编码后的值，最后将这两列的数据写回 `dict_table`。
1. 将 `dict_table `与 `hive_table` 进行 Join，完成 `hive_table` 中原始值替换成整型编码值的工作。
1. `hive_table `会被下一步数据预处理的流程所读取，经过计算后导入到 Doris 中。
### 1.2.2、数据预处理（DPP）

1. 从数据源读取数据，上游数据源可以是 HDFS 文件，也可以是 Hive 表。
1. 对读取到的数据进行字段映射，表达式计算以及根据分区信息生成分桶字段 `bucket_id`。
1. 根据 Doris 表的 Rollup 元数据生成 RollupTree。
1. 遍历 RollupTree，进行分层的聚合操作，下一个层级的 Rollup 可以由上一个层的 Rollup 计算得来。
1. 每次完成聚合计算后，会对数据根据 `bucket_id `进行分桶然后写入 HDFS 中。
1. 后续 Broker 会拉取 HDFS 中的文件然后导入 Doris Be 中。
# 二、使用 SOP

注意：

- 对于处在**Spark load 导入过程中的表**，请**不要**通过alter 语句进行**schema change **等操作，**否则不保证导入成功**
- **不要同时 spark load 导入同一张表**，即**不要起两个作业（重导也算）同时对同一个Doris 表进行spark load**
- 确认集群是否有 Spark Load 功能
- 创建一个 Doris 外部表
- 其schema 与hive 表一致， hive 表中的分区列也要包含在doris 外部表中。
- **hive 中列是Doris 列的超集，那么外部表不要求完全和hive 一致，只要保证doris 里有的列，都在外部表里就好**
- 如果Doris 中有一些特殊类型的字段，如**HLL\Bitmap** 等，在**外部表中可以设置成varchar(1)**即可
- 创建 Hive2Doris 任务