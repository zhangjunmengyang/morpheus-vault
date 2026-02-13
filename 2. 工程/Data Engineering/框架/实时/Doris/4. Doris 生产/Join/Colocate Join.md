---
title: "Colocate Join"
category: "工程"
tags: [Doris, Join, 分布式训练]
created: "2026-02-13"
updated: "2026-02-13"
---

# Colocate Join

https://km.sankuai.com/collabpage/71030685

## 一、基本理论

### 1.1、基本概念

- **Colocation Group（CG）**：一个 CG 中会包含一张及以上的 Table。一个CG内的Table 按相同的分桶方式和副本放置方式, 使用Colocation Group Schema描述.   (组合分组）
- **Colocation Group Schema（CGS）**： 包含CG的分桶键，分桶数以及副本数等信息。
- **Colocate Parent Table**：将决定一个 Group 数据分布的 Table 称为 Parent Table。（当创建表时, 通过表的 PROPERTIES 的属性"colocate_with" = "group_name" 指定表归属的CG; 如果CG不存在, 说明该表为CG的第一张表, 称之为Parent Table,   Parent Table的数据分布(分桶键的类型,数量和顺序, 副本数和分桶数)决定了CGS; 如果CG存在, 则检查表的数据分布是否和CGS一致.）
- **Colocate Child Table**：将一个 Group 中除 Parent Table 之外的 Table 称为 Child Table。
- **Bucket Seq**：如下图，如果一个表有 N 个 Partition, 则每个 Partition 的第 M 个 bucket 的 Bucket Seq 是 M。
（图片）

### 1.2、原理

- **目的**：将拥有相同 CGS 的 table 组成 一个 group，保证这些 table 对应的分桶副本会落在相同的 BE 节点上，当在分桶列上 Join 时，方便直接本地 Join，减少传输开销
为了保证同 CG 相同数据分布，必须保证下列**约束**：

1. 分桶键的类型, 数量和顺序完全一致，保证多张表的数据分片能够一一对应的进行分布控制。
1. 副本数必须一致。如果不一致，可能出现某一个 Tablet 的某一个副本，在同一个 BE 上没有其他的表分片的副本对应。
1. 分区键无关：同一个 CG 内所有表的分区键, 分区数量可以不同.
核心思想：

所以我们在数据导入时保证本地性的核心思想就是两次映射，对于 colocate tables，我们保证相同 Distributed Key 的数据映射到相同的 Bucket Seq，再保证相同 Bucket Seq 的 buckets 映射到相同的 BE

具体实现

第一步：我们计算 Distributed Key 的 hash 值，并对 bucket num 取模，保证相同 Distributed Key 的数据映射到相同的 Bucket Seq。

第二步：将同一个 Colocate Group 下所有相同 Bucket Seq 的 Bucket 映射到相同的 BE，方法如下：

**（child ---> parrent ---> first partition ---> Round Robin）**

1、Group 中所有 Table 的 Bucket Seq 和 BE 节点的映射关系和 Parent Table 一致

2、Parent Table 中所有 Partition 的 Bucket Seq 和 BE 节点的映射关系和第一个 Partition 一致

3、Parent Table 第一个 Partition 的 Bucket Seq 和 BE 节点的映射关系利用原生的 Round Robin 算法决定

## 二、生产使用

### 1.1、使用命令

- 建表：建表时，可以在 PROPERTIES 中指定属性 **"colocate_with" = "group_name"**，表示这个表是一个 Colocation Join 表，并且归属于一个指定的 Colocation Group
- 如果 **group_name **不存在，会新建一个 group
- 如果已存在，会检查当前表是否满足 CGS，如果满足会创建然后加入，根据已存在的数据分布进行映射
- 修改 group：修改其 Colocation Group 属性。示例：
```
ALTER TABLE tbl SET ("colocate_with" = "group2");
```

- 删除 group：删除一个表的 Colocation 属性：
```
ALTER TABLE tbl SET ("colocate_with" = "");
```

### 1.2、副本均衡 & 修复

**触发条件**：BE 数变化，节点挂掉，修改 CG 属性

Colocation 表的副本分布需要遵循 Group 中指定的分布，所以在副本修复和均衡方面和普通分片有所区别。=

Group Stable 属性：

- 当 Stable 为 true 时，表示当前 Group 内的表的所有分片没有正在进行变动，Colocation 特性可以正常使用。
- 当 Stable 为 false 时（Unstable），表示当前 Group 内有部分表的分片正在做修复或迁移，此时，相关表的 Colocation Join 将退化为普通 Join。
当某个 BE 不可用时（宕机、Decommission 等），需要寻找一个新的 BE 进行替换。DorisDB 会优先寻找负载最低的 BE 进行替换。替换后，该 Bucket 内的所有在旧 BE 上的数据分片都要做修复。迁移过程中，Group 被标记为 Unstable。

注意点：

1. 当前的 Colocation 副本均衡和修复算法，对于异构部署的 DorisDB 集群效果可能不佳。所谓异构部署，即 BE 节点的磁盘容量、数量、磁盘类型（SSD 和 HDD）不一致。在异构部署情况下，可能出现小容量的 BE 节点和大容量的 BE 节点存储了相同的副本数量。
1. 当一个 Group 处于 Unstable 状态时，其中的表的 Join 将退化为普通 Join。此时可能会极大降低集群的查询性能。如果不希望系统自动均衡，可以设置 FE 的配置项 disable_colocate_balance 来禁止自动均衡。然后在合适的时间打开即可。
1. **与普通分片 balance 的区别**：粒度不同，普通就是 bucket 分桶粒度，而 colocate 是以 group 为单位的，保证每个 group 下的数据分布合理。
balance 流程

1. 为需要复制或迁移的 Bucket 选择目标 BE
1. 标记 colocate group 的状态为 balancing（ubstable）
1. 对于需要复制或迁移的 Bucket，发起 Clone Job，Clone Job 会从 Bucket 的现有副本复制一个新副本目标 BE
1. 更新 backendsPerBucketSeq（维护 Bucket Seq 到 BE 映射关系的元数据）
1. 当一个 colocate group 下的所有 Clone Job 都完成时，标记 colocate group 的转态为 stable
1. 删除冗余的副本