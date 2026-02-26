---
title: "1. Doris 架构"
type: concept
domain: engineering/doris
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris
  - type/concept
---
# 1. Doris 架构

Doris是一个MPP的OLAP系统，以较低的成本提供在大数据集上的高性能分析和报表查询功能。

MPP (Massively Parallel Processing)，即大规模并行处理。简单来说，MPP是将任务并行的分散到多个服务器和节点上，在每个节点上计算完成后，将各自部分的结果汇总在一起得到最终的结果(与Hadoop相似)

现在大数据存储与处理趋势：MPPDB+Hadoop混搭使用，用MPP处理PB级别的、高质量的结构化数据，同时为应用提供丰富的SQL和事物支持能力

Doris 实现低成本，可线性扩展，高可用，高查询、加载性能的服务，更适合多表关联和多维分析场景

## 一、Doris 引擎架构

### 1.1、架构组成

- Doris主要整合了Google Mesa（数据模型），Apache Impala（MPP Query Engine) 和 Apache ORCFile (存储格式，编码和压缩) 的技术。 
1. 分布式存储引擎：Mesa是一种高度可扩展的分析数据存储系统，用于存储与Google的互联网广告业务有关的关键测量数据。Mesa旨在满足一系列复杂且具有挑战性的用户和系统需求，包括接近实时的数据提取和查询能力，以及针对大数据和查询量的高可用性，可靠性，容错性和可伸缩性。
1. MPP SQL查询引擎：Impala是为Hadoop数据处理环境从头开始构建的现代开源MPP SQL引擎。
1. ORCFile：采用列式存储（只访问查询涉及的列，能大量降低系统I/O；列数据相对来说比较类似，压缩比更高；每一列由一个线索来处理，更有利于查询的并发处理）。
![image](C7NKdWQIFoOmNqxu0SPcNQfpnrg.png)

**FE**：负责存储和维护元数据、接收查询请求、生成查询计划、调度查询执行、返回查询结果

- 主要有三个角色：Leader、Follower和Observer。
- Leader是从Follower中选主获得的，三个Follower能保证元数据读写的高可用。
- Observer用于扩展接收查询请求的能力，不参与选主
**BE**：负责物理数据的存储和计算，依赖FE生成的查询计划，分布式地执行查询

- 数据存储一般都是三副本，会分布在3台Host不同的BE上，保证了数据的可靠性。
**详细解释下 FE 和 BE ？**

- FE 主要负责元数据的管理、存储，以及查询的解析等；一个用户请求经过 FE 解析、规划后，具体的执行计划会发送给 BE，BE 则会完成查询的具体执行。
- 管理元数据, 执行SQL DDL命令, 用Catalog记录库, 表, 分区, tablet副本等信息。
- FE高可用部署, 使用复制协议选主和主从同步元数据, 所有的元数据修改操作, 由FE leader节点完成, FE follower节点可执行读操作。 元数据的读写满足顺序一致性。  FE的节点数目采用2n+1, 可容忍n个节点故障。  当FE leader故障时, 从现有的follower节点重新选主, 完成故障切换。
- FE的SQL layer对用户提交的SQL进行解析, 分析, 改写, 语义分析和关系代数优化, 生产逻辑执行计划。
- FE的Planner负载把逻辑计划转化为可分布式执行的物理计划, 分发给一组BE。
- FE监督BE, 管理BE的上下线, 根据BE的存活和健康状态, 维持tablet副本的数量。
- FE协调数据导入, 保证数据导入的一致性。
- BE 节点主要负责数据的存储、以及查询计划的执行
- BE管理tablet副本, tablet是table经过分区分桶形成的子表, 采用列式存储。
- BE受FE指导, 创建或删除子表。
- BE接收FE分发的物理执行计划并指定BE coordinator节点, 在BE coordinator的调度下, 与其他BE worker共同协作完成执行。
- BE读本地的列存储引擎, 获取数据, 通过索引和谓词下沉快速过滤数据。
- BE后台执行compact任务, 减少查询时的读放大。
- 数据导入时, 由FE指定BE coordinator, 将数据以fanout的形式写入到tablet多副本所在的BE上。
**FE 包含的三种角色的理解？**

- leader跟follower，主要是用来达到元数据的高可用，保证单节点宕机的情况下，元数据能够实时地在线恢复，而不影响整个服务。
- observer只是用来扩展查询节点，就是说如果在发现集群压力非常大的情况下，需要去扩展整个查询的能力，那么可以加observer的节点。observer不参与任何的写入，只参与读取。
### 1.2、表存储

![image](SRycdEh3yooAB1xoh5VcKouonxh.png)

核心层级：**Table -> Rollup -> partition -> tablet -> rowset -> segment**

- Index：这里的 index 主要是指物化视图（Rollup），因为属于存储的一部分，所以写入是和表的 schema 同步写入的（而不是通过什么同步机制异步任务等）
- partition：最小的逻辑和管理单位，因为数据 导入/ 重导 最细粒度也只能针对 partition
- tablet：也就是分桶的存储，对分桶列 hash 得到，是数据移动、复制的最小单位，比如 balance 的过程也是以分桶为单位
- rowset：一个 rowset 代表 一个 tablet 单次导入的数据，从最旧一直读到最新才能得到正确的查询结果，也是 compaction 存在的原因，可以读时合并、写时合并
- segment： 如果单次导入数据量大，一个 rowset 可能有多个 segment
注意：

1. 聚合模型和Uniq模型需要根据Key对Value进行聚合或Replace，根据 Key 在写入的时候进行聚合，最终对于一个Segment文件是没有重复Key的。查询的时候通过Key对所有读取上来的数据进行Merge，返回正确的结果。
1. 同样原理，只有Key上能谓词下推到存储层，而 Value 不能，因为Value上的谓词由于可能上一个Rowset的数据没有被过滤&下一个Rowset的数据被过滤了导致语义不正确，所以只会下推到 Base Rowset
1. （前缀索引的基石）对于所有模型，单个Segment在写入完成后会**保证有序**，单个Rowset在Compaction完成后会保证**多个Segment有序**。由于Segment的有序性，Doris能够基于Key的前缀列构建前缀索引。在查询时会将前缀Key上的条件拆分成**多个Range**进行查询，并且该Range能够命中前缀索引加速过滤。
借助 MySQL 协议用户使用任意 MySQL 的 ODBC/JDBC以及MySQL 的客户端，都可以直接访问 Doris。
