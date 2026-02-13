---
title: "数据工程"
type: moc
domain: engineering
tags:
  - engineering
  - type/reference
---

# 🔧 数据工程

> Flink / Spark / Doris — 实时 + 离线数据基础设施

## Apache Flink
- [[Engineering/Flink/Flink 概述|Flink 概述]]
- [[Engineering/Flink/Flink 内存机制|内存机制]]
- [[Engineering/Flink/Flink 窗口和时间机制|窗口 & 时间]]
- [[Engineering/Flink/Flink CEP CDC|CEP & CDC]]
- **生产**：[[Engineering/Flink/生产/Flink 生产|生产概述]] / [[Engineering/Flink/生产/调优/Flink 调优|调优]] / [[Engineering/Flink/生产/监控指标|监控]]
- **CheckPoint**：[[Engineering/Flink/CheckPoint/Flink CheckPoint|CheckPoint]] / [[Engineering/Flink/CheckPoint/Chandy-Lamport 算法|Chandy-Lamport]] / [[Engineering/Flink/CheckPoint/Exactly Once 语义|Exactly Once]] / [[Engineering/Flink/CheckPoint/RocksDB 原理|RocksDB]]
- **开发**：[[Engineering/Flink/开发/Flink 开发|开发概述]] / [[Engineering/Flink/开发/Flink SQL UDF|SQL & UDF]] / [[Engineering/Flink/开发/Flink API（Java）|Java API]]
- **运行**：[[Engineering/Flink/运行/Flink 运行|运行概述]] / [[Engineering/Flink/运行/Flink 提交流程|提交流程]] / [[Engineering/Flink/运行/Flink 重启策略|重启策略]]

## Apache Spark
- [[Engineering/Spark/Spark 概述|Spark 概述]] / [[Engineering/Spark/Spark 算子|算子]]
- **存储**：[[Engineering/Spark/存储/Spark 存储|存储概述]] / [[Engineering/Spark/存储/RDD DataFrame DataSet|RDD vs DataFrame vs DataSet]] / [[Engineering/Spark/存储/Spark 内存模型|内存模型]]
- **RDD**：[[Engineering/Spark/存储/RDD/Spark RDD|RDD]] / [[Engineering/Spark/存储/RDD/Spark Lineage|Lineage]] / [[Engineering/Spark/存储/RDD/Spark CheckPoint|CheckPoint]]
- **SQL**：[[Engineering/Spark/SQL/Spark SQL|SQL]] / [[Engineering/Spark/SQL/Spark Join|Join]] / [[Engineering/Spark/SQL/Spark Shuffle|Shuffle]] / [[Engineering/Spark/SQL/Spark AQE + DDP|AQE]] / [[Engineering/Spark/SQL/谓词下推|谓词下推]]
- **生产运维**：[[Engineering/Spark/生产运维/Spark 生产运维|概述]] / [[Engineering/Spark/生产运维/调优/Spark 调优|调优]] / [[Engineering/Spark/生产运维/数据倾斜优化/数据倾斜优化|数据倾斜]] / [[Engineering/Spark/生产运维/小文件问题/小文件问题|小文件]]

## Apache Doris
- [[Engineering/Doris/Doris 概述|Doris 概述]] / [[Engineering/Doris/Doris 架构|架构]]
- **底层存储**：[[Engineering/Doris/底层存储/Doris 底层存储|存储概述]] / [[Engineering/Doris/底层存储/存储结构|存储结构]] / [[Engineering/Doris/底层存储/读取流程|读取]] / [[Engineering/Doris/底层存储/删除流程|删除]]
- **写入**：[[Engineering/Doris/底层存储/写入流程/写入流程|写入流程]] / [[Engineering/Doris/底层存储/写入流程/数据导入/Doris 数据导入|数据导入]]
- **治理**：[[Engineering/Doris/治理/Doris 治理|治理]] / [[Engineering/Doris/治理/Doris 查询计划|查询计划]] / [[Engineering/Doris/治理/Doris 看板指标分析|看板分析]]
- **生产**：[[Engineering/Doris/生产/Doris 生产|生产]] / [[Engineering/Doris/生产/Join/Join|Join]] / [[Engineering/Doris/生产/索引/索引|索引]]

## 相关 MOC
- ↑ 上级：[[00-Home/HOME]]
