---
title: "Spark"
category: "工程"
tags: [Hadoop, Hive, Join, RDD, SQL]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark

## **一、Spark 框架**

### **1.1、Spark VS. Hadoop**

 

Hadoop

Spark

类型

分布式基础平台，包含计算、存储、调度

分布式计算工具

场景

大数据集批处理

迭代计算，交互式计算，流计算

价格

对机器要求低，便宜

对内存有要求，相对较贵

编程范式

只有map和reduce两个操作，欠缺表达能力，算法适应性差；API较底层，编码难以上手

使用RDD统一抽象，API较为顶层，方便使用；提供了很多转换，实现了很多基本操作，如sort、join等

延迟

中间结果存在HDFS磁盘上，延迟大

中间结果存在内存中，延迟小，内存不足时溢写磁盘

运行方式

 

reduce task需要等所有map task全部执行完才能开始执行

迭代计算，一个分区内的转换可以以流水线的方式并行执行，只有分区不同的转换，才需要shuffle操作

### 1.2、**Spark VS. MapReduce**

1. MapReduce将计算中间结果保存在磁盘，IO开销大，如果配置在HDFS，网络传输时间长，且需要进行数据块切分、备份等工作，耗时长。Spark 基于内存。
1. Spark使用DAG进行查询优化，减少磁盘IO次数、不必要的计算过程（流水线式并行计算）。
1. 支持缓存RDD多次复用。
1. Spark采用多线程模型，相比 MapReduce 每个 task 都是进程可以大幅减少开销。
### **1.3、Spark VS. Hive**

- Spark 相比 Hive 消除了冗余的 HDFS 读写：Hadoop 每次 shuffle 操作后，必须写到磁盘，而 Spark 在 shuffle 后不一定落盘，可以 cache 到内存中，以便迭代时使用。如果操作复杂，很多的 shufle 操作，那么 Hadoop 的读写 IO 时间会大大增加。
- 消除了冗余的 MapReduce 阶段：Hadoop 的 shuffle 操作一定连着完整的 MapReduce 操作，冗余繁琐。而 Spark 基于 RDD 形成 DAG 迭代计算，尽量合并过程。
- 基于线程：Hadoop 每次 MapReduce 操作的 Task 基于进程。而 Spark 每次 MapReduce 操作是基于线程的，只在启动 Executor 是启动一次 JVM。
如果 shuffle 次数很少，hive 性能也可能比 spark 更优，一般情况下 spark 更好，我们生产环境也是用 spark3

### 1.4、**Spark 特点**

Spark是基于MR实现的分布式计算框架。

- **速度快**：Spark基于内存运算。
- **易用**：支持Java、Python、R和Scala的API，还支持交互式的Python和Scala的shell。
- **通用**：提供统一的解决方案，可以用于批处理、交互式查询（Spark SQL）、实时流处理（Spark Streaming）、机器学习（Spark MLlib）和图计算（GraphX）。
- **部署方便**：可以使用Yarn调度，可以处理Hadoop支持的数据，包括HDFS、HBase等。Spark也可不依赖第三方资源管理框架，它实现了Standalone作为其内置的资源管理框架。
## 二、**Spark架构**

### 2.1、**基本组件**

- Application：应用程序。
- Driver：运行main()函数，创建SparkContext。SparkContext与Cluster Manager通信，负责资源申请、任务分配和监控等。
- Cluster Manger：分配和管理Worker上运行所需资源（Yarn Resource Manager）。
- Executor：是Application运行在Worker上的一个进程，负责运行Task，每个Application有各自的一批Executor，每个Executor包含一定的资源来运行分配给它的任务。
- Worker：集群中可以运行Application代码的节点（Yarn Node Manager）。
![image](assets/Y9z4dKh1Mot06gxrLsAconSmnxD.png)

### **2.2、存储组件**

- 存储对象：RDD缓存、shuffle中间文件、广播变量。
- BlockManager：在Executor端负责统一管理和协调数据的本地存取与跨节点传输。
- 与 Driver 端 BlockManagerMaster 通信，定期汇报本地数据元信息、拉取全局数据存储状态。不同 Executor 的 BlockManager 之间会跨节点推送和拉取数据。
- BlockManager 组合存储系统内部组件来实现数据存取，提供两种存储组件：MemoryStore 和 DiskStore，分别用于管理数据在内存和磁盘的存取。
1. 解释一下Spark Master的选举过程
参考：https://blog.csdn.net/zhanglh046/article/details/78485745
