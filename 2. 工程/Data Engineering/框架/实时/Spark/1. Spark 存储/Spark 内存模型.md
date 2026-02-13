---
title: "Spark 内存模型"
category: "工程"
tags: [Join, RAG, RDD, Shuffle, Spark]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark 内存模型

## 一、**内存介绍**

spark.executor.memory：配置Executor堆内存大小，如果是压缩表如orc，则需要对文件大小乘2~3倍，因为文件解压后所占空间会增长2~3倍。

![image](assets/PjWTdelbUoRLCdxEzMOc2goDnBf.png)

- Storage内存：缓存数据，例如缓存RDD、广播变量。
- Execution内存：存储shuffle、join、sort等计算过程中的临时数据。
- User内存：存储用户定义的对象实例。
- Reserved内存：预留内存，用来存储Spark内部对象实例。
优缺点：

- 优点：**内存存算分离，**实现对 Storage 和 Execution 内存各自独立的规划管理。
- 缺点：无法准确统计非序列化对象占用内存大小，标记释放的对象也可能没有被JVM回收，所以不能准确记录实际可用堆内存空间，无法避免OOM。
## 二、内存参数

- spark.memory.fraction：配置堆内存中统一内存比例大小，默认0.6。
- spark.dynamicAllocation.enabled：开启动态资源分配，spark可以根据当前作业负载动态申请和释放资源。
- spark.dynamicAllocation.maxExecutors：同一时刻最多可申请的Executor数量。
- spark.dynamicAllocation.minExecutors：同一时刻Executor的最小数量。
- spark.executor.cores：设置Executor中同时运行的Task数量（CPU core核数），多个Task共享一个Executor的内存，适当提高可以增加程序并发，执行更快，但会增加Executor内存压力，容易出现OOM。
- spark.memory.offHeap.enabled=true、spark.memory.offHeap.size：配置Executor堆外内存大小，可以直接使用工作节点的系统内存，存储序列化的二进制数据，存储结构只有Storage和Execution
注意：

- 堆外内存不归JVM管理，减少了不必要的内存开销，避免了频繁GC。
- spark.yarn.executor.memoryOverhead在2.3被废弃，使用spark.executor.memoryOverhead代替，表示Container预留内存，形式是堆外，用来保证稳定性，spark无法使用。
- 堆内和堆外内存以Job粒度划分，同一个Job要么全用堆外内存，要么全用堆内存，无法共享堆内堆外内存。
![image](assets/JSxodgd20oLknVxUrspcIxnZnNh.png)

**动态占用机制**

- 双方空间都不足时，存储到磁盘中，按照LRU置换；若己方空间不足而对方空间剩余时，可抢占对方空间。
- Execution内存被Storage占用后，可让对方将占用部分转存到磁盘或清除，然后归还借用空间。
- Storage内存被Execution占用后，需要等待Execution任务执行完毕后才释放。
![image](assets/BkludKrfmohuhyxEKCacrSJJnAe.png)
