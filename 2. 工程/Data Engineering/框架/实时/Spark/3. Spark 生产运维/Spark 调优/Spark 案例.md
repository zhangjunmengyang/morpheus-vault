# Spark 案例

## 一、必要知识

**我们用的是hive on spark吗？还是Spark SQL？**

我们**使用的是Spark SQL**，两者很相似，但仍然有区别。

1. 首先不是同一家公司的产品
1. hive on spark是以hive sql为api，底层 spark core 来替换 mapreduce 作为计算引擎。Spark SQL是spark sql的api，spark core作为执行引擎。有极个别的语法，两边是不同的
钨丝计划（Tungsten）做了什么使得spark2.x比spark1.x速度有极大提升？

1. 对Memory的使用，Tungsten使用了off-heap，也就是JVM之外的内存空间（这就好像C语言对内存的分配、使用和销毁），此时Spark实现了自己的独立的内存管理，就避免了JVM的GC引发的性能问题，其实还包含避免序列化和反序列化；
1. 对于Memory管理方面一个至关重要的内容Cache，Tungsten提出了Cache-aware computation，也就是说使用对缓存友好的算法和数据结构来完成数据的存储和复用；
1. 对于CPU而言，Tungsten提出了CodeGeneration，其首先在Spark SQL使用，通过Tungsten要把该功能普及到Spark的所有功能中；
原理参考 

## 二、案例分析

### 2.1、Shuffle Spark任务中间文件过多

#### 2.1.1、问题定位

发现任务执行时间超时

特征：发现其中一个stage耗时过多(1.9h)，但是读取的数据量并不大，启动的task数量也不多

![image](assets/SP8nd5TJdoQE9Qx2jrPcE7U4nfh.png)

查看具体的 stage summary 也没有发现数据倾斜，shuffle read size / record 的 max 分位数等都比较接近，没有倾斜。

但是发现 每个task的执行时间大多消耗在 shuffle read blocked time上了（shuffle read blocked time表示reduce在拉取文件中的时间开销）

日志中发现有大量的 fetch block 也印证这点。这么多 block 肯定是来源于 spark 的 shuffle 过程，所以核心还是对中间文件的治理。

小文件的常见场景：

- 上游表有大量小文件，map stage读取时没有合并文件，一个文件创建一个task，于是产生了 M 个map task，落地了 M 个中间结果文件，如果 spark.sql.shuffle.partitions 设置为 N，就会产生 M  *N 次fetch block。
- 当前任务有M个子查询的union all，那么 shuffle 产生的文件数也是会随之膨胀
### 2.1.2、解决方案

参考 
