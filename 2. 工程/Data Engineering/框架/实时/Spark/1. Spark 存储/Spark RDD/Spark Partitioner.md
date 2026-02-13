---
title: "Spark Partitioner"
category: "工程"
tags: [RDD, Shuffle, Spark, 数据倾斜]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark Partitioner

## 一、基本概念

### 1.1、定义

- Partitioner对象定义了 Pair RDD基于Key分区的实现；
- 将每一个Key映射到一个分区ID，编号[0, 分区数 - 1)
### 1.2、作用

- 决定shuffle过程的分区个数；
- 决定map端的每一条数据记录应该分配到哪一个分区；
- 决定RDD分区数量；
## 二、分类

有两种实现：Hash、Range

### 2.1、HashPartitioner

使用key值的hashcode除以子RDD分区个数并取余

### 2.2、RangePartitioner

- 基于对传入RDD内容的采样得到一系列key对应的边界值范围，基于该边界值序列，对可排序的数据记录进行粗略分区；
- 需要注意的是：在抽取样本的个数少于指定分区个数的情况下，实际通过RangePartitioner创建的分区个数不一定完全与指定的分区个数相同
### 2.3、对比

- HashPartitioner，无需遍历整个RDD数据，尽可能使得同一分区获得相同数量的Key对应的数据
- RangePartitioner，采用水塘采样会遍历整个RDD数据一次，获取其数据count值，尽可能使得所有分区分配得到相同多的数据
1. 在数据分布比较均匀的情况下，使用HashPartitioner可以具有更好的运行效率；
1. 而在数据分布不均匀，key值对应数据量差距较大，而数据总量又较大的情况下，HashPartitioner会使某一分区数据倾斜，在shuffle过程中，由于fetch数据过多而容易造成某节点rpc过高，造成executor heartbeat time out，出现Lost Executor的问题，从而该task失败，进而引发整个作业失败；而使用RangePartitioner，由于数据均匀分配到各节点，使得rpc请求压力分散，不易出现该问题。