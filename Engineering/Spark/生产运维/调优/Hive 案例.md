---
title: "Hive 案例"
type: concept
domain: engineering/spark/生产运维/调优
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/生产运维/调优
  - type/concept
---
# Hive 案例

## 一、必要知识

Hive 内核基于 MapReduce（虽然现在基本公司平台会替换为 Spark，但还是看下优化思路）

MR 原理：

## 二、优化 case

背景：Hive 性能差，调优过程

### 2.1、长时间没有 start job

原因多数是由于job的split阶段耗时长，一般由下面两个具体原因共同决定

- 查询sql中所涉及的表存储文件数过多（超过1w认为是过多，超过10万就非常多了，可能会卡住几个小时）
- 所归属的Name Node压力比较大
### 2.2、长时间没有打印mapper/reducer数量

一般是当前yarn的负载比较高，waiting for AM container to be allocated, launched and register with RM.

该原因通常不会卡住太久时间

### 2.3、长时间卡在map = 0%,  reduce = 0%,没有打印Cumulative CPU耗时

这是是卡在task创建、资源申请阶段，可能由于下面的原因之一

- 需要的task数量较多，创建task需要耗时较长(hive的机制是先把所有task进程都创建出来，再统一分配资源)
- 队列资源不足，task创建完成后一直在等待(pending)
### 2.4、长时间卡在map = 99%,  reduce = 33%

当前集群参数（mapred.reduce.slowstart.completed.maps）设置，map阶段完成95%后，reduce阶段就可以开始执行了

但是由于存在慢节点/某些map计算时被杀掉重启/map处理数据量分布不均等原因，个别map晚于整体进度。所以reduce任务无法完全执行完成。

注意：MR中的shuffle阶段在日志打印中归属于reduce过程（推测reduce在33%进度以内都是在做拉取map文件），之所以reduce卡在33%是因为有 **部分map任务没有结束**，shuffle的copy阶段无法全部完成

该原因一般来说不会卡住太久，如果是由于map处理数据量分布不均的原因，可以在sql前加上一些参数调整

```
set mapred.min.split.size.per.node=64000000;-- 小文件合成后分配给一个map
set mapred.min.split.size.per.rack=64000000;-- 小文件合成后分配给一个map
```

### 2.5、长时间卡在map = 100%,  reduce = 99%

一般是由于数据倾斜造成的，也是hive慢查询的最常见问题

reduce阶段的数据倾斜一般是进行了distinct/全局order by /join/group by 等操作

### 2.6、map 端进度由 100% 回退到 99%，reduce端进度不变

map节点连接不上或者结果文件有问题，所以触发了map端 **重新计算 **的容错机制，进度回退（当前配置是最多4次回退）。一般为偶发，不需要处理。如果稳定复现，一般是内存溢出的原因导致，需要具体问题具体分析
