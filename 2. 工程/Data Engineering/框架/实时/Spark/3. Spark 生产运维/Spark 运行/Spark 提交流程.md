---
title: "Spark 提交流程"
category: "工程"
tags: [Spark, 优化]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark 提交流程

## **一、基本运行流程**

1. 客户端提交 Spark Application ，Driver运行main()函数生成SparkContext，SparkContext向Cluster Manager注册，并申请Executor运行资源。
1. Cluster Manager 为 Executor 分配资源并启动，Executor运行情况随着心跳发送给Cluster Manager。
1. SparkContext构建DAG，DAGScheduler将DAG拆分为Stage，并创建计算任务Tasks和任务组TaskSet。
1. SchedulerBackend获取Executor的资源情况，通过WorkerOffer提供空闲Executor计算资源。
1. TaskScheduler按照调度规则决定优先调度哪些任务/组。不同stage之间调度模式有FIFO和Fair；同一stage内不同任务会按照本地性级别进行分发，移动数据不如移动计算。
1. TaskScheduler对选出的任务序列化后，交给SchedulerBackend，SchedulerBackend将任务分发到Executor中，SparkContext将计算代码发送给Executor。
1. Task在Executor上运行，并通过心跳将执行状态反馈给TaskScheduler，后者会重试运行失败的Task。
1. 所有Task运行完毕后，SparkContext向Cluster Manager注销并释放所有资源。
## **二、运行特点**

- 每个Application有专属Executor进程，并以多线程方式执行Task，减少多进程启动开销，也意味着不能跨应用共享数据，除非使用外部存储。
- Spark与Cluster Manager无关，只要能获取Executor进程，并保持相互通信即可（Master切换不会影响计算）。
- 提交SparkContext的Client应该靠近Worker节点，最好在一个机架中，因为在运行过程中，SparkContext和Executor之间有大量信息交换。
- Task采用了数据本地性和推测执行优化机制，前者指尽量将计算移动到数据所在节点，移动计算比移动数据的网络开销要小的多；后者指一个Stage里运行慢的Task，会在其他Executor上再次启动，最终保留最先完成的Task计算结果，干掉其他Executor上运行的相同Task实例。