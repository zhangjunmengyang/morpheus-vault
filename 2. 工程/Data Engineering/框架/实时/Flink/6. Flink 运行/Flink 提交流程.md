---
title: "Flink 提交流程"
category: "工程"
tags: [Flink, YARN, 流处理]
created: "2026-02-13"
updated: "2026-02-13"
---

# Flink 提交流程

## **提交流程**

---

**程序起点：CliFrontend**

---

**CliFrontend**创建 yarn 客户端：执行起点：flink bin 中 flink 脚本，找到 **cliFrontend 中的 main 方法**

1. **参数解析**：run 里面调用 getCommandLine 方法，然后getCommandLine调用的是封装了好几层的 parse，分别解析“--”和“-”形式的参数获取有效配置
1. **封装**成 commandline；run 提供三个客户端接口：依次添加 Generic、Yarn、Default 三种客户端，依次添加，判断有没有 yarn ID 等；
1. **参数封装、执行 execute()、生成 StreamGraph**
1. run 里面 executeProgram **调用用户代码 main 方法**
1. 在 stream execution environment 里面**调用 execute 方法**，**生成 StreamGraph**，然后用这个 **StreamGraph** 根据提交模式选择匹配的工厂、执行器
**YarnJobClusterExecutor**

1. Executer：**生成 JobGraph**
1. **集群描述器：**获取集群配置参数、上传 jar 包到 HDFS、封装 AM 参数和命令
1. 最后 **YarnClient 提交应用 到 YRM** Yarn Resourse Manager
---

**YarnJobClusterEntryPoint：AM 执行的入口类**

---

1. 创建 Dispatcher、ResourseManager：
1. 创建启动 **RM（Yarn Resourse Manager）**，里面  **SM（Slot Manager）真正管理资源**
1. 创建 Yarn 的 RM 和 NM 客户端
1. 启动 SM：leadership 选举、超时检查、如果没有 job 在执行释放 TM
1.  创建启动 Dispatcher
1. Dispatcher 启动 **JM（JobManager）**，**里面的 SlotPool 真正申请资源，同时转换为 ExecutorGraph**
1. JM 的 SlotPool 向 RM 的 SlotManager 申请 slot，SlotManager 向 Yarn 申请资源（启动新节点 TM）
1. 这些申请通信通过 RPC 调用，从 gateway 访问
---

**YarnTaskExecutorRunner：Yarn 模式下 TM 的入口类**

---

1. TM启动，并向 RM 注册 slot
1. RM 分配 slot，TM 需要根据 RM 指令提供 offset 给 JM 的 SlotPool
1. JM 提交任务给 TaskExecutor 去执行