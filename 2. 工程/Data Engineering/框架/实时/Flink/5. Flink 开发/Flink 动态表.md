# Flink 动态表

## 1、简介

## 2、动态表

动态表是Flink Table API和Flink SQL支持流数据的核心概念，与静态表比，动态表会随时间变化，但可以像静态表一样查询动态表，只不过需要产生连续查询。

![image](assets/VV6bdLVumoieNuxn7Dbcpf2anfh.png)

动态表的连续查询，与物化视图相似：每天等数据源就绪后，调度物化视图sql查询产生结果后缓存。连续查询就是一种实时视图维护技术，会不断查询动态输入表，更新动态结果表。

流程：输入流->动态输入表->执行连续查询->动态结果表->输出流。

动态结果表可能是只有一行不断更新的changelog，也可能是没有更新和删除的仅插入表，或者介于两者之间。将动态表转为流或写入外部系统时，需要对更改进行编码描述动态表变化：

- Append-only：输出结果仅有insert操作。
- Retract：包含两种消息类型，即添加消息和撤回消息，将insert编码为添加消息，将delete编码为撤回消息，将update编码为对先前行的撤回消息和对新增行的添加消息，从而将动态表转为回撤流。
- 输出结果有「-」，「+」两种，「-」表示撤回消息，「+」表示添加消息，两种数据都会写入下游，下游需要正确处理「-」，「+」，防止重复计算。
- 不需要主键，撤回消息明确指定了需要撤回的数据。
![image](assets/UEJ6drVFaog27yxpYRIcpRernrb.png)

- Upsert：包含两种消息类型，upsert和delete，转为upsert流的动态表需要唯一键（可以是复合主键，主键用于确定新消息影响哪一行）。将insert和update编码为upsert消息，将delete编码为delete消息。与Retract区别在于update使用单个消息编码，不会产生回撤，效率更高。
![image](assets/WPp8d3LqQoGQ3qxRZ2ac8HAbncb.png)
