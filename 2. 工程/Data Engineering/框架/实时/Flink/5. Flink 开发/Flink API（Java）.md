---
title: "Flink API（Java）"
category: "工程"
tags: [Flink, Hive, SQL, 分布式训练, 反压]
created: "2026-02-13"
updated: "2026-02-13"
---

# Flink API（Java）

### **Flink API（DataSet/DataStream）**

DataSet：source 部分来源于文件、表、Java 集合

DataStream：source 部分来源于消息中间件

编程模型：起始于一个或多个 source，经过 transformation，按照 DAG 流转，终止与一个或多个 sink

1. map：接受一个元素输入，根据自定义处理逻辑，映射输出
```
SingleOutputStreamOperator<Object> mapItems = items.map(
    item -> item.getName()
);
// 也可以写自己的 map 函数
SingleOutputStreamOperator<Object> mapItems = items.map(
    new MyMapFunction()
);
```

1. flatmap：接受一个元素，返回 0 到多个元素，和 map 类似，但当返回值是列表的时候会将列表平铺（逐个输出）
1. filter：筛掉某些元素
1. KeyBy：按照某些 key 进行分组，使用KeyBy时会把 datastream 转成 KeyedStream，其中元素会根据用户传入参数进行分组，容易导致**反压和数据倾斜**
1. Aggregations：聚合函数，sum, max, min, maxby, minby
1. min 和 minby 区别：min 或者 max 在数据流中，只会更新关注的字段位置的值比如第三位(0, 2, 3)， 可能这个 3 是来自第三个数据，但 2 还停留在第一个数据；minby 则不同，返回数据是准确的，返回整个元素
1. 注意！！！：
1. 尽量避免在一个无限流上使用 Aggregations，因为状态数据不会清理，无限增长
1. 同一个 KeyedStream 只能调用一次Aggregations函数，同样原因，可能造成混乱
1. reduce：在每一个 keyedstream 上生效，按照用户自定义的聚合逻辑进行分组聚合
### **Flink API Table**

跟Hive一样：底层的 SQL 解析、优化、执行都是 Calcite，过程：无论流批，都是经过对应的转换器 parser 转换为节点树，SQLNode tree，然后生成逻辑执行计划，优化后生成物理执行计划提供给 DataSet/DataStream

**分布式缓存使用**：

1. 在环境中注册，可以是本地文件，也可以是 HDFS，然后可以命名
1. 使用的时候直接就可以用命名读取，getdistributedcache， 然后 getfile