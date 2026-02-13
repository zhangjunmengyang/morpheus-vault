---
title: "Flink SQL / UDF"
category: "工程"
tags: [Flink, Hive, Join, Planner, SQL]
created: "2026-02-13"
updated: "2026-02-13"
---

# Flink SQL / UDF

## 11、Flink SQL

### 11.1、背景

Flink SQL的出现是为了简化计算模型，降低使用门槛，由阿里巴巴的Blink实现。

Flink本身是流批统一的计算框架，Table API是内嵌在Java和Scala中的查询API，它允许组合一些关系运算符的查询，比如select、filter、join等；Flink SQL支持直接在代码中写SQL，实现查询操作。

Flink planner：提供运行时环境和生成执行计划功能，分为old planner和blink planner。

- 区别：blink将批处理作业视为流处理的特殊情况，所以blink不支持表和DateSet之间转换，批处理作业不会被转换为DataSet，而是跟流处理一样，转换为DataStream。
### 11.2、原理

Flink SQL基于Apache Calcite实现，Apache Calcite作为SQL计算引擎，支持标准SQL语言，提供多种查询优化和连接各种数据源的能力，功能：

- SQL解析：通过JavaCC编写SQL语法描述文件，将SQL解析成AST语法树。
- SQL校验：
- 验证SQL语句是否规范。
- 与元数据结合验证SQL的Schema、Field、Function是否存在，输入输出是否相符。
- SQL优化：对逻辑计划进行优化，得到优化后的物理执行计划。
- RBO基于规则的优化器：裁剪原有表达式，遍历规则，只要满足条件就转化，生成最终执行计划。规则包括分区裁剪、列裁剪、谓词下推、常量折叠等。
- CBO基于代价的优化器：保留原有表达式，基于统计和代价模型，生成等价关系表达式，最终取代价最小的执行计划。
- SQL生成：将物理执行计划转为特定平台的可执行程序，如Flink、Hive等。
## 12、Flink UDF

### 12.1、自动类型推导

函数的参数和返回类型都必须映射到数据类型，自动类型推导会检查求值方法，派生出函数结果类型，使用@FunctionHint和@DataTypeHint实现。

@DataTypeHint用于修饰求值方法，@FunctionHint用于修饰函数类或求值方法

示例：

```
*// 为函数类的所有求值方法指定同一个输出类型*
**@FunctionHint(**output **=** **@DataTypeHint(**"ROW<s STRING, i INT>"**))**
**public** **static** **class** **OverloadedFunction** **extends** TableFunction**<**Row>
```

### 12.2、函数类别

标量函数：

- 输入参数：0个或多个标量值
- 输出：1个标量值
- 继承类：ScalarFunction
- 重写方法：eval
- 使用方式：`select func(input...)`
表值函数：

- 输入参数：0个或多个标量值
- 输出：返回任意多行，返回的每一行可以包含1或多列，如果输出只有一列，会省略结构化信息并生成标量值。
- 继承类：TableFunction<T>
- 重写方法：eval
- 使用方式：`join/left join lateral table func(input...) as alias(output...) on true`
- on true实际就是on 1=1
聚合函数：

- 输入参数：1行或多行，每行有1个或多列
- 输出：1个标量值
- 继承类：AggregateFunction<output type, accumulator>
- 重写方法：
- `createAccumulator()`：创建数据结构accumulator用于存储聚合中间结果
- `accumulate(accumulator, input...)`：每来一条数据，更新accumulator
- `getValue(accumulator)`：计算和返回最终结果
- 使用方式：`select a, func(input) from table group by a`