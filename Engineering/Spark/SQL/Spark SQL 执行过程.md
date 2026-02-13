---
title: "Spark SQL 执行过程"
type: concept
domain: engineering/spark/sql
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/sql
  - type/concept
---
# Spark SQL 执行过程

**Spark SQL **是Spark中用于处理结构化数据的一个模块，可以将计算任务通过SQL形式转化为RDD再提交到集群执行计算，类似于Hive通过SQL形式将数据计算任务转化为MapReduce，简化了编写Spark数据计算程序的复杂性，且执行效率比MapReduce高。

**Spark SQL架构由两部分组成：**

- Core：负责处理数据输入/输出，从不同数据源获取数据，然后转化为DataFrame。
- Catalyst（核心）：负责处理整个查询语句的执行过程，包括解析（Parser）、分析（Analyzer）、优化（Optimizer）、生成物理计划（Planner）等，可以将DataFrame、Dataset转化为RDD。
## 一、**执行流程**

1. 使用 SessionCatalog **保存元数据**，包括数据库名、表名、字段名、字段类型等。
1. 使用 Antlr 进行SQL语法解析，构建未解析的逻辑计划（Unresolved Logical Plan）。
**未解析的逻辑计划 ****内容**：

1. **Unresolved Relation：**尚未与具体数据源或数据库表绑定的**表或视图**。
1. **Unresolved Function：**尚未被解析的**函数调用**，没有确认是否存在，以及具体实现。
1. **Unresolved Attribute：**尚未被解析的列，仅包含**列名信息**，没有确定这个列名是否有效，以及属于哪个表或者视图
**Q：为什么要未解析的逻辑计划？**

1. **平台无关性**：不包含任何数据库的执行细节，更容易适应不同数据源。
1. **方便优化**：是SQL语句的结构化表示，方便后续处理（语法验证、分析、优化等）。
1. 使用Analyzer组件中定义的分析规则，结合SessionCatalog元数据，将未解析的逻辑计划转换为已解析的逻辑计划（Analyzed Logical Plan），涉及识别表、列和函数，并将他们与数据库中实际元数据匹配，从而确保引用的数据库对象都是有效的。
分析规则都有哪些？

1. ResolveRelations：将UnresolvedRelation转为具体的数据源。
1. ResolveReferences：将UnresolvedAttribute转为具体的列引用。
1. ResolveFunctions：将UnresolvedFunction解析为具体的函数调用。
1. ResolveAliases：解析列别名。
1. Typecoerction：进行数据类型转化和兼容性检查。
1. 使用Optimizer组件中定义的优化规则（逻辑优化和物理优化），对已解析的逻辑计划进行优化，完成列裁剪、谓词下推等工作，生成优化的逻辑计划（Optimized Logical Plan）。
**逻辑优化规则都有哪些？**

1. **谓词下推（Predicate Pushdown）**：将过滤条件尽可能下推到接近数据源的位置，减少扫描的数据量。
1. **常量折叠（Constant Folding）**：预计算常量表达式的值。
1. **列裁剪（Projection Pushdown）**：扫描数据源时，只读取与查询相关的列，减少扫描的数据量。
**物理优化规则都有哪些？**

1. **Join Selection：**选择最高效的Join算法，执行效率排序：Broadcast Hash Join（BHJ）、Shuffle Sort Merge Join（SMJ）、Shuffle Hash Join（SHJ）、Broadcast Nested Loop Join（BNLJ）、Cartesian Product Join（CPJ）
1. 使用SparkPlanner组件中定义的规则（Preparation rules），将优化的逻辑计划转换为可执行的物理计划（Physical Plan）。
SparkPlanner 规则

1. **EnsureRequirements**：补充必要的操作步骤，比如添加shuffle（验证输出分区数，如果不一致，则添加shuffle重分区）、sort（满足排序需求）等操作。
1. **CollapseCodegenStages**：将连续的支持代码生成的物理计划整合到一起，即全阶段代码生成（WholeStageCodegen），可以将多个算子合并为一个java函数，提高执行速度。
1. **ReuseExchange**：共享广播变量和Shuffle中间结果，避免重复Shuffle。
1. **ReuseSubquery**：复用同样的子查询结果，避免重复计算。
1. **PlanSubquery**：对子查询应用Preparation rules。
1. 调用SparkPlan的execute()，执行物理计划计算RDD。