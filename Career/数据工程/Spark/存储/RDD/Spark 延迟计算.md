---
title: "Spark 延迟计算"
type: concept
domain: engineering/spark/存储/rdd
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark/存储/rdd
  - type/concept
---
# Spark 延迟计算

在 RDD 的编程模型中，一共有两种算子，Transformations 类算子和 Actions 类算子。开发者需要使用 Transformations 类算子，定义并描述数据形态的转换过程，然后调用 Actions 类算子，将计算结果收集起来、或是物化到磁盘。

在这样的编程模型下，Spark 在运行时的计算被划分为两个环节。

1. 构建 DAG：基于不同数据形态之间的转换，构建计算流图（DAG，Directed Acyclic Graph）；
1. Actioin 触发：通过 Actions 类算子，以回溯的方式去触发执行这个计算流图。
换句话说，开发者调用的各类 Transformations 算子，并不立即执行计算，当且仅当开发者**调用** Actions 算子时，之前调用的转换算子才会付诸执行。在业内，这样的计算模式有个专门的术语，叫作“延迟计算”（Lazy Evaluation）。

**为什么Spark 在执行的过程中，只有最后一行代码会花费很长时间，而前面的代码都是瞬间执行完毕的呢？**

这里的答案正是 Spark 的延迟计算。flatMap、filter、map 这些算子，仅用于构建计算流图，因此，当你在 spark-shell 中敲入这些代码时，spark-shell 会立即返回。只有在你敲入最后那行包含 take 的代码时，Spark 才会触发执行从头到尾的计算流程，所以直观地看上去，最后一行代码是最耗时的。

Spark 程序的整个运行流程如下图所示：

![image](WGsxdhM5voPOAMxnXaLcnhvZnzh.png)

常见算子见 
