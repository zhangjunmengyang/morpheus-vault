# Spark 海量数据多分区落表小文件治理

场景：知识图谱场景，百亿级，一级分区日期，二级分区tuple类型

问题发现：任务执行时间不长，但总时间很长比如两个小时，发现向 HDFS 提交花了1 个小时，小文件过多，有 5w 多个小文件。

### 一、尝试

首先，任务有 shuffle 阶段

```
set spark.sql.adaptive.shuffle.targetPostShuffleInputSize=134217728;-- 128m
```

即使调大，依旧效果微乎其微。

再尝试调小并行度，减半，小文件数也减少到 2w 多了，任务执行时间（不算提交）有所变长

```
set spark.sql.shuffle.partitions=1000;-- 集群默认是2000，减小了一半
```

**疑惑1**：为什么各方都说好使的参数spark.sql.adaptive.shuffle.targetPostShuffleInputSize，在我用来完全不起作用呢？

**疑惑2**：各方资料都在说spark.sql.shuffle.partitions的值等于最终的文件数，为什么我的任务文件数却增加了几十倍？

### 二、探索

首先仔细研究了spark.sql.adaptive.shuffle.targetPostShuffleInputSize这个参数，很多网上的资料说这个参数和spark.sql.adaptive.enabled配合使用

*当开启spark.sql.adaptive.enabled后，两个partition的和低于该参数设置的阈值会合并到一个reducer中处理*

这句话的描述让我先入为主的认为该参数功能上类似于hive的hive.merge.size.per.task，先正常生成小文件然后再合并成设置的文件大小。

这种认知的推论是，不会在任务最终落表时存在多个文件，大小小于设置的阈值。（因为两个相加小于阈值是会被合并的）

但是这与我实际观察到的结果不符合

在某一个分区中，一共有1100个文件，总大小是531.9k，平均每个文件5个byte，小文件并没有合并，与上面的认知不符。

后来查阅了公司内各个wiki以及网上的资料，发现有人提及但自己一直忽略的一句话

*spark.sql.adaptive.shuffle.targetPostShuffleInputSize等价于hive中的hive.exec.reducers.bytes.per.reducer*

也就是说 spark.sql.adaptive.shuffle.targetPostShuffleInputSize 这个参数 其实并不起合并partition的作用，只是起到计算partition数量的作用（想想hive中的参数是怎么用的）。

经与数据平台的同学沟通，得知spark.sql.shuffle.partitions显式的指定了partition数量。但当设置了spark.sql.adaptive.shuffle.targetPostShuffleInputSize时，在最后一个stage（即ResultStage）中，由该参数计算出partition的数量（如果根据设置计算出来的partition数量小于指定的partition数量，则相当于变相的合并了？？）。

计算公式是

`min(Shuffle Read/spark.sql.adaptive.shuffle.targetPostShuffleInputSize, spark.sql.shuffle.partitions)`

后面经过多次测试验证，得出以下结论：

1. 在任意有shuffle的stage(ShuffleMapStage或者ResultStage)中，参数spark.sql.shuffle.partitions控制了Shuffle Read的最大partition数量
1. 在ResultStage(即最后一个stage)中，通过spark.sql.adaptive.shuffle.targetPostShuffleInputSize参数计算出partition数量，如果比spark.sql.shuffle.partitions数值大，则取小
1. 在ShuffleMapStage中，partition的数量由spark.sql.shuffle.partitions决定，设置多少就有多少（key值数量小于partition数量除外）
- 疑问 1 解答：总文件大小很大，以及除以设置的目标大小参数，得到分区数量还是大于spark.sql.shuffle.partitions，二者取小
- 疑问 2 解答：spark在ResultStage阶段 partition 和 task 是一一对应的，最终一个task会写一个结果文件，但这里有一个隐含的条件：在同一个目录下一个task会写一个结果文件。
那么对于动态分区插入的方式，最终结果集会进入多表分区，也就是多个目录，所以每一个目录都会有接近partition数量的文件数，总体的文件数大约等于partition数量*表分区数量。

**以上面的case为例，partition数量=2000，表分区数=30，计算可得总体结果为6w个文件，实际结果是5.7w个文件，基本吻合。**

### 三、解决

**使用 distribute by 进行打散**

推荐使用rand来进行数据打散，可以使用hash(id)%散列数的方式进行替换。

这样处理很好的解决了大小表分区区别对待的方案，但是有些场景却没法处理。比如开发人员并不知道每个表分区的数据量，也就没法设置各个表分区的partition值。另外，如果每个表分区的数据量是动态变化的，上面的代码可能就需要时常的修改维护了。为了优化这个问题，我们可以先统计结果集中每个表分区的数量或者近似数量，然后动态的规划partition值。

```
distribute by 
partition_column,
ceil(rand()*if(data_amount/10000000>1000,1000,data_amount/10000000)) --一个partition处理1000w条数据,最多一个分区划分1000个partition
```

其中：

- data_amount 是预先统计的各个表分区数据量，需要提前探查一下
```
SELECT partition_column, COUNT(*) as data_amount FROM your_table GROUP BY partition_column
```

- 10000000 是我们期望的每个 partition 处理的数据量
- 1000 是我们期望一个分区最多划分的 partition 数量
- 使用 `ceil(rand() * if(data_amount/10000000 > 1000, 1000, data_amount/10000000))`
- `rand()` 生成 [0,1) 范围内的一个随机数。
- `data_amount/10000000` ：期望每个分区处理 1000 万条数据。
- `if(data_amount/10000000 > 1000, 1000, data_amount/10000000)` ：确保每个分区最多划分 1000 个 partition。
- 计算得到的分区值取整后进行 `ceil()` 函数取上限，保证分区数量足够。
**这样的代码可以灵活的应对不同表分区，数据量不同，同时可以确保每个partition处理的数据量近似，不会发生较大的倾斜。**

**该方案的成本 & 收益**：多增加了一轮shuffle操作，任务的执行时间和资源消耗会增加一些，不涉及计算，所以资源开销的增长并不多。但文件数减少提交时间大幅减少，总体的执行时间还是大量降低的。而且 由于partition数量和最终落表文件没有了关系，可以通过增加partition数量来提高任务的执行速度。
