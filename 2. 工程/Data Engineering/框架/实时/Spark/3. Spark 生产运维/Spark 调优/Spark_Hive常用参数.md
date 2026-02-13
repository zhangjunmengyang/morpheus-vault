---
title: "Spark/Hive常用参数"
category: "工程"
tags: [AI安全, ETL, Hadoop, Hive, RDD]
created: "2026-02-13"
updated: "2026-02-13"
---

# Spark/Hive常用参数

## 一、Spark3

万金油配置

```
## 提升资源利用率，降低内存申请量，绝大多数任务memory+memoryOverhead<=2304M是可行的，既不会延长任务执行时间，也可以节省计算资源开销
SET spark.executor.memory=1536M;## 对于资源利用率低的可以调小至1024M甚至更低，不够用时再调大
SET spark.yarn.executor.memoryOverhead=768M;## 大部分任务其实设置为512M就够用，但部分大任务在大促等场景可能会oom，因此在768M更为安全
SET spark.memory.fraction=0.6;## 提高统一内存的比例，0.5-0.7都可以，如果使用udf导致oom可以适当调小
## 控制map和reduce处理的文件大小
SET spark.hadoopRDD.targetBytesInPartition=134217728;## 既控制了spark.hadoop.hive.exec.orc.split.strategy=ETL时的文件切分大小，也一定程度上控制了map stage的输出大小
SET spark.sql.adaptive.shuffle.targetPostShuffleInputSize=134217728;## shuffle read一般在128-256m之间，如果有group操作且聚合粒度很大可以调大；如果有数据膨胀可以调小（以shuffle write的数据量为判断依据）
## 小文件合并
SET spark.sql.mergeSmallFileSize=33554432;## 小于32m启动合并，对于多分区的大表来说低于100m都可以考虑启动文件合并
SET spark.sql.targetBytesInPartitionWhenMerge=134217728;## 128m，设置为64-256m之间比较合适
## 控制申请的executor的并发数量
## 这是一个根据任务调整的参数，不是对时效性要求高的任务不设置太大的并发executor。非SLA任务executor和partition数量控制在1:5或1:10，SLA任务控制在1:5到1:2之间。
## 特殊情况是input task数特别多，shuffle task数不大，为了追求速度可以适当调大
SET spark.dynamicAllocation.maxExecutors=500;
##控制partition的最大并行度
SET spark.sql.shuffle.partitions=2000;## 这是一个根据任务调整的参数，spark3中这里设置的是最大partition数量，如果明确知道未来也用不了这么大，可以调小至合理值；如果数据量很大（尤其是peak day数据量很大）也可以调大
## SET spark.shuffle.manager=rss; ##只有在SLA中的任务，RSS参数才是有效的，可以根据任务权重是否有A或S判断，比如任务权重是 0S_15A_742B，则是SLA任务
```

Spark3 为前提，这套模板的特点是对绝大多数离线数仓任务的成本、效率、小文件等问题做了均衡取舍，表现不佳可以酌情调整。对算法类任务或者有特殊需求可能并不适用。

总体资源降低 50%左右，平均时长降低 20%左右

**参数分类**

**场景**

**参数**

**参数含义**

executor申请&并行度

一般需要大数量下，需要提升任务并行度时可以考虑修改这些参数

spark.dynamicAllocation.enabled

是否开启动态资源分配，平台默认开启，同时强烈建议用户不要关闭。理由：开启动态资源分配后，Spark可以根据当前作业的负载动态申请和释放资源

spark.dynamicAllocation.maxExecutors

开启动态资源分配后，同一时刻，最多可申请的executor个数。平台默认设置为1000。当在Spark UI中观察到task较多时，可适当调大此参数，保证task能够并发执行完成，缩短作业执行时间。但同时申请过多的executor会带来资源使用的开销，所以要多方面考虑来设置（可以参考万金油版参数的设置思路）

spark.dynamicAllocation.minExecutors

和maxExecutors相反，此参数限定了某一时刻executor的最小个数。平台默认设置为3，即在任何时刻，作业都会保持至少有3个及以上的executor存活，保证任务可以迅速调度。部分小任务有时会出现申请不到资源而一直等待，可以尝试设置该参数为1，减少pending的概率

spark.dynamicAllocation.initialExecutors

初始化的时候的executor数量，仅在动态资源分配时生效

spark.executor.instances

初始化的时候的executor数量，动态和非动态资源分配均生效

spark.executor.cores

单个executor上可以同时运行的task数。Spark中的task调度在线程上，该参数决定了一个executor上可以并行执行几个task。这几个task共享同一个executor的内存（spark.executor.memory+spark.yarn.executor.memoryOverhead）。适当提高该参数的值，可以有效增加程序的并行度，是作业执行的更快，但会使executor端的日志变得不易阅读，同时增加executor内存压力，容易出现OOM，所以一般需要配合的增加executor的内存。在作业executor端出现OOM时，如果不能增大spark.executor.memory，可以适当降低该值。平台默认设置为1。该参数是executor的并发度，和spark.dynamicAllocation.maxExecutors配合，可以提高整个作业的并发度。

内存分配

一般出现了内存溢出，可以考虑修改这些参数

spark.executor.memory

executor用于缓存数据、代码执行的堆内存以及JVM运行时需要的内存。当executor端由于OOM时，多数是由于spark.executor.memory设置较小引起的。该参数一般可以根据表中单个文件的大小进行估计，但是如果是压缩表如ORC，则需要对文件大小乘以2~3倍，这是由于文件解压后所占空间要增长2~3倍。平台默认设置为2G。

spark.yarn.executor.memoryOverhead

Spark运行还需要一些堆外内存，直接向系统申请，如数据传输时的netty等。Spark根据spark.executor.memory+spark.yarn.executor.memoryOverhead的值向RM申请一个容器，当executor运行时使用的内存超过这个限制时，会被yarn kill掉，最大值是16GB

spark.driver.memory

driver使用内存大小， 平台默认为10G，根据作业的大小可以适当增大或减小此值。一般有大表的广播可以考虑增加这个数值

spark.yarn.driver.memoryOverhead

类似于spark.yarn.executor.memoryOverhead，即Driver Java进程的off-heap内存

spark.memory.fraction

存储+执行内存占节点总内存的大小，社区版是0.6。平台为了方便的把hive任务迁移到spark任务，把该区域的内存比例调小至0.3，给other区留取更大的内存空间（UDF的影响）。个人建议非udf的任务中可以调整到0.6，减少spill的次数，提升性能

spark.memory.storageFraction

内存模型中存储内存占存储+执行内存的比例，由于在同一内存管理下可以动态的占用，该参数保持不变即可

spark.sql.windowExec.buffer.spill.threshold

当用户的SQL中包含窗口函数时，并不会把一个窗口中的所有数据全部读进内存，而是维护一个缓存池，当池中的数据条数大于该参数表示的阈值时，spark将数据写到磁盘。该参数如果设置的过小，会导致spark频繁写磁盘，如果设置过大则一个窗口中的数据全都留在内存，有OOM的风险。但是，为了实现快速读入磁盘的数据，spark在每次读磁盘数据时，会保存一个1M的缓存。举例：当spark.sql.windowExec.buffer.spill.threshold为10时，如果一个窗口有100条数据，则spark会写9（(100 - 10)/10）次磁盘，在读的时候，会创建9个磁盘reader，每个reader会有一个1M的空间做缓存，也就是说会额外增加9M的空间。当某个窗口中数据特别多时，会导致写磁盘特别频繁，就会占用很大的内存空间作缓存。因此如果观察到executor的日志中存在大量“spilling data because number of spilledRecords crossed the threshold”日志，则可以考虑适当调大该参数，平台默认该参数为40960。

文件输入输出与合并

当出现map端数据倾斜，map端由于小文件启动大量task，或者结果生成大量小文件时，可以考虑修改这些参数

spark.hadoop.hive.exec.orc.split.strategy

BI策略以文件为粒度进行split划分；ETL策略会将文件进行切分，多个stripe组成一个split；HYBRID策略为：当文件的平均大小大于hadoop最大split值（默认256M）时使用ETL策略，否则使用BI策略。该参数只对orc格式生效注意：当ETL策略生效时，如果输入文件的数量以及每个文件的stripe数量过多，有可能会导致driver压力过大，出现长时间计算不出task数量，甚至OOM的情况。当BI策略生效时，也有可能会出现输入数据倾斜。

spark.hadoop.mapreduce.input.fileinputformat.split.minsize

map端输入文件的切分和合并参数，可以把小文件进行合并，大文件进行切割其中spark.hadoopRDD.targetBytesInPartition是美团自己开发的参数，社区版不存在。这个参数的值用来标识一个hadoopRDD的大小，当同分区的多个文件小于参数设置值时，可以进行合并（注意：不能跨分区合并）spark.hadoop.mapreduce.input.fileinputformat.split.minsizespark.hadoop.mapreduce.input.fileinputformat.split.maxsize这两个参数控制了单个文件的切分和合并大小，跨文件、跨分区不行maxsize控制了split的最大值，minsize控制了最小值从有限的测试结果中发现，hadoopRDD参数要比另两个参数表现好

spark.hadoop.mapreduce.input.fileinputformat.split.maxsize

spark.hadoopRDD.targetBytesInPartition

spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version

文件提交算法，MapReduce-4815 详细介绍了 fileoutputcommitter 的原理，version=2 是批量按照目录进行提交，version=1是一个个的按照文件提交。设置 version=2 可以极大的节省文件提交至hdfs的时间，减轻nn压力。

spark.sql.adaptive.shuffle.targetPostShuffleInputSize

开启spark.sql.adaptive.enabled后，最后一个stage在进行动态合并partition时，会根据shuffle read的数据量除以该参数设置的值来计算合并后的partition数量。所以增大该参数值会减少partition数量，反之会增加partition数量。注意：对于shuffle read数据量很大，但是落地文件数很小无法很好的处理，例如：小表left join大表

spark.sql.mergeSmallFileSize

与 hive.merge.smallfiles.avgsize 类似，写入hdfs后小文件合并的阈值。如果生成的文件平均大小低于该参数配置，则额外启动一轮stage进行小文件的合并

spark.sql.targetBytesInPartitionWhenMerge

与hive.merge.size.per.task 类似，当额外启动一轮stage进行小文件的合并时，会根据该参数的值和input数据量共同计算出partition数量

mapjoin

想使用自动mapjoin时，需要考虑的参数

spark.sql.autoBroadcastJoinThreshold

当执行join时，小表被广播的阈值。当被设置为-1，则禁用广播。表大小需要从 Hive Metastore 中获取统计信息。该参数设置的过大会对driver和executor都产生压力。注意，由于我们的表大部分为ORC压缩格式，解压后的数据量3-5倍甚至10倍

spark.sql.statistics.fallBackToHdfs

当从Hive Metastore中没有获取到表的统计信息时，返回到hdfs查看其存储文件大小，如果低于spark.sql.autoBroadcastJoinThreshold依然可以走mapjoin。该参数在社区版默认是true，由于会增大对hdfs的压力，被平台改成false了，个人建议打开

spark.sql.broadcastTimeout

在broadCast join时 ，广播等待的超时时间

shuffle阶段

shuffle阶段的调优，出现fetchfailed可以考虑修改

spark.sql.shuffle.partitions

reduce阶段(shuffle read)的数据分区，分区数越多，启动的task越多（1:1），“一般”来说速度会变快，同时生成的文件数也会越多。个人建议一个partition保持在256mb左右的大小就好

spark.sql.adaptive.enabled

是否开启调整partition功能，如果开启，spark.sql.shuffle.partitions设置的partition可能会被合并到一个reducer里运行。平台默认开启，同时强烈建议开启。理由：更好利用单个executor的性能，还能缓解小文件问题。

spark.shuffle.adaptive.all

美团独有的参数，true表示所有stage都会动态调整partition，false表示只有最后一个stage开启。目前不建议打开，但如果只是聚合操作的话可以打开。如果有很多join会增加额外的shuffle

spark.sql.adaptive.shuffle.targetPostShuffleInputSize

开启spark.sql.adaptive.enabled后，最后一个stage在进行动态合并partition时，会根据shuffle read的数据量除以该参数设置的值来计算合并后的partition数量。所以增大该参数值会减少partition数量，反之会增加partition数量。注意：对于shuffle read数据量很大，但是落地文件数很小无法很好的处理，例如：小表left join大表

spark.sql.adaptive.minNumPostShufflePartitions

开启spark.sql.adaptive.enabled后，合并之后保底会生成的分区数

spark.shuffle.service.enabled

启用外部shuffle服务，这个服务会安全地保存shuffle过程中，executor写的磁盘文件，因此executor即使挂掉也不要紧，必须配合spark.dynamicAllocation.enabled属性设置为true，才能生效，而且外部shuffle服务必须进行安装和启动，才能启用这个属性

spark.reducer.maxSizeInFlight

同一时刻一个reducer可以同时拉取的数据量大小

spark.reducer.maxReqsInFlight

同一时刻一个reducer可以同时产生的请求数

spark.reducer.maxBlocksInFlightPerAddress

同一时刻一个reducer向同一个上游executor最多可以拉取的数据块数

spark.reducer.maxReqSizeShuffleToMem

shuffle请求的文件块大小 超过这个参数值，就会被强行落盘，防止一大堆并发请求把内存占满，社区版默认Long.MaxValue，美团默认512M

spark.shuffle.io.connectionTimeout

客户端超时时间，超过该时间会fetchfailed

spark.shuffle.io.maxRetries

shuffle read task从shuffle write task所在节点拉取属于自己的数据时，如果因为网络异常导致拉取失败，是会自动进行重试的。该参数就代表了可以重试的最大次数。如果在指定次数之内拉取还是没有成功，就可能会导致作业执行失败。

推测执行

推测执行相关的参数，一般不需要特别关注

spark.speculation

spark推测执行的开关，作用同hive的推测执行。（注意：如果task中有向外部存储写入数据，开启推测执行则可能向外存写入重复的数据，要根据情况选择是否开启）

spark.speculation.interval

开启推测执行后，每隔多久通过checkSpeculatableTasks方法检测是否有需要推测式执行的tasks

spark.speculation.quantile

当成功的Task数超过总Task数的spark.speculation.quantile时(社区版默认75%，公司默认80%)，再统计所有成功的Tasks的运行时间，得到一个中位数，用这个中位数乘以spark.speculation.multiplier（社区版默认1.5，公司默认3）得到运行时间门限，如果在运行的Tasks的运行时间超过这个门限，则对它启用推测。如果资源充足，可以适当减小spark.speculation.quantile和spark.speculation.multiplier的值

spark.speculation.multiplier

解释见上面spark.speculation.quantile

谓词下推

如果出现谓词没有下推，可以考虑修改这些参数

spark.sql.parquet.filterPushdown

parquet格式下的谓词下推开关

spark.sql.orc.filterPushdown

orc格式下的谓词下推开关

spark.sql.hive.metastorePartitionPruning

当为true，spark sql的谓语将被下推到hive metastore中，更早的消除不匹配的分区

## 二、Hive

万金油配置

```
set hive.exec.dynamic.partition=true;                                                                
set hive.exec.dynamic.partition.mode=nonstrict;                        
set hive.exec.parallel=true;                                                                                                
set mapred.max.split.size=64000000;                                                                        
set mapred.min.split.size.per.node=64000000;                                
set mapred.min.split.size.per.rack=64000000;                                
set hive.exec.reducers.bytes.per.reducer=256000000;        
set hive.exec.reducers.max=2000;                                                                                
set hive.merge.mapredfiles=true;                                                                                
set hive.merge.smallfiles.avgsize        =128000000;                                
set hive.merge.size.per.task=128000000;                
```

大概率优于集群默认配置，可以少踩一些坑（小文件、map 倾斜、reduce 倾斜、分配资源时间）

**参数分类**

**参数**

**参数含义**

基本设置

hive.mapred.mode

严格模式开关，当设置为nonstrict时代表非严格模式，当设置为strict时，说明是严格模式。严格模式下，对于分区表的查询，笛卡尔积join和order by都有一些限制

dfs.block.size 

hdfs一个文件块的大小

mapred.reduce.slowstart.completed.maps

map阶段完成设置的比例后，reduce阶段就开始申请资源并执行。该比例设置的较小有类似“死锁”的风险

hive.exec.max.created.files

在一个job中，所有的map或者reduce能够创建的最多文件数

hive.input.format

hive文件的输入格式。默认配置下，该输入格式支持小文件的合并输入。与org.apache.hadoop.hive.ql.io.HiveInputFormat具有不同的map/reduce数量的计算公式

动态分区

hive.exec.dynamic.partition

是否允许动态分区，建议开启

hive.exec.dynamic.partition.mode

动态分区的模式，strict表示必须指定至少一个分区为静态分区，nonstrict模式表示允许所有的分区字段都可以使用动态分区，建议nonstrict

hive.exec.max.dynamic.partitions

动态分区最大分区数量

hive.exec.max.dynamic.partitions.pernode

每个节点（map or reduce task）可以写入多少个分区。注意：该参数在mt集群中在静态参数列表中，不可以运行时修改（xt上设置了也不起效果）

map端聚合

hive.map.aggr 

map端聚合开关，设置为true时有可能在map端进行聚合

hive.groupby.mapaggr.checkinterval

map端聚合开关打开时，预先拿100000条数据做聚合测试

hive.map.aggr.hash.min.reduction

聚合后的数据条数/聚合前的数据条数>0.5，则不进行map端聚合

任务并行

hive.exec.parallel

任务并行开关，当设置为true时，多个job可以同时启动执行

hive.exec.parallel.thread.number

并发执行任务的最大数量

推测执行

mapred.map.tasks.speculative.execution

map端推测执行开关

mapred.reduce.tasks.speculative.execution

haddop中的reduce端的推测执行开关，由于会被覆盖，所以设置无效

hive.mapred.reduce.tasks.speculative.execution

hive中的reduce端推测执行开关（会覆盖mapred.reduce.tasks.speculative.execution）

谓词下推

hive.optimize.ppd

谓词下推开关

hive.ppd.remove.duplicatefilters

谓词下推生效后，filter被下推到距离数据源更近的位置，那么原始位置是否还保留该filter

数据倾斜

hive.optimize.skewjoin 

倾斜连接优化开关，打开后join操作【可能】会（根据hive.skewjoin.key设置的阈值）生成两个job。点击展开内容看起来hive上的skewjoin 原理是shuffle阶段还是把相同的key分发到同一个reducer，然后在reducer端对于key的数量进行统计，发现超过hive.skewjoin.key设置的阈值，就把这些key对应的数据写到本地一个temp文件，然后copy到hdfs上的一个临时目录。最后再启动一个job，针对大key进行mapjoin的操作。
这样做只是减少了在同一个reducer中计算的耗时（通过另起job，split到多个maptask中进行），但是并没有节省当前job的shuffle阶段耗时（copy、sort都是在单一reducer进行）
因此，据我理解hive.optimize.skewjoin这个参数只能解决数据量大造成的计算耗时较长以及最后一个结果文件落盘的写入耗时（结果数据倾斜），但是无法解决数据量大造成的数量统计耗时和shuffle阶段文件传输耗时该参数打开时要注意，和mapjoin一同使用会有问题（报错或数据结果异常）

hive.skewjoin.key 

join时倾斜key的识别阈值，超过这个阈值，认为出现了一个倾斜key，配合hive.optimize.skewjoin使用。

hive.optimize.skewjoin.compiletime 
