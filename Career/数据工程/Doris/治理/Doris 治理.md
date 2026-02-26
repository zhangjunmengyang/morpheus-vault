---
title: "3. Doris 治理"
type: concept
domain: engineering/doris/治理
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris/治理
  - type/concept
---
# 3. Doris 治理

## 5、查询优化

### 5.1、查询计划

FE将SQL转化为Fragment并下发给BE执行，BE执行完成后，将结果汇聚返回给FE。

分布式查询计划由多个Fragment组成，每个Fragment负责一部分查询，之间通过ExchangeNode进行数据传输。

Fragment可以划分为多个执行实例Instance，从而提升Fragment的执行并发度，默认一个Instance扫描一个分桶。

### 5.2、Profile

FE将查询计划拆分为Fragment下发到BE执行任务，BE会记录运行状态信息输出到日志中，即Profile。

关键Profile指标：

- hostname：BE节点
- Active：节点执行总时长
- **RowsKeyRangeFiltered**：前缀索引过滤的行数
- **RowsBloomFilterFiltered**：BloomFilter索引过滤的行数
- **RawRowsRead**：通过各种索引过滤后读取的行数
- RowsRead：返回给Scanner的行数，通常小于RawRowsRead，是因为返回过程中可能会经过数据聚合。
- RowsReturned：ScanNode返回给上层节点的行数，RowsReturned通常小于RowsRead，一些没有下推的谓词条件，会在内存中进行过滤。
查询分析指标：

- ScanTime：查询扫描消耗的时间，包括等待IO以及实际扫描数据所花费的时间。
- ScanWaitTime：查询扫描等待的总时间，如读磁盘等待、网络传输等待等。
- ScanCpuTime：查询扫描过程中消耗的CPU时间，如内存中根据运算符比较过滤数据、聚合数据等时间，可以反映前缀索引或Rollup的过滤效果。
- FragmentCpuTime：查询过程中所有Fragment消耗的CPU时间，用于反映整个查询的计算复杂度。
- ScanWorkerWaitTime：查询扫描过程中，等待scanner调度的时间，等待其他资源或者任务完成的时间。
- 大查询数、扫描行数、超时查询数、查询实例数、可优化数据量（单查询所有Instance的RawRowsRead-RowsReturned之和）。
查询分析粒度：

- 单查询：定位单查询问题出现环节。
- 单BE：BE的负载=单BE max(1天内所有30s窗口实际ScanTime)/单BE Scanner数*30s。
- 集群：集群负载=max(集群所有BE的负载)
### 5.3、分析查询

治理重点：

- 大查询
- 大量热点可优化查询，不一定是大查询
大查询影响：

- 假设扫描1000个分区，每个分区48分桶，1个scanner扫描1个分桶，则需要48000个scanner调度查询，而目前1台BE默认配置48个scanner，1个50BE的集群也就2400个scanner同时运行，这会导致集群下所有BE的大部分scanner都在为这个大查询服务。大查询会引发资源竞争，导致BE IO被打满、查询响应时间变长甚至超时、BE负载过大甚至集群故障等问题。
大查询定义：

- 扫描行数多，过滤效果差：
- 查询时间长：超过集群TP99或查询时长>5s
- 扫描数据量大：单BE扫描行数>50w
- 过滤效果差：索引过滤数据行数<扫描总行数的50%
- 过滤效果好，但扫描行数还是多：
- 查询时间长：超过集群TP99
- 扫描数据量大：单BE前缀索引过滤后扫描行数>100w
- 扫描分区数多：
- 查询时间长：超过集群TP99
- 扫描分区多：一年以上
查询慢原因定位：

1. 集群是否存在其他大查询影响？
1. 是否数据倾斜？
1. Rollup是否可优化？（调整前缀索引、上卷聚合）
1. 是否SQL可优化？
1. 分桶数是否可优化？
### 5.4、优化步骤

1. 先整体看集群负载，分析集群机器CPU、IO使用率现状，确定是IO密集型（扫描数据量大）还是CPU密集型（计算复杂）。
1. 根据集群监控定位Scan Time/Scan Cpu Time/Fragment Cpu Time有尖刺的时间段。
1. 定位问题SQL，找出对性能优化影响最大的查询，根据查询次数、扫描数据量扫描耗时等指标进行排序。
1. 分析表的不同查询场景或产品的谓词组合，通过查询次数&过滤数据量确定根据哪些谓词构建rollup roi最高。
1. 单SQL关注：是否命中Rollup、扫描行数RawRowsRead、前缀索引过滤行数RowsKeyRangeFiltered、查询扫描时间ScanTime。
1. 优化经验：
1. 前缀索引第一列很重要，需要保证命中，且尽量将扫描数据量减少50%以上。
1. 分区列不放在前缀索引中，只要在key列，就可以分区裁剪。
1. 前缀索引尽量不要出现字符串类型，如果枚举值有限，可以使用整型替代。
1. 如果是关联查询，可以考虑使用Colocate Join或者Runtime Filter优化。
1. 治理完成后观测集群性能指标，负载低于60%则可进行集群节点下线、集群合并等动作。
## 一、背景

现在 hotel Doris 单日成本 10k 以上，成本比较高，同时大部分集群此前也没有做过优化，查询性能很低。此处专项需要对实时、离线数仓的集群进行资源治理。

## 二、问题分析

### 2.1、现状

**经过对实时doris集群分析，发现集群存在的主要问题集中在IO上（对于hbdataflow集群单次扫描的数据量在千万至亿级别。**

集群IO如下图，可以看到该集群的平均IO达到了50%上下，同时存在很多将IO打满的情况，如果战役期间这种大查询增多，会对集群性能产生不小的影响。

![image](VyD7dKpmZo2WRdxKwa5c4LfanJY.png)

对比治理后：明显减少尖刺、峰值

![image](WuP0ds19ZoZrtfxbFJFcGPlKnAc.png)

特点：

1. 计算逻辑少：与CPU相关的聚合操作等较少
1. 数据量大：往往在千万至亿级别
1. 大多数表无 rollup：战役前都没有创建 rollup
1. 扫所有分桶：**分桶字段大部分为poi_id，当查询的poi比较多时，几乎扫描所有分桶，也就是扫描整个分区**
### 2.2、治理必要性

BE IO打满（BE IO指标为100%）指的是某个机器的**磁盘读写速度达到了它最大可承受的速度上限**

- falcon上的doris集群监控默认取的是**一个****机器所有磁盘IO的max值**，也就是说，一个BE有12个磁盘的情况下，如果有一个磁盘IO被打满，那么该BE对应的IO指标在上图就会显示为100%
- 假如一个表A的副本分别存在a, b, c三台BE上，a的IO打满无法读取数据时，还可以从b,c上读取副本进行查询，即使出现某个副本所在 BE 真正打满，但仍然可以正常查询（读取其他副本）
如果 IO 持续打满，导致相应增加，用户体验变差，严重时因为缺少资源，新查询在队列中不断积压，BE 节点不可用，最终集群故障（集群打挂了）

### 2.3、大查询

初期治理动作不规范，都是调“比较严重”的问题进行治理，缺少量化动作和挖掘规则

后期在工具化以后才对详细指标阈值等给出定义，次数仅做大查询的分析。以某个大查询为例：

假设扫描3年的数据，1095个的partition，每个partition有48个分桶，一个scanner线程扫描一个分桶，那么至少是50000个scanner，也就是50000个线程调度

而一个BE默认配置48个scanner线程，一个100BE的集群也就5000个scanner线程能够同时运行，这将会导致这个集群下全部BE的**大部分**scanner线程都在为这一个大查询服务。

不考虑执行速度，单纯按照 BE 数都需要 10 个 batch（5w / 5k），一次这样的大查询就给整个集群“阻塞”2s

如果再来几个大查询...如果查询几次？

为什么是大部分 BE，因为还有防饿死策略，调度算法不展开细讲。

![image](EMQhdh5cgotczzxx4rxcvyL5nDb.png)

**scan线程调度流程：**

对于查询扫描，每个instance都会起一个transform线程用于scan线程调度：

- **等待调度：**将scanner放到阻塞队列中进行调度（一次加的scanner有限，和内存有关），对应**scan worker wait time**（开始等待时就进行计算，而非放入阻塞队列中才计算）
- **执行扫描：**scan worker pool从阻塞队列中获取scanner线程，然后执行扫描，对应**scan time** （扫描用时）和**scan cpu time**（该CPU利用率起来后，判断哪个才能占用的比较多）
- **获取结果：**读到的结果由transform线程放入instance中
## 三、治理目标

**主要目标是提升TP95查询性能，进而提高集群查询时能够支撑的QPS**，包含以下两个方面：

- **减小扫描量：**保证scan的过程不要阻塞（scan wait time、scan time）、shuffle时间不要太长
- **减小SQL复杂度：**fragment的层级需要少一点，因为不同层级之间的数据交换会导致exchange算子的waiting time比较长
## 四、治理动作

### 4.1、分析问题

寻找查询时长、扫描行数较大的 SQL，看看是否有共性，或者有哪些明显的问题

#### 代码块

```
SELECT tb.`table`,
       ta.tp95_duration,
       cast(tb.query_sum / 7 as int) avg_query_sum,
       tb.query_sum,
       ta.query_success_sum,
       ta.query_success_sum / tb.query_sum as query_success_rate,
       ta.max_duration,
       ta.min_duration,
       ta.avg_duration,
       ta.max_instances_num,
       ta.min_instances_num,
       ta.avg_instances_num,
       ta.max_scan_rows,
       ta.min_scan_rows,
       ta.avg_scan_rows,
       ta.tp90_duration,
       ta.tp99_duration,
       ta.tp999_duration,
       ta.avg_cpu_ms,
       tc.sql
  FROM (
        SELECT `table`,
               count(*) AS query_success_sum,
               percentile(cast(duration as bigint),array(0.90))[1] as tp90_duration,
               percentile(cast(duration as bigint),array(0.95))[1] as tp95_duration,
               percentile(cast(duration as bigint),array(0.99))[1] as tp99_duration,
               percentile(cast(duration as bigint),array(0.999))[1] as tp999_duration,
               max(cast(duration as bigint)) AS max_duration,
               min(cast(duration as bigint)) AS min_duration,
               avg(cast(duration as bigint)) AS avg_duration,
               cast(max(instances_num) AS bigint) AS max_instances_num,
               cast(min(instances_num) as bigint) AS min_instances_num,
               cast(avg(instances_num) as bigint) AS avg_instances_num,
               cast(max(scan_rows) as bigint) AS max_scan_rows,
               cast(min(scan_rows) as bigint) AS min_scan_rows,
               cast(avg(scan_rows) as bigint) AS avg_scan_rows,
               avg(cpu_ms) as avg_cpu_ms
          FROM log.data_doris_audit 
    -- datekey筛选时可以自行选择集群查询较多的时间段
         WHERE dt BETWEEN 20230401 AND 20230407
           AND CLUSTER = 'hbdataflow'
           AND db = 'hbdataflow' 
    -- `user`的值取决于提供的Doris连接账号，有些可能不是XX:XX的形式，而是XX:XX_r
           AND `user` = 'hbdataflow:hbdataflow'
           AND success = 1
         GROUP BY `table`
       ) ta
  LEFT JOIN (
        SELECT `table`,
               count(*) AS query_sum
          FROM log.data_doris_audit
         WHERE dt BETWEEN 20230401 AND 20230407
           AND CLUSTER = 'hbdataflow'
           AND db = 'hbdataflow'
           AND `user` = 'hbdataflow:hbdataflow'
         GROUP BY `table`
       ) tb
    ON ta.`table` = tb.`table`
  left join (
        select tt.`table`,
               tt.`sql` from(
                SELECT `table`,
                       `sql`,
                       row_number() over(PARTITION BY `table` order by cast(duration AS bigint) desc) as rn
                  FROM log.data_doris_audit
                 WHERE dt BETWEEN 20230401 AND 20230407
                   AND CLUSTER = 'hbdataflow'
                   AND db = 'hbdataflow'
                   AND `user` = 'hbdataflow:hbdataflow'
                   AND success = 1
               ) tt
         where rn = 1
       ) tc
    on ta.`table` = tc.`table`
```

### 4.2、指标分析

- **Scan time：**表示查询计划在进行扫描操作消耗的总时间，包括等待IO以及实际扫描数据所花费的时间。该指标衡量了查询的整体执行效率。
- 扫描总时长，用于初步筛选大查询
- **Scan wait time：**表示查询计划在等待IO操作(例如读磁盘)的总时间。该指标反映了IO速度的影响。
- **Scan cpu time：**表示查询计划在CPU处理相关操作的时间。例如，算术运算、常量表达式或者条件判断等。该指标反映了服务器负载以及查询在执行过程中的计算复杂度。
- **scan time 和 scan cpu time 的值基本一样，大概率是因为查询计划没有遇到需要等待IO操作的瓶颈**。这意味着，数据已经存储在内存或缓存中（例如，使用了 MemTable），所以查询可以快速扫描数据，而无需等待从磁盘或其他介质读取数据。
- **Fragment cpu time：**表示查询计划在所有Fragments（即分区）间分配的处理器时间。Doris通常使用Fragment来存储和处理数据集的一个子集，而每个Fragment可能由多个不同的节点共享。该指标反映了不同片段之间数据交换的开销、子查询和其他操作的效率等。
- 考虑优化SQL结构，减少fragment
- **Scan worker wait time：**等待scan线程调度的时间
- 分析当前集群性能，等待时间过长会导致查询阻塞
- **Shuffle rows：**代表了这个操作节点需要进行的 Shuffle 行数。比如在一次 JOIN 操作中将两表 join 在一起可能会有一个 Shuffle 阶段
### 4.3、SQL 问题分析

**扫描指标**

1. 先看扫描行数：看 profile，**RawRowsRead**
1. 再看是否命中 rollup：看查询计划，OlapScanNode -> **rollup**
1. 看前缀索引过滤掉多少数据：看 profile，OlapScanner -> SegmentIterator -> **RowsKeyRangeFiltered**
1. 看扫描时长：profile -> OlapScanner -> **ScanTime**
**数据交换指标**

1. shuffle数据量有多少
1. fragment之间处理耗时
经验值

- Exchange node
- 同一节点的不同实例间执行耗时的方差值>10000
- Exchange node的执行耗时占比>20%
- 
根据，分析单个 SQL 的查询问题，具体情况具体分析

类型

原因

判断依据

查询问题

查全表

explain显示OlapScanNode的partitions较多例：OlapScanNode.partitions=276/318

没有命中rollup

desc `TABLE NAME` all 发现存在rollup，但explain显示OlapScanNode中的rollup还是基表，并且PreAggregation未OFF，后边显示原因

没有列裁剪

PREAGGREGATION值为OFF，其后显示原因例1：PREAGGREGATION: OFF. Reason: conjunct on `allocated_memory` which is StorageEngine value column例2：PREAGGREGATION: OFF. Reason: No AggregateInfo 没有聚合算子

使用了错误的join算法

使用HashJoin时广播了大表profile显示HASH_JOIN_NODE的BuildRows指标过大（大于百万行）使用ShuffleJoin处理小右表profile显示HASH_JOIN_NODE的BuildRows指标过小（小于1万行）出现了CROSS_JOIN_NODE(笛卡尔积）需要检查join条件写得是否对（比如存在or，或非等值连接）

不好的inner join顺序

profile中看到前面的join后产生了较多的行数，而后面的join产生的较小的行数。

join数据倾斜

profile的某个instance的HASH_JOIN_NODE的ProbeRows较多，其他instance的较少

没有命中runtime filter

doris版本>=0.14，并且sql中有inner join。并且explain没有RF.

命中了runtime filter但还是很慢

profile中可能有runtime filter的过滤率，尝试切换in/bitmap过滤类型测试看是否变快

大基数bitmap执行union很慢

缺少指标，无法查看

union比grouping sets慢

union生成过多的OlapScanNode的instance，占用了scanner线程池资源

没有下推谓词

explain发现单表的过滤条件不在OlapScanNode节点

compaction不及时

数据导入当天查询可能很慢，第二天查询却又很快，很可能是数据没有及时compaction造成的，可以通过segment 文件数量对比查看

模型问题

分桶数过少

profile看到OlapScanNode的RowsReturned与instance数的比值过大（大于百万行）

分桶数过多

profile看到单个instance的OlapScanNode的RowsReturned过小，并且同一个fragment的instance数过多

分桶键不合理/分桶数据倾斜

profile看到OlapScanNode在不同instance中的RowsReturned值浮动较大

选择模型不对

对于数据量超大的表选择了Duplicate模型，导致查询时数据量非常大。

集群性能问题

慢节点

相同的数据量级、相同的fragment，某个instance的执行时间长(profile的Active * nonChild)，其他instance的执行时间短。有的时候会表现为exchange node比较慢，但是上游算子很快

BE数太少

profile看到RowsReturned与OlapScanNode的be数的比值过大explain中看到的“tabletRatio=实际扫描副本数/总副本数” 中的实际扫描很大，而numNodes很少，即实际扫描副本数/numNodes的值很大，大几十甚至几百时

大查询/大导入影响

平常sql很快，在数据量级、表结构没发生变化的情况下相同sql变慢了。通常表现在profile的OLAP_SCAN_NODE.RowsRead较少（小于1万行），但是ScannerWorkerWaitTime或ScannerBatchWaitTime（大于5、6秒）较高的情况。
