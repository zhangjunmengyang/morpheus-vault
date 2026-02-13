# Doris 数据导入

[Spark Load - Apache Doris](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2F1.2%2Fdata-operate%2Fimport%2Fimport-way%2Fspark-load-manual)

DorisDB提供了多种导入方式，用户可以根据数据量大小、导入频率等要求选择最适合自己业务需求的导入方式。

# 一、导入方式

##  1.1、常见场景

- **离线数据导入**：
- Spark Load 导入：单表的数据量特别大、或者需要做全局数据字典来精确去重、且聚合逻辑多
- Broker Load：因为 Doris 表里的数据是有序的，所以 Broker load 在导入数据的时是要利用doris 集群资源对数据进行排序，相对于 Spark load 来完成海量历史数据迁移，对 Doris 的集群资源占用要比较大，这种方式是在用户没有 Spark 这种计算资源的情况下使用，如果有 Spark 计算资源建议使用 [Spark load](https%3A%2F%2Fdoris.apache.org%2Fzh-CN%2Fdocs%2F1.2%2Fdata-operate%2Fimport%2Fimport-way%2Fspark-load-manual)。
- Stream Load：比如库存，每天 60 亿数据量，目前还是采用 方式
- 实时数据导入：
- 日志数据和业务数据库的binlog同步到Kafka以后，优先推荐通过 Routine load 导入DorisDB
- 如果导入过程中有复杂的多表关联和ETL预处理可以使用Flink处理以后用 stream load 写入DorisDB
- Mysql数据导入：推荐使用Mysql外表，insert into new_table select * from external_table 的方式导入
- d2d 任务导入：
- 原生导入：doris 进行读写，交给 doris 引擎来执行，在写法上和 Spark on doris 存在区别：
- 需要写临时分区，doris 是先写到临时分区
- 不需要写 dsn，直接写db
- Spark on doris：需要注意 dsn 和 db 的区分的写法，因为在Spark on doris中是**根据写法来指定引擎**
- dsn vs db
- dsn的作用是唯一的标识一个引擎实例下的一个db，而一个db下的表名/视图名称是不能重复的，因此dsn.table能够唯一的标识一张表
- 而db.table不能唯一的标识不同引擎下的同名表，比如hive、mysql可能都有dim.city这张表，如果用dsn.table来表示，那么hive中的叫hdim.city，mysql中的叫dim.city
- 在spark on doris引擎中，既能够用jdbc去执行doris sql（主要用于DDL，如alter table语句），也能在spark引擎中执行sparksql（主要用于跨引擎查询与导入，如insert into语句）。
- 由于需要使用jdbc去执行doris的DDL语句，所以需要遵循doris语法，只能用db.table的格式，而不能用dsn.table的形式，否则会找不到表。
- 而在spark中执行跨数据源查询时，我们必须要唯一找出来源表是什么引擎，什么db下的，所以必须要用dsn.table的格式。
- **因此用户需要熟记：在执行Alter table等DDL语句（例：Delete）时，需要写table；在执行Insert into语句时，需要写dsn.table。**
外部表：

DorisDB支持以外部表的形式，接入其他数据源。

外部表指的是保存在其他数据源中的数据表。

目前DorisDB已支持的第三方数据源包括 MySQL、HDFS、ElasticSearch，Hive。

1.1  Broker Load

在Broker Load模式下，通过部署的 Broker 程序，DorisDB可读取对应数据源（如HDFS, S3、阿里云 OSS、腾讯 COS）上的数据，利用自身的计算资源对数据进行预处理和导入。这是一种异步的导入方式，用户需要通过MySQL协议创建导入，并通过查看导入命令检查导入结果。 

1、名词解释

Broker：Broker 为一个独立的无状态进程，封装了文件系统接口，为 DorisDB 提供读取远端存储系统中文件的能力。

Plan：导入执行计划，BE会执行导入执行计划将数据导入到DorisDB系统中。

说明：

1）、Label：导入任务的标识。每个导入任务，都有一个数据库内部唯一的Label。Label是用户在导入命令中自定义的名称。

通过这个Label，用户可以查看对应导入任务的执行情况，并且Label可以用来防止用户导入相同的数据。

当导入任务状态为FINISHED时，对应的Label就不能再次使用了。

当 Label 对应的导入任务状态为CANCELLED时，可以再次使用该Label提交导入作业。

2）、data_desc：每组 data_desc表述了本次导入涉及到的数据源地址，ETL 函数，目标表及分区等信息。

1.2 Spark Load

Spark Load 通过外部的 Spark 资源实现对导入数据的预处理，提高 DorisDB 大数据量的导入性能并且节省 Doris 集群的计算资源。主要用于初次迁移、大数据量导入 DorisDB 的场景（数据量可到TB级别）

Spark Load 任务的执行主要分为以下几个阶段： 

1、用户向 FE 提交 Spark Load 任务；

2、FE 调度提交 ETL 任务到 Spark 集群执行。

3、Spark 集群执行 ETL 完成对导入数据的预处理。包括全局字典构建（BITMAP类型）、分区、排序、聚合等。

4、ETL 任务完成后，FE 获取预处理过的每个分片的数据路径，并调度相关的 BE 执行 Push 任务。

5、BE 通过 Broker 读取数据，转化为 DorisDB 存储格式。

6、FE 调度生效版本，完成导入任务。

2、预处理流程：

1、从数据源读取数据，上游数据源可以是HDFS文件，也可以是Hive表。

2、对读取到的数据完成字段映射、表达式计算，并根据分区信息生成分桶字段bucket_id。

3、根据DorisDB表的Rollup元数据生成RollupTree。

4、遍历RollupTree，进行分层的聚合操作，下一个层级的Rollup可以由上一个层的Rollup计算得来。

5、每次完成聚合计算后，会对数据根据bucket_id进行分桶然后写入HDFS中。

6、后续Broker会拉取HDFS中的文件然后导入DorisDB BE节点中。

1.3 Stream Load

Stream Load 是一种同步的导入方式，用户通过发送 HTTP 请求将本地文件或数据流导入到 DorisDB 中。Stream Load 同步执行导入并返回导入结果。用户可直接通过请求的返回值判断导入是否成功。

说明： 

Stream Load 中，用户通过HTTP协议提交导入命令。

如果提交到FE节点，则FE节点会通过HTTP redirect指令将请求转发给某一个BE节点，用户也可以直接提交导入命令给某一指定BE节点。

该BE节点作为Coordinator节点，将数据按表schema划分并分发数据到相关的BE节点。

导入的最终结果由 Coordinator节点返回给用户。

1.4 Routine Load

Routine Load 是一种例行导入方式，DorisDB通过这种方式支持从Kafka持续不断的导入数据，并且支持通过SQL控制导入任务的暂停、重启、停止

1、基本原理

导入流程说明：   

1、用户通过支持MySQL协议的客户端向 FE 提交一个Kafka导入任务。

2、FE将一个导入任务拆分成若干个Task，每个Task负责导入指定的一部分数据。

3、每个Task被分配到指定的 BE 上执行。在 BE 上，一个 Task 被视为一个普通的导入任务，通过 Stream Load 的导入机制进行导入。

4、BE导入完成后，向 FE 汇报。

5、FE 根据汇报结果，继续生成后续新的 Task，或者对失败的 Task 进行重试。

6、FE 会不断的产生新的 Task，来完成数据不间断的导入。

2.1 MySQL外部表

星型模型中，数据一般划分为维度表和事实表。维度表数据量少，但会涉及UPDATE操作。目前DorisDB中还不直接支持UPDATE操作（可以通过Unique数据模型实现），在一些场景下，可以把维度表存储在MySQL中，查询时直接读取维度表。

在使用MySQL的数据之前，需在DorisDB创建外部表，与之相映射。DorisDB中创建MySQL外部表时需要指定MySQL的相关连接信息，如下图
