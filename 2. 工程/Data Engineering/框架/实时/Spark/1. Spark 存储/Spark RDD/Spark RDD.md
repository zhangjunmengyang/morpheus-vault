# Spark RDD

## 一、RDD 定义

RDD（Resilient Distributed Dataset）弹性分布式数据集，是Spark最基本的数据抽象，是一种只读、可分区、有容错机制的分布式数据集。

- Resilient：
- 自动进行内存和磁盘数据存储切换。
- 基于Lineage的高效容错，第n个节点出错，会从n-1节点恢复。
- Task失败会自动重试，默认4次。
- Stage失败会自动重试，只运行计算失败的分区。
- Distributed：可以分布在多台机器上并行处理。
- Dataset：数据集，可存放很多元素。
- Read-only：不能修改，只能通过转化操作生成新的RDD。
**RDD为什么是只读的？**

- 保证容错，可以随时通过转换操作恢复丢失数据。
- 保证并行处理的安全性，分布式处理不变数据是安全的。
- 提高处理效率，不变数据可以缓存在内存中复用。
## **二、RDD主要属性**

**一组分区（A list of partitions）**：数据集的基本组成单位，每个分区都会被一个计算任务处理，spark.default.parallelism可以指定默认分区数。

**一个计算每个分区的函数（A function for computing each split）**：RDD的计算是以分区为单位的，计算函数会被作用到每个分区上。

**RDD之间的依赖关系（A list of dependencies on other RDDs）**：RDD每次转换都会生成新的RDD，所以RDD之间就会形成流水线一样的前后依赖关系。当部分分区数据丢失时，可以通过这个依赖关系重新计算丢失的分区数据，而不是对所有分区重新计算。 

**RDD的分区函数（Partitioner for key-value RDDs）**：基于KV类型的RDD会有一个，默认为HashPartitioner（根据key进行哈希计算，然后对分区数取模），还有基于范围的分区函数RangePartitioner（根据key排序，根据范围将数据分配到不同分区）。

**一个列表，存储每个Partition的优先位置（A list of preferred locations to compute each split on）**：对于HDFS，保存每个Partition所在块的位置。按照“移动数据不如移动计算”理念，在进行调度时，会尽可能将计算任务分配到其所要处理数据块的存储位置。只存在于（K,V）类型的 RDD 中，非（K,V）类型的 partitioner 的值就是 None。

RDD不仅表示一个数据集，还表示这个数据集的来源，如何计算。

- 分区列表、分区函数、优先位置：数据集在哪、如何分区、在哪计算更合适。
- 分区计算函数、依赖关系：转换关系、数据集来源。
- 不实际存储真正要计算的数据。
## **三、RDD算子**

分为两类：Transformation、Action。

action 会触发真正的作业提交，transformation 算子是不会立即触发（延迟计算）作业提交的。每一个 transformation 方法返回一个新的 RDD。只是某些 transformation 比较复杂，会包含多个子 transformation，因而会生成多个 RDD。这就是实际 RDD 个数比我们想象的多一些 的原因。通常是，当遇到 action 算子时会触发一个job的提交，然后反推回去看前面的 transformation 算子，进而形成一张  DAG。

### 3.1、Transformation

Transformation：返回一个新RDD，具有lazy特性（惰性求值/延迟执行），只有当发生要求返回结果的action动作时，才会执行转换代码。

**为什么要惰性求值？**

- 根据DAG进行优化，提高执行效率。
- 减少中间计算过程，减少中间结果落盘，降低磁盘IO次数。
常见Transform算子：

- map(func)：返回新RDD，由每个输入元素经过func转换组成。
- filter(func)：返回新RDD，由计算func返回true的的输入元素组成。
- flatMap(func)：返回新RDD，每个输入元素可被映射为0个或多个输出元素。
- groupByKey(func)：返回新RDD，根据key对值进行分组，将所有具有相同key的值收集到一个集合中，只能分组不能聚合。
- reduceByKey(func)：返回新RDD，与groupByKey不同的是，会在shuffle前对分区内相同key数据集进行预聚合，从而减少网络传输数据量，提升性能。
### 3.2、Action

Action：无返回值或返回其他的，不返回RDD，至少需要有一个action算子。

常见Action算子：

- count()：返回元素个数。
- reduce(func)：通过func聚合所有元素。
## **四、RDD 高可用**

### 4.1、持久化 / 缓存

如果RDD会频繁地被使用，那么可以通过persist()或cache()方法，持久化在内存或磁盘中，有lazy特性。

`**cache()**`** 和 **`**persist()**`** 区别**

- `cache()` 计算结果将保存在节点的内存中，它底层调用了 `persist()` 方法，并使用默认的存储级别 `MEMORY_ONLY`。且使用未序列化的 Java 对象格式存储数据，这可能会占用较多的内存空间。
- `persist()` 方法允许用户选择不同的存储级别，如 `MEMORY_AND_DISK`、`MEMORY_ONLY_SER` 等。且支持序列化存储，可以节省内存空间，但读取时会增加 CPU 的计算负担。
如果RDD引用次数大于1，且运行成本（计算RDD耗时与作业执行时间的比值）超过30%，可以考虑Cache。

通过最近最少使用算法（LRU）将旧数据移除，也可以手动调用unpersist()方法删除。

### 4.2、**Checkpoint**

为了更加可靠的持久化数据，在Checkpoint时会将数据放在HDFS上，借助HDFS实现容错和高可用。

Checkpoint流程：

1. 将调用了Checkpoint的rdd标记。
1. 重新新任务，重新计算被标记的rdd，将rdd结果写入到HDFS。
1. 优化：重新计算意味着该rdd会被计算2次，最好在Checkpoint前进行cache，重启的任务只需要将内存中的数据拷贝到HDFS上即可，从而省去计算过程。
持久化与Checkpoint区别：

- 位置：持久化只能保存数据在本地磁盘或内存中；Checkpoint可以保存数据到HDFS等可靠存储上。
- 生命周期：持久化数据会在程序结束后被清除，或手动调用unpersist；Checkpoint在程序结束后依然存在，不会被删除。
- Lineage：持久化不会丢掉rdd间的依赖关系，因为如果数据丢失，需要通过依赖链重新计算；Checkpoint会斩断依赖链，因为数据已经被物理存储，不需要通过lineage重新计算。
### 4.3、Lineage 容错

Spark RDD 是怎么容错的，基本原理是什么？

一般来说，分布式数据集的容错性有两种方式：**数据检查点 **和 **记录数据的更新**。

面向大规模数据分析，数据检查点操作成本很高，需要通过数据中心的网络连接在机器之间复制庞大的数据集，而网络带宽往往比内存带宽低得多，同时还需要消耗更多的存储资源。

因此，Spark选择记录更新的方式。但是，如果更新粒度太细太多，那么记录更新成本也不低（**比如记录所有更新点**）。

因此，RDD只支持粗粒度转换，即只记录单个块上执行的单个操作，然后将创建RDD的一系列变换序列（每个RDD都包含了他是如何由其他RDD变换过来的以及如何重建某一块数据的信息。因此RDD的容错机制又称“血统(Lineage)”容错）记录下来，以便恢复丢失的分区。

Lineage本质上很类似于数据库中的重做日志（Redo Log），只不过这个重做日志粒度很大，是对全局数据做同样的重做进而恢复数据。

**Spark 粗粒度和细粒度是什么？**

1. 粗粒度模式（Coarse-grained Mode）: 每个应用程序的运行环境由一个 dirver 和若干个 executor 组成，其中，每个 executor 占用若干资源，内部可运行多个 task（对应多少个 slot）。应用程序的各个任务正式运行之前，需要将运行环境中的资源全部申请好，且运行过程中要一直占用这些资源，即使不用，也需要到最后程序运行结束后才回收这些资源。
1. 细粒度模式（Fine-grained Mode）: 鉴于粗粒度模式会造成大量资源浪费，Spark On Mesos 还提供了另外一种调度模式：细粒度模式，这种模式类似于现在的云计算，思想是**按需分配**。与粗粒度模式一样，应用程序启动时，先会启动 executor，但每个 executor 占用资源仅仅是自己运行所需的资源，之后，Mesos 会为每个 executor 动态分配资源，每分配一些，便可以运行一个新任务，单个 Task 运行完之后可以马上释放对应的资源。每个 Task 会汇报状态给 Mesos slave 和 Mesos Master，便于更加细粒度管理和容错。这种调度模式类似于 MapReduce 调度模式，每个 task 完全独立，优点是便于资源控制和隔离，但缺点也很明显，短作业运行延迟大。
在实际应用中，Spark支持多种部署模式，如Standalone、Spark on YARN和Spark on Mesos等，其中Spark on Mesos支持粗粒度和细粒度两种资源调度模式，而Spark on YARN目前仅支持粗粒度模式

## **五、RDD依赖关系**

RDD和它依赖的父RDDs关系有两种类型：

![image](assets/KTT2d1RcgoTAKLxJN8kcObq7nYb.png)

- 宽依赖：父RDD的一个分区会被子RDD的多个分区依赖（groupByKey、reduceByKey等），涉及跨节点通信。
- 窄依赖可以在同一个节点上，以 pipeline 形式执行多条命令（同一个 stage），例如在执行了 map 后，紧接着执行 filter
- 窄依赖：父RDD的一个分区只会被子RDD的一个分区依赖（map、filter等），不需要跨节点通信。
- 宽依赖需要所有的父分区都是可用的
其次，则是从失败恢复的角度考虑。窄依赖的恢复更快，因为它只需要重新计算丢失的 parent partition 即可，而且可以并行地在不同节点进行重计算（一台机器太慢就会分配到多个节点进行），相反，宽依赖牵涉 RDD 各级的多个 parent partition。

## 六、DAG

DAG（Directed Acyclic Graph）有向无环图，包含顶点和边，顶点是RDD，边是RDD之间的依赖与转换关系。DAG的构建是通过在分布式数据集上迭代调用算子来完成的，一个 Spark 程序有几个 action 就有几个 DAG。

DAG Spark 中使用 DAG 对 RDD 的关系进行建模，描述了 RDD 的依赖关系，这种关系也被称之为 lineage（血缘），RDD 的依赖关系使用 Dependency 维护。DAG 在 Spark 中的对应的实现为 DAGScheduler。

### 2.1、**Stage**

在 DAG 中又进行 stage 的划分，划分的依据是依赖是否是 shuffle 的，每个 stage 又可以划分成若干 task。接下来的事情就是 driver 发送 task 到 executor，executor 自己的线程池去执行这些 task，完成之后将结果返回给 driver。action 算子是划分不同 job 的依据。

划分 stage 的优点：实现 **流水线计算 **和 **并行计算**

- **流水线式内存计算**：在一个stage中，所有算子融合为一个函数，stage的输出结果由该函数一次性作用在输入数据集产生，提高数据在内存中的转换效率。
- **并行计算**：stage中每一个分区对应一个task，一个stage中可以并行运行多个task。
划分算法

回溯：**从后往前回溯，遇到窄依赖加入本stage，遇到宽依赖进行stage的划分。**

1. 从触发Action操作的RDD开始从后往前推，先为最后一个RDD创建一个stage，然后倒推。
1. 如果遇到窄依赖，就把该RDD加入到本stage中，如果遇到宽依赖，就从宽依赖切开，结束当前stage。
1. 创建新stage，按照步骤2继续倒推，直到最开始的RDD。
## 七、**Shuffle**

Shuffle分为两个阶段：Map（Shuffle Write）和Reduce（Shuffle Read）。

Map 流程

1. 计算分片数据得到目标分区，并填充到PartitionedPairBuffer。
1. PartitionedPairBuffer是数组形式的缓存结构，每条数据记录会占用数组中相邻的两个元素空间，第一个元素是(目标分区ID, key)，第二个元素是Value。
1. reduceByKey使用PartitionedAppendOnlyMap填充数据，Map的Value是可累加、更新的，适合聚合类计算场景，如求和、极值等。能够减少磁盘溢写文件数量和文件大小，降低Shuffle过程中磁盘和网络开销。
1. Buffer填满后，如果还有未处理的数据，就对Buffer中的数据按(目标分区ID, key)进行排序，将所有数据溢写到磁盘临时文件，并清空缓存。
1. 重复1、2，直到所有数据都被处理。
1. 对所有临时文件 和 PartitionedPairBuffer 归并排序，生成最终数据文件和索引文件。
Reduce环节：

- 每个 Map Task 生成的数据文件，都包含所有 Reduce Task 所需的部分数据。
- Reduce Task 需要从所有 Map Task 中拉取属于自己分区的部分数据，索引文件用于判定 Reduce Task 拉取数据的偏移地址范围。
一个形象类比：做薯片

![image](assets/OGXedhNbJoOvSixSAr0cKh4KnJg.png)
