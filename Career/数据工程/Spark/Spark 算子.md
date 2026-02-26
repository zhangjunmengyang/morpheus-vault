---
title: "Spark 算子"
type: concept
domain: engineering/spark
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/spark
  - type/concept
---
# Spark 算子

RDD内部的数据转换算子

掌握 RDD 常用算子是做好 Spark 应用开发的基础，而数据转换类算子则是基础中的基础，因此我们优先来学习这类 RDD 算子。

在这些算子中，我们重点讲解的就是 map、mapPartitions、flatMap、filter。这 4 个算子几乎囊括了日常开发中 99% 的数据转换场景，剩下的 mapPartitionsWithIndex，我把它留给你作为课后作业去探索。

RDD算子分类表

![image](CIlGdYIGhoIgFPxJtiPc2S8AnUg.png)

开发注意点

使用 **reduceByKey、join、distinct、repartition** 等会进行 shuffle 的算子，需要考虑性能问题和必要性。减少 shuffle 可以大大减少性能开销。

俗话说，巧妇难为无米之炊，要想玩转厨房里的厨具，我们得先准备好米、面、油这些食材。学习 RDD 算子也是一样，要想动手操作这些算子，咱们得先有 RDD 才行。

所以，接下来我们就一起来看看 RDD 是怎么创建的。

创建 RDD

在 Spark 中，创建 RDD 的典型方式有两种：

通过 SparkContext.parallelize 在内部数据之上创建 RDD；

通过 SparkContext.textFile 等 API 从外部数据创建 RDD。

这里的内部、外部是相对应用程序来说的。开发者在 Spark 应用中自定义的各类数据结构，如数组、列表、映射等，都属于“内部数据”；而“外部数据”指代的，是 Spark 系统之外的所有数据形式，如本地文件系统或是分布式文件系统中的数据，再比如来自其他大数据组件（Hive、Hbase、RDBMS 等）的数据。

第一种创建方式的用法非常简单，只需要用 parallelize 函数来封装内部数据即可，比如下面的例子：

你可以在 spark-shell 中敲入上述代码，来直观地感受 parallelize 创建 RDD 的过程。通常来说，在 Spark 应用内定义体量超大的数据集，其实都是不太合适的，因为数据集完全由 Driver 端创建，且创建完成后，还要在全网范围内跨节点、跨进程地分发到其他 Executors，所以往往会带来性能问题。因此，parallelize API 的典型用法，是在“小数据”之上创建 RDD。

要想在真正的“大数据”之上创建 RDD，我们还得依赖第二种创建方式，也就是通过 SparkContext.textFile 等 API 从外部数据创建 RDD。由于 textFile API 比较简单，而且它在日常的开发中出现频率比较高，因此我们使用 textFile API 来创建 RDD。在后续对各类 RDD 算子讲解的过程中，我们都会使用 textFile API 从文件系统创建 RDD。

为了保持讲解的连贯性，我们还是使用之前的源文件 wikiOfSpark.txt 来创建 RDD，代码实现如下所示：

好啦，创建好了 RDD，我们就有了可以下锅的食材。接下来，咱们就要正式地走进厨房，把铲子和炒勺挥起来啦。

RDD 内的数据转换类算子

首先，我们先来认识一下 map 算子。毫不夸张地说，在所有的 RDD 算子中，map“出场”的概率是最高的。因此，我们必须要掌握 map 的用法与注意事项。

map：以元素为粒度的数据转换

我们先来说说 map 算子的用法：给定映射函数 f，map(f) 以元素为粒度对 RDD 做数据转换。其中 f 可以是带有明确签名的带名函数，也可以是匿名函数，它的形参类型必须与 RDD 的元素类型保持一致，而输出类型则任由开发者自行决定。

这种照本宣科的介绍听上去难免会让你有点懵，别着急，接下来我们用些小例子来更加直观地展示 map 的用法。

我们使用如下代码，把包含单词的 RDD 转换成元素为（Key，Value）对的 RDD，后者统称为 Paired RDD。

在上面的代码实现中，传递给 map 算子的形参，即：word => （word，1），就是我们上面说的映射函数 f。只不过，这里 f 是以匿名函数的方式进行定义的，其中左侧的 word 表示匿名函数 f 的输入形参，而右侧的（word，1）则代表函数 f 的输出结果。

如果我们把匿名函数变成带名函数的话，可能你会看的更清楚一些。这里我用一段代码重新定义了带名函数 f。

可以看到，我们使用 Scala 的 def 语法，明确定义了带名映射函数 f，它的计算逻辑与刚刚的匿名函数是一致的。在做 RDD 数据转换的时候，我们只需把函数 f 传递给 map 算子即可。不管 f 是匿名函数，还是带名函数，map 算子的转换逻辑都是一样的，你不妨把以上两种实现方式分别敲入到 spark-shell，去验证执行结果的一致性。

到这里为止，我们就掌握了 map 算子的基本用法。现在你就可以定义任意复杂的映射函数 f，然后在 RDD 之上通过调用 map(f) 去翻着花样地做各种各样的数据转换。

比如，通过定义如下的映射函数 f，我们就可以改写 Word Count 的计数逻辑，也就是把“Spark”这个单词的统计计数权重提高一倍：

尽管 map 算子足够灵活，允许开发者自由定义转换逻辑。不过，就像我们刚刚说的，map(f) 是以元素为粒度对 RDD 做数据转换的，在某些计算场景下，这个特点会严重影响执行效率。为什么这么说呢？我们来看一个具体的例子。

比方说，我们把 Word Count 的计数需求，从原来的对单词计数，改为对单词的哈希值计数，在这种情况下，我们的代码实现需要做哪些改动呢？我来示范一下：

由于 map(f) 是以元素为单元做转换的，那么对于 RDD 中的每一条数据记录，我们都需要实例化一个 MessageDigest 对象来计算这个元素的哈希值。

在工业级生产系统中，一个 RDD 动辄包含上百万甚至是上亿级别的数据记录，如果处理每条记录都需要事先创建 MessageDigest，那么实例化对象的开销就会聚沙成塔，不知不觉地成为影响执行效率的罪魁祸首。

那么问题来了，有没有什么办法，能够让 Spark 在更粗的数据粒度上去处理数据呢？还真有，mapPartitions 和 mapPartitionsWithIndex 这对“孪生兄弟”就是用来解决类似的问题。相比 mapPartitions，mapPartitionsWithIndex 仅仅多出了一个数据分区索引，因此接下来我们把重点放在 mapPartitions 上面。

mapPartitions：以数据分区为粒度的数据转换

按照介绍算子的惯例，我们还是先来说说 mapPartitions 的用法。mapPartitions，顾名思义，就是以数据分区为粒度，使用映射函数 f 对 RDD 进行数据转换。对于上述单词哈希值计数的例子，我们结合后面的代码，来看看如何使用 mapPartitions 来改善执行性能：

可以看到，在上面的改进代码中，mapPartitions 以数据分区（匿名函数的形参 partition）为粒度，对 RDD 进行数据转换。具体的数据处理逻辑，则由代表数据分区的形参 partition 进一步调用 map(f) 来完成。你可能会说：“partition. map(f) 仍然是以元素为粒度做映射呀！这和前一个版本的实现，有什么本质上的区别呢？”

仔细观察，你就会发现，相比前一个版本，我们把实例化 MD5 对象的语句挪到了 map 算子之外。如此一来，以数据分区为单位，实例化对象的操作只需要执行一次，而同一个数据分区中所有的数据记录，都可以共享该 MD5 对象，从而完成单词到哈希值的转换。

通过下图的直观对比，你会发现，以数据分区为单位，mapPartitions 只需实例化一次 MD5 对象，而 map 算子却需要实例化多次，具体的次数则由分区内数据记录的数量来决定。

map与mapPartitions的区别

对于一个有着上百万条记录的 RDD 来说，其数据分区的划分往往是在百这个量级，因此，相比 map 算子，mapPartitions 可以显著降低对象实例化的计算开销，这对于 Spark 作业端到端的执行性能来说，无疑是非常友好的。

实际上。除了计算哈希值以外，对于数据记录来说，凡是可以共享的操作，都可以用 mapPartitions 算子进行优化。这样的共享操作还有很多，比如创建用于连接远端数据库的 Connections 对象，或是用于连接 Amazon S3 的文件系统句柄，再比如用于在线推理的机器学习模型，等等，不一而足。

相比 mapPartitions，mapPartitionsWithIndex 仅仅多出了一个数据分区索引，这个数据分区索引可以为我们获取分区编号，当你的业务逻辑中需要使用到分区编号的时候，不妨考虑使用这个算子来实现代码。除了这个额外的分区索引以外，mapPartitionsWithIndex 在其他方面与 mapPartitions 是完全一样的。

介绍完 map 与 mapPartitions 算子之后，接下来，我们趁热打铁，再来看一个与这两者功能类似的算子：flatMap。

flatMap：从元素到集合、再从集合到元素

flatMap 其实和 map 与 mapPartitions 算子类似，在功能上，与 map 和 mapPartitions 一样，flatMap 也是用来做数据映射的，在实现上，对于给定映射函数 f，flatMap(f) 以元素为粒度，对 RDD 进行数据转换。

不过，与前两者相比，flatMap 的映射函数 f 有着显著的不同。对于 map 和 mapPartitions 来说，其映射函数 f 的类型，都是（元素） => （元素），即元素到元素。而 flatMap 映射函数 f 的类型，是（元素） => （集合），即元素到集合（如数组、列表等）。因此，flatMap 的映射过程在逻辑上分为两步：

以元素为单位，创建集合；

去掉集合“外包装”，提取集合元素。

这么说比较抽象，我们还是来举例说明。假设，我们再次改变 Word Count 的计算逻辑，由原来统计单词的计数，改为统计相邻单词共现的次数，如下图所示：

变更Word Count计算逻辑

对于这样的计算逻辑，我们该如何使用 flatMap 进行实现呢？这里我们先给出代码实现，然后再分阶段地分析 flatMap 的映射过程：

在上面的代码中，我们采用匿名函数的形式，来提供映射函数 f。这里 f 的形参是 String 类型的 line，也就是源文件中的一行文本，而 f 的返回类型是 Array[String]，也就是 String 类型的数组。在映射函数 f 的函数体中，我们先用 split 语句把 line 转化为单词数组，然后再用 for 循环结合 yield 语句，依次把单个的单词，转化为相邻单词词对。

注意，for 循环返回的依然是数组，也即类型为 Array[String]的词对数组。由此可见，函数 f 的类型是（String） => （Array[String]），也就是刚刚说的第一步，从元素到集合。但如果我们去观察转换前后的两个 RDD，也就是 lineRDD 和 wordPairRDD，会发现它们的类型都是 RDD[String]，换句话说，它们的元素类型都是 String。

回顾 map 与 mapPartitions 这两个算子，我们会发现，转换前后 RDD 的元素类型，与映射函数 f 的类型是一致的。但在 flatMap 这里，却出现了 RDD 元素类型与函数类型不一致的情况。这是怎么回事呢？其实呢，这正是 flatMap 的“奥妙”所在，为了让你直观地理解 flatMap 的映射过程，我画了一张示意图，如下所示：

不难发现，映射函数 f 的计算过程，对应着图中的步骤 1 与步骤 2，每行文本都被转化为包含相邻词对的数组。紧接着，flatMap 去掉每个数组的“外包装”，提取出数组中类型为 String 的词对元素，然后以词对为单位，构建新的数据分区，如图中步骤 3 所示。这就是 flatMap 映射过程的第二步：去掉集合“外包装”，提取集合元素。

得到包含词对元素的 wordPairRDD 之后，我们就可以沿用 Word Count 的后续逻辑，去计算相邻词汇的共现次数。你不妨结合文稿中的代码与第一讲中 Word Count 的代码，去实现完整版的“相邻词汇计数统计”。

filter：过滤 RDD

在今天的最后，我们再来学习一下，与 map 一样常用的算子：filter。filter，顾名思义，这个算子的作用，是对 RDD 进行过滤。就像是 map 算子依赖其映射函数一样，filter 算子也需要借助一个判定函数 f，才能实现对 RDD 的过滤转换。

所谓判定函数，它指的是类型为（RDD 元素类型） => （Boolean）的函数。可以看到，判定函数 f 的形参类型，必须与 RDD 的元素类型保持一致，而 f 的返回结果，只能是 True 或者 False。在任何一个 RDD 之上调用 filter(f)，其作用是保留 RDD 中满足 f（也就是 f 返回 True）的数据元素，而过滤掉不满足 f（也就是 f 返回 False）的数据元素。

老规矩，我们还是结合示例来讲解 filter 算子与判定函数 f。

在上面 flatMap 例子的最后，我们得到了元素为相邻词汇对的 wordPairRDD，它包含的是像“Spark-is”、“is-cool”这样的字符串。为了仅保留有意义的词对元素，我们希望结合标点符号列表，对 wordPairRDD 进行过滤。例如，我们希望过滤掉像“Spark-&”、“|-data”这样的词对。

掌握了 filter 算子的用法之后，要实现这样的过滤逻辑，我相信你很快就能写出如下的代码实现：

掌握了 filter 算子的用法之后，你就可以定义任意复杂的判定函数 f，然后在 RDD 之上通过调用 filter(f) 去变着花样地做数据过滤，从而满足不同的业务需求。

数据聚合算子

这部分的算子都会引入繁重的 Shuffle 计算。这些算子分别是 groupByKey、reduceByKey、aggregateByKey 和 sortByKey，也就是表格中加粗的部分。

我们知道，在数据分析场景中，典型的计算类型分别是分组、聚合和排序。而 groupByKey、reduceByKey、aggregateByKey 和 sortByKey 这些算子的功能，恰恰就是用来实现分组、聚合和排序的计算逻辑。

尽管这些算子看上去相比其他算子的适用范围更窄，也就是它们只能作用（Apply）在 Paired RDD 之上，所谓 Paired RDD，它指的是元素类型为（Key，Value）键值对的 RDD。

但是在功能方面，可以说，它们承担了数据分析场景中的大部分职责。因此，掌握这些算子的用法，是我们能够游刃有余地开发数据分析应用的重要基础。那么接下来，我们就通过一些实例，来熟悉并学习这些算子的用法。

我们先来说说 groupByKey，坦白地说，相比后面的 3 个算子，groupByKey 在我们日常开发中的“出镜率”并不高。之所以要先介绍它，主要是为后续的 reduceByKey 和 aggregateByKey 这两个重要算子做铺垫。

groupByKey：分组收集

groupByKey 的字面意思是“按照 Key 做分组”，但实际上，groupByKey 算子包含两步，即分组和收集。

具体来说，对于元素类型为（Key，Value）键值对的 Paired RDD，groupByKey 的功能就是对 Key 值相同的元素做分组，然后把相应的 Value 值，以集合的形式收集到一起。换句话说，groupByKey 会把 RDD 的类型，由 RDD[(Key, Value)]转换为 RDD[(Key, Value 集合)]。

这么说比较抽象，我们还是用一个小例子来说明 groupByKey 的用法。还是我们熟知的 Word Count，对于分词后的一个个单词，假设我们不再统计其计数，而仅仅是把相同的单词收集到一起，那么我们该怎么做呢？按照老规矩，咱们还是先来给出代码实现：

结合前面的代码可以看到，相比之前的 Word Count，我们仅需做两个微小的改动，即可实现新的计算逻辑。第一个改动，是把 map 算子的映射函数 f，由原来的 word => （word，1）变更为 word => （word，word），这么做的效果，是把 kvRDD 元素的 Key 和 Value 都变成了单词。

紧接着，第二个改动，我们用 groupByKey 替换了原先的 reduceByKey。相比 reduceByKey，groupByKey 的用法要简明得多。groupByKey 是无参函数，要实现对 Paired RDD 的分组、收集，我们仅需在 RDD 之上调用 groupByKey() 即可。

尽管 groupByKey 的用法非常简单，但它的计算过程值得我们特别关注，下面我用一张示意图来讲解上述代码的计算过程，从而让你更加直观地感受 groupByKey 可能存在的性能隐患。

从图上可以看出，为了完成分组收集，对于 Key 值相同、但分散在不同数据分区的原始数据记录，Spark 需要通过 Shuffle 操作，跨节点、跨进程地把它们分发到相同的数据分区。Shuffle 是资源密集型计算，对于动辄上百万、甚至上亿条数据记录的 RDD 来说，这样的 Shuffle 计算会产生大量的磁盘 I/O 与网络 I/O 开销，从而严重影响作业的执行性能。

虽然 groupByKey 的执行效率较差，不过好在它在应用开发中的“出镜率”并不高。原因很简单，在数据分析领域中，分组收集的使用场景很少，而分组聚合才是统计分析的刚需。

为了满足分组聚合多样化的计算需要，Spark 提供了 3 种 RDD 算子，允许开发者灵活地实现计算逻辑，它们分别是 reduceByKey、aggregateByKey 和 combineByKey。

reduceByKey 我们并不陌生，Word Count 实现就用到了这个算子，aggregateByKey 是 reduceByKey 的“升级版”，相比 reduceByKey，aggregateByKey 用法更加灵活，支持的功能也更加完备。

接下来，我们先来回顾 reduceByKey，然后再对 aggregateByKey 进行展开。相比 aggregateByKey，combineByKey 仅在初始化方式上有所不同，因此，我把它留给你作为课后作业去探索。

reduceByKey：分组聚合

reduceByKey 的字面含义是“按照 Key 值做聚合”，它的计算逻辑，就是根据聚合函数 f 给出的算法，把 Key 值相同的多个元素，聚合成一个元素。

在Word Count 的实现中，我们使用了 reduceByKey 来实现分组计数：

重温上面的这段代码，你有没有觉得 reduceByKey 与之前讲过的 map、filter 这些算子有一些相似的地方？没错，给定处理函数 f，它们的用法都是“算子 (f)”。只不过对于 map 来说，我们把 f 称作是映射函数，对 filter 来说，我们把 f 称作判定函数，而对于 reduceByKey，我们把 f 叫作聚合函数。

在上面的代码示例中，reduceByKey 的聚合函数是匿名函数：(x, y) => x + y。与 map、filter 等算子的用法一样，你也可以明确地定义带名函数 f，然后再用 reduceByKey(f) 的方式实现同样的计算逻辑。

需要强调的是，给定 RDD[(Key 类型，Value 类型)]，聚合函数 f 的类型，必须是（Value 类型，Value 类型） => （Value 类型）。换句话说，函数 f 的形参，必须是两个数值，且数值的类型必须与 Value 的类型相同，而 f 的返回值，也必须是 Value 类型的数值。

咱们不妨再举一个小例子，让你加深对于 reduceByKey 算子的理解。

接下来，我们把 Word Count 的计算逻辑，改为随机赋值、提取同一个 Key 的最大值。也就是在 kvRDD 的生成过程中，我们不再使用映射函数 word => (word, 1)，而是改为 word => (word, 随机数)，然后再使用 reduceByKey 算子来计算同一个 word 当中最大的那个随机数。

你可以先停下来，花点时间想一想这个逻辑该怎么实现，然后再来参考下面的代码：

观察上面的代码片段，不难发现，reduceByKey 算子的用法还是比较简单的，只需要先定义好聚合函数 f，然后把它传给 reduceByKey 算子就行了。那么在运行时，上述代码的计算又是怎样的一个过程呢？

我把 reduceByKey 的计算过程抽象成了下图：

从图中你可以看出来，尽管 reduceByKey 也会引入 Shuffle，但相比 groupByKey 以全量原始数据记录的方式消耗磁盘与网络，reduceByKey 在落盘与分发之前，会先在 Shuffle 的 Map 阶段做初步的聚合计算。

比如，在数据分区 0 的处理中，在 Map 阶段，reduceByKey 把 Key 同为 Streaming 的两条数据记录聚合为一条，聚合逻辑就是由函数 f 定义的、取两者之间 Value 较大的数据记录，这个过程我们称之为“Map 端聚合”。相应地，数据经由网络分发之后，在 Reduce 阶段完成的计算，我们称之为“Reduce 端聚合”。

你可能会说：“做了 Map 聚合又能怎样呢？相比 groupByKey，reduceByKey 带来的性能收益并不算明显呀！”确实，就上面的示意图来说，我们很难感受到 reduceByKey 带来的性能收益。不过，量变引起质变，在工业级的海量数据下，相比 groupByKey，reduceByKey 通过在 Map 端大幅削减需要落盘与分发的数据量，往往能将执行效率提升至少一倍。

应该说，对于大多数分组 & 聚合的计算需求来说，只要设计合适的聚合函数 f，你都可以使用 reduceByKey 来实现计算逻辑。不过，术业有专攻，reduceByKey 算子的局限性，在于其 Map 阶段与 Reduce 阶段的计算逻辑必须保持一致，这个计算逻辑统一由聚合函数 f 定义。当一种计算场景需要在两个阶段执行不同计算逻辑的时候，reduceByKey 就爱莫能助了。

比方说，还是Word Count，我们想对单词计数的计算逻辑做如下调整：

在 Map 阶段，以数据分区为单位，计算单词的加和；

而在 Reduce 阶段，对于同样的单词，取加和最大的那个数值。

显然，Map 阶段的计算逻辑是 sum，而 Reduce 阶段的计算逻辑是 max。对于这样的业务需求，reduceByKey 已无用武之地，这个时候，就轮到 aggregateByKey 这个算子闪亮登场了。

aggregateByKey：更加灵活的聚合算子

老规矩，算子的介绍还是从用法开始。相比其他算子，aggregateByKey 算子的参数比较多。要在 Paired RDD 之上调用 aggregateByKey，你需要提供一个初始值，一个 Map 端聚合函数 f1，以及一个 Reduce 端聚合函数 f2，aggregateByKey 的调用形式如下所示：

初始值可以是任意数值或是字符串，而聚合函数我们也不陌生，它们都是带有两个形参和一个输出结果的普通函数。就这 3 个参数来说，比较伤脑筋的，是它们之间的类型需要保持一致，具体来说：

初始值类型，必须与 f2 的结果类型保持一致；

f1 的形参类型，必须与 Paired RDD 的 Value 类型保持一致；

f2 的形参类型，必须与 f1 的结果类型保持一致。

不同类型之间的一致性描述起来比较拗口，咱们不妨结合示意图来加深理解：

aggregateByKey参数之间的类型一致性

熟悉了 aggregateByKey 的用法之后，接下来，我们用 aggregateByKey 这个算子来实现刚刚提到的“先加和，再取最大值”的计算逻辑，代码实现如下所示：

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

// 把RDD元素转换为（Key，Value）的形式

val kvRDD: RDD[(String, Int)] = cleanWordRDD.map(word => (word, 1))

 

// 显示定义Map阶段聚合函数f1

def f1(x: Int, y: Int): Int = {

return x + y

}

// 显示定义Reduce阶段聚合函数f2

def f2(x: Int, y: Int): Int = {

return math.max(x, y)

}

 

// 调用aggregateByKey，实现先加和、再求最大值

val wordCounts: RDD[(String, Int)] = kvRDD.aggregateByKey(0) (f1, f2)

怎么样？是不是很简单？结合计算逻辑的需要，我们只需要提前定义好两个聚合函数，同时保证参数之间的类型一致性，然后把初始值、聚合函数传入 aggregateByKey 算子即可。按照惯例，我们还是通过 aggregateByKey 在运行时的计算过程，来帮你深入理解算子的工作原理：

不难发现，在运行时，与 reduceByKey 相比，aggregateByKey 的执行过程并没有什么两样，最主要的区别，还是 Map 端聚合与 Reduce 端聚合的计算逻辑是否一致。值得一提的是，与 reduceByKey 一样，aggregateByKey 也可以通过 Map 端的初步聚合来大幅削减数据量，在降低磁盘与网络开销的同时，提升 Shuffle 环节的执行性能。

sortByKey：排序

我们再来说说 sortByKey 这个算子，顾名思义，它的功能是“按照 Key 进行排序”。给定包含（Key，Value）键值对的 Paired RDD，sortByKey 会以 Key 为准对 RDD 做排序。算子的用法比较简单，只需在 RDD 之上调用 sortByKey() 即可：

在默认的情况下，sortByKey 按照 Key 值的升序（Ascending）对 RDD 进行排序，如果想按照降序（Descending）来排序的话，你需要给 sortByKey 传入 false。总结下来，关于排序的规则，你只需要记住如下两条即可：

升序排序：调用 sortByKey()、或者 sortByKey(true)；

降序排序：调用 sortByKey(false)。

数据的准备、重分布与持久化算子

在 RDD 常用算子的前两部分中，我们分别介绍了用于 RDD 内部转换与聚合的诸多算子，今天这一讲，我们继续来介绍表格中剩余部分的算子。

按照惯例，表格中的算子我们不会全都介绍，而是只挑选其中最常用、最具代表性的进行讲解。今天要讲的算子，我用加粗字体进行了高亮显示，你不妨先扫一眼，做到心中有数。

数据准备

首先，我们先来说说数据准备阶段的 union 和 sample。

union

在我们日常的开发中，union 非常常见，它常常用于把两个类型一致、但来源不同的 RDD 进行合并，从而构成一个统一的、更大的分布式数据集。例如，在某个数据分析场景中，一份数据源来自远端数据库，而另一份数据源来自本地文件系统，要将两份数据进行合并，我们就需要用到 union 这个操作。

具体怎么使用呢？我来举个例子。给定两个 RDD：rdd1 和 rdd2，调用 rdd1.union(rdd2) 或是 rdd1 union rdd2，其结果都是两个 RDD 的并集，具体代码如下：

需要特别强调的是，union 操作能够成立的前提，就是参与合并的两个 RDD 的类型必须完全一致。也就是说，RDD[String]只能与 RDD[String]合并到一起，却无法与除 RDD[String]以外的任何 RDD 类型（如 RDD[Int]、甚至是 RDD[UserDefinedClass]）做合并。

对于多个类型一致的 RDD，我们可以通过连续调用 union 把所有数据集合并在一起。例如，给定类型一致的 3 个 RDD：rdd1、rdd2 和 rdd3，我们可以使用如下代码把它们合并在一起。

不难发现，union 的典型使用场景，是把多份“小数据”，合并为一份“大数据”，从而充分利用 Spark 分布式引擎的并行计算优势。

与之相反，在一般的数据探索场景中，我们往往只需要对一份数据的子集有基本的了解即可。例如，对于一份体量在 TB 级别的数据集，我们只想随机提取其部分数据，然后计算这部分子集的统计值（均值、方差等）。

那么，面对这类把“大数据”变成 “小数据”的计算需求，Spark 又如何进行支持呢？这就要说到 RDD 的 sample 算子了。

sample

RDD 的 sample 算子用于对 RDD 做随机采样，从而把一个较大的数据集变为一份“小数据”。相较其他算子，sample 的参数比较多，分别是 withReplacement、fraction 和 seed。因此，要在 RDD 之上完成数据采样，你需要使用如下的方式来调用 sample 算子：sample(withReplacement, fraction, seed)。

其中，withReplacement 的类型是 Boolean，它的含义是“采样是否有放回”，如果这个参数的值是 true，那么采样结果中可能会包含重复的数据记录，相反，如果该值为 false，那么采样结果不存在重复记录。

fraction 参数最好理解，它的类型是 Double，值域为 0 到 1，其含义是采样比例，也就是结果集与原数据集的尺寸比例。seed 参数是可选的，它的类型是 Long，也就是长整型，用于控制每次采样的结果是否一致。光说不练假把式，我们还是结合一些示例，这样才能更好地理解 sample 算子的用法。

我们的实验分为 3 组，前两组用来对比添加 seed 参数与否的差异，最后一组用于说明 withReplacement 参数的作用。

不难发现，在不带 seed 参数的情况下，每次调用 sample 之后的返回结果都不一样。而当我们使用同样的 seed 调用算子时，不论我们调用 sample 多少次，每次的返回结果都是一致的。另外，仔细观察第 3 组实验，你会发现结果集中有重复的数据记录，这是因为 withReplacement 被置为 true，采样的过程是“有放回的”。

好啦，到目前为止，数据准备阶段常用的两个算子我们就讲完了。有了 union 和 sample，你就可以随意地调整分布式数据集的尺寸，真正做到收放自如。

数据预处理

接下来，在数据预处理阶段，我们再来说说负责数据重分布的两个算子：repartition 和 coalesce。

在了解这两个算子之前，你需要先理解并行度这个概念。所谓并行度，它实际上就是 RDD 的数据分区数量。还记得吗？RDD 的 partitions 属性，记录正是 RDD 的所有数据分区。因此，RDD 的并行度与其 partitions 属性相一致。

开发者可以使用 repartition 算子随意调整（提升或降低）RDD 的并行度，而 coalesce 算子则只能用于降低 RDD 并行度。显然，在数据分布的调整方面，repartition 灵活度更高、应用场景更多，我们先对它进行介绍，之后再去看看 coalesce 有什么用武之地。

repartition

一旦给定了 RDD，我们就可以通过调用 repartition(n) 来随意调整 RDD 并行度。其中参数 n 的类型是 Int，也就是整型，因此，我们可以把任意整数传递给 repartition。按照惯例，咱们还是结合示例熟悉一下 repartition 的用法。

首先，我们通过数组创建用于实验的 RDD，从这段代码里可以看到，该 RDD 的默认并行度是 4。在我们分别用 2 和 8 来调整 RDD 的并行度之后，通过计算 RDD partitions 属性的长度，我们发现新 RDD 的并行度分别被相应地调整为 2 和 8。

看到这里，你可能还有疑问：“我们为什么需要调整 RDD 的并行度呢？2 和 8 看上去也没什么实质性的区别呀”。

每个 RDD 的数据分区，都对应着一个分布式 Task，而每个 Task 都需要一个 CPU 线程去执行。

因此，RDD 的并行度，很大程度上决定了分布式系统中 CPU 的使用效率，进而还会影响分布式系统并行计算的执行效率。并行度过高或是过低，都会降低 CPU 利用率，从而白白浪费掉宝贵的分布式计算资源，因此，合理有效地设置 RDD 并行度，至关重要。

这时你可能会追问：“既然如此，那么我该如何合理地设置 RDD 的并行度呢？”坦白地说，这个问题并没有固定的答案，它取决于系统可用资源、分布式数据集大小，甚至还与执行内存有关。

不过，结合经验来说，把并行度设置为可用 CPU 的 2 到 3 倍，往往是个不错的开始。例如，可分配给 Spark 作业的 Executors 个数为 N，每个 Executors 配置的 CPU 个数为 C，那么推荐设置的并行度坐落在 NC2 到 NC3 这个范围之间。

尽管 repartition 非常灵活，你可以用它随意地调整 RDD 并行度，但是你也需要注意，这个算子有个致命的弊端，那就是它会引入 Shuffle。

我们知道由于 Shuffle 在计算的过程中，会消耗所有类型的硬件资源，尤其是其中的磁盘 I/O 与网络 I/O，因此 Shuffle 往往是作业执行效率的瓶颈。正是出于这个原因，在做应用开发的时候，我们应当极力避免 Shuffle 的引入。

但你可能会说：“如果数据重分布是刚需，而 repartition 又必定会引入 Shuffle，我该怎么办呢？”如果你想增加并行度，那我们还真的只能仰仗 repartition，Shuffle 的问题自然也就无法避免。但假设你的需求是降低并行度，这个时候，我们就可以把目光投向 repartition 的孪生兄弟：coalesce。

coalesce

在用法上，coalesce 与 repartition 一样，它也是通过指定一个 Int 类型的形参，完成对 RDD 并行度的调整，即 coalesce (n)。那两者的用法到底有什么差别呢？我们不妨结合刚刚的代码示例，来对比 coalesce 与 repartition。

可以看到，在用法上，coalesce 与 repartition 可以互换，二者的效果是完全一致的。不过，如果我们去观察二者的 DAG，会发现同样的计算逻辑，却有着迥然不同的执行计划。

在 RDD 之上调用 toDebugString，Spark 可以帮我们打印出当前 RDD 的 DAG。尽管图中的打印文本看上去有些凌乱，但你只要抓住其中的一个关键要点就可以了。

这个关键要点就是，在 toDebugString 的输出文本中，每一个带数字的小括号，比如 rdd1 当中的“(2)”和“(4)”，都代表着一个执行阶段，也就是 DAG 中的 Stage。而且，不同的 Stage 之间，会通过制表符（Tab）缩进进行区分，比如图中的“(4)”显然要比“(2)”缩进了一段距离。

对于 toDebugString 的解读，你只需要掌握到这里就足够了。学习过调度系统之后，我们已经知道，在同一个 DAG 内，不同 Stages 之间的边界是 Shuffle。因此，观察上面的打印文本，我们能够清楚地看到，repartition 会引入 Shuffle，而 coalesce 不会。

那么问题来了，同样是重分布的操作，为什么 repartition 会引入 Shuffle，而 coalesce 不会呢？原因在于，二者的工作原理有着本质的不同。

给定 RDD，如果用 repartition 来调整其并行度，不论增加还是降低，对于 RDD 中的每一条数据记录，repartition 对它们的影响都是无差别的数据分发。

具体来说，给定任意一条数据记录，repartition 的计算过程都是先哈希、再取模，得到的结果便是该条数据的目标分区索引。对于绝大多数的数据记录，目标分区往往坐落在另一个 Executor、甚至是另一个节点之上，因此 Shuffle 自然也就不可避免。

coalesce 则不然，在降低并行度的计算中，它采取的思路是把同一个 Executor 内的不同数据分区进行合并，如此一来，数据并不需要跨 Executors、跨节点进行分发，因而自然不会引入 Shuffle。

这里我还特意准备了一张示意图，更直观地为你展示 repartition 与 coalesce 的计算过程，图片文字双管齐下，相信你一定能够更加深入地理解 repartition 与 coalesce 之间的区别与联系。

好啦，到此为止，在数据预处理阶段，用于对 RDD 做重分布的两个算子我们就讲完了。掌握了 repartition 和 coalesce 这两个算子，结合数据集大小与集群可用资源，你就可以随意地对 RDD 的并行度进行调整，进而提升 CPU 利用率与作业的执行性能。

结果收集

预处理完成之后，数据生命周期的下一个阶段是数据处理，在这个环节，你可以对数据进行各式各样的处理，比如数据转换、数据过滤、数据聚合，等等。完成处理之后，我们自然要收集计算结果。

在结果收集方面，Spark 也为我们准备了丰富的算子。按照收集路径区分，这些算子主要分为两类：第一类是把计算结果从各个 Executors 收集到 Driver 端，第二个类是把计算结果通过 Executors 直接持久化到文件系统。在大数据处理领域，文件系统往往指的是像 HDFS 或是 S3 这样的分布式文件系统。

first、take 和 collect

我们今天要介绍的第一类算子有 first、take 和 collect，它们的用法非常简单，按照老规矩，我们还是使用代码示例进行讲解。这里我们结合Word Count入门程序，分别使用 first、take 和 collect 这三个算子对不同阶段的 RDD 进行数据探索。

其中，first 用于收集 RDD 数据集中的任意一条数据记录，而 take(n: Int) 则用于收集多条记录，记录的数量由 Int 类型的参数 n 来指定。

不难发现，first 与 take 的主要作用，在于数据探索。对于 RDD 的每一步转换，比如 Word Count 中从文本行到单词、从单词到 KV 转换，我们都可以用 first 或是 take 来获取几条计算结果，从而确保转换逻辑与预期一致。

相比之下，collect 拿到的不是部分结果，而是全量数据，也就是把 RDD 的计算结果全量地收集到 Driver 端。在上面 Word Count 的例子中，我们可以看到，由于全量结果较大，屏幕打印只好做截断处理。

为了让你更深入地理解 collect 算子的工作原理，我把它的计算过程画在了后面的示意图中。

结合示意图，不难发现，collect 算子有两处性能隐患，一个是拉取数据过程中引入的网络开销，另一个 Driver 的 OOM（内存溢出，Out of Memory）。

网络开销很好理解，既然数据的拉取和搬运是跨进程、跨节点的，那么和 Shuffle 类似，这个过程必然会引入网络开销。

再者，通常来说，Driver 端的预设内存往往在 GB 量级，而 RDD 的体量一般都在数十 GB、甚至上百 GB，因此，OOM 的隐患不言而喻。collect 算子尝试把 RDD 全量结果拉取到 Driver，当结果集尺寸超过 Driver 预设的内存大小时，Spark 自然会报 OOM 的异常（Exception）。

正是出于这些原因，我们在使用 collect 算子之前，务必要慎重。不过，你可能会问：“如果业务逻辑就是需要收集全量结果，而 collect 算子又不好用，那我该怎么办呢？”别着急，我们接着往下看。

saveAsTextFile

对于全量的结果集，我们还可以使用第二类算子把它们直接持久化到磁盘。在这类算子中，最具代表性的非 saveAsTextFile 莫属，它的用法非常简单，给定 RDD，我们直接调用 saveAsTextFile(path: String) 即可。其中 path 代表的是目标文件系统目录，它可以是本地文件系统，也可以是 HDFS、Amazon S3 等分布式文件系统。

为了让你加深对于第二类算子的理解，我把它们的工作原理也整理到了下面的示意图中。可以看到，以 saveAsTextFile 为代表的算子，直接通过 Executors 将 RDD 数据分区物化到文件系统，这个过程并不涉及与 Driver 端的任何交互。

由于数据的持久化与 Driver 无关，因此这类算子天然地避开了 collect 算子带来的两个性能隐患。
