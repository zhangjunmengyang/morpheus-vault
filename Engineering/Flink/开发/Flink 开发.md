---
title: "5. Flink 开发"
type: concept
domain: engineering/flink/开发
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/开发
  - type/concept
---
# 5. Flink 开发

## 开发模式选择

Flink 提供多层 API，选择的核心依据是 **业务复杂度 vs 开发效率** 的权衡：

```
SQL / Table API      ← 80% 的场景用这个就够了
      ↓
DataStream API       ← 需要精细控制状态、窗口、时间
      ↓
ProcessFunction      ← 最底层，完全自定义
```

我的经验是：**能用 SQL 就不要用 DataStream**。Flink SQL 的优化器（Calcite-based）越来越成熟，手写 DataStream 大概率跑不过 SQL 优化后的执行计划。只有在需要复杂事件处理（CEP）、自定义窗口、精确状态控制时才下沉到 DataStream。

## DataStream API 核心模式

### Source → Transform → Sink

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Source
DataStream<String> source = env.fromSource(
    KafkaSource.<String>builder()
        .setBootstrapServers("kafka:9092")
        .setTopics("input-topic")
        .setGroupId("my-group")
        .setStartingOffsets(OffsetsInitializer.committedOffsets(OffsetResetStrategy.LATEST))
        .setValueOnlyDeserializer(new SimpleStringSchema())
        .build(),
    WatermarkStrategy.noWatermarks(),
    "Kafka Source"
);

// Transform
DataStream<Event> events = source
    .map(json -> JSON.parseObject(json, Event.class))
    .filter(e -> e.getType() != null)
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new EventAggregator());

// Sink
events.sinkTo(
    KafkaSink.<Event>builder()
        .setBootstrapServers("kafka:9092")
        .setRecordSerializer(
            KafkaRecordSerializationSchema.builder()
                .setTopic("output-topic")
                .setValueSerializationSchema(new EventSerializationSchema())
                .build()
        )
        .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
        .build()
);

env.execute("My Flink Job");
```

### ProcessFunction：最灵活的武器

```java
public class OrderTimeoutFunction extends KeyedProcessFunction<String, Order, Alert> {

    private ValueState<Order> orderState;

    @Override
    public void open(Configuration parameters) {
        orderState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("order", Order.class));
    }

    @Override
    public void processElement(Order order, Context ctx, Collector<Alert> out) throws Exception {
        orderState.update(order);
        // 注册 30 分钟后的定时器
        ctx.timerService().registerEventTimeTimer(
            order.getCreateTime() + 30 * 60 * 1000L);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<Alert> out) throws Exception {
        Order order = orderState.value();
        if (order != null && order.getStatus().equals("UNPAID")) {
            out.collect(new Alert(order.getId(), "ORDER_TIMEOUT"));
        }
        orderState.clear();
    }
}
```

ProcessFunction 的核心能力：**状态 + 定时器 + Side Output**。这三个组合能覆盖绝大多数复杂业务逻辑。

## Flink SQL 开发要点

### DDL 定义

```sql
CREATE TABLE kafka_orders (
    order_id STRING,
    user_id STRING,
    amount DECIMAL(10, 2),
    order_time TIMESTAMP(3),
    WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'orders',
    'properties.bootstrap.servers' = 'kafka:9092',
    'format' = 'json',
    'scan.startup.mode' = 'latest-offset'
);
```

### 常见陷阱

1. **Retract 机制**：Flink SQL 的动态表是 Append + Retract 模式。GROUP BY 后的结果会不断更新，下游要能处理 Retract 消息。
2. **状态 TTL**：忘记设置 `table.exec.state.ttl` 会导致状态无限增长，最终 OOM。

```sql
SET 'table.exec.state.ttl' = '24h';
```

3. **时间属性**：Event Time 必须配 Watermark，Processing Time 不需要但无法保证结果确定性。

## 开发调试技巧

### 本地调试

```java
// MiniCluster 模式，不需要部署集群
StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironment(4);
// 或者用 LocalStreamEnvironment 配合 Web UI
Configuration conf = new Configuration();
conf.setInteger(RestOptions.PORT, 8081);
StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(conf);
```

### 状态查询

```bash
# 通过 Queryable State 查询运行时状态（适合调试）
# 生产慎用，性能开销大
```

### 日志埋点

```java
// 不要用 System.out，用 SLF4J
private static final Logger LOG = LoggerFactory.getLogger(MyFunction.class);

@Override
public void processElement(Event event, Context ctx, Collector<Result> out) {
    LOG.debug("Processing event: {}", event.getId());
    // 生产环境关闭 DEBUG，保留 WARN/ERROR
}
```

## 序列化选型

| 序列化框架 | 性能 | 易用性 | 适用场景 |
|-----------|------|--------|---------|
| Flink PojoSerializer | 中 | 高（自动推断） | 简单 POJO |
| Kryo | 中 | 高（兜底方案） | 复杂对象 |
| Avro | 高 | 中 | Schema Evolution |
| Protobuf | 高 | 低 | 跨语言、高性能 |

**建议**：核心数据路径用 Avro 或 Protobuf，开发阶段用 POJO + Kryo 快速迭代。如果发现 `KryoSerializer` 出现在执行计划里，要警惕——它说明 Flink 没能推断出高效的 TypeSerializer。

## 项目结构建议

```
flink-job/
├── src/main/java/
│   ├── job/           # Job 主入口
│   ├── source/        # 自定义 Source
│   ├── function/      # UDF / ProcessFunction
│   ├── model/         # 数据模型
│   ├── sink/          # 自定义 Sink
│   └── util/          # 工具类
├── src/main/resources/
│   └── log4j2.properties
└── pom.xml
```

## 相关

- [[Flink 概述]]
- [[Flink API（Java）]]
- [[Flink SQL UDF]]
- [[Flink CEP CDC]]
- [[Flink 窗口和时间机制]]
- [[Flink 内存机制]]
- [[Flink CheckPoint]]
- [[Flink 生产]]
- [[Flink SideOutput]]
