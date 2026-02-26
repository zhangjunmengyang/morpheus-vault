---
title: "Flink Partitioner"
type: concept
domain: engineering/flink/开发
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/flink/开发
  - type/concept
---
# Flink Partitioner

### 分区策略

1. globalpt：分发到下游第一个实例
1. forwardpt： API 用，用于同一个 operatorChain 上下游数据转发，要求并行度一样
1. ShufflePt：随机分区
1. RebalancePt：轮询分发
1. rescale：按比例
1. broadcast：广播
1. keyGroupStreampt：API 用，按照 key hash 分区
1. Custom：自定义分区接口