---
title: "Doris 看板指标分析"
type: concept
domain: engineering/doris/治理
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris/治理
  - type/concept
---
# Doris 看板指标分析

> 内部参考：https://km.sankuai.com/page/1342529639

## 为什么需要看板

Doris 集群的健康状态不是"能查就行"这么简单。很多问题在爆发前就有征兆——Compaction 积压、内存水位上升、慢查询增多。**看板的核心价值是提前发现问题，而不是出了事再去翻日志。**

## 核心指标分类

### 一、集群资源指标

```
# BE 节点 CPU
doris_be_cpu_user_rate          # 用户态 CPU 使用率
doris_be_cpu_system_rate        # 内核态 CPU 使用率

# BE 内存
doris_be_process_mem_bytes      # BE 进程内存使用
doris_be_mem_limit              # 内存限制
doris_be_memory_pool_bytes_total # 内存池使用

# 磁盘
doris_be_disks_avail_capacity   # 磁盘可用空间
doris_be_disks_total_capacity   # 磁盘总容量
doris_be_disks_state            # 磁盘状态（0=正常，1=异常）
```

**告警阈值建议**：
- CPU > 80% 持续 5 分钟 → Warning
- 内存 > 90% mem_limit → Critical
- 磁盘使用率 > 85% → Warning，> 95% → Critical

### 二、查询性能指标

```
# QPS 与延迟
doris_fe_query_total             # 查询总数
doris_fe_query_err               # 查询错误数
doris_fe_query_latency_ms        # 查询延迟分布

# 重点关注 P99 延迟，而非平均值
# P50 = 100ms, P99 = 5s → 有 1% 的查询很慢，需要排查
```

我在看板上通常这么分面板：

```
┌──────────────┬──────────────┬──────────────┐
│   QPS 趋势   │  P99 延迟    │  错误率      │
├──────────────┼──────────────┼──────────────┤
│  慢查询 Top10 │ 查询类型分布  │ 并发连接数   │
└──────────────┴──────────────┴──────────────┘
```

### 三、导入指标

```
# Stream Load
doris_be_stream_load_rows_total   # 导入行数
doris_be_stream_load_bytes_total  # 导入字节数
doris_be_stream_load_duration_ms  # 导入耗时

# Routine Load
doris_fe_routine_load_rows        # Routine Load 消费行数
doris_fe_routine_load_error_rows  # 错误行数

# 导入延迟（Kafka Lag）
# 需要额外监控 Kafka Consumer Lag
```

**导入健康指标**：
- 导入成功率 > 99.9%
- 单次导入耗时 < 预期 SLA
- Kafka Lag 不持续增长

### 四、Compaction 指标

```
doris_be_compaction_deltas_total           # 待合并的版本数
doris_be_compaction_bytes_total            # Compaction 处理的数据量
doris_be_tablet_cumulative_max_compaction_score  # 最大 Compaction 分数
doris_be_tablet_base_max_compaction_score        # Base Compaction 分数
```

**这是最容易被忽视但最关键的指标**。Compaction Score 持续升高意味着合并跟不上导入速度，最终会触发 `-235` 错误。

```
Compaction Score 解读：
< 10   → 健康
10-50  → 正常范围
50-100 → 需要关注，考虑降低导入频率或增加 Compaction 线程
> 100  → 危险，可能很快触发版本限制
```

### 五、Tablet 指标

```
doris_be_tablet_num                   # Tablet 总数
doris_be_tablet_rowset_num            # Rowset 总数
doris_fe_scheduled_tablet_num         # 正在调度的 Tablet 数
```

Tablet 数量过多（比如超过 10 万/BE）会导致：
- FE 元数据管理压力大
- Compaction 调度变慢
- BE 启动时间变长

## Grafana 看板设计

### 总览 Dashboard

```json
// 核心面板布局
{
  "panels": [
    {"title": "集群 QPS", "type": "graph", "metric": "rate(doris_fe_query_total[5m])"},
    {"title": "查询 P99 延迟", "type": "graph", "metric": "histogram_quantile(0.99, doris_fe_query_latency_ms)"},
    {"title": "BE CPU 使用率", "type": "graph", "metric": "doris_be_cpu_user_rate"},
    {"title": "BE 内存使用率", "type": "gauge", "metric": "doris_be_process_mem_bytes / doris_be_mem_limit"},
    {"title": "磁盘使用率", "type": "gauge", "metric": "1 - doris_be_disks_avail_capacity / doris_be_disks_total_capacity"},
    {"title": "Compaction Score", "type": "graph", "metric": "doris_be_tablet_cumulative_max_compaction_score"}
  ]
}
```

### 导入 Dashboard

重点关注：
- Stream Load / Routine Load 的吞吐量趋势
- 导入错误率和错误类型
- Kafka Consumer Lag 趋势
- 导入与 Compaction 的平衡（导入速度 vs 合并速度）

### 慢查询分析

```sql
-- 从 FE audit log 中分析慢查询
-- 通常配置 audit_log_slow_query_time = 5000（5 秒）

-- 常见慢查询原因：
-- 1. 未命中分区裁剪 → 全表扫描
-- 2. JOIN 顺序不优 → 大表驱动小表
-- 3. 缺少合适的索引
-- 4. Compaction 滞后 → 查询需要 merge 大量版本
```

## 告警规则设计

```yaml
# 核心告警规则
alerts:
  - name: "Doris BE 内存告警"
    expr: doris_be_process_mem_bytes / doris_be_mem_limit > 0.9
    duration: 5m
    severity: critical
    
  - name: "Compaction Score 过高"
    expr: doris_be_tablet_cumulative_max_compaction_score > 100
    duration: 10m
    severity: warning
    
  - name: "查询错误率升高"
    expr: rate(doris_fe_query_err[5m]) / rate(doris_fe_query_total[5m]) > 0.01
    duration: 5m
    severity: warning
    
  - name: "磁盘空间不足"
    expr: 1 - doris_be_disks_avail_capacity / doris_be_disks_total_capacity > 0.85
    duration: 10m
    severity: warning
    
  - name: "BE 节点下线"
    expr: doris_be_alive == 0
    duration: 1m
    severity: critical
```

## 日常巡检清单

每日：
- [ ] QPS 和延迟趋势是否正常
- [ ] 导入是否有积压
- [ ] 磁盘和内存水位

每周：
- [ ] Compaction Score 趋势
- [ ] 慢查询 Top 20 分析
- [ ] Tablet 数量增长趋势
- [ ] 副本健康状态

## 相关

- [[Doris 治理]]
- [[Doris 架构|Doris 架构]]
- [[Doris 底层存储|Doris 底层存储]]
- [[Doris 生产|Doris 生产]]
- [[grafana 看板|grafana 看板]]
- [[监控指标|监控指标]]
