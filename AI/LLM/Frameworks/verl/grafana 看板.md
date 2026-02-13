---
title: "grafana 看板"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# grafana 看板

> 参考：https://verl.readthedocs.io/en/latest/advance/grafana_prometheus.html

## 为什么需要监控

RL 训练动辄几十个小时，出了问题不可能靠 `print` 来调。verl 内置了 Prometheus + Grafana 的监控方案，能实时观察训练过程中的关键指标，比如 reward 曲线、GPU 利用率、rollout 延迟等。

个人经验：**reward 曲线不涨**和**某个 worker 卡住**是最常见的两类问题，有了看板能在几分钟内定位。

## 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  verl Worker │────▶│  Prometheus  │────▶│   Grafana    │
│  (metrics)   │     │  (scrape)    │     │  (dashboard) │
└──────────────┘     └──────────────┘     └──────────────┘
```

verl 的每个 worker（Actor、Critic、Rollout、Reward）都暴露了一个 `/metrics` 端点，Prometheus 定时抓取，Grafana 负责可视化。

## 部署步骤

### 1. 启动 Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'verl'
    static_configs:
      - targets: ['localhost:9090']  # verl metrics port
    metrics_path: '/metrics'
```

```bash
# Docker 方式
docker run -d --name prometheus \
  -p 9090:9090 \
  -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 2. 启动 Grafana

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

登录后添加 Prometheus 数据源，URL 填 `http://host.docker.internal:9090`。

### 3. verl 侧开启 metrics

在训练脚本里通过配置开启：

```python
# 在 verl 配置中启用 prometheus exporter
trainer_config = {
    "metrics": {
        "enable_prometheus": True,
        "prometheus_port": 9090,
    }
}
```

## 关键指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `reward_mean` | 平均 reward | 核心指标，应该稳步上升 |
| `reward_std` | reward 标准差 | 太大说明不稳定，考虑降 LR |
| `kl_divergence` | 策略与参考模型的 KL 散度 | 过大 = reward hacking |
| `rollout_latency_seconds` | rollout 单步耗时 | 波动大可能是 OOM 或网络问题 |
| `gpu_utilization` | GPU 利用率 | 低于 70% 说明有瓶颈 |
| `critic_loss` | Critic 损失 | 应该平稳下降 |
| `entropy` | 策略熵 | 过低 = 模式坍塌 |

## 实用告警规则

```yaml
# Grafana Alert: reward 停滞
- alert: RewardStagnation
  expr: increase(reward_mean[30m]) < 0.01
  for: 1h
  annotations:
    summary: "Reward 已停滞超过 1 小时"

# Grafana Alert: KL 散度爆炸
- alert: KLExplosion
  expr: kl_divergence > 15
  for: 10m
  annotations:
    summary: "KL 散度过大，可能存在 reward hacking"
```

## 调试经验

1. **reward 突然掉崖**：大概率是 Critic 学崩了。看 critic_loss 是否发散，是的话降低 Critic LR 或加 gradient clipping
2. **GPU 利用率周期性掉底**：通常是在做 rollout generation，这是正常的 — 因为 generation 阶段是自回归，计算密度低
3. **某个 rank 延迟特别高**：检查该节点的网络和磁盘 I/O，NCCL 通信可能在等它
4. **entropy 快速降到 0**：模型坍塌到固定输出，需要增大 entropy bonus 或降低 KL penalty

## 我的看板布局建议

```
Row 1: reward_mean (折线) | kl_divergence (折线) | entropy (折线)
Row 2: critic_loss (折线) | actor_loss (折线)
Row 3: GPU utilization (per rank, 堆叠) | rollout_latency (热力图)
Row 4: throughput (tokens/sec) | memory usage (per rank)
```

把 reward 和 KL 放第一行，因为这两个是你最常看的。GPU 和延迟放第三行做性能诊断。

## 相关

- [[AI/LLM/Frameworks/verl/verl 概述|verl 概述]]
- [[AI/LLM/Frameworks/verl/性能调优|性能调优]]
- [[AI/LLM/Frameworks/verl/verl 训练参数|verl 训练参数]]
- [[AI/LLM/Frameworks/verl/硬件资源预估|硬件资源预估]]
- [[AI/LLM/Frameworks/verl/Reward Function|Reward Function]]
