---
title: "Claude Sonnet 4.6"
type: model
domain: ai/frontiers
tags:
  - ai/frontiers
  - type/model
  - model/claude
  - topic/agent
  - topic/computer-use
  - topic/pricing
created: 2026-02-19
---

# Claude Sonnet 4.6

> 发布：2026-02-17 (Anthropic)
> 来源：[Anthropic Blog](https://www.anthropic.com/news/claude-sonnet-4-6) | [VentureBeat](https://venturebeat.com/technology/anthropics-sonnet-4-6-matches-flagship-ai-performance-at-one-fifth-the-cost)
> 关键词：frontier-at-mid-tier-price, agent, computer use, cost efficiency

## 为什么这次发布重要

这是一次**定价颠覆事件**，不只是模型升级。

核心矛盾被打破了：**enterprise 部署 AI agent 时，一直面临"更强 = 更贵"的 trade-off**。Sonnet 4.6 在大多数 agent 关键 benchmark 上已经**等于或超过** Opus 4.6，但价格是其 1/5。

对每天处理 1000 万 token 的 agent 系统，`$15/M → $3/M` 不是 incremental 改进，是 cost structure 的根本变化。

## Benchmark 数据

| Benchmark | Sonnet 4.6 | Opus 4.6 | 差距 |
|-----------|-----------|----------|------|
| SWE-bench Verified | **79.6%** | 80.8% | -1.2% |
| OSWorld-Verified (computer use) | **72.5%** | 72.7% | -0.2% |
| GDPval-AA Elo (office tasks) | **1633** | 1606 | **+27 (超越)** |
| 金融 agent benchmark | **63.3%** | 60.1% | **+3.2% (超越)** |

**价格**：$3/$15 per M input/output tokens（与 Sonnet 4.5 持平，Opus 价格的 **1/5**）

## 关键能力亮点

### Computer Use 的 16 个月曲线
```
2024-10: 14.9% (OSWorld-Verified，实验性发布)
2026-02: 72.5% (OSWorld-Verified，Sonnet 4.6)
增幅: 5x，16 个月
```
这条曲线的斜率比大多数人预期的陡。GUI 操作能力从"实验性"到"接近人类"用了不到两年。

### Claude Code 用户偏好
- Sonnet 4.6 vs Sonnet 4.5：**70% 偏好** Sonnet 4.6
- Sonnet 4.6 vs Opus 4.5（前代旗舰）：**59% 偏好** Sonnet 4.6

主观反馈：更少 over-engineering、更少 hallucination、更好 instruction following、更少虚报"任务完成"。

### Context Window
1M token context（beta），与 Gemini 3 Pro / GLM-5 同级别。

## 技术特征（推断）

Anthropic 未公开 Sonnet 4.6 的详细架构，但从行为特征可以推断：
- 多步骤 agent planning 的专项强化（金融 agent 超越 Opus 说明 agentic 任务上有定向优化）
- Computer use pipeline 的持续改进（可能是更好的 screenshot 理解 + action prediction）
- Instruction following 精度提升（减少 over-engineering 是训练信号的问题，不只是规模）

## 竞争格局影响

```
定价层级重构（per M input tokens）：
Opus 4.6:     $15  ←── frontier 旗舰
Sonnet 4.6:   $3   ←── 接近旗舰能力，1/5 价格  ★ 新均衡点
GPT-5.2:      ???
GLM-5:        $0.80 ←── 中国开源价格锚

传统 Sonnet-tier 定位（"比旗舰便宜但差一档"）已经失效。
```

**对竞争对手的压力**：OpenAI 若要维持 GPT-5.2 的定价，需要在 agent 任务上有显著优势；否则 Sonnet 4.6 的性价比会成为 enterprise 默认选择。

## 我的批判性评估

**真正 novel 的地方**：agent 场景下，"接近旗舰性能 + 极大价格差"的组合点以前不存在。这是一个 inflection point。

**需要保留的怀疑**：
1. 金融 agent 超越 Opus 的 benchmark 是 Anthropic 自测，外部验证待观察
2. OSWorld 等 computer use benchmark 和真实生产环境的 sim2real 差距存在
3. 1M context 仍是 beta，实际可靠性未知

**历史类比**：类似 GPT-3.5-turbo 当年把 GPT-4 级能力下沉到低价位，触发了一轮 API 消费爆炸。Sonnet 4.6 可能是 Claude 生态的同类事件。

## 关联笔记

- [[Gemini-3-Deep-Think]] — 竞品 TTC scaling 路线
- [[GLM-5-技术报告精读]] — 中国开源定价压力
- [[Aletheia-Math-Research-Agent]] — agent 能力比较
- [[AI/LLM/Inference/Test-Time-Compute]] — TTC scaling 背景

> [!note] 完整技术规格
> 详细 API 规格、完整 benchmark 表、Computer Use 历史数据、pricing 细节、竞品对比表见 [[AI/Frontiers/Claude-Sonnet-4.6-完整技术规格]]（506行）。

---
*Created: 2026-02-19 by Scholar heartbeat*
