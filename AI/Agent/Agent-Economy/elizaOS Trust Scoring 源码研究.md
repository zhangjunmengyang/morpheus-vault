---
title: "elizaOS Trust Scoring 源码研究"
type: research
domain: ai/agent/agent-economy
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/agent/agent-economy
  - elizaOS
  - trust-scoring
  - reputation
  - 量化
  - status/active
source: "elizaOS/eliza 0.x branch, v0.1.7-alpha.2"
---

# elizaOS Trust Scoring 源码研究

> elizaOS 的去中心化信誉评分系统源码分析。承接 [[ai16z 竞品分析]] 中的"下一步"，为量化 Agent 策略框架移植信任评分机制提供技术依据。

## 代码版本

- **v0.1.7-alpha.2** (0.x branch)
- Trust Scoring 在 v1.x 重构中被移出核心，移至外部插件
- 主要源码：`@elizaos/plugin-trustdb`（数据库层）、`@elizaos/plugin-solana`（评分逻辑）、`@elizaos-plugins/plugin-rabbi-trader`（简化版）
- GitHub: https://github.com/elizaOS/eliza (branch `0.x`)

## 架构总览

三层架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading Actions Layer                       │
│  TAKE_ORDER / executeSwap / EXTRACT_RECOMMENDATIONS             │
│  (conviction: none/low/medium/high → 买入金额 0/1%/5%/10% 流动性) │
├─────────────────────────────────────────────────────────────────┤
│                    TrustScoreManager                            │
│  generateTrustScore() / updateRecommenderMetrics()              │
│  calculateTrustScore() / calculateRiskScore()                   │
│  SimulationSellingService (RabbitMQ 消息驱动卖出)                │
├─────────────────────────────────────────────────────────────────┤
│                    TrustScoreDatabase (SQLite)                  │
│  recommenders / recommender_metrics / token_performance         │
│  token_recommendations / trade / simulation_trade / transactions│
└─────────────────────────────────────────────────────────────────┘
```

## 数据结构

### Recommender（推荐人）

多渠道身份绑定——同一推荐人可通过 Telegram/Discord/Twitter/钱包地址识别。与 [[ERC-8004 Trustless Agents]] 的 Identity Registry 设计异曲同工，但 elizaOS 的实现是中心化 SQLite，非链上注册表。

### RecommenderMetrics（推荐人指标）

核心字段：`trustScore`（综合信任分）、`successfulRecs`（成功推荐数）、`avgTokenPerformance`（推荐 token 平均表现）、`riskScore`（风险分）、`consistencyScore`（一致性分）、`virtualConfidence`（虚拟信心 = 持仓余额/1M）、`trustDecay`（衰减后分数）。

### TokenPerformance（Token 表现）

包含链上指标（liquidity, holderChange, rugPull, isScam）和衍生指标（sustainedGrowth, rapidDump, suspiciousVolume, validationTrust）。

## 算法详解

### Trust Score：设计 vs 实现的差距

**文档描述的 5 因子加权模型：**

```
trustScore = (successRate × 0.3 + normalizedPerformance × 0.2 
             + consistencyScore × 0.2 + (1 - riskScore) × 0.15
             + timeDecayFactor × 0.15) × 100
```

**实际源码实现：**

```typescript
calculateTrustScore(tokenPerformance, recommenderMetrics) {
    const riskScore = this.calculateRiskScore(tokenPerformance);
    const consistencyScore = this.calculateConsistencyScore(
        tokenPerformance, recommenderMetrics
    );
    return (riskScore + consistencyScore) / 2;  // 简单平均！
}
```

⚠️ **关键发现**：实际实现远比文档简单。且 `consistencyScore = |priceChange24h - avgPerformance|` 越大分越高——语义上应表示**不一致**，疑似 bug。

### Risk Score

加法计分：`rugPull +10, isScam +10, rapidDump +5, suspiciousVolume +5`。范围 0-30。

### 时间衰减

```
DECAY_RATE = 0.95 (每天衰减 5%)
MAX_DECAY_DAYS = 30
30天不活跃 → 0.95^30 ≈ 0.215，分数降到原来的 21.5%
```

### Validation Trust

所有推荐过该 token 的推荐人的平均信任分。类似 [[Virtuals Protocol]] ACP 中的 Agent 信誉聚合思路。

### Virtual Confidence (Skin in the Game)

`virtualConfidence = recommenderBalance / 1,000,000`。推荐人自持仓位量化信心程度——好的反垃圾推荐设计。

## 交易决策链路

### Conviction-Based Position Sizing

| 信念级别 | 占流动性比例 |
|---------|------------|
| LOW     | 1%         |
| MEDIUM  | 5%         |
| HIGH    | 10%        |

### Rabbi-Trader 安全限制

关键参数：`MIN_TRUST_SCORE: 0.4`、`STOP_LOSS: 20%`、`TAKE_PROFIT: 12%`、`MAX_ACTIVE_POSITIONS: 5`、`MIN_LIQUIDITY: $1,000`。

### Rabbi-Trader 简化版 Trust Score

纯基于 token 市场数据的三因子模型：Liquidity (40%) + Volume (40%) + MarketCap (20%)。不涉及推荐人历史，适合冷启动。

## 信誉积累与降低

**积累**：成功推荐（非 rug pull）+1、推荐 token 表现好提升平均值、持续活跃避免衰减、自持仓位提升信心、推荐被多人验证提升 validationTrust。

**降低**：推荐 rug pull/骗局 → riskScore +10、推荐暴跌 → +5、不活跃 → 每天衰减 5%。

## 模拟交易系统

新推荐默认走模拟交易 (`is_simulation=true`)，通过 RabbitMQ 异步处理卖出决策，与 Sonar Backend 集成监控。模拟结果影响推荐人 trust score。**模拟优先**是关键风控设计。

## 移植到 Python 量化框架

### 可行性：✅ 高

核心算法极简（加法/除法/指数衰减），SQLite 数据库 Python 原生支持，不依赖 Solana 链上操作。参考 [[Coinbase AgentKit 技术评估]] 中的 Python 栈，可直接集成。

### 改进建议

1. **采用文档版 5 因子加权模型**——实际实现过于简化
2. **贝叶斯更新**——用 Beta 分布建模成功率，替代简单计数
3. **归一化处理**——riskScore 和 consistencyScore 量纲不同，应归一化到 [0,1]
4. **共谋检测**——推荐人网络关系分析，防止协同操纵
5. **数据源适配**——替换 DexScreener/Birdeye 为 CoinGecko API / CEX API
6. **与 ERC-8004 集成**——链上信誉注册表替代中心化 SQLite（→ [[ERC-8004 Trustless Agents]]）

## 关键洞察

1. **设计 vs 实现差距大**——文档精心设计的 5 因子模型在代码中变成简单平均，consistencyScore 语义可能反转
2. **两层信任**——推荐人级别 (Recommender Trust) + Token 级别 (Validation Trust)
3. **Skin in the Game** 设计值得借鉴——virtualConfidence 要求推荐人自持仓位
4. **模拟优先**——新推荐默认纸面交易，验证后才可能实盘
5. **v1.x 中被降级**——从核心移至外部插件，说明社区对该模块的维护投入有限

## 参考源码路径

| 模块 | 包 |
|------|---|
| TrustScoreDatabase | `@elizaos/plugin-trustdb@0.1.7-alpha.2` |
| TrustScoreManager | `@elizaos/plugin-solana@0.1.7-alpha.2` |
| SimulationSellingService | `@elizaos/plugin-solana@0.1.7-alpha.2` |
| Rabbi-Trader Trust | `@elizaos-plugins/plugin-rabbi-trader` |
| Trust Engine 文档 | `docs/docs/advanced/trust-engine.md` |

## 相关

- [[ai16z 竞品分析]] — 本笔记的上游，elizaOS 全貌分析
- [[ERC-8004 Trustless Agents]] — 链上信誉标准，与 Trust Scoring 互补
- [[Coinbase AgentKit 技术评估]] — 我们的 Python 技术栈，移植目标
- [[Virtuals Protocol]] — Agent 信誉聚合的另一实现路径
- [[Agent 经济基础设施]] — 全景综述
- [[Agentic Spring]] — 市场趋势背景
