---
title: "Verifier/Reviewer Scorecard for Auto-Research (Quant & AI4Science)"
brief: "把“可验证、可自由探索”落地成评分卡：Verifier 负责硬门禁（可复现/可对账/无泄漏），Reviewer 负责软质量（新颖性/严谨性/稳健性/可迁移）。"
date: 2026-03-01
tags:
  - evaluation
  - ai4science
  - verifiable
  - research-factory
  - quant
status: draft
---

# Verifier/Reviewer Scorecard

## 1. 为什么需要 Scorecard
自动探索会天然产生海量“看似有效”的结果；没有统一评分基准线，系统会被过拟合与口径漂移淹没。

Scorecard = 机器可读的科研质量度量，支持：门禁、排序、看板聚合、经验生命周期（validated/deprecated）。

---

## 2. Verifier（硬门禁，pass/fail）
### 2.1 Reproducibility
- data_version 固化
- code_version 固化
- config_hash 固化
- seed/engine 固化

### 2.2 Auditability（可对账）
- 交易日志/持仓序列/费用明细齐全
- PnL 可重算一致（误差阈值）

### 2.3 Leakage / Lookahead
- 特征对齐检查
- label/target 泄漏扫描

### 2.4 Feasibility
- 成本/滑点/最小下单/杠杆约束满足
- 容量粗检（冲击成本/换手）

输出：verifier_pass: bool + failure_reasons[]

---

## 3. Reviewer（软评分 0-10）
建议维度：
- novelty：对 baseline 的增量（不接受“指标堆叠”）
- rigor：多重比较校正、walk-forward、置信区间
- robustness：regime 分段 + 成本扰动 + 参数扰动
- transfer：跨市场/跨周期/跨币池
- clarity：机制解释与失败模式清晰度

输出：dimension_scores + overall_score + short_rationale

---

## 4. Failure taxonomy（让失败可学习）
- leakage
- regime_overfit
- cost_sensitive
- capacity_limited
- unstable_pnl
- non_stationary_feature
- metric_gaming (Goodhart)

---

## 5. 与经验库（experience-hub）联动
- overall_score 超过入库线 → promote 为 tactical/strategic 经验
- 多次复现验证 → validate_experience
- 后续证伪/失效 → deprecate_experience
