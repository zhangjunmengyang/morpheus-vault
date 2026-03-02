---
title: "FARS-style Research Factory for Quant Platform Pro"
brief: "把 quant platform pro 从工具箱升级为可持续运行的科研工厂：Project workspace=共享文件系统协议，四阶段流水线（Ideation→Planning→Experiment→Writing），Verifier/Reviewer scorecard 门禁，Dashboard 全程可观测，允许负结果并沉淀为经验。"
date: 2026-03-01
tags:
  - ai4science
  - automated-research
  - multi-agent
  - research-factory
  - quant
  - platform-design
status: draft
related:
  - "[[AI/6-应用/FARS-Fully-Automated-Research-System]]"
  - "[[AI/6-应用/LLMOps]]"
---

# FARS-style Research Factory for Quant Platform Pro

## 0. One-liner
把“量化研究”当成科研：每个结论都必须由 **可复现的 RunSpec + 可审计的 Artifact + 可验证的 Scorecard** 支撑；探索可以自由，但**入库必须过门禁**。

---

## 1. 从 FARS 复制的不是 Agent，而是“工厂结构件”
FARS 的关键工程决策：**四阶段固定骨架 + 共享文件系统代替消息传递 + 自动化评分基准线 + 允许负结果 + 持续运行 + 全程可观测**。

我们要复刻的结构件：
1) 研究项目目录（workspace）= 文件协议（file-as-API）
2) 流水线状态机（Ideation/Planning/Experiment/Writing）
3) Verifier/Reviewer 的 scorecard（量化版 paperreview.ai）
4) Dashboard（吞吐/质量/失败/知识沉淀）
5) 跨项目记忆（避免 FARS 的“每个课题独立”）

---

## 2. Research Project Workspace：共享文件系统协议（file-as-API）
**目标**：不引入复杂消息框架，用目录结构就能编排多 agent 并行。

建议每个项目一个目录：
```
projects/<project_id>/
  hypothesis.md
  plan.md
  run_specs/
  artifacts/
  scorecards/
  report.md
  status.json
  links.json
```

### status.json（最小状态机）
- stage: ideation|planning|experiment|writing|done|blocked
- current_run_id
- last_update
- blockers[]

### links.json（证据链）
- hypothesis_id → plan_id → run_id → artifacts → scorecard → experience_ids

**关键原则**：任何“结论”必须能从 links 追溯到 run_specs + artifacts。

---

## 3. 四阶段流水线：Quant 版本语义对齐
### 3.1 Ideation（选题/假设）
输出：hypothesis.md（必须可证伪）
- statement（可检验陈述）
- metrics（目标指标）
- boundary（适用条件）
- falsifiers（反例条件）
- baseline（最小对照）

### 3.2 Planning（实验设计）
输出：plan.md + ExperimentSpec
- split（walk-forward/时间分段/跨市场）
- ablations（消融清单）
- robustness（成本扰动/参数扰动/regime 分段）
- acceptance_criteria（入库线）

### 3.3 Experiment（执行）
输出：Artifact + Scorecard
- 每次 run 强制写 RunSpec（data_version/code_version/config/seed）
- 产物必须包含“可对账工件”（交易日志/持仓序列/费用明细），否则 Verifier fail

### 3.4 Writing（报告）
输出：report.md（不是故事）
- delta vs baseline
- failure analysis（失败=知识）
- next experiments（下一轮计划）

---

## 4. Verifier/Reviewer：量化版“paperreview.ai”
### 4.1 Verifier（硬门禁，pass/fail）
- lookahead/leak 检测（未来函数、错位对齐）
- PnL 可重算一致性（基于交易日志/持仓）
- 成本/撮合/杠杆约束一致性
- 数据版本不可变性

### 4.2 Reviewer（软评分，形成质量基准线）
建议维度（0-10）：
- novelty：相对 baseline 的信息增量
- rigor：检验完整性（多重比较、walk-forward）
- robustness：regime/成本/参数敏感性
- transfer：可迁移性（跨市场/跨周期）
- clarity：可解释的机制与失败模式

输出：scorecard.json + 可读摘要（写入 report.md）。

---

## 5. Dashboard：科研工厂的“生产看板”
五块固定区域：
1) Throughput：projects/runs per day，平均耗时，排队/失败率
2) Quality：scorecard 分布 + 入库线对比
3) Exploration Map：方向/因子族/策略族覆盖与重复度
4) Failure Museum：top failure modes（泄漏/过拟合/容量/成本敏感…）
5) Knowledge Flywheel：validated/deprecated experiences 数量与引用排行

---

## 6. 关键升级点：跨项目学习（避免 FARS 的最大缺陷）
FARS 的缺陷之一是“每个课题独立、不从历史中学习”。

Quant 平台应该把每次实验的失败模式、有效条件、参数敏感性沉淀到 experience-hub，并在 Ideation/Planning 阶段自动召回：
- 先 query_experiences → 再生成 plan
- 失败原因标准化（taxonomy）→ 形成 curriculum（下一轮搜索空间约束）

---

## 7. 落地到 quant-platform-pro：最小闭环（两周）
P0：ResearchProject workspace + RunSpec/Artifact/Scorecard schema（强制落库）
P1：Verifier v1（泄漏 + 可对账）+ baseline/ablation 自动派生
P2：Reviewer + Dashboard + experience flywheel 自动触发

---

## 8. Open questions（值得当成研究课题）
- 如何定义“quant 的可验证 reward”，避免指标 Goodhart？
- 失败模式 taxonomy 如何设计，才能指导下一轮探索（课程学习）？
- 在不训练的情况下，如何做跨项目 self-evolution（in-context policy update）？
