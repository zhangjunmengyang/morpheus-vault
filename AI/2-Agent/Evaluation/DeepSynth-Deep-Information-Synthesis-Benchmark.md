---
title: "DeepSynth: 深度信息综合 Agent Benchmark"
brief: 120 道需跨源信息综合的 Agent benchmark，覆盖 67 国 7 领域，SOTA（o3-deep-research）LLM-Judge 仅 17.5%。核心发现：瓶颈是规划能力而非推理能力（提供步骤后性能+170%）。测量真实 Agent 能力边界的新维度：数据导航 + 跨源综合，不是多跳 QA。
arxiv: "2602.21143"
date: 2026-02-25
rating: ★★★★☆
affiliation: Huawei Noah's Ark Lab + Imperial College London + UCL + Cambridge
tags:
  - benchmark
  - agent
  - information-synthesis
  - evaluation
  - web-agent
  - real-world
  - agent-evaluation
sources:
  - "arXiv:2602.21143 — DeepSynth: Deep Information Synthesis Benchmark"
related:
  - "[[AI/2-Agent/Agentic-RL/Agent-RL-环境工程系统论]]"
  - "[[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization]]"
  - "[[AI/2-Agent/Evaluation/Aletheia-Gemini3-DeepThink-FirstProof]]"
  - "[[AI/2-Agent/Evaluation/Agent评估体系批判-Goodhart法则与Benchmark陷阱]]"
  - "[[AI/2-Agent/Agentic-RL/Search-R1-Reasoning-Search-Engine-RL]]"
---

# DeepSynth: 深度信息综合 Agent Benchmark

---

## TL;DR

120 道需要跨源信息综合的 Agent benchmark，覆盖 7 个领域、67 个国家。SOTA 模型（包括 GPT-5.2-Pro、o3-deep-research）的最高 LLM-Judge 分仅 17.5%，精确匹配率接近 0。核心难点不是推理能力不足，而是**规划能力和跨源信息导航能力**的系统性缺失。

---

## 动机：现有 Benchmark 的三个盲点

当前 Agent 评估 benchmark 的普遍问题：
1. **浅层事实检索**：问答可以通过单次搜索解决，不需要真正的信息综合
2. **单一来源**：只从 Wikipedia 等已知来源取信息，不测试真实的多源整合
3. **英语中心**：忽略了全球化的信息生态（不同语言、不同地区的官方数据源）

DeepSynth 的设计原则：答案**必须无法通过直接搜索或记忆检索获得**，必须通过多步推理和跨源综合才能得到。

---

## 任务设计

### 构建流程（人工专家驱动，4 阶段）

```
数据源识别 → 假设生成 → 假设验证（人工分析）→ 任务制定（双重标注）
```

- 16 名专家，每道题平均耗时 **5.5 小时**
- 初始收集 223 个数据源，经过过滤留 130 个，最终 120 道题
- 双重标注验证：两位标注者独立作答，答案一致才保留

### 任务特征

| 统计项 | 值 |
|--------|-----|
| 总任务数 | 120 |
| 平均问题长度 | 78.49 tokens |
| 平均中间步骤数 | 7.54 |
| 平均需访问网页数 | 4.2 |
| 覆盖国家数 | 67 |
| 覆盖领域数 | 7（社会经济/金融/环境/科学/教育/交通/政治） |

### 综合操作类型分布

| 操作类型 | 比例 |
|---------|------|
| Counting & Comparing | 33.72% |
| Trend Detection | 20.93% |
| Ranking | 19.77% |
| Average | 11.05% |
| Correlation | 6.98% |
| Anomaly Detection | 6.98% |
| Filtering | 0.58% |

### 任务示例

> "哪些非东盟国家在 2023 年实现了显著的后 COVID 恢复——赴新加坡游客达到 2019 年水平的 95% 以上——其主要旅行目的（商务 or 旅游）是什么？"

这道题需要：（1）识别东盟成员国；（2）从多个来源提取旅客数据；（3）筛选非东盟国家；（4）计算恢复率；（5）分析旅行目的分布。**不可能通过单次搜索解决**。

---

## 评估结果

### 主榜成绩（Pass@1）

| 模型 | F1 | EM | LLM-Judge |
|-----|-----|-----|-----------|
| **o3-deep-research** | **8.97** | 2.50 | **17.5%** |
| GPT-5.2-Pro | 8.70 | **6.25** | 6.67% |
| Gemini-Pro-2.5 | 6.25 | 0.0 | 5.0% |
| Smolagent (GPT-5) | 6.42 | 1.67 | 2.5% |
| GPT-5.1 | 3.83 | 0.0 | 0.0% |
| OWL (GPT-4.1) | 5.41 | 1.67 | 12.5% |
| o4-mini | 3.05 | 0.0 | 0.0% |
| DeepSeek-R1-Reasoner | 2.80 | 2.50 | 6.67% |

**关键数字**：最好的 agent（o3-deep-research）LLM-Judge 只有 17.5%，严格精确匹配只有 2.5%。换言之，**97.5% 的任务没有一个模型能完整正确完成**。

### Best-of-N 实验（DeepSynth-Dev 子集）

- Smolagents + GPT-4.1 在 Best@5 下达到 25.0% LLM-Judge
- 但 Self-Consistency@5 只有 5%（多次运行答案不一致，consistency score = 0.27）
- **结论**：偶尔能成功，但不可靠。高方差 = 运气而非能力。

---

## 关键分析发现

### RQ1：性能随中间步骤数上升而下降
所有模型随任务复杂度（中间步骤数）增加，性能单调下降。7.54 步的平均复杂度已超出当前 SOTA 的稳定处理范围。

### RQ2：提供规划步骤大幅提升性能
给模型提供 ground truth 中间步骤（不给答案）后：
- GPT-4.1：F1 从 3.46 → **9.36**（+170%）
- Smolagents：F1 从 3.75 → **10.50**（+180%）

**这说明核心瓶颈是规划能力，不是推理能力**。模型知道怎么做每一步，但不知道应该做哪些步骤、顺序是什么。

### RQ3：合成操作的差异化难度
- 异常检测（Anomaly Detection）：o3 表现最好（26.51%），有结构化模式识别能力
- 过滤操作（Filtering）：所有模型 0 分
- 趋势检测和排名：Gemini-2.5-Pro 和 o3 相对较好

### RQ4：错误类型分析（OWL + GPT-4.1 的 32 道题）
| 错误类型 | 出现次数 |
|---------|---------|
| 合成错误（找到信息但推理出错） | 16 |
| 导航错误（找不到正确数据源） | 15 |
| 无答案 | 4 |
| 技术问题 | 4 |

**两大主要错误**：导航失败（找不到正确页面/数据源）和综合推理失败（拿到数据但算错）。各占约 50%，说明两个能力层都有显著缺陷。

### RQ5：地理偏差
- **非洲相关任务：所有模型 F1 = 0.0**
- 欧洲/亚洲任务表现相对较好（数据覆盖多）
- 地理偏差 = 训练数据分布偏差的直接映射

---

## 对 Agent 评估体系的深层洞察

### DeepSynth 填补的空白

与已有 benchmark 对比：

| Benchmark | 信息来源 | 多源综合 | 全球覆盖 | 防记忆污染 |
|-----------|---------|---------|---------|----------|
| GAIA | 多样 | 部分 | 有限 | 部分 |
| WebArena | 网页交互 | ✗ | ✗ | ✗ |
| SWE-bench | 代码库 | ✗ | ✗ | ✓ |
| HotpotQA | 多段落 | 部分 | ✗ | ✗ |
| **DeepSynth** | **官方多源** | **✓** | **✓ (67国)** | **✓** |

### 重要的方法论启示

**"信息综合"不等于"多跳问答"**：

多跳问答（HotpotQA/2WikiQA）的答案通常通过两次搜索即可获得。DeepSynth 要求：
1. 识别相关数据源（官方统计机构、政府报告）
2. 处理非结构化表格数据（PDF/Excel）
3. 执行统计计算（趋势分析、相关性、排名）
4. 跨多个中间步骤保持上下文

这更接近**数据分析师/研究员的工作方式**，而非传统 QA 的检索-生成。

### 与 Benchmark Goodhart's Law 的关系

DeepSynth 在设计上刻意防止 Goodhart's Law：
- 答案不可直接搜索（必须计算/推导）
- 时间稳定（官方数据不频繁变化）
- 双重验证确保 gold standard 质量

但注意：**规模只有 120 题，统计显著性有限**。17.5% vs 12.5% 的差异（o3-deep-research vs OWL）在 120 题下可能不稳定。未来需要扩大到 500+ 题才有可靠排名。

---

## 对 Agent RL 训练方向的影响

DeepSynth 揭示的核心能力缺口：

1. **规划能力**：给定目标，分解成正确的执行子步骤（提供步骤后性能 +170%）
2. **数据导航能力**：找到正确的官方数据源（15/32 任务的导航错误）
3. **表格/数值推理**：对真实世界数据执行统计计算
4. **地理/语言偏差校正**：非洲相关任务全部失败

对 Agent RL 训练的启示：
- **Reward 设计**：DeepSynth 风格的任务可作为 verifiable reward 的优质来源（JSON 格式输出，可自动验证）
- **训练任务**：多步数据分析任务可能是提升 Agent 规划能力的关键数据类型
- **CSO / Credit Assignment**：导航错误和综合错误是两种不同的失败模式，需要不同的 credit 信号

---

## 批判性评价

**优秀的地方**：
- 真正测量了 Agent 的信息综合能力（不是单跳 QA 的变体）
- 67 国覆盖是目前 benchmark 中地理多样性最高的
- 双重标注 + 人工验证确保质量

**值得质疑的地方**：
- **规模偏小（120 题）**：统计显著性不足，排名稳定性差
- **任务构建成本极高**（5.5h/题 × 16 专家），扩展困难
- **领域覆盖不均**：Filtering 操作只有 0.58%，统计不可靠
- **非洲任务全部失败**：可能是数据源本身难以访问（防火墙/认证），不一定是模型能力问题

---

## See Also

- [[AI/2-Agent/Agentic-RL/Agent-RL-环境工程系统论]] — 真实世界任务的 Reward 设计挑战；DeepSynth 类任务的 JSON 验证器可作为 Reward 信号来源
- [[AI/2-Agent/Agentic-RL/CSO-Verified-Critical-Step-Optimization]] — 反事实 credit assignment：导航错误（找不到正确数据源）和综合错误（拿到数据但推理出错）是两种不同的失败模式，对应 CSO 的不同 credit 维度
- [[AI/2-Agent/Evaluation/Aletheia-Gemini3-DeepThink-FirstProof]] — 研究级任务评估对比：数学（形式可验证，Lean 4）vs 信息综合（JSON 可验证）；两种验证器的工程差异
- [[AI/2-Agent/Evaluation/Agent评估体系批判-Goodhart法则与Benchmark陷阱]] — DeepSynth 在设计上对抗 Goodhart's Law（不可直接搜索 + 双重标注），但规模偏小是统计弱点
- [[AI/2-Agent/Agentic-RL/Search-R1-Reasoning-Search-Engine-RL]] — 搜索 RL 训练方法；DeepSynth 的多源导航任务可作为更难的 Search-RL 训练数据

## 推荐阅读

1. **arXiv:2602.21143** — DeepSynth 原文，重点看 RQ2（提供步骤+170%）和 RQ4（导航/综合错误各占 50%）
2. **GAIA benchmark**（arXiv:2311.12983）— 对比：GAIA vs DeepSynth 在多源综合深度上的差异
3. **[[AI/2-Agent/Agentic-RL/Agentic-RL-2026前沿综合分析]]** — Reward 设计挑战章节（DeepSynth 提供了新的真实 verifiable reward 来源）
4. **[[AI/3-LLM/Evaluation/LLM 评测体系]]** — Benchmark 生态全景，DeepSynth 填补信息综合维度缺口
