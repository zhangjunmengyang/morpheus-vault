---
tags:
  - claude
  - anthropic
  - sonnet
  - llm-release
date: 2026-02-17
model_name: Claude Sonnet 4.6
model_id: claude-sonnet-4-6
knowledge_cutoff: 2025-08
context_window: 1000000
pricing_input: "$3/M tokens"
pricing_output: "$15/M tokens"
---

> [!note] 关联版本
> 本文为完整技术规格版（503行）。精华分析版见 [[AI/Frontiers/Claude-Sonnet-4.6]]（102行，含竞争格局评估）。

# Claude Sonnet 4.6

## 概述

2026 年 2 月 17 日，Anthropic 发布 **Claude Sonnet 4.6**——仅在 Opus 4.6 发布 12 天之后。这是 Sonnet 产品线迄今最强的版本，被 Anthropic 称为"our most capable Sonnet model yet"。

核心定位：**以中端价格交付接近旗舰级智能**。它在 coding、computer use、long-context reasoning、agent planning、knowledge work、design 六个维度全面升级，1M token context window（beta），且价格维持 Sonnet 4.5 水平——$3/$15 per million tokens。

关键数字速览：

| 指标 | 数值 |
|------|------|
| SWE-bench Verified | 79.6% |
| OSWorld-Verified | 72.5% |
| ARC-AGI-2 | 60.4% |
| MATH | 88.0% |
| GDPval-AA Elo（office tasks） | 1633（超过 Opus 4.6 的 1606） |
| 用户偏好 vs Sonnet 4.5 | 70% |
| 用户偏好 vs Opus 4.5 | 59% |
| 价格 | $3 input / $15 output per M tokens |
| Context Window | 1M tokens（beta） |
| Knowledge Cutoff | August 2025 |

发布即成为 Free 和 Pro 用户的**默认模型**，同时在 Claude API、Claude Code、Claude Cowork、Amazon Bedrock、Google Vertex AI、Azure AI Foundry 全线可用。

---

## 定位：中端 "Workhorse" 模型

Anthropic 的模型家族沿用三层架构：

- **Opus**：旗舰，最深度推理，最贵
- **Sonnet**：中端主力，性价比最优
- **Haiku**：轻量快速，低延迟场景

Sonnet 4.6 的战略意义在于**模糊了 Sonnet 与 Opus 的界限**。FinancialContent 的分析文章直接称其为"The Workhorse AI Model That Outpaces Flagships"——这不是夸张，在多个 benchmark 上 Sonnet 4.6 确实超越了上一代旗舰 Opus 4.5（2025 年 11 月发布），甚至在 office tasks 基准 GDPval-AA 上超过了当代旗舰 Opus 4.6。

从市场叙事看，Anthropic 正在执行一个"**tier collapse**"策略：每一代 Sonnet 都在逼近上一代 Opus 的能力，而价格保持不变。这意味着：

1. **对大多数生产负载，Sonnet 已经足够好**——不再需要为 Opus 买单
2. **Opus 被推向更极端的使用场景**——深度代码重构、多 agent 协调、需要绝对精确的任务
3. **中端价位的能力天花板在快速抬升**——对下游 SaaS 和传统软件构成巨大压力

CNBC 报道指出，Sonnet 4.6 的发布加速了软件股的抛售，iShares Expanded Tech-Software Sector ETF (IGV) 年初至今已跌超 20%。Salesforce (-2.7%)、Oracle (-3.4%)、Intuit (-5.2%)、Adobe (-1.4%) 都在发布当日应声下跌。

---

## 关键能力与 Benchmark

### Coding

Sonnet 4.6 在 **SWE-bench Verified** 上拿到 **79.6%**，这是衡量 real-world software engineering 任务完成率的标准 benchmark。这个分数：

- 逼近 GPT-5.2 的 80.0%
- 显著超过 Sonnet 4.5
- 接近 Opus 4.6 水平

但 benchmark 只是一面。来自 Claude Code 的**实际用户反馈**更能说明问题：

- **70% 的用户偏好 Sonnet 4.6** 而非 Sonnet 4.5
- 用户报告它**更有效地阅读上下文再修改代码**，而不是盲目改写
- **合并共享逻辑**而非重复它——这是之前所有 Claude 模型的痛点
- **显著减少 overengineering 和 "laziness"**——这两个词是 Sonnet 4.5 时期开发者最常见的吐槽
- **更少的虚假成功声明**，更少幻觉，multi-step task 的 follow-through 更一致

Rakuten AI 的评测称 Sonnet 4.6 "produced the best iOS code we've tested"，在 spec compliance、architecture、modern tooling 使用上都超出预期。Replit 总裁 Michele Catasta 评价其"performance-to-cost ratio is extraordinary"。

在 bug detection 方面，Sonnet 4.6 "meaningfully closed the gap with Opus"，使团队可以并行运行更多 reviewer，捕获更多种类的 bug，且不增加成本。

### Computer Use

这是 Sonnet 4.6 最引人注目的进步。Anthropic 在 2024 年 10 月首次推出 computer use 时承认它"still experimental—at times cumbersome and error-prone"。16 个月后，Sonnet 4.6 在这个方向上取得了质的飞跃。

**OSWorld-Verified** 成绩：

| 模型 | OSWorld-Verified |
|------|------------------|
| Claude 3.5 Sonnet（2024.10） | 14.9% |
| Claude Sonnet 4.5（2025.09） | 61.4% |
| Claude Opus 4.5（2025.11） | 66.3% |
| **Claude Sonnet 4.6**（2026.02） | **72.5%** |
| Claude Opus 4.6（2026.02） | 72.7% |

注意：Sonnet 4.6 与 Opus 4.6 在 OSWorld 上仅差 0.2%。从 3.5 Sonnet 的 14.9% 到 4.6 的 72.5%，16 个月提升了近 **5 倍**。

实际表现上，早期用户报告 Sonnet 4.6 在以下任务中达到了 **human-level capability**：

- 导航复杂的 spreadsheet
- 填写多步骤 web form
- 跨多个浏览器 tab 协调工作
- Browser-based testing 和 scraping

一家保险公司在其 computer use benchmark 上达到了 **94% 的准确率**，称其为"mission-critical to workflows like submission intake and first notice of loss"。

Computer use 的工作方式是**模拟人类操作**——点击虚拟鼠标、敲虚拟键盘，不依赖 API 或专用连接器。这意味着它可以操作任何有 GUI 的遗留系统，这在企业环境中价值巨大。

### Long-Context Reasoning（1M Token Context Window）

Sonnet 4.6 的 1M token context window（beta）足以容纳：

- 完整的大型代码库
- 长篇合同文档
- 数十篇研究论文

更重要的是，模型能**有效地在整个上下文中推理**，而不只是检索。这在 **Vending-Bench Arena** 中表现得尤为明显——这是一个模拟经营类 benchmark，让 AI 模型相互竞争经营虚拟企业：

Sonnet 4.6 发展出了一个**出人意料的策略**：前 10 个模拟月大举投资产能（支出远超竞争对手），然后在最后阶段急转向盈利。这个战略转换的时机帮助它远远领先竞争对手——展示了真正的 long-horizon planning 能力。

### Enterprise Document Comprehension

在 **OfficeQA** 上（衡量模型读取企业文档——图表、PDF、表格——提取事实并推理的能力），Sonnet 4.6 **匹配了 Opus 4.6 的表现**。

Box 的评测发现 Sonnet 4.6 在 heavy reasoning Q&A 上比 Sonnet 4.5 **高出 15 个百分点**。金融服务客户报告在其 Financial Services Benchmark 上看到了"significant jump in answer match rate"。

Databricks 的 Neural Networks CTO Hanling Tang 评价："It's a meaningful upgrade for document comprehension workloads."

### Frontend Development 与 Design

多位早期客户**独立地**描述 Sonnet 4.6 的视觉输出"notably more polished"——更好的布局、动画和设计感。达到生产质量所需的迭代轮次更少。

有客户直接说："Sonnet 4.6 has perfect design taste when building frontend pages and data reports, and it requires far less hand-holding to get there than anything we've tested before."

### ARC-AGI-2

TechCrunch 特别提到 Sonnet 4.6 在 **ARC-AGI-2** 上拿到 **60.4%**——这个 benchmark 旨在衡量"skills specific to human intelligence"。这个分数超过了大多数同级模型，但仍落后于 Opus 4.6、Gemini 3 Deep Think 和 GPT-5.2 的某些优化版本。

---

## 与 Sonnet 4.5 对比

Sonnet 4.6 是 Sonnet 4.5（2025 年 9 月发布）以来首次重大升级，间隔约 5 个月。

| 维度 | Sonnet 4.5 | Sonnet 4.6 | 变化 |
|------|-----------|-----------|------|
| Context Window | 200K → 500K | 1M（beta） | 2x+ |
| SWE-bench Verified | 较低 | 79.6% | 显著提升 |
| OSWorld-Verified | 61.4% | 72.5% | +11.1pp |
| GDPval-AA Elo | 较低 | 1633 | 大幅提升 |
| 用户偏好（Claude Code） | baseline | 70% prefer 4.6 | — |
| Enterprise Q&A（Box） | baseline | +15pp | 重大提升 |
| Prompt injection 抵抗 | 较弱 | 显著改善，接近 Opus 4.6 | 重大提升 |
| Overengineering 问题 | 严重 | 显著减少 | — |
| "Laziness" 问题 | 常见 | 大幅改善 | — |
| 价格 | $3/$15 per M | $3/$15 per M | 不变 |
| 默认模型 | 否 | 是（Free/Pro） | — |
| Knowledge Cutoff | July 2025 | August 2025 | +1 月 |

核心改进总结：

1. **Coding 质量飞跃**——不只是 benchmark 分数，实际使用中的一致性、指令遵循、减少幻觉
2. **Computer use 从实验走向生产**——OSWorld 从 61.4% 到 72.5%
3. **Context window 翻倍+**——从 500K 到 1M
4. **Document comprehension 达到 Opus 水平**——OfficeQA 匹配 Opus 4.6
5. **安全性提升**——prompt injection 抵抗力大幅增强
6. **设计感知能力**——前端输出质量独立被多客户认可

---

## 与 Opus 4.6 对比

Opus 4.6 于 2026 年 2 月 5 日发布，比 Sonnet 4.6 早 12 天。两者是同代模型，构成当前 Claude 家族的"旗舰+主力"组合。

| 维度 | Sonnet 4.6 | Opus 4.6 | 谁赢？ |
|------|-----------|---------|--------|
| SWE-bench Verified | 79.6% | 更高 | Opus |
| OSWorld-Verified | 72.5% | 72.7% | 基本持平 |
| GDPval-AA Elo（office tasks） | 1633 | 1606 | **Sonnet** |
| Finance Agent v1.1 | 更高 | 较低 | **Sonnet** |
| OfficeQA | 匹配 | 匹配 | 持平 |
| ARC-AGI-2 | 60.4% | 更高 | Opus |
| Prompt Injection Resistance | 接近 | 略好 | 基本持平 |
| 深度代码重构 | 可用 | 明显更强 | Opus |
| Multi-agent 协调 | 可用 | 明显更强 | Opus |
| Context Window | 1M（beta） | 1M | 持平 |
| Knowledge Cutoff | August 2025 | August 2025 | 持平 |
| 价格 (input) | $3/M | $5/M | Sonnet 便宜 40% |
| 价格 (output) | $15/M | $25/M | Sonnet 便宜 40% |
| Adaptive Thinking | ✅ | ✅ | 持平 |
| Extended Thinking | ✅ | ✅ | 持平 |
| Context Compaction | ✅（beta） | ✅ | 持平 |

**关键观察**：

1. **OSWorld 几乎打平**——0.2% 的差距在统计上可以忽略
2. **Office tasks Sonnet 反超**——GDPval-AA 和 Finance Agent 上 Sonnet 竟然赢了
3. **OfficeQA 完全持平**——企业文档理解能力一致
4. **价格差 40%**——Opus 的 input 贵 67%，output 贵 67%
5. **Opus 的护城河在"最深度推理"**——代码重构、multi-agent workflow、需要绝对精确的场景

Anthropic 自己的官方建议是："We find that Opus 4.6 remains the strongest option for tasks that demand the deepest reasoning, such as codebase refactoring, coordinating multiple agents in a workflow, and problems where getting it just right is paramount."

IT Pro 计算了成本差异：Opus 4.6 定价 $5/$25 per M tokens，是 Sonnet 的 **1.67x**。对于大多数不需要"最深度推理"的生产负载，Sonnet 4.6 提供了几乎等价的能力，成本却低 40%。

---

## 价格与经济性

### API 定价

| 模型 | Input | Output | Batch Input | Batch Output |
|------|-------|--------|-------------|-------------|
| Sonnet 4.6 | $3/M | $15/M | 预计 $1.5/M | 预计 $7.5/M |
| Opus 4.6 | $5/M | $25/M | — | — |
| GPT-5.2 | 更高 | 更高 | — | — |

### 成本分析

假设一个典型的 agentic coding session：
- Input: ~50K tokens（代码库 + 指令 + 上下文）
- Output: ~10K tokens（代码 + 解释）

**单次调用成本**：
- Sonnet 4.6: $0.15 + $0.15 = **$0.30**
- Opus 4.6: $0.25 + $0.25 = **$0.50**

每次调用节省 40%。如果每天跑 100 次 agentic task，月节省 **$600**。规模化后差异更大。

### 免费层升级

Sonnet 4.6 发布同时，Anthropic 升级了免费层——Free 用户现在获得：
- Sonnet 4.6 作为默认模型
- File creation 能力
- Connectors
- Skills
- Context compaction

这是一个激进的市场策略：让免费用户也能体验到接近旗舰级的能力，从而拉动付费转化和 API 使用。

---

## Agentic 能力

Sonnet 4.6 在 agentic AI 方面的进步是此次发布的核心叙事之一。

### Adaptive Thinking Engine

Sonnet 4.6 引入了 **Adaptive Thinking**——一种动态推理模式，允许模型：

- 根据任务复杂度调整推理努力级别（Low / Medium / High / Max）
- 在内部进行"暂停"式的自我反思和逻辑自纠正
- 替代静态 prompting，实现 real-time recursive reasoning

Anthropic 建议开发者"explore across the thinking effort spectrum to find the ideal balance of speed and reliable performance"——Sonnet 4.6 在任何 thinking effort 级别都表现强劲，甚至在 extended thinking 关闭时也是如此。

### Context Compaction

**Context Compaction**（beta）是解决长 agent session 的关键功能：当对话接近 context limit 时，自动总结旧的上下文，有效延长可用 context 长度。这让 agent 可以进行理论上"无限"的长时间 session——对 agentic coding 和 autonomous workflow 至关重要。

### Multi-step Task Execution

相比 Sonnet 4.5，4.6 在以下 agentic 场景中表现显著改善：

- **Contract routing** 和 **conditional template selection**
- **CRM coordination**——需要 strong model sense 和 reliability 的多步任务
- **Autonomous browser workflows**——跨 tab 协调、form 填写、数据抓取
- **Agentic financial analysis**——在 Finance Agent v1.1 上甚至超过了 Opus 4.6

### Web Search + Code Execution

Claude 的 web search 和 fetch 工具现在会**自动编写和执行代码来过滤和处理搜索结果**，只保留相关内容在 context 中。这同时提升了响应质量和 token 效率。

此外以下能力现已 GA（Generally Available）：
- Code execution
- Memory
- Programmatic tool calling
- Tool search
- Tool use examples

### Claude Code 集成

Sonnet 4.6 在 Claude Code 中的表现尤其亮眼。Wikipedia 记载 Claude Code "was widely considered the best AI coding assistant, when paired with Opus 4.5"，现在 Sonnet 4.6 在用户偏好测试中甚至超过了 Opus 4.5（59% 偏好率）。

这意味着对于大多数 Claude Code 用户来说，Sonnet 4.6 + Claude Code 的组合已经足够——不再需要 Opus 级别的模型来获得顶级编码体验。

---

## API / 平台可用性

### 发布渠道

Sonnet 4.6 于 2026 年 2 月 17 日在以下平台**同步上线**：

| 平台 | 状态 |
|------|------|
| claude.ai（Free/Pro/Max/Team/Enterprise） | ✅ 默认模型 |
| Claude Cowork | ✅ 默认模型 |
| Claude Code | ✅ |
| Claude API | ✅ model id: `claude-sonnet-4-6` |
| Amazon Bedrock | ✅ |
| Google Vertex AI | ✅ |
| Azure AI Foundry | ✅ |

### API Features

| Feature | 状态 |
|---------|------|
| Adaptive Thinking | ✅ |
| Extended Thinking | ✅ |
| Context Compaction | ✅（beta） |
| 1M Context Window | ✅（beta） |
| Web Search Tool | ✅（带动态过滤） |
| Web Fetch Tool | ✅（带代码执行过滤） |
| Code Execution | ✅ GA |
| Memory | ✅ GA |
| Programmatic Tool Calling | ✅ GA |
| Tool Search | ✅ GA |
| Tool Use Examples | ✅ GA |
| MCP Connectors（Excel） | ✅ Pro/Max/Team/Enterprise |

### Claude in Excel 新能力

值得特别一提的是 **Claude in Excel** 的 MCP connectors 升级：现在可以在 Excel 中直接连接外部工具——S&P Global、LSEG、Daloopa、PitchBook、Moody's、FactSet。在 claude.ai 中配置的 MCP connectors 会自动同步到 Excel。

这对金融和研究工作流的影响巨大——不离开 Excel 就能调用外部数据源和 AI 推理能力。

---

## 安全评估

Anthropic 对 Sonnet 4.6 进行了广泛的安全评估，发布了完整的 [System Card](https://anthropic.com/claude-sonnet-4-6-system-card)。

### 安全评级

- **ASL-3 rated**（Anthropic 四级安全量表中的第三级，表示"significantly higher risk"但在可控范围内）
- 安全研究员总结：Sonnet 4.6 具有"a broadly warm, honest, prosocial, and at times funny character, very strong safety behaviors, and no signs of major concerns around high-stakes forms of misalignment"

### Prompt Injection Resistance

这是 computer use 场景的关键安全维度。恶意行为者可以在网站上隐藏指令，试图劫持模型（prompt injection attack）。Sonnet 4.6 在这方面**相比 Sonnet 4.5 有重大改善**，表现**接近 Opus 4.6**。

### CyberGym

在网络安全相关的 CyberGym benchmark 上得分 65.2%——这是一个需要关注的双刃剑：更强的能力意味着更大的 dual-use 风险。

### 仍然存在的风险

- Computer use 仍然落后于最熟练的人类用户
- Real-world computer use 比 benchmark 环境更混乱、更模糊，错误的代价更高
- 长 session 中的"context rot"——上下文降解问题
- Dual-use 风险——相同的能力可用于攻击

---

## 竞争格局

### 与主要竞品对比

| 模型 | SWE-bench | OSWorld | ARC-AGI-2 | LMArena Elo | 定价 |
|------|-----------|---------|-----------|-------------|------|
| Claude Sonnet 4.6 | 79.6% | 72.5% | 60.4% | — | $3/$15 |
| Claude Opus 4.6 | 更高 | 72.7% | 更高 | — | $5/$25 |
| GPT-5.2 | 80.0% | 38.2% | 更高（某些版本） | — | 更高 |
| Gemini 3 Pro | — | — | — | 1501 | — |
| Gemini 3 Deep Think | — | — | 更高 | — | — |

**关键竞争动态**：

1. **vs GPT-5.2**：SWE-bench 几乎持平（79.6% vs 80.0%），但 OSWorld computer use 上 Sonnet 4.6 碾压（72.5% vs 38.2%）。据报道 OpenAI 正在赶制 GPT-5.3 Codex 作为回应。

2. **vs Gemini 3 Pro**：Gemini 3 Pro 在 LMArena Elo 上领先（1501），且有 2M context window 优势。但在 agentic planning 上落后。

3. **vs open-source（Llama 4 等）**：Meta 推进 Llama 4 开源路线，但在 agentic 能力上仍有差距。

4. **市场影响**：Anthropic 的 multi-cloud 策略（Bedrock + Vertex AI + Azure 同步上线）和 $30B Series G（$380B 估值）巩固了其市场地位。

---

## 对我们的意义

### 当前状况

我们目前使用 **Opus 4.6** 作为主力模型（见 Runtime 配置中的 `model=anthropic/claude-opus-4-6`）。这是当前最强的 Claude 模型，但也是最贵的。

### Sonnet 4.6 能否作为低成本替代？

**短回答：对大多数任务，是的。**

根据收集的数据，以下是我的分析：

#### ✅ 可以迁移到 Sonnet 4.6 的场景

1. **日常 coding 任务**——Sonnet 4.6 在 Claude Code 中已被 70% 的用户偏好，且 bug detection 能力接近 Opus
2. **文档理解和 Q&A**——OfficeQA 上与 Opus 4.6 持平
3. **Office 自动化任务**——GDPval-AA 上 Sonnet 反而更好（1633 vs 1606）
4. **Financial analysis**——Finance Agent v1.1 上 Sonnet 超过了 Opus
5. **Computer use / browser automation**——OSWorld 差距仅 0.2%
6. **Frontend 开发**——设计感知能力获得独立验证
7. **一般性的 agentic workflow**——在大多数 orchestration eval 上表现优异

#### ❌ 应继续使用 Opus 4.6 的场景

1. **深度代码重构**——需要跨整个代码库的深层理解和改造
2. **Multi-agent 协调**——复杂 workflow 中多个 agent 的编排
3. **极高精确度要求**——"getting it just right is paramount" 的场景
4. **ARC-AGI-2 类型的抽象推理**——Opus 仍有明显优势
5. **最复杂的长链推理任务**——需要 deepest reasoning 的场景

#### 💰 成本节省估算

假设我们当前 70% 的 Opus 调用可以迁移到 Sonnet：

- Input 成本降低：($5 - $3) × 0.7 = **$1.4/M tokens** 节省
- Output 成本降低：($25 - $15) × 0.7 = **$7/M tokens** 节省
- 综合节省率：约 **28-30%** 的总 API 支出

#### 🎯 建议的迁移策略

1. **混合部署**：默认使用 Sonnet 4.6，仅在以下场景自动升级到 Opus 4.6：
   - 检测到 codebase refactoring 类任务
   - Multi-agent orchestration workflow
   - 用户显式要求最高质量
   
2. **A/B 测试期**：在我们的实际工作流中运行 1-2 周，对比：
   - 代码质量
   - 任务完成率
   - 用户满意度
   - 成本

3. **渐进迁移**：
   - Phase 1: 子代理（subagent）全部切换到 Sonnet 4.6（它们通常执行独立、明确的任务）
   - Phase 2: Heartbeat 和 cron job 切换
   - Phase 3: 主 session 根据 A/B 结果决定

### Thinking Effort 优化

Sonnet 4.6 支持 adaptive thinking，可以根据任务复杂度调整推理努力。这意味着简单任务可以用低 effort 快速完成，复杂任务再调高——进一步优化成本和延迟。

---

## 时间线与上下文

| 日期 | 事件 |
|------|------|
| 2024.10 | Anthropic 首次推出 computer use（Claude 3.5 Sonnet） |
| 2025.05 | Claude Sonnet 4 / Opus 4 发布 |
| 2025.08 | Opus 4.1 发布 |
| 2025.09 | **Sonnet 4.5 发布** |
| 2025.10 | Haiku 4.5 发布 |
| 2025.11 | Opus 4.5 发布 |
| 2026.01 | Claude Cowork 发布（research preview） |
| 2026.02.05 | **Opus 4.6 发布**——agent team、PowerPoint 集成 |
| 2026.02.12 | Anthropic 完成 $30B Series G，$380B 估值 |
| 2026.02.17 | **Sonnet 4.6 发布** |
| 2026 Q1/Q2（预期） | Haiku 4.6 |
| 2027（预期） | Claude 5——"emotional intelligence" and superhuman feats（Dario Amodei） |

---

## 技术规格总结

| 规格 | 值 |
|------|-----|
| Model ID | `claude-sonnet-4-6` |
| Context Window | 1,000,000 tokens（beta） |
| Max Output | >128K tokens |
| Knowledge Cutoff | August 2025 |
| Input Pricing | $3 / M tokens |
| Output Pricing | $15 / M tokens |
| Thinking | Adaptive + Extended |
| Safety Rating | ASL-3 |
| Computer Use | ✅（human-level on many tasks） |
| Context Compaction | ✅（beta） |
| 1M Context | ✅（beta） |
| Web Search | ✅（with dynamic filtering） |
| Code Execution | ✅ GA |
| MCP Support | ✅ |

---

## 参考来源

- [Anthropic 官方博客：Introducing Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)
- [Anthropic System Card](https://anthropic.com/claude-sonnet-4-6-system-card)
- [CNBC: Anthropic releases Claude Sonnet 4.6](https://www.cnbc.com/2026/02/17/anthropic-ai-claude-sonnet-4-6-default-free-pro.html)
- [TechCrunch: Anthropic releases Sonnet 4.6](https://techcrunch.com/2026/02/17/anthropic-releases-sonnet-4-6/)
- [Winbuzzer: Claude Sonnet 4.6 with Near-Opus Level Scores](https://winbuzzer.com/2026/02/17/anthropic-claude-sonnet-4-6-flagship-performance-mid-tier-pricing-xcxwbn/)
- [IT Pro: Opus-level reasoning at lower cost](https://www.itpro.com/technology/artificial-intelligence/anthropic-promises-opus-level-reasoning-claude-sonnet-4-6-model-at-lower-cost)
- [AdwaitX: Claude Sonnet 4.6 Features](https://www.adwaitx.com/claude-sonnet-4-6-features/)
- [FinancialContent: The Workhorse AI Model](https://markets.financialcontent.com/stocks/article/tokenring-2026-2-18-anthropic-unleashes-claude-sonnet-46-the-workhorse-ai-model-that-outpaces-flagships-and-ignites-the-agentic-revolution)
- [Wikipedia: Claude (language model)](https://en.wikipedia.org/wiki/Claude_(language_model))

---
> [!warning] 已迁移
> 正式版本位于 [[AI/Frontiers/Claude-Sonnet-4.6]]，本文件为历史遗留，不再维护。

---
> [!warning] 已迁移
> 完整技术规格请见本文件，精炼分析版（竞争影响、批判评估）位于 [[AI/Frontiers/Claude-Sonnet-4.6]]，不再重复维护。本文件保留为技术规格存档。
