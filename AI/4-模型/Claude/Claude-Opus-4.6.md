---
brief: "Claude Opus 4.6——Anthropic 最 Agentic 的旗舰模型；超大 context window + RL 对齐的 Agent 能力；首次证明 AI Agent 可在高复杂度代码任务（ARC-AGI-2）超越人类；★★★★★，标志性 Agent 能力里程碑。"
title: "Claude Opus 4.6 — Anthropic 最 Agentic 的 Frontier 模型"
date: 2026-02-05
source: "https://philippdubach.com/posts/claude-opus-4.6-anthropics-new-flagship-ai-model-for-agentic-coding/"
tags: [anthropic, claude, frontier-model, agentic, context-window, RL, agent-teams]
rating: ★★★★★
---

# Claude Opus 4.6 — Anthropic 最 Agentic 的 Frontier 模型

**发布时间**: 2026-02-05  
**价格**: $5/$25 per million tokens（input/output）  
**Context Window**: 1M tokens（Opus 系列首次）  
**Max Output**: 128K tokens

## 核心评价

这不是 Opus 4.5 的小幅升级。三件事情的组合彻底改变了 agent 工作流的边界：

1. **ARC-AGI-2 68.8%** — 从 37.6% 跳到 68.8%，超越 GPT-5.2 (54.2%) 和 Gemini 3 Pro (45.1%) 超过 14-23 个点。ARC-AGI-2 专门抵抗记忆，测试的是真实泛化能力，这个分数有真实意义。
2. **MRCR v2 76%** — 长上下文检索能力对比 Sonnet 4.5 的 18.5%，不是增量改进，是能力级别的跃升。1M context 在这个基础上才有实用价值。
3. **Agent Teams + Context Compaction** — 技术上解决了长任务的两大瓶颈：并发和上下文耗尽。

## Benchmark 全景

| 任务类型 | Benchmark | Opus 4.6 | 对比 |
|---|---|---|---|
| 抽象推理 | ARC-AGI-2 | **68.8%** | GPT-5.2: 54.2%, Gemini 3 Pro: 45.1%, Opus 4.5: 37.6% |
| 专业知识 | GDPval-AA Elo | **SOTA** | 超 GPT-5.2 ~144 Elo（44 种职业知识工作评估）|
| 长上下文检索 | MRCR v2 | **76%** | Sonnet 4.5: 18.5% |
| Agent 终端编码 | Terminal-Bench 2.0 | **65.4%** | Opus 4.5: 59.8% |
| 计算机操控 | OSWorld | **72.7%** | Opus 4.5: 66.3%，领先 GPT-5.2 和 Gemini 3 Pro |
| 代码修复 | SWE-bench Verified | ~80.9% | 轻微倒退（vs Opus 4.5），但代表更真实工作的 Terminal-Bench 上升 |
| Agent 网络搜索 | BrowseComp | **SOTA** | — |

## 技术创新

### 1. 1M Token Context Window + Context Compaction
- 1M tokens ≈ 750 本小说，或整个企业代码库数千个文件
- **Context Compaction**：接近上限时自动 summarize 旧上下文 → agent 可理论上无限长运行
- **实际注意点**：
  - prefill latency 在 1M token 下可超过 2 分钟
  - 超 200K token 的 prompt 需要付更高价格（$10/$37.50）
  - 1M context vs RAG pipeline：取决于任务，不是无条件胜出

### 2. Agent Teams（Claude Code 研究预览）
- 多个 subagent 并行协调，自主完成大型代码库审查等 read-heavy 任务
- 每个 agent 通过 **git worktrees** 处理不同分支，最后 merge
- 这是 PARL（Kimi K2.5）、GitHub Agentic Workflows 同一趋势的 Anthropic 实现

### 3. Adaptive Thinking
- 模型根据上下文语境动态决定"思考深度"
- Effort controls（low/medium/high/max）让开发者精细控制智能/速度/成本 tradeoff

## 与其他同期模型的对比（2026-02 Frontier 格局）

| 模型 | ARC-AGI-2 | OSWorld | SWE-bench |
|---|---|---|---|
| Claude Opus 4.6 | **68.8%** | **72.7%** | ~80.9% |
| GPT-5.2 | 54.2% | — | — |
| Gemini 3 Pro | 45.1% | — | — |
| Claude Sonnet 4.6 | — | 72.5% | 79.6% |
| GLM-5 | — | — | 77.8% |

> Sonnet 4.6（发布于 2/17）在 agent 任务上几乎追平 Opus 4.6，但价格是 Opus 的 1/5。这说明 Anthropic 的模型系列正在发生定价重构。

## 我的判断

**ARC-AGI-2 这个分数意味着什么**：
ARC-AGI 系列设计原则是"人类一眼能解决，AI 不能靠记忆作弊"。从 37.6% → 68.8% 不是训练数据更多，是 test-time compute 或 post-training 方法的突破。具体技术细节 Anthropic 未公开，但 adaptive thinking + effort control 的组合暗示 TTC scaling 在 Opus 4.6 中有实质性提升。

**MRCR v2 76% vs 18.5% 的含义**：
这是长上下文能力的真正分水岭。1M context window 对很多模型来说只是"支持"，但不能"利用"——信息在上下文深处就丢失了。Opus 4.6 的 76% 表明它能在 massive prompts 里找到并推理特定事实。这才是企业级 agent（代码库分析/法律文档/医疗记录）场景的核心能力。

**Agent Teams 的工程意义**：
git worktrees 并行 + 自动 merge 是软件工程场景的第一个真实可用的多 agent 架构。比起学术论文里的 PARL / MARL，这是直接在 Claude Code 用户面前可用的产品能力。

**潜在 concern**：
- SWE-bench 轻微倒退值得关注 — 极端优化 ARC-AGI-2 等推理 benchmark 是否以牺牲某些代码生成能力为代价？
- 1M context 实用性存疑：2 分钟 prefill 延迟 + 高价 对于追求吞吐的 agent 场景是真实摩擦。Context Compaction 的质量（summary 是否保留关键信息）是关键变量。

## 关联

- [[AI/4-模型/Claude/Claude-Sonnet-4.6]] — 发布于 2/17，在 agent 任务上追平 Opus 4.6，但 1/5 价格
- [[AI/2-Agent/Multi-Agent/Kimi-K2.5-PARL]] — 同期 Agent Swarm，PARL 学术路线 vs Anthropic Agent Teams 产品路线
- [[AI/2-Agent/Agentic-RL/EnterpriseGym-Corecraft]] — Corecraft benchmark 中 Opus 4.6 < 30% pass rate，说明当前 frontier 在真实企业 agent 任务上仍远未饱和
- [[Gemini-3-Deep-Think]] — Gemini 3 Pro ARC-AGI-2 45.1% vs Opus 4.6 68.8%，差距显著
- [[AI/4-模型/Gemini/Gemini-3.1-Pro|Gemini 3.1 Pro]] — ARC-AGI-2 77.1% 反超 Opus 4.6（68.8%），格局逆转，2026-02-19 发布

## 备注

EnterpriseGym Corecraft (arXiv 2602.16179) 将 Opus 4.6 作为基准之一：在高保真企业 agent 模拟中，frontier 模型（GPT-5.2 和 Opus 4.6）仍只能解决 <30% 任务。这是 2026 年 agentic RL 研究的核心动力。
