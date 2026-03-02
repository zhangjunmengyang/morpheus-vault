---
title: "INBOX"
type: inbox
tags:
  - type/inbox
updated: 2026-02-27
---

# 📥 INBOX — 待处理区

> 新内容落地处。三类内容：待读论文、待炼化笔记、知识缺口。
> 处理原则：读完 → 写独立笔记 → 从这里删掉。不处理不等于没看到。

---

## 待读论文

### Agent 进化模式（2026-02-27 老板指令，高优先级）

> 研究主线：不依赖重训模型、靠 in-context 经验积累实现推理时持续进化（贾维斯正在做的事）。
> 当前 Vault 缺口：训练时进化（ERL/SELAUR/SCoRe 已有）和运行时 prompt 工程（10模式已有），
> 中间层——in-context 经验积累进化范式——完全空白。

- [x] **Reflexion** ✅ 已写精读笔记（Reflexion-Verbal-Reinforcement-Learning.md，NeurIPS 2023，92行）
- [x] **ExpeL** ✅ 已写精读笔记（ExpeL-Experiential-Learning-Agent.md，AAAI 2024，101行）
- [x] **AgentQ** ✅ 已写精读笔记（AgentQ-MCTS-Self-Critique-DPO.md，2024，105行）
- [x] **RAGEN/StarPO** ✅ 已有完整精读笔记（★★★★★，Echo Trap + StarPO-S，RAGEN-StarPO-Multi-Turn-RL-Self-Evolution.md）
- [ ] **EWoK / Self-Taught Reasoner** — 待哨兵 ArXiv 扫描确认论文入口（非馆长任务）

> 产出目标：一篇「Agent 进化模式谱系」整合笔记
> 输出路径：AI/2-Agent/Agentic-RL/Agent-进化模式谱系.md
> 性质：wisdom 层，不是综述——要有判断框架和对贾维斯实践的映射

### LLM / RL

- [x] REINFORCE++ ✅ 已有笔记（REINFORCE-Plus-Plus-Global-Advantage-Normalization.md）
- [x] GRPO / DeepSeekMath ✅ 已有深度笔记（GRPO 深度理解.md，含 arXiv:2402.03300 出处）
- [x] IOPO ✅ 已有笔记（IOPO-Input-Output-Preference-Optimization.md）+ 链接修复完成
- [x] LLMs Get Lost In Multi-Turn Conversation — [arXiv:2505.06120](https://arxiv.org/abs/2505.06120) ✅ 笔记：AI/3-LLM/Evaluation/LLMs-Get-Lost-In-Multi-Turn-Conversation-2505.06120.md
- [x] GRPO 完整流程实践 ✅ 已有（GRPO-verl实践.md / GRPO-Unsloth实践.md / GRPO-TRL实践.md 三份实践笔记）
- [x] 知乎 GRPO 分析 ✅ 内容已被 GRPO深度理解.md 覆盖
- [x] 知乎 GRPO 分析 2 ✅ 内容已被 GRPO深度理解.md 覆盖

### Prompt Engineering

- [x] Chain-of-Thought Prompting ✅ 已覆盖（Prompt-Engineering-基础.md，含原论文引用）
- [x] Self-Consistency ✅ 已覆盖（Prompt-Engineering-基础.md，含 Wang et al. 2022 引用）
- [x] Least-to-Most ✅ 已覆盖（Prompt Engineering 高级.md）
- [x] ReAct ✅ 已有独立笔记（ReAct-推理模式.md + ReAct 与 CoT.md）
- [x] Tree of Thoughts ✅ 已覆盖（Prompt-Engineering-基础.md，含 Yao et al. 2023 引用）
- [x] ART ✅ 与 Reflexion 共用 arXiv:2303.11366，待确认（Reflexion 笔记写完后可合并）

### 论文阅读资源

- 李沐论文精读笔记：https://github.com/Tramac/paper-reading-note
- 选读列表（LLMs）：https://github.com/km1994/llms_paper
- 李沐视频精读：https://github.com/mli/paper-reading
- ML Papers of the Week：https://github.com/dair-ai/ML-Papers-of-the-Week

---

## 知识缺口（待入库）

> 来源：全景类文件扫描后提炼的缺口。有对应全景文件的先从全景中提取，全部覆盖后删全景文件。

### 高优先级（与核心方向直接相关）

- [x] **Agent 进化模式谱系** ✅ 2026-02-27 完成（Agent-进化模式谱系.md，200行，wisdom层，含贾维斯映射+选型决策树）
- [ ] **Intermediate Verification Signal 自动化** — 开放任务 auxiliary reward 自动生成（从“人工 checklist”走向“自动 checklist/约束生成”）
  - ✅ CM2（arXiv:2602.12268）已有独立笔记（17233 bytes，★★★★☆，AI/2-Agent/Agentic-RL/，2026-02-25）
  - ✅ ACE-RL（arXiv:2509.04903）长文生成：instruction → 自动 constraints checklist → verifier → reward → RL（AI/3-LLM/RL/Fundamentals/）
  - ✅ TICK（arXiv:2410.03608）评测：instruction → 自动 YES/NO checklist → 结构化 LLM-as-judge；并用于 STICK self-refine/BoN（AI/3-LLM/Evaluation/）
  - 🧭 路线图（wisdom）：AI/2-Agent/Agentic-RL/Intermediate-Verification-Signal-自动化-路线图.md
  - 真正缺口（仍未解决）：
    - checklist/constraint 的 **coverage 质量控制**（漏项如何发现）
    - verifier 的 **对抗鲁棒性**（防 verifier hacking）
    - 从 checklist 到 reward shaping 的 **理论保证/偏差分析**（避免 correlation trap）
- [x] **预训练数据工程** — ✅ 已存在：（误报缺口，2026-02-27 确认）

### 中优先级

- [x] **Tool Use / Function Calling 深度** ✅ 已有（Agent-Tool-Use.md 295行 + Tool Use.md 156行 + Chrome-DevTools-MCP.md）

### 低优先级（非 AI 核心，按需）

- [ ] **Crypto 量化交易** — 资金费率/波动率/多空策略
- [ ] **AI 搜索与推荐系统** — 向量检索/召回精排
- [ ] **LLM 应用部署与工程化** — vLLM/Triton/监控

---

## 待炼化笔记

> Scholar/哨兵写入但尚未系统炼化的内容。馆长每次心跳 P0 必读此区块。
> Scholar 写完新笔记后，在此追加一行：
> - [ ]  — 一句话描述（写入时间）
> 馆长炼化完成后删除该条目。

_（当前为空）_

---

## 待转「思考」的全景文件

> 这些领域独立深度笔记已充分，全景文件的价值只剩"串联视角"——提炼成一篇能发出去的思考文章后，原文件删除。

- [x] Transformer 架构深度解析（32篇独立笔记覆盖）→ 写「Transformer架构演化的逻辑」✅
- [x] LLM 推理优化全景（36篇覆盖）→ 写「LLM推理优化的本质」✅
- [x] Prompt Engineering 全景（14篇覆盖）→ 写「Prompt工程的边界在哪里」✅
- [x] LLM 微调实战全景（12篇覆盖）→ 写「微调选型决策框架」✅
- [x] 多智能体系统全景（16篇覆盖）→ 写「多Agent协作的核心设计问题」✅
- [x] AI 安全与对齐全景（18篇覆盖）→ 写「对齐问题的本质」✅
- [x] RAG 全景（11篇覆盖）→ 写「RAG的适用边界」✅
- [x] 多模态大模型全景（22篇覆盖）→ 写「多模态融合的几种路线」✅ 2026-02-27
- [x] RLHF/DPO 全景（RL 有大量手撕覆盖）→ 写「从RLHF到DPO的设计演化」✅

---

## 持续追踪（外部系统）

### FARS — Analemma AI 自动科研系统

> 状态：已完成 100 篇（228h），系统仍在运行，人工质量评估进行中。

**笔记路径**：`AI/2-Agent/Fundamentals/FARS-Fully-Automated-Research-System.md`

- [ ] **人工质量评估报告** — 发布后立即读取，关键数据点：AI 评分 vs 人类评分的差距分布，是否有任何论文通过顶会审稿
- [ ] **系统停止后总结报告** — Analemma 团队发布后精读，重点：他们自己对 FARS 局限的分析
- [ ] **GitLab 代码精读** — https://gitlab.com/fars-a，重点：Ideation Agent 的文献调研实现 + Experiment Agent 的 GPU 工具封装方式
- [ ] **OpenFARS 开源版与原版差距** — https://github.com/open-fars/openfars，确认开源版是否包含完整流水线

**关注信号**：analemma.ai 官方 Blog / X (@AnalemmaAI) / 36kr AI 频道
