---
brief: "豆包大模型 2.0（Doubao-Seed-2.0）——字节跳动大模型旗舰；融合 Seed1.5 多模态能力+高效推理架构；在中文理解/代码/Agent 任务上的能力分析；国内 AI 产品格局中字节的技术底座评估。"
title: 豆包大模型 2.0 (Doubao-Seed-2.0)
aliases:
  - Seed 2.0
  - Doubao 2.0
  - 豆包2.0
company: 字节跳动 (ByteDance)
team: Seed Team
release_date: 2026-02-14
category: LLM / VLM / Agent
variants:
  - Seed2.0 Pro
  - Seed2.0 Lite
  - Seed2.0 Mini
  - Seed2.0 Code
modality: 文字 + 图片 + 视频 → 文字
positioning: 对标 GPT-5.2 / Gemini 3 Pro / Claude Opus 4.5
api_platform: 火山引擎 (Volcano Engine)
tags:
  - AI
  - LLM
  - Agent
  - 多模态
  - 字节跳动
created: 2026-02-18
updated: 2026-02-18
---

# 豆包大模型 2.0 (Doubao-Seed-2.0)

## 概述

2026 年 2 月 14 日，字节跳动正式发布**豆包大模型 2.0**（Doubao-Seed-2.0），宣布豆包全面进入 **Agent 时代**。这是自 2024 年 5 月豆包大模型首发以来的首次跨代升级。

Seed2.0 围绕**大规模生产环境**的真实需求做了系统性优化，三大核心方向：

1. **更稳健的视觉与多模态理解** — 复杂文档、图表、视频解析能力显著提升
2. **更可靠的复杂指令执行** — 多约束、多步骤、长链路任务的理解与执行能力强化
3. **更灵活的推理选择** — Pro/Lite/Mini/Code 四档模型覆盖不同场景

关键成绩：豆包 App 已服务超 **2 亿用户**；LMSYS Chatbot Arena 文本排名第 6、视觉排名第 3；Token 定价比海外同级模型**低约一个数量级**。

---

## 模型家族

四款模型全部支持**文字 + 图片 + 视频输入，文字输出**。

| 维度 | **Pro** | **Lite** | **Mini** | **Code** |
|---|---|---|---|---|
| **定位** | 旗舰全能，深度推理 + 长链路 Agent | 性能成本均衡，通用生产级 | 低时延、高并发、成本敏感 | 编程场景优化 |
| **对标** | GPT-5.2 / Gemini 3 Pro | 超越 Seed 1.8 | — | 配合 TRAE IDE |
| **模型 ID** | `doubao-seed-2-0-pro-260215` | `doubao-seed-2-0-lite-260215` | `doubao-seed-2-0-mini-260215` | `doubao-seed-2-0-code-preview-260215` |
| **输入 (¥/M tok)** | 3.2 | 0.6 | 0.2 | 3.2 |
| **输出 (¥/M tok)** | 16 | 3.6 | 2 | 16 |
| **缓存命中 (¥/M tok)** | 0.64 | 0.12 | 0.04 | 0.64 |
| **上下文窗口** | — | — | 256k, 4 档思考长度 | — |
| **上线渠道** | 豆包 App「专家」模式 / API | API | API | TRAE IDE / API |

---

## 架构亮点与技术创新

### 训练方法论：从真实需求出发

Seed 团队分析了 MaaS 服务的真实调用场景分布（来自火山方舟协作奖励计划），发现：
- **最高比例需求**：处理混杂图表、文档等非结构化信息的知识内容
- **企业典型模式**：先让模型做"读得多、想得多"的任务，再进入复杂专业的流程型工作
- 训练路径高度贴合真实业务反馈，而非单纯追求竞赛分数

### 多模态融合

- 统一的文字 + 图片 + 视频输入架构
- 视觉推理与感知能力系统性强化，降低幻觉风险
- **VideoCut 工具**：扩展长视频处理时长范围，提升推理精度；在 ZeroVideo 任务中准确率几乎翻倍
- 流式实时视频分析能力（从被动问答到主动指导）

### Agent 能力设计

- **从短链路问答 → 长链路智能系统**：核心转型方向
- 系统性补强长尾领域知识（各行业经验往往不在训练语料高频区）
- 强化指令遵循一致性与可控性，为多步骤任务执行奠定基础
- 多工具协同调度框架（搜索 → 归纳 → 结论的连续工作流）

### Code 模型特色

- 分析开发者行为发现：**前端开发是 Agent 编程主战场**
  - JS/TS/CSS/HTML 占据绝对主导，Vue.js 使用率约为 React 的 3 倍
  - Bug 修复与调试类任务排名第一
- 针对性强化：前端语义理解、CSS 布局推理、报错诊断能力
- 增强 Agent 工作流中的纠错能力

### 成本优势

- Token 单价较海外模型低约 **一个数量级**
- Agent 跑一次 workflow 消耗的 token 是人类对话的几十倍，成本变量至关重要
- API 兼容 OpenAI SDK，可通过更换 base_url 直接迁移

---

## 评测成绩

### 核心 Benchmark（Seed2.0 Pro vs 竞品）

#### 科学与推理

| Benchmark | Pro | Lite | Mini | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| MMLU-Pro | 87.0 | 87.7 | 83.6 | 85.9 | 89.3 | 90.1 |
| HLE (text) | 32.4 | 28.2 | 13.3 | 29.9 | 23.7 | 33.3 |
| SuperGPQA | **68.7** | 67.5 | 61.6 | 67.9 | 70.6 | 73.8 |
| HealthBench | **57.7** | 51.2 | 30.0 | 63.3 | 36.3 | 37.9 |
| GPQA Diamond | 88.9 | 85.1 | 79.0 | 92.4 | 86.9 | 91.9 |

#### 数学竞赛

| Benchmark | Pro | Lite | Mini | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| AIME 2025 | **98.3** | 93.0 | 87.0 | 99.0 | 91.3 | 95.0 |
| AIME 2026 | 94.2 | 88.3 | 86.7 | 93.3 | 92.5 | 93.3 |
| HMMT Feb 2025 | 97.3 | 90.0 | 70.0 | 100 | 92.9 | 97.3 |
| IMOAnswerBench | **89.3** | 81.6 | 71.6 | 86.6 | 72.6 | 83.3 |
| MathArenaApex | 20.3 | 4.7 | 4.2 | 18.2 | 1.6 | 24.5 |

#### 代码

| Benchmark | Pro | Lite | Mini | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Codeforces Elo | **3020** | 2233 | 1644 | 3148 | 1701 | 2726 |
| LiveCodeBench v6 | 87.8 | 81.7 | 64.1 | 87.7 | 84.8 | 90.7 |
| SWE-Bench Verified | 76.5 | 73.5 | 67.9 | 80.0 | 80.9 | 76.2 |

#### 多模态视觉

| Benchmark | Pro | Lite | Mini | GPT-5.2 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|
| MathVista | **89.8** | 89.0 | 85.5 | 83.1 | 89.8 |
| MathVision | **88.8** | 86.4 | 78.1 | 86.8 | 86.1 |
| MMMU | 85.4 | 83.7 | 79.7 | 83.7 | 87.0 |
| VLMsAreBlind | **98.6** | 97.0 | 93.1 | 84.2 | 97.5 |
| VLMsAreBiased | **77.4** | 74.8 | 58.4 | 28.0 | 50.6 |

#### 视频理解

| Benchmark | Pro | Lite | Mini | Seed 1.8 | Gemini 3 Pro | 人类 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| VideoMME | **89.5** | 87.7 | 81.2 | 87.8 | 88.4 | — |
| EgoTempo | **71.8** | 61.8 | 67.2 | 67.0 | 65.4 | 63.2 |
| MotionBench | **75.2** | 70.9 | 64.4 | 70.6 | 70.3 | — |
| TVBench | **75.0** | 71.5 | 70.5 | 71.5 | 71.1 | 94.8 |

### 关键里程碑

- 🏅 **IMO / CMO / ICPC 金牌** 成绩
- 📊 **19 项多模态基准中 12 项第一**
- 🧬 **HealthBench 第一名**，SuperGPQA 超过 GPT-5.2
- 🎥 **EgoTempo 超过人类分数**（71.8 vs 63.2）
- 🔍 **HLE-text 最高分 54.2**（使用搜索 Agent）
- 🏗️ **BrowseComp 77.3 / BrowseComp-zh 82.4**

---

## Agent 能力

### 深度研究 (Deep Research)

| Benchmark | Pro | Lite | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|
| DeepConsult | **61.1** | 60.3 | 54.3 | 61.0 | 48.0 |
| DeepResearchBench | 53.3 | **54.4** | 52.2 | 50.6 | 49.6 |
| ResearchRubrics | **50.7** | 50.8 | 42.3 | 45.0 | 37.7 |

### 搜索与工具使用

| Benchmark | Pro | Lite | GPT-5.2 | Claude Opus 4.5 | Gemini 3 Pro |
|---|:---:|:---:|:---:|:---:|:---:|
| BrowseComp | 77.3 | 72.1 | 77.9 | 67.8 | 59.2 |
| BrowseComp-zh | **82.4** | 82.0 | 76.1 | 62.4 | 66.8 |
| HLE-text (Agent) | **54.2** | 49.5 | 45.5 | 43.2 | 46.9 |
| HLE-Verified | 73.6 | 70.7 | 68.5 | 56.6 | 67.5 |
| MCP-Mark | 54.7 | 46.7 | 57.5 | 42.3 | 53.9 |
| τ²-Bench (retail) | **90.4** | 90.9 | 82.0 | 88.9 | 85.3 |

### 科学发现

- 在 **FrontierSci-research** 上表现强劲（25.0），部分场景超越 Gemini 3 Pro（15.0）
- **AInstein Bench** 领先（47.7），展现假设驱动式推理能力
- 能将研究想法推进为**可落地的实验方案**（如高尔基体蛋白分析的完整实验流程）

### 已知局限

官方坦言的结构性限制：
- 超长周期任务中自组织能力仍不足
- 极端专业化领域与真正专家级理解有差距
- 多模态统一表征尚未完全成熟
- 部分高难编码基准与国际领先模型仍有差距
- 缺乏基于物理世界的常识性直觉（"过度推理"问题）

---

## 未来方向

Seed 团队明确的核心发展路线：

1. **能力层**：强化长链路推理，跨阶段任务中形成稳定认知结构，逐步实现经验内化
2. **系统层**：深化 Agent 框架与工具体系融合，构建多工具协同调度机制
3. **数据层**：加大真实行业场景、专业知识体系、高质量长文档数据建设
4. **安全层**：完善对齐机制、风险控制框架和行为评估体系

---

## 参考链接

- 📄 [官方项目主页](https://seed.bytedance.com/zh/seed2)
- 📋 [79 页 Model Card (PDF)](https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/Seed2.0%20Model%20Card.pdf)
- 📝 [官方技术博客](https://seed.bytedance.com/zh/blog/seed2-0-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83)
- 🔥 [火山引擎 API](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-seed-2-0-pro)
- 📰 [腾讯新闻 · 全信息整理](https://news.qq.com/rain/a/20260214A04AJ700)
- 📰 [IT之家报道](https://news.qq.com/rain/a/20260214A03X3W00)
- 📰 [新浪财经](https://finance.sina.com.cn/tech/2026-02-14/doc-inhmuhwr9351190.shtml)
- 📰 [AI 前线深度拆解](https://www.163.com/dy/article/KLO8JFGI05566ZHB.html)
- 📰 [爱范儿实测体验](https://www.ifanr.com/1655221)
- 📰 [TechNode (EN)](https://technode.com/2026/02/14/bytedance-releases-doubao-seed-2-0-positions-pro-model-against-gpt-5-2-and-gemini-3-pro/)
- 📰 [DigitalApplied Benchmark Guide (EN)](https://www.digitalapplied.com/blog/bytedance-seed-2-doubao-ai-model-benchmarks-guide)

## See Also

- [[AI/Frontiers/目录|Frontiers MOC]] — 前沿模型追踪索引
- [[Doubao-Seed-2.0|Doubao Seed 2.0 技术分析]] — 本笔记的深度技术版（馆长炼化版）
- [[AI/目录|AI MOC]] — AI 知识域全索引
- [[AI/3-LLM/目录|LLM MOC]] — LLM 技术基础
- [[GLM-5|GLM-5]] — 同期竞品参考
