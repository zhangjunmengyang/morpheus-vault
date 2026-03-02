---
title: 张钧梦阳简历
type: career
date: 2026-02-28
updated: 2026-02-28
tags: [resume, career]
brief: 张钧梦阳个人简历，唯一维护版本，面向 AI 应用算法工程师岗位，可直接复制投递。
---

# 张钧梦阳

📱 18804466177 ｜ 📮 e0983185@u.nus.edu ｜ 🐙 [github.com/zhangjunmengyang](https://github.com/zhangjunmengyang)

---

## 教育背景

**新加坡国立大学（NUS）** — 人工智能系统 硕士 ｜ 2022.08 – 2023.11

**电子科技大学（UESTC）** — 软件工程 学士 ｜ 2018.09 – 2022.06

---

## 工作经历

### 美团 · 本地核心商业 ｜ AI 应用算法工程师 ｜ 2024.11 – 至今

**取数 Agent（NL2SQL / NL2Param）**

- 主导住宿酒旅业务取数 Agent 建设，构建业务黑话知识库 + 元数据治理 + 指标枚举值规范化体系，解决用户意图模糊与数据口径不一致两大落地瓶颈
- 将方案从直接生成 SQL 演进至 NL2Param 架构，查询准确率从 **~40% 提升至 80%+**
- 引入 Doris 分区键 SQL 优化 + 异步并发取数，查询性能显著提升

**分析 Agent（BA Agent）**

- 主导节假日复盘、酒旅夏战等场景的 Multi-Agent 分析系统：三层架构（主 Planner → 分析子 Agent → 报告生成 Agent），逐章节生成防幻觉
- 攻克上下文崩溃：外循环每章清空历史、内循环过滤失败消息，System Prompt 从 **3w token 压缩至 1.5w**；引入 DAG 并行调度替换串行，报告耗时从 **2h+ 压缩至 30 秒内**
- 主导垂类分析模型后训练全链路（SFT + AgentRL），节假日评测集从 **55 分提升至 74 分**（商业对标 QwQ-32B 76 分）；训练数据从人工 9 条迭代至真实 Agent 日志蒸馏（v0.1.0 → v0.3.0）
- 设计三类自动化评测方案（封闭式 100% / 半封闭式 95% / 开放主观题 80%+），整体评测提效 **40%**
- 酒旅夏战报告已投产，策略知识注入后业务洞察准确率达 **80%**

**Agent 架构与安全**

- 深度参与架构演进全过程（单 Agent → Multi-Agent Planner/Supervisor/Swarm → 超级单 Agent），形成判断框架：高泛化选 Agent，快落地选 Workflow
- 设计并落地 Prompt Injection 多层防御（输入过滤 → Instruction Hierarchy → 行为审计）

---

### 美团 · 到店住宿业务 ｜ 大数据开发工程师 ｜ 2023.10 – 2024.11

- 建设住宿核心交易链路实时数仓；主导 Flink + Spark + Doris 全栈资源治理并推广至多 BU
- 住宿 2025 H1 主 R **AI 战役保障工具**，战役提效 **30%**，全年节省 **50 PD+**

---

### 算法实习生 ｜ 新加坡科技研究局 A\*STAR MedImage A3 Lab ｜ 2023.03 – 2023.08

- 设计多模态心脏病风险预测框架（MAE 预训练 + MMAN 多模态融合网络），用于冠状动脉疾病自动诊断
- **并列第一作者**发表 IEEE JBHI（[DOI: 10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502)）

---

### 产品经理实习 ｜ 爱奇艺 ｜ 2021.09 – 2022.01

- 负责视频审核 / 内容安全方向，独立上线智能客服中台 OKR

---

## 个人项目

**多 Agent 自进化系统**（个人项目，基于 OpenClaw 运行时）

- 设计并运行 6-Agent 长期协作系统（总协调 + 5 个专职 Agent），实验横向信息流、故障驱动进化、跨 Agent 记忆继承等机制
- 量化结论：公告板协作机制将 Agent 间信息同步延迟从 ∞ 降至 **<1h**，跨 Agent 协作任务完成率从 **12% 提升至 73%**

---

## 技能

| 方向 | 技能 |
|------|------|
| **AI / Agent** | Agent 架构设计（单 / 多 / Supervisor / Swarm）、NL2SQL、NL2Param、Prompt 工程、ReAct、工具调用、上下文管理 |
| **后训练** | SFT（全参 / LoRA / QLoRA）、DPO、GRPO、PPO、verl + Ray + vLLM 分布式训练框架 |
| **分布式训练** | 数据并行（ZeRO-1/2/3）、张量并行（TP）、流水线并行（PP）、MoE 训练 |
| **大数据** | Flink、Spark、Doris、Kafka、Hive |
| **编程** | Python、Java、Scala、Shell、Git |
| **语言** | 英语（CET-4 603 / CET-6 471 / 雅思口语 6.5，全英文环境实习经历） |

---

## 论文发表

**Junmengyang Zhang**\*, Xiaohong Wang\*, Xuefen Teng, Kok Wei Aik, Larry Natividad, Charmaine Cheng.
"A Multi-Modality Attention Network for Coronary Artery Disease Evaluation From Routine Myocardial Perfusion Imaging and Clinical Data."
*IEEE Journal of Biomedical and Health Informatics (JBHI)*, Vol. 29, Issue 5, pp. 3272–3281, May 2025.
[DOI: 10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502)（\* 并列第一作者）
