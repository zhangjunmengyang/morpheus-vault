---
title: 张钧梦阳简历
type: career
date: 2026-02-28
updated: 2026-02-28
tags: [resume, career]
brief: 张钧梦阳个人简历，唯一维护版本，随新项目/经历持续打磨，可直接复制投递。
---

# 张钧梦阳

- **电话**：18804466177
- **邮箱**：e0983185@u.nus.edu
- **GitHub**：[github.com/zhangjunmengyang](https://github.com/zhangjunmengyang)

---

## 教育背景

**新加坡国立大学（NUS）** — 人工智能系统硕士 | 2022.08 - 2023.11

**电子科技大学（UESTC）** — 软件工程学士 | 2018.09 - 2022.06

---

## 论文发表

**Junmengyang Zhang**\*, Xiaohong Wang\*, Xuefen Teng, Kok Wei Aik, Larry Natividad, Charmaine Cheng.
"A Multi-Modality Attention Network for Coronary Artery Disease Evaluation From Routine Myocardial Perfusion Imaging and Clinical Data."
*IEEE Journal of Biomedical and Health Informatics (JBHI)*, Vol. 29, Issue 5, pp. 3272–3281, May 2025.
[DOI: 10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502)（\* 并列第一作者）

---

## 工作经历

### 美团 · 本地核心商业 | AI 应用算法工程师 | 2024.11 - 至今

**取数 Agent**

- 主导 NL2SQL / NL2Param 取数 Agent 建设，覆盖住宿酒旅业务核心指标体系
- 深度解决三类落地难题：用户意图不明确（业务黑话知识库）、数据口径不一致（元数据治理 + 指标枚举值规范化）、查询性能（Doris 分区键 SQL 优化 + 异步并发取数）
- 从直接生成 SQL 演进至 NL2Param 方案，准确率从 ~40% 提升至 80%+

**分析 Agent**

- 落地节假日分析、酒旅夏战等场景，单 Agent 挂多工具，全流程自动产出结构化分析报告
- 解决分析延迟（流式输出 + 预计算缓存），端到端响应从 7-8 分钟降至 30 秒内
- 设计并落地 Prompt Injection 多层防御（输入过滤 → Instruction Hierarchy → 行为审计）

**Agent 架构**

- 深度参与 Agent 架构演进全过程：单 Agent → 多 Agent（supervisor/swarm）→ 回归超级单 Agent
- 形成系统性判断：高泛化选 Agent，快落地选 Workflow，两者不互斥

---

### 美团 · 到店住宿业务 | 大数据开发工程师 | 2023.10 - 2024.11

**住宿交易实时数仓建设**

- 建设住宿核心交易链路实时数仓，保障关键数据的时效性与稳定性

**到店数据资源治理专项**

- 主导住宿实时（Flink）+ 离线（Spark）+ OLAP（Doris）全栈资源治理，沉淀方法论、SOP 和工具，推广至到综到餐等其他 BU
- 建设 Flink 压测工具，分场景对存量作业进行资源治理与性能调优；治理前住宿 Doris 单日成本 10k+，治理后显著缩减
- 住宿 2025 H1 主 R AI 战役保障工具，推广内部各战役场景，战役提效 30%，全年节省 50 PD+

---

### 算法实习生 · 新加坡科技研究局（A\*STAR MedImage A3 Lab）| 2023.03 - 2023.08

- 设计多模态心脏病风险预测框架：基于 MAE 架构预训练提取心肌灌注影像特征，结合临床数据设计多模态融合网络（MMAN），用于冠状动脉疾病（CAD）自动诊断
- **并列第一作者**发表 IEEE JBHI（[DOI: 10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502)）

---

### 产品经理实习 · 爱奇艺 | 2021.09 - 2022.01

- 负责视频审核 / 内容安全方向，独立上线智能客服中台 OKR

---

### 算法实习 · 四川省网络空间与数据安全重点实验室 | 2020.12 - 2021.02

- 广告 / 推荐数据清洗与特征工程

---

## 技能

**AI / Agent**：AI Agent 架构（单 / 多 / supervisor / swarm）���NL2SQL、NL2Param、Prompt 工程、ReAct、多模态深度学习

**后训练**：SFT（全参 / LoRA / QLoRA）、DPO、GRPO、PPO、verl + Ray + vLLM 分布式训练框架

**分布式训练**：数据并行（ZeRO-1/2/3）、张量并行（TP）、流水线并行（PP）、MoE 训练

**大数据**：Flink（源码级）、Spark、Doris、Hudi、Hadoop / HDFS / Hive / HBase、Lambda 架构

**编程**：Python、Java（并发 / 集合 / 内存）、Scala、Shell、Git

**语言**：CET-4 603 / CET-6 471 / 雅思口语 6.5，全英文环境实习经历
