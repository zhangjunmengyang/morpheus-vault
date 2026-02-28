---
title: 老板个人简历（完整版）
type: career
date: 2026-02-27
tags: [resume, career, master-copy]
source: 微信文件 木及简历.md
updated: 2026-02-27
---

# 张钧梦阳 - 个人简历

- 出生：2000.01
- 电话：18804466177
- 邮箱：e0983185@u.nus.edu
- GitHub：https://github.com/zhangjunmengyang

---

## 教育背景

**新加坡国立大学** — 人工智能系统硕士 | 2022.08 - 2023.11
**电子科技大学** — 软件工程学士 | 2018.09 - 2022.06

---

## 工作经历

### 美团 - 本地核心商业 | AI 应用算法工程师

#### 整体演进路线

**Workflow 时代**：产物——业务分析小助手
- 走向一个分歧：未来是单 agent 还是 workflow 的路由？
- 当时判断是一定是 agent，agent 是未来，后来落地证明二者不应该互斥：想泛化、灵活一定是 agent，想被业务快速使用、快速落地一定是 workflow
- 23 年 24 年整个行业都在争论这个东西

**Agent 时代**：技术演进路线——单 agent → 多 agent → 单 agent

1. **第一个时代**：简单工具调用，ReAct，没有上下文管理概念，更多关注提示词工程。因 agent 能力不足、无法身兼多职，演进向第二个时代——拆
2. **第二个时代**：解耦成多个 agent，各自独立管理上下文。出现很多架构（supervisor、swarm 等），但多 agent 在于可控性和灵活性的 trade-off：
   - 最高度可控 = 类似 agentic workflow（流程固化）
   - 自由灵活 = planner 下发任务
   - 很多工程优化（观察者等）用于解决特定问题
   - 多 agent 执行效率较低，比较重
3. **第三个时代**：为业务"可用性"再次回到单 agent 形态。未来愿景：一个超级单 agent 有足够强大的上下文处理能力和推理能力来承担复杂任务。Agent as tool 情况下与多 agent 界限逐渐模糊，现在不会特别关注单还是多

#### 取数方向

**NL2SQL（Workflow 方式）**
- 指标维度写死，简易 NL2SQL
- 问题：实现过于简陋，没有 ReAct 能力和 human-in-the-loop 能力，落地用途有限。是很好的 presentation，但不具备可落地和可泛化性

**NL2SQL（Agent 方式）**
- 与 workflow 没有本质区别
- 宏观方向：重心都在取数上，后来逐渐发现取数的"坑"：
  - **用户意图不明确性**（黑话 → 知识库）
  - **SQL 语言不友好性**：SQL 更贴近自然语言/英语母语表达，SQL 本身就是一种自然语言（特殊口径 → few-shot/知识库，复杂计算/函数使用 → few-shot/知识库）。SQL 的语义逻辑、书写顺序、执行顺序都是错位的
    - pySQL
    - NL2Param（现在的方式，依赖起源模型/数据集）
    - 有数据库公司从语法树层开始做更底层方案
  - **最大填坑——数据治理**：
    - 人为错误案例：日期分区 date，表里写了 data，对 AI 是灾难
    - 数据口径：住宿酒旅 DAU 指标上百个，指标名不规范；维度名有问题，维度值一个月一变，不同时间分区内维度枚举值不同
    - 维度不可下钻、指标维度矩阵非对齐，很多不支持交叉的"死点"
  - **性能问题**：
    - 取数：AI 友好的元数据建设（别名、维度枚举值等），各业务推了一波又一波；数仓建设目标是 for AI 的 AI 表，口径不统一，就绪时间不统一，SLA 可能到下午。目前用 Doris 大宽表，但难用
      - Doris 表优化
      - SQL 写法优化（需要 group by 日期分区键）
    - 模型性能：Claude 生成一次 SQL 需两三分钟，有 ReAct 更慢
      - NL2SQL 方案：多模型、异步生成+执行、few-shot，整体优化生成 & 执行两阶段
    - 明细数据自由探查/分析：时间太长，NL2SQL 查不起 Hive 表；面向消费端太慢用户等不了；损失即席数据加工可能性

**核心难点**：取数准确率要求 100%

**两种路线**：
- API：100% 准，远古做法
- NL2Param：折中做法

**Spider 打榜评价**：距离真正可用差得远。数仓日积月累的复杂度远比互联网材料高——信息高度压缩/不可见，信息流通依赖口口相传，大量数据孤岛，走了就没人维护。最后演变成只有几张表可用。

#### 分析方向

**分析 Agent 落地验证**
- 节假日分析：单 agent 挂取数工具，不断取数，写出分析报告
- 酒旅夏战（待补充）

---

### 美团 - 到店住宿业务 | 大数据开发工程师 | 2023.10 - 2024.11

#### 住宿交易实时数仓数据建设
- （关键词、背景、内容待补充）

#### 到店数据资源治理专项
- **关键词**：Flink、Spark、Doris、成本治理、稳定性保障
- **背景**：住宿大数据成本较高，需对实时和离线存算资源进行治理，沉淀治理方法论、SOP 和工具，推全到店及其他 BU
- **内容**：
  - **Flink 实时资源治理**：日常资源治理 + 战役期间稳定性保障。建设压测工具，分场景对存量作业进行资源治理和性能调优。从数仓建设角度优化作业链路
  - **Spark 离线资源治理**：压缩、小文件治理等层面对存量作业治理
  - **Doris 治理工具**：治理前住宿 Doris 单日成本 10k+，战役期间性能差。与多业务线及数平沟通，建设治理工具，打通问题发现和治理动作环节
  - **稳定性保障**：实时作业稳定性要求高，下游数据应用和看板面向分析师、管理者
- **收益**：Flink 节省 xxx vcore/cu，Spark 治理 xxx cu，Doris 缩减 xx BE。整体效果明显，治理工具和 SOP 已推广至到综到餐等其他业务线
- **AI 延伸**：积累 AI 落地思维，住宿 2025 H1 主 R **AI 战役保障工具**，推广住宿内部各战役场景，战役提效 30%，住宿全年战役场景节省 50 PD 以上

---

### 算法实习生 - 新加坡科技研究局（MedImage A3 Lab - Dr. Huang Weimin）| 2023.03 - 2023.08

- 设计多模态风险预测框架，基于 MAE 架构预训练提取图像信息，使用多种特征融合方式结合 Clinic Data 设计多模态风险预测框架
- **成果**：第一作者发表 IEEE JBHI（DOI: [10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502)）

## 论文发表

**Junmengyang Zhang**, Xiaohong Wang, Xuefen Teng, Kok Wei Aik, Larry Natividad, Charmaine Cheng.
"A Multi-Modality Attention Network for Coronary Artery Disease Evaluation From Routine Myocardial Perfusion Imaging and Clinical Data."
*IEEE Journal of Biomedical and Health Informatics (JBHI)*, Vol. 29, Issue 5, pp. 3272–3281, May 2025.
DOI: [10.1109/JBHI.2024.3523476](https://ieeexplore.ieee.org/document/10817502) | PubMed: 40030779

> 第一作者。提出多模态注意力网络（MMAN），融合心肌灌注影像（MPI）与临床数据用于冠状动脉疾病（CAD）诊断。
> 核心模块：图像相关交叉注意力（ICCA）+ 临床数据引导注意力（CDGA）+ MAE 自监督预训练。
> 新加坡科技研究局 MedImage A3 Lab 实习产出，2024-12-27 在线发表。

---

## 技能

### 基础知识
- 熟悉 Java 基础（并发编程、集合、内存区域等），了解 Python、Scala
- 熟悉计算机网络、操作系统、数据库、软件工程、敏捷开发
- 熟悉 Linux 编程及常见命令，了解 Shell 编程。熟悉 Git、Maven 等开发工具

### 实时数仓
- 熟悉 Flink 提交流程、组件通信、任务调度、内存模型等核心机制。熟悉 CheckPoint、Windows、时间等核心概念。熟悉常见 Flink 线上运维及性能调优
- 熟悉 Lambda 数仓架构及架构演进方向，了解流批一体数仓
- 熟悉 Hudi 读写流程、索引机制、MOR/COW 表类型选择策略。了解部分列更新解决方案
- 了解 Spark 常见算子、Shuffle 机制、CheckPoint 机制

### 离线数仓
- 熟悉数仓建模理论、分层模型，了解维度建模、范式建模
- 熟悉 Hadoop 工作原理（NN、2NN、MR 等），HDFS 读写流程
- 了解 HiveSQL，能编写常见查询。了解 HBase 基本原理（flush、compact 等）

### 其他技能
- **钻研能力**：热爱技术，学习力强，近期主要阅读 Flink 1.13 核心源码，持续了解数据湖
- **语言能力**：四级 603 分，六级 471 分，雅思口语单项 6.5 分，全英文环境实习经历
- **产品思维**：独立负责并上线爱奇艺 2021 Q4 智能客服中台 OKR；熟悉推荐、内容审核业务
- **项目推进**：PMP 认证在考，熟悉多方对接和沟通；带领 13 名同学搭建大学生求职就业服务平台（用户 3000+，直播观看 10000+）

---

> **维护说明**：此文件为老板简历的完整存档（master copy），后续根据新信息持续更新。
> 原始文件路径：`/Users/peterzhang/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_f2m38ath9rjk22_82ce/msg/file/2026-02/木及简历.md`
