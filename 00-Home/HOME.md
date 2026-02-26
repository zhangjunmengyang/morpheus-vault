---
title: "HOME"
type: moc
domain: home
tags:
  - type/reference
---

# 🏠 Morpheus Vault

> 个人知识体系中枢 — 按知识主题组织，而非内容类型

## 核心知识域

### 🤖 AI — 人工智能
- [[AI/目录]] — AI 总览
  - [[AI/Foundations/目录]] — 数学基础 / ML / DL
  - [[AI/LLM/目录]] — 大语言模型（核心领域）⭐
    - [[AI/LLM/RL/目录]] — ⭐ 强化学习 for LLM（重点方向）
  - [[AI/MLLM/目录]] — 多模态大模型
  - [[AI/Agent/目录]] — Agent 智能体
  - [[AI/RAG/目录]] — 检索增强生成（RAG）
  - [[AI/Safety/目录]] — AI 安全与对齐
  - [[AI/Frontiers/目录]] — 前沿方向（Embodied AI / World Models）
  - [[AI/CV/目录]] — 计算机视觉

### 🔧 Engineering — 数据工程
- [[Engineering/目录]] — 工程总览（Flink / Spark / Doris）

### 📈 Quant — 量化研究
- Quant/ — Crypto 量化交易 2026 全景（待建 MOC，1篇）

### 🛠️ Tools — 工具与系统
- [[Tools/OpenClaw/目录]] — OpenClaw / MCP / DevTools
- [[Tools/观猹-Watcha-AI产品聚合平台|观猹（Watcha）]] — 中文 AI 产品发现与点评，Product Hunt 平替

### 💼 Career — 职业发展
- [[Career/目录]] — 述职 / 求职 / PMO / 方法论

### 🚀 Projects — 项目实战
- [[Projects/目录]] — 企业级项目 + 实验

### 📚 Resources — 学习资源
- [[Resources/目录]] — 课程索引 / 论文列表

### 📰 Newsloom — 每日情报
- Newsloom/ — Sentinel 每日 AI 情报归档（无 MOC，按日期检索）

## 工作区入口

- [[00-Home/Inbox|📥 Inbox]] — 新内容临时收集箱，整理后归入知识域

## 快速查询

```dataview
TABLE type, domain, file.mtime as "更新时间"
FROM ""
WHERE type = "paper"
SORT file.mtime DESC
LIMIT 10
```

## 最近更新

```dataview
TABLE type, domain
FROM "" AND -"Templates" AND -"00-Home"
SORT file.mtime DESC
LIMIT 15
```
