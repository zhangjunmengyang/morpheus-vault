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
- [[AI/_MOC]] — AI 总览
  - [[AI/Foundations/_MOC]] — 数学基础 / ML / DL
  - [[AI/LLM/_MOC]] — 大语言模型（核心领域）
    - [[AI/LLM/RL/_MOC]] — ⭐ 强化学习 for LLM（重点方向）
  - [[AI/MLLM/_MOC]] — 多模态大模型
  - [[AI/Agent/_MOC]] — Agent 智能体
  - [[AI/CV/_MOC]] — 计算机视觉

### 🔧 Engineering — 数据工程
- [[Engineering/_MOC]] — 工程总览
  - Flink / Spark / Doris

### 💼 Career — 职业发展
- [[Career/_MOC]] — 述职 / PMO / 方法论

### 🚀 Projects — 项目实战
- [[Projects/_MOC]] — 企业级项目 + 实验

### 📚 Resources — 学习资源
- [[Resources/_MOC]] — 课程索引 / 论文列表

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
