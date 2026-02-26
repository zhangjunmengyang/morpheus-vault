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
  - [[AI/1-Foundations/目录]] — 数学基础 / ML / DL
  - [[AI/3-LLM/目录]] — 大语言模型（核心领域）⭐
    - [[AI/3-LLM/RL/目录]] — ⭐ 强化学习 for LLM（重点方向）
  - [[AI/3-LLM/MLLM/目录]] — 多模态大模型
  - [[AI/2-Agent/目录]] — Agent 智能体
  - [[AI/6-应用/RAG/_MOC]] — 检索增强生成（RAG）
  - [[AI/5-AI 安全/目录]] — AI 安全与对齐
  - [[AI/Frontiers/目录]] — 前沿方向（Embodied AI / World Models）
  - [[AI/CV/_MOC]] — 计算机视觉

### 🔧 Engineering — 数据工程
- [[Career/数据工程/目录]] — 工程总览（Flink / Spark / Doris）

### 📈 Quant — 量化研究
- Quant/ — Crypto 量化交易 2026 全景（待建 MOC，1篇）

### ✍️ Output — 输出与发布
- [[思考/目录]] — 发布文章 & 社区分享

### 🔍 外部资源
- [[观猹-Watcha-AI产品聚合平台|观猹（Watcha）]] — 中文 AI 产品发现与点评，Product Hunt 平替

### 💼 Career — 职业发展
- [[Career/目录]] — 述职 / 求职 / PMO / 方法论

### 🚀 Projects — 项目实战
- [[Projects/0-目录]] — 企业级项目 + 实验

### 📚 Resources — 学习资源
- [[Resources/0-目录]] — 课程索引 / 论文列表

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


# 论文待学习列表

李沐：https://github.com/Tramac/paper-reading-note?tab=readme-ov-file

选读列表：https://github.com/km1994/llms_paper

https://github.com/mli/paper-reading

仓库 top papers：https://github.com/dair-ai/ML-Papers-of-the-Week?tab=readme-ov-file

LLM 思想

- [LLMs Get Lost In Multi-Turn Conversation](https%3A%2F%2Farxiv.org%2Fabs%2F2505.06120)
 RL

- REINFORCE++： [REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models](https%3A%2F%2Farxiv.org%2Fabs%2F2501.03262)
- GPRO（deepseekmath）：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https%3A%2F%2Farxiv.org%2Fabs%2F2402.03300)
- [zhuanlan.zhihu.com](https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F20021693569)
- https://zhuanlan.zhihu.com/p/21046265072
- https://swift.readthedocs.io/zh-cn/latest/BestPractices/GRPO%E5%AE%8C%E6%95%B4%E6%B5%81%E7%A8%8B.html
- [IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimizatio](https%3A%2F%2Farxiv.org%2Fabs%2F2411.06208)
Prompt
*https://arxiv.org/abs/2201.11903*
*https://arxiv.org/abs/2205.11916*
*https://arxiv.org/abs/2203.11171*
*https://arxiv.org/abs/2210.03493*
*https://arxiv.org/abs/2305.10601*
*https://arxiv.org/pdf/2303.11366.pdf*