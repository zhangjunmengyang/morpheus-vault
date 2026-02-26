---
title: "VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use"
brief: "Sea AI：基于 verl 框架的 holistic Agentic RL with Tool Use；统一单步/多步工具调用的 RL 训练流程；工具调用 reward 与任务 reward 联合优化（arXiv:2509.01055）"
type: paper
domain: ai/agent/agentic-rl
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/agent/agentic-rl
  - type/paper
sources:
  - "arXiv:2509.01055 | Sea AI Lab"
  - "https://hf.co/papers/2509.01055"
---
# VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use

by：Sea AI

论文链接：https://hf.co/papers/2509.01055

PaperScope.ai 解读：https://paperscope.ai/hf/2509.01055

具有可验证奖励的强化学习（RLVR）已证明其在增强 LLM 推理能力方面的成功，但仍然局限于没有工具集成的单轮交互。虽然最近出现了用于处理多轮工具交互的智能体强化学习（ARLT）方法，但现有工作开发的是特定任务的代码库，存在碎片化、同步执行瓶颈以及跨领域扩展性有限的问题。

最近的研究通过将工具使用与 RLVR 相结合，开始弥合这一差距，由此产生了一种我们称之为 **ARLT**—**A**gent **R**einforcement **L**earning with **T**ool use 的新范式。在 ARLT 中，LLM 可以积极与外部工具交互，如代码执行环境、搜索引擎、图像操作器和特定领域的 API。这种交互将训练转变为一个多轮、富含反馈的过程，不仅提高了效率并减少了 token 使用，还培养了更稳健的代理行为。

然而，实现 ARLT 从系统角度来看面临着重大挑战。

- 首先，*** 部署效率 *****变得至关重要**：多工具轨迹异步展开，不同工具产生结果的速度各异，需要可扩展的异步执行。
- 其次，* ****工具管理 *****仍然分散**：现有的 RLAT 代码库通常针对特定工具定制，使得扩展或复现结果变得困难。
- 最后，* ****多模态支持 *****仍不成熟**，虽然大多数 RL 框架狭隘地关注文本，但新兴的多模态推理代理需要在统一设计中处理包含图像、视频或其他结构化模态的工具输出。

## 相关

- [[verl 概述|verl 概述]]
- [[GRPO 深度理解|GRPO 深度理解]]
