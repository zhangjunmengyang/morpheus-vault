---
title: "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"
brief: "Oxford+上海AI Lab+NUS：Agentic RL 系统性综述，将 MDP 扩展为 POMDP 多步框架；双维度分类（能力视角：规划/工具/记忆/自我改进）；覆盖 GRPO/ASPO 算法变体、WebEnv/ToolEmu 环境、SWE-Bench/GAIA 基准（arXiv:2509.02547）"
type: paper
domain: ai/agent/agentic-rl
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/agent/agentic-rl
  - type/paper
sources:
  - "arXiv:2509.02547 | Oxford + 上海AI Lab + NUS"
  - "https://hf.co/papers/2509.02547"
---
# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

by：牛津大学、上海人工智能实验室、新加坡国立大学等

论文链接：https://hf.co/papers/2509.02547

PaperScope.ai 解读：https://paperscope.ai/hf/2509.02547

![image](assets/FKKmdD1Mjo6M2CxCI4jczLAFntc.png)

该工作系统性地定义了大语言模型的智能体增强范式（Agentic RL），通过将传统单步马尔可夫决策过程（MDP）扩展为部分可观测的多步POMDP框架，推动LLM从静态文本生成器进化为具备规划、工具使用、记忆、推理等动态决策能力的自主智能体。

研究团队构建了双维度分类体系：从能力视角解析了Agentic RL在规划（包括蒙特卡洛树搜索引导与策略梯度优化）、工具调用（从ReAct式提示到多轮工具集成推理）、记忆管理（检索增强到结构化图记忆）、自我改进（基于DPO的反射机制与自演化课程）等核心模块的优化路径。

论文特别强调RL在解决长时程信用分配（如多轮工具调用的稀疏奖励问题）、构建动态记忆管理系统（如层级化图结构记忆）及实现跨模态主动认知（视觉-语言模型的接地推理）中的关键作用。研究还整合了LLM智能体开发所需的开源环境（WebEnv、ToolEmu等）、RL框架（GRPO、ASPO等算法变体）及评估基准（SWE-Bench、GAIA等），并指出可信赖性、训练规模化与环境复杂度提升是未来核心挑战。

## 相关

- [[AI/LLM/RL/GRPO/GRPO 深度理解|GRPO 深度理解]]
- [[AI/LLM/RL/PPO/PPO 原理|PPO 原理]]
- [[AI/LLM/RL/Fundamentals/RL 概览|RL 概览]]
