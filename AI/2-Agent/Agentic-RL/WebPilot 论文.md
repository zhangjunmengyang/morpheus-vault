---
title: "WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration"
brief: "多 Agent 系统用于 web 任务执行，结合 MCTS 战略探索；Planner+Executor 分工架构；在 WebArena/Mind2Web 等 benchmark 上验证自主 web 导航能力"
type: paper
domain: ai/agent/agentic-rl
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/agent/agentic-rl
  - type/paper
---
# WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration

前置：蒙特卡洛树搜索

# 二、multi-agent 系统中应用

论文：WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration

## 2.1、原理

*我们的目标是让基于LLM的网页代理能够在网页环境中通过模拟人类网页导航策略有效地解决任务。网页环境本质上具有部分可观测性，这限制了代理可获取的信息，并使得问题解决变得复杂。这种部分可观测性出现是因为网页内容可以动态变化，意味着**代理无法完全预测或知道某些元素（如更新的内容或可用性）的状态，直到与之交互****。****因此，代理经常需要在不确定性和不完整信息的条件下做出决策。遵循WebArena（Zhou et al. 2023b）的方法，我们使用了一种称为actree的可访问性树来表示观察，它捕捉了网页的结构和交互元素。然而，由于缺乏特定的网页领域知识，LLM通常难以识别或利用各种网页元素的功能。因此，代理必须主动探索环境，收集关于任务和网页元素功能的至关重要的信息，以在这些不确定性和不完整信息的情况下做出明智的决策。*

*具体来说，这个过程可以建模为一个部分可观测马尔可夫决策过程（POMDP）。****环境由状态空间、动作空间和观测空间定义****。*

![image](GF4LddQkso9JogxyqOPcceIgnof.png)

*转移函数指定了状态如何基于采取的动作进行演化，通常是在环境控制下以一种确定性的方式。任务的执行需要代理在每个时间步 (t) 基于部分观测 (o_{t}) 做出决策。每个动作都会导致一个新的状态和更新的观测。评估函数由环境定义，用于评估任务执行的成功程度。在这里*

![image](ZrJDdBtFmocXW7xHUM6cClYdnAh.png)

*表示执行的动作序列*

![image](ZQkndAADPo5STHxLSaWcm987ngb.png)

*表示相应的中间状态序列。该函数评估状态转换是否满足任务 T 设定的标准。*

## 2.2、方法

![image](JHQYd3ZiSoGZSwxw9wzc60uwnKe.png)

*Figure 1: An overview of WebPilot. *

- *GOS: Goal-Oriented Selection;*
- *RENE: Reflection-Enhanced Node Expansion; *
- *DES: Dynamic  Evaluation  and  Simulation;  *
- *MVB:  Maximal  Value  Backpropagation;  *
- *HTD:  Hierarchical  Task  Decomposition;  *
- *RTA:  Reflective Task Adjustmen*
奖励函数在蒙特卡洛树搜索（MCTS）中至关重要，但之前的方法往往依赖于来自环境的直接反馈（Zhou等人2023a），这**对于现实中的网络任务来说是不切实际的**，或者使用二元成功/失败结果或过于简单的中间状态（Koh等人2024）。这些方法经常误判网络环境的模糊和演变性质，导致评估不准确。此外，网络上的中间步骤很难简单地归类为正确或错误，因为它们对最终任务结果的有效贡献可能不会立即显现。受到A*算法（Hart, Nilsson和Raphael 1968）的启发，评估者（Appraiser）评估了执行动作 的效果以及产生观察结果  达到预期目标的潜力，提供了一个更加细致和动态的评估。这种方法通过使用精细的0-10分评分系统进行了优化，允许对动作影响进行更精确的评估，捕捉到不断变化和不确定的网络环境中的细微差别。

“其中 (*) 是基于已执行的动作和当前观察得出的继续原因。如果子任务确认为完成，则搜索终止。否则，DES 在 RENE 的指导下进行一步向前模拟，生成一个模拟反思 ( *)，这作为一次浅层尝试，为下一次探索提供洞见。这一过程使得智能体通过模拟潜在结果来更好地理解不可预测的转换，有助于澄清模糊性和不确定性。通过这些模拟赋予的定量值指导智能体做出更明智的决策，有效地在管理不可预测性和不完整信息的同时，导航最有前景的路径。”

最大值反向传播（Maximal Value Backpropagation，MVB）通过优先考虑最有希望的道路来增强传统的蒙特卡洛树搜索（MCTS）的反向传播步骤。与典型的蒙特卡洛树搜索不同，后者通常平均子节点的值，并可能导致探索较不优的道路，MVB使用所有子节点中的最大值，即  ，其中 Q(s) 代表在状态 s 完成子任务的潜在价值。对于首次访问的状态，Q 值初始化为在DES中评估的总得分 。这种方法在整个决策树中累积价值，始终针对具有长期成功最大潜力的策略。通过关注这些高价值路径，WebPilot 确保与最终目标保持一致，而不仅仅是推进到下一个直接步骤。

---

## See Also

- [[Agentic-RL-2026前沿综合分析|Agentic RL 2026 前沿综合分析]] — WebPilot 在 Web Automation RL 方向的位置
- [[Multi-Agent 概述|Multi-Agent 概述]] — WebPilot 是 Multi-Agent 系统的 Web 特化版
- [[Tool Use|Tool Use]] — Web 导航依赖的 tool use 能力
-  — Agent 知识全图谱
